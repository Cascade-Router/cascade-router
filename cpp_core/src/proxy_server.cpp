/**

 * cascade-router HTTP proxy — OpenAI-compatible routing endpoint.

 *

 * Tokenizes prompts, runs ONNX embedding inference, applies logistic

 * regression routing weights, mutates the upstream JSON model field, and forwards
 * the request to OpenAI when OpenSSL is available (Docker/Linux).

 *

 * Usage:

 *   proxy_server [path/to/model.onnx]

 */



#ifdef _WIN32

#ifndef _WIN32_WINNT

#define _WIN32_WINNT 0x0A00

#endif

#endif

// CPPHTTPLIB_OPENSSL_SUPPORT is set by CMake when OpenSSL is linked (Docker/Linux).

#include "tokenizer.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <simdjson.h>



#include <array>

#include <chrono>

#include <cmath>

#include <cstdint>

#include <cctype>

#include <ctime>

#include <filesystem>

#include <fstream>

#include <iomanip>

#include <iostream>

#include <memory>

#include <mutex>

#include <sstream>

#include <string>

#include <string_view>

#include <thread>

#include <vector>



namespace {



constexpr int kListenPort = 8000;

constexpr int kBatchSize = 1;

constexpr int kRoutingSeqLen = 16;

constexpr int kEmbeddingDim = 384;

constexpr int kFeatureDim = 2 + kEmbeddingDim;



constexpr const char* kDefaultModelPath = "../models/all-MiniLM-L6-v2-int8.onnx";

constexpr const char* kDefaultVocabPath = "../models/vocab.txt";

constexpr const char* kDefaultWeightsPath = "../models/router_weights.json";

constexpr const char* kInputIdsName = "input_ids";

constexpr const char* kAttentionMaskName = "attention_mask";
constexpr const char* kTokenTypeIdsName = "token_type_ids";
constexpr const char* kSmallModel = "gpt-4o-mini";
constexpr const char* kLargeModel = "gpt-4o";
constexpr const char* kTrafficLogPath = "../logs/cascade_traffic.log";



struct TokenUsage {
    int input_tokens = 0;
    int output_tokens = 0;
};



struct RouterWeights {

    std::vector<float> weights;

    float intercept = 0.0f;

};



#ifdef _WIN32

std::wstring to_wide(std::string_view path) {

    std::wstring wide;

    wide.reserve(path.size());

    for (char ch : path) {

        wide.push_back(static_cast<wchar_t>(ch));

    }

    return wide;

}

#endif



std::string json_escape(std::string_view text) {

    std::string escaped;

    escaped.reserve(text.size());

    for (char ch : text) {

        switch (ch) {

            case '"':

                escaped += "\\\"";

                break;

            case '\\':

                escaped += "\\\\";

                break;

            case '\n':

                escaped += "\\n";

                break;

            case '\r':

                escaped += "\\r";

                break;

            case '\t':

                escaped += "\\t";

                break;

            default:

                escaped += ch;

                break;

        }

    }

    return escaped;

}



std::string select_target_model(float p_success) {
    return p_success >= 0.5f ? kSmallModel : kLargeModel;
}

std::string rewrite_model_in_payload(
    const std::string& raw_body,
    const std::string& target_model,
    std::string& error) {
    try {
        nlohmann::json json_body = nlohmann::json::parse(raw_body);
        if (!json_body.is_object()) {
            error = "Request body must be a JSON object";
            return {};
        }

        // Force the routed model before upstream forwarding (OpenAI requires this field).
        json_body["model"] = target_model;
        return json_body.dump();
    } catch (const nlohmann::json::exception& ex) {
        error = ex.what();
        return {};
    }
}

bool header_equals_ignore_case(std::string_view lhs, std::string_view rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

bool should_strip_upstream_header(std::string_view name) {
    return header_equals_ignore_case(name, "Content-Length") ||
           header_equals_ignore_case(name, "Host") ||
           header_equals_ignore_case(name, "Content-Type") ||
           header_equals_ignore_case(name, "Accept-Encoding") ||
           header_equals_ignore_case(name, "Transfer-Encoding");
}

httplib::Headers build_upstream_headers(const httplib::Request& req) {
    httplib::Headers upstream_headers;
    for (const auto& header : req.headers) {
        if (should_strip_upstream_header(header.first)) {
            continue;
        }
        upstream_headers.emplace(header.first, header.second);
    }
    return upstream_headers;
}

std::string format_latency_header(double latency_ms) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << latency_ms;
    return out.str();
}

std::string utc_timestamp_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void ensure_traffic_log_directory() {
    const std::filesystem::path log_path(kTrafficLogPath);
    if (log_path.has_parent_path()) {
        std::filesystem::create_directories(log_path.parent_path());
    }
}

TokenUsage extract_usage_from_response(const std::string& body) {
    TokenUsage usage;
    try {
        const auto parsed = nlohmann::json::parse(body);
        if (parsed.contains("usage") && parsed["usage"].is_object()) {
            usage.input_tokens = parsed["usage"].value("prompt_tokens", 0);
            usage.output_tokens = parsed["usage"].value("completion_tokens", 0);
        }
    } catch (const std::exception&) {
        // Best-effort telemetry only.
    }
    return usage;
}

int estimate_input_tokens_from_request(const std::string& body) {
    try {
        const auto parsed = nlohmann::json::parse(body);
        int total_chars = 0;
        if (parsed.contains("messages") && parsed["messages"].is_array()) {
            for (const auto& message : parsed["messages"]) {
                if (message.contains("content") && message["content"].is_string()) {
                    total_chars += static_cast<int>(
                        message["content"].get<std::string>().size());
                }
            }
        }
        return std::max(1, total_chars / 4);
    } catch (const std::exception&) {
        return 0;
    }
}

void append_traffic_log_async(
    const std::string& model_routed,
    int input_tokens,
    int output_tokens,
    double routing_latency_ms) {
    const double rounded_latency =
        std::round(routing_latency_ms * 10.0) / 10.0;
    const std::string line = nlohmann::json{
        {"timestamp", utc_timestamp_iso8601()},
        {"model_routed", model_routed},
        {"input_tokens", input_tokens},
        {"output_tokens", output_tokens},
        {"routing_latency_ms", rounded_latency},
    }.dump();

    std::thread([line]() {
        try {
            ensure_traffic_log_directory();
            std::ofstream out(kTrafficLogPath, std::ios::app);
            if (out) {
                out << line << '\n';
            }
        } catch (const std::exception&) {
            // Telemetry must never break the proxy hot path.
        }
    }).detach();
}


bool is_punctuation_char(char c) {

    return c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?';

}



bool is_bracket_char(char c) {

    return c == '[' || c == ']' || c == '{' || c == '}' || c == '(' || c == ')' ||

           c == '<' || c == '>' || c == '`';

}



bool is_code_hint_char(char c) {

    return c == '\\' || c == '|' || c == '/' || c == '@' || c == '#' || c == '$' ||

           c == '%' || c == '^' || c == '&' || c == '*' || c == '~' || c == '_' ||

           c == '+' || c == '=';

}



/** Mirrors src/extract_features.py structural_complexity heuristic. */

float calculate_structural_complexity(const std::string& text) {

    if (text.empty()) {

        return 0.0f;

    }



    int hits = 0;

    for (char c : text) {

        if (is_punctuation_char(c) || is_bracket_char(c) || c == '\n' || c == '\t' ||

            is_code_hint_char(c)) {

            ++hits;

        }

    }



    const float score = static_cast<float>(hits) / static_cast<float>(text.size());

    return std::round(score * 1'000'000.0f) / 1'000'000.0f;

}



int count_non_pad_tokens(const std::vector<int64_t>& input_ids) {

    int count = 0;

    for (int64_t id : input_ids) {

        if (id != BertWordPieceTokenizer::kPadId) {

            ++count;

        }

    }

    return count;

}



RouterWeights load_router_weights(const std::string& path) {

    simdjson::dom::parser parser;

    simdjson::padded_string json = simdjson::padded_string::load(path);

    simdjson::dom::element doc = parser.parse(json);



    RouterWeights router;

    for (double value : doc["weights"]) {

        router.weights.push_back(static_cast<float>(value));

    }

    router.intercept = static_cast<float>(double(doc["intercept"]));



    if (static_cast<int>(router.weights.size()) != kFeatureDim) {

        throw std::runtime_error(

            "Expected " + std::to_string(kFeatureDim) + " weights, got " +

            std::to_string(router.weights.size()));

    }



    return router;

}



std::vector<float> mean_pool_and_normalize(

    const float* hidden_state,

    int seq_len,

    const std::vector<int64_t>& attention_mask) {

    std::vector<float> embedding(static_cast<std::size_t>(kEmbeddingDim), 0.0f);

    int valid_tokens = 0;



    for (int t = 0; t < seq_len; ++t) {

        if (attention_mask[static_cast<std::size_t>(t)] == 0) {

            continue;

        }

        ++valid_tokens;

        const float* token_vec = hidden_state + static_cast<std::size_t>(t * kEmbeddingDim);

        for (int d = 0; d < kEmbeddingDim; ++d) {

            embedding[static_cast<std::size_t>(d)] += token_vec[d];

        }

    }



    if (valid_tokens > 0) {

        const float denom = static_cast<float>(valid_tokens);

        for (float& value : embedding) {

            value /= denom;

        }

    }



    float norm = 0.0f;

    for (float value : embedding) {

        norm += value * value;

    }

    norm = std::sqrt(norm);

    if (norm > 1e-12f) {

        for (float& value : embedding) {

            value /= norm;

        }

    }



    return embedding;

}



float sigmoid(float z) {

    return 1.0f / (1.0f + std::exp(-z));

}



float predict_pass_probability(

    const std::vector<float>& features,

    const RouterWeights& router) {

    float z = router.intercept;

    for (std::size_t i = 0; i < features.size(); ++i) {

        z += features[i] * router.weights[i];

    }

    return sigmoid(z);

}



/** ONNX session + logistic regression routing. */

class RoutingBrain {

public:

    RoutingBrain(

        std::string model_path,

        BertWordPieceTokenizer tokenizer,

        RouterWeights router_weights)

        : env_(ORT_LOGGING_LEVEL_WARNING, "cascade_router_proxy"),

          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),

          model_path_(std::move(model_path)),

          tokenizer_(std::move(tokenizer)),

          router_weights_(std::move(router_weights)) {

        Ort::SessionOptions options;

        options.SetIntraOpNumThreads(1);

        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);



#ifdef _WIN32

        const std::wstring wide_path = to_wide(model_path_);

        session_ = std::make_unique<Ort::Session>(env_, wide_path.c_str(), options);

#else

        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), options);

#endif



        const std::size_t output_count = session_->GetOutputCount();

        output_name_storage_.reserve(output_count);

        output_names_.reserve(output_count);

        for (std::size_t i = 0; i < output_count; ++i) {

            auto name = session_->GetOutputNameAllocated(i, allocator_);

            output_name_storage_.emplace_back(name.get());

            output_names_.push_back(output_name_storage_.back().c_str());

        }



        std::cout << "[routing] ONNX model loaded: " << model_path_ << '\n';

        std::cout << "[routing] Vocab loaded: " << tokenizer_.vocab_path() << '\n';

        std::cout << "[routing] Router weights: " << router_weights_.weights.size()

                  << " coefficients\n";

    }



    struct InferenceResult {

        std::vector<int64_t> input_ids;

        int token_count = 0;

        float structural_complexity = 0.0f;

        float p_success = 0.0f;

        double latency_ms = 0.0;

    };



    InferenceResult run_inference(const std::string& prompt) {

        std::lock_guard<std::mutex> lock(mutex_);



        InferenceResult result;

        result.structural_complexity = calculate_structural_complexity(prompt);



        result.input_ids = tokenizer_.tokenize(prompt, kRoutingSeqLen);

        result.token_count = count_non_pad_tokens(result.input_ids);



        std::vector<int64_t> attention_mask = tokenizer_.attention_mask(result.input_ids);

        std::vector<int64_t> token_type_ids =

            tokenizer_.token_type_ids(result.input_ids.size());



        const std::array<int64_t, 2> shape = {kBatchSize, kRoutingSeqLen};



        std::vector<Ort::Value> inputs;

        inputs.reserve(3);

        inputs.push_back(make_int64_tensor(result.input_ids, shape));

        inputs.push_back(make_int64_tensor(attention_mask, shape));

        inputs.push_back(make_int64_tensor(token_type_ids, shape));



        const char* input_names[] = {

            kInputIdsName,

            kAttentionMaskName,

            kTokenTypeIdsName,

        };



        const auto start = std::chrono::high_resolution_clock::now();

        auto outputs = session_->Run(

            Ort::RunOptions{nullptr},

            input_names,

            inputs.data(),

            inputs.size(),

            output_names_.data(),

            output_names_.size());

        const auto end = std::chrono::high_resolution_clock::now();



        result.latency_ms =

            std::chrono::duration<double, std::milli>(end - start).count();



        const float* hidden_state = outputs[0].GetTensorData<float>();

        const auto shape_info = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        const int seq_len = static_cast<int>(shape_info[1]);



        std::vector<float> embedding =

            mean_pool_and_normalize(hidden_state, seq_len, attention_mask);



        std::vector<float> features;

        features.reserve(static_cast<std::size_t>(kFeatureDim));

        features.push_back(static_cast<float>(result.token_count));

        features.push_back(result.structural_complexity);

        features.insert(features.end(), embedding.begin(), embedding.end());



        result.p_success = predict_pass_probability(features, router_weights_);

        return result;

    }



private:

    Ort::Value make_int64_tensor(

        std::vector<int64_t>& backing_store,

        const std::array<int64_t, 2>& shape) {

        return Ort::Value::CreateTensor<int64_t>(

            memory_info_,

            backing_store.data(),

            backing_store.size(),

            shape.data(),

            shape.size());

    }



    Ort::Env env_;

    Ort::MemoryInfo memory_info_;

    Ort::AllocatorWithDefaultOptions allocator_;

    std::unique_ptr<Ort::Session> session_;

    std::string model_path_;

    BertWordPieceTokenizer tokenizer_;

    RouterWeights router_weights_;

    std::vector<std::string> output_name_storage_;

    std::vector<const char*> output_names_;

    std::mutex mutex_;

};



std::string extract_last_user_content(std::string_view body, std::string& error) {

    static thread_local simdjson::dom::parser parser;



    simdjson::dom::element doc;

    if (auto err = parser.parse(body).get(doc)) {

        error = std::string("JSON parse error: ") + simdjson::error_message(err);

        return {};

    }



    simdjson::dom::array messages;

    if (auto err = doc["messages"].get_array().get(messages)) {

        error = "Missing or invalid 'messages' array";

        return {};

    }



    const uint64_t count = messages.size();

    if (count == 0) {

        error = "'messages' array is empty";

        return {};

    }



    simdjson::dom::element last_message = messages.at(count - 1);

    std::string_view content;

    if (auto err = last_message["content"].get_string().get(content)) {

        error = "Last message has no string 'content' field";

        return {};

    }



    return std::string(content);

}

void handle_chat_completions(
    const httplib::Request& req,
    httplib::Response& res,
    RoutingBrain& brain) {
    std::string parse_error;
    const std::string prompt = extract_last_user_content(req.body, parse_error);
    if (prompt.empty() && !parse_error.empty()) {
        res.status = 400;
        res.set_content(
            std::string("{\"error\":\"") + json_escape(parse_error) + "\"}",
            "application/json");
        return;
    }

    const auto result = brain.run_inference(prompt);
    const std::string target_model = select_target_model(result.p_success);

    res.set_header("X-Cascade-Latency", format_latency_header(result.latency_ms));

    std::string response_body;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    const auto auth_it = req.headers.find("Authorization");
    if (auth_it == req.headers.end()) {
        res.status = 401;
        res.set_content(
            "{\"error\":\"Missing Authorization header (OpenAI API key).\"}",
            "application/json");
        return;
    }

    std::string rewrite_error;
    const std::string upstream_payload =
        rewrite_model_in_payload(req.body, target_model, rewrite_error);
    if (upstream_payload.empty()) {
        res.status = 400;
        res.set_content(
            std::string("{\"error\":\"") + json_escape(rewrite_error) + "\"}",
            "application/json");
        return;
    }

    httplib::SSLClient cli("api.openai.com", 443);
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(120, 0);
    cli.enable_server_certificate_verification(true);

    httplib::Headers filtered_headers = build_upstream_headers(req);

    std::cout << "[DEBUG] Sending rewritten payload to OpenAI: " << upstream_payload
              << std::endl;

    auto upstream = cli.Post(
        req.path.c_str(),
        filtered_headers,
        upstream_payload,
        "application/json");

    if (!upstream) {
        res.status = 502;
        res.set_content(
            "{\"error\":\"Upstream request to OpenAI failed.\"}",
            "application/json");
        return;
    }

    res.status = upstream->status;
    const std::string content_type = upstream->get_header_value("Content-Type");
    response_body = upstream->body;
    res.set_content(
        response_body,
        content_type.empty() ? "application/json" : content_type);
#else
    // Local Windows builds without OpenSSL: return mutated payload for testing.
    std::string rewrite_error;
    const std::string upstream_payload =
        rewrite_model_in_payload(req.body, target_model, rewrite_error);
    if (upstream_payload.empty()) {
        res.status = 400;
        res.set_content(
            std::string("{\"error\":\"") + json_escape(rewrite_error) + "\"}",
            "application/json");
        return;
    }

    res.status = 200;
    response_body = upstream_payload;
    res.set_content(response_body, "application/json");
#endif

    TokenUsage usage = extract_usage_from_response(response_body);
    if (usage.input_tokens == 0) {
        usage.input_tokens = estimate_input_tokens_from_request(req.body);
    }

    append_traffic_log_async(
        target_model,
        usage.input_tokens,
        usage.output_tokens,
        result.latency_ms);
}

}  // namespace



int main(int argc, char* argv[]) {

    try {

        const std::string model_path =

            (argc > 1) ? argv[1] : std::string(kDefaultModelPath);

        const std::string vocab_path = kDefaultVocabPath;

        const std::string weights_path = kDefaultWeightsPath;



        BertWordPieceTokenizer tokenizer(vocab_path);

        RouterWeights router_weights = load_router_weights(weights_path);

        auto brain = std::make_shared<RoutingBrain>(

            model_path,

            std::move(tokenizer),

            std::move(router_weights));



        httplib::Server server;



        server.Post("/v1/chat/completions", [brain](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completions(req, res, *brain);
        });



        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {

            res.set_content("{\"status\":\"ok\"}", "application/json");

        });



        std::cout << "[proxy] cascade-router listening on http://0.0.0.0:" << kListenPort << '\n';
        std::cout << "[proxy] POST /v1/chat/completions\n";
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        std::cout << "[proxy] Mode: upstream forwarding (OpenSSL)\n";
#else
        std::cout << "[proxy] Mode: mock forwarding (no OpenSSL)\n";
#endif

        if (!server.listen("0.0.0.0", kListenPort)) {

            std::cerr << "[proxy] Failed to bind to localhost:" << kListenPort << '\n';

            return 1;

        }



        return 0;

    } catch (const Ort::Exception& ex) {

        std::cerr << "ONNX Runtime error: " << ex.what() << '\n';

        return 1;

    } catch (const std::exception& ex) {

        std::cerr << "Error: " << ex.what() << '\n';

        return 1;

    }

}


