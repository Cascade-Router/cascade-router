/**

 * cascade-router HTTP proxy — OpenAI-compatible routing endpoint.

 *

 * Tokenizes prompts, runs ONNX embedding inference, applies logistic

 * regression routing weights, mutates the upstream JSON model field, and returns
 * the rewritten payload (mock forward for local testing).

 *

 * Usage:

 *   proxy_server [path/to/model.onnx]

 */



#ifdef _WIN32

#ifndef _WIN32_WINNT

#define _WIN32_WINNT 0x0A00

#endif

#endif



#include "tokenizer.hpp"



#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <simdjson.h>



#include <array>

#include <chrono>

#include <cmath>

#include <cstdint>

#include <fstream>

#include <iomanip>

#include <iostream>

#include <memory>

#include <mutex>

#include <sstream>

#include <string>

#include <string_view>

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

std::string mutate_request_payload(const std::string& raw_body, const std::string& target_model) {
    nlohmann::json payload = nlohmann::json::parse(raw_body);
    payload["model"] = target_model;
    return payload.dump();
}

std::string format_latency_header(double latency_ms) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << latency_ms;
    return out.str();
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

            std::string parse_error;

            const std::string prompt = extract_last_user_content(req.body, parse_error);

            if (prompt.empty() && !parse_error.empty()) {

                res.status = 400;

                res.set_content(

                    std::string("{\"error\":\"") + json_escape(parse_error) + "\"}",

                    "application/json");

                return;

            }



            const auto result = brain->run_inference(prompt);

            const std::string target_model = select_target_model(result.p_success);



            try {

                const std::string mutated_body = mutate_request_payload(req.body, target_model);

                res.status = 200;

                res.set_header("X-Cascade-Latency", format_latency_header(result.latency_ms));

                res.set_content(mutated_body, "application/json");

            } catch (const nlohmann::json::exception& ex) {

                res.status = 400;

                res.set_content(

                    std::string("{\"error\":\"") + json_escape(ex.what()) + "\"}",

                    "application/json");

            }

        });



        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {

            res.set_content("{\"status\":\"ok\"}", "application/json");

        });



        std::cout << "[proxy] cascade-router listening on http://localhost:" << kListenPort << '\n';

        std::cout << "[proxy] POST /v1/chat/completions\n";



        if (!server.listen("127.0.0.1", kListenPort)) {

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


