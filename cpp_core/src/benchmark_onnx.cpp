/**
 * cascade-router — ONNX Runtime embedding inference micro-benchmark
 *
 * Profiles latency across sequence lengths to inform prompt truncation cutoffs.
 *
 * Usage:
 *   benchmark_onnx [path/to/model.onnx]
 *
 * Default model path: ../models/all-MiniLM-L6-v2-int8.onnx
 */

#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kBatchSize = 1;
constexpr int kNumInferences = 1000;
constexpr int kWarmupInferences = 50;
constexpr double kLatencyBudgetMs = 5.0;

constexpr const char* kDefaultModelPath = "../models/all-MiniLM-L6-v2-int8.onnx";

constexpr const char* kInputIdsName = "input_ids";
constexpr const char* kAttentionMaskName = "attention_mask";
constexpr const char* kTokenTypeIdsName = "token_type_ids";

const std::vector<int> kSeqLengths = {16, 32, 64, 128};

struct SeqLenResult {
    int seq_len;
    double avg_latency_ms;
};

std::vector<int64_t> make_random_input_ids(std::mt19937_64& rng, int seq_len) {
    std::uniform_int_distribution<int64_t> dist(0, 30521);
    std::vector<int64_t> data(static_cast<std::size_t>(kBatchSize * seq_len));
    for (auto& value : data) {
        value = dist(rng);
    }
    return data;
}

std::vector<int64_t> make_ones(std::size_t count) {
    return std::vector<int64_t>(count, 1);
}

std::vector<int64_t> make_zeros(std::size_t count) {
    return std::vector<int64_t>(count, 0);
}

Ort::Value make_int64_tensor(
    Ort::MemoryInfo& memory_info,
    std::vector<int64_t>& backing_store,
    const std::array<int64_t, 2>& shape) {
    return Ort::Value::CreateTensor<int64_t>(
        memory_info,
        backing_store.data(),
        backing_store.size(),
        shape.data(),
        shape.size());
}

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

std::vector<const char*> resolve_output_names(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& storage) {
    const std::size_t output_count = session.GetOutputCount();
    storage.clear();
    storage.reserve(output_count);
    std::vector<const char*> names;
    names.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        storage.emplace_back(name.get());
        names.push_back(storage.back().c_str());
    }
    return names;
}

double benchmark_seq_len(
    Ort::Session& session,
    Ort::MemoryInfo& memory_info,
    const std::vector<const char*>& output_names,
    int seq_len,
    std::mt19937_64& rng) {
    const std::array<int64_t, 2> shape = {kBatchSize, seq_len};
    const std::size_t flat_size = static_cast<std::size_t>(kBatchSize * seq_len);

    std::vector<int64_t> input_ids_data = make_random_input_ids(rng, seq_len);
    std::vector<int64_t> attention_mask_data = make_ones(flat_size);
    std::vector<int64_t> token_type_ids_data = make_zeros(flat_size);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(3);
    input_tensors.push_back(
        make_int64_tensor(memory_info, input_ids_data, shape));
    input_tensors.push_back(
        make_int64_tensor(memory_info, attention_mask_data, shape));
    input_tensors.push_back(
        make_int64_tensor(memory_info, token_type_ids_data, shape));

    const char* input_names[] = {
        kInputIdsName,
        kAttentionMaskName,
        kTokenTypeIdsName,
    };

    auto run_once = [&]() {
        (void)session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());
    };

    for (int i = 0; i < kWarmupInferences; ++i) {
        run_once();
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kNumInferences; ++i) {
        run_once();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / static_cast<double>(kNumInferences);
}

void print_results_table(const std::vector<SeqLenResult>& results) {
    std::cout << "\nSeq_Len | Avg_Latency_ms | Status\n";
    std::cout << "--------|----------------|--------\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& result : results) {
        const char* status =
            result.avg_latency_ms < kLatencyBudgetMs ? "[PASS]" : "[FAIL]";
        std::cout << std::setw(7) << result.seq_len << " | "
                  << std::setw(14) << result.avg_latency_ms << " | "
                  << status << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string model_path =
            (argc > 1) ? argv[1] : std::string(kDefaultModelPath);

        std::cout << "cascade-router ONNX sequence-length profiler\n";
        std::cout << "  model:       " << model_path << '\n';
        std::cout << "  batch:       " << kBatchSize << '\n';
        std::cout << "  seq_lengths: ";
        for (std::size_t i = 0; i < kSeqLengths.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << kSeqLengths[i];
        }
        std::cout << '\n';
        std::cout << "  iterations:  " << kNumInferences << " (+ "
                  << kWarmupInferences << " warmup per seq_len)\n";
        std::cout << "  budget:      " << kLatencyBudgetMs << " ms\n";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cascade_router_benchmark");
        Ort::SessionOptions session_options;
        // Single thread per request — avoids hogging cores in a concurrent proxy.
        session_options.SetIntraOpNumThreads(1);
        // Sequential execution — no background thread pool for graph management.
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        // Maximum ORT graph optimizations (constant folding, fusion, etc.).
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        const std::wstring wide_model_path = to_wide(model_path);
        Ort::Session session(env, wide_model_path.c_str(), session_options);
#else
        Ort::Session session(env, model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> output_name_storage;
        std::cout << "  onnx inputs: " << session.GetInputCount() << '\n';

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::vector<const char*> output_names =
            resolve_output_names(session, allocator, output_name_storage);

        std::mt19937_64 rng(42);
        std::vector<SeqLenResult> results;
        results.reserve(kSeqLengths.size());

        for (int seq_len : kSeqLengths) {
            std::cout << "\nProfiling seq_len=" << seq_len << " ...\n";
            const double avg_ms =
                benchmark_seq_len(session, memory_info, output_names, seq_len, rng);
            results.push_back({seq_len, avg_ms});
        }

        print_results_table(results);
        return 0;
    } catch (const Ort::Exception& ex) {
        std::cerr << "ONNX Runtime error: " << ex.what() << '\n';
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
