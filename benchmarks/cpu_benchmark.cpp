#include "transformer/Transformer.h"
#include "utils/Timer.h"
#include "utils/Profiler.h"

#include <iostream>

using namespace transformer;


//Benchmark result structure: Stores the results of a benchmark so that main() can later compare different inference paths.
struct BenchmarkResult {

    double total_time;
    double average_time;
};


// Benchmark 1: Full-sequence inference
BenchmarkResult benchmarkFullSequence(
    Transformer& model,
    const Tensor& token_ids,
    int iterations
) {

    // Warm-up: Run a few iterations of the model to ensure that any initial setup costs (like memory allocation) do not affect the benchmark results.
    for (int i = 0;
         i < 3;
         ++i) {

        model.forward(token_ids);
    }


    Timer timer;

    timer.start();

    for (int i = 0;
         i < iterations;
         ++i) {

        model.forward(token_ids);
    }

    const double total_time =
        timer.stopMilliseconds();

    const double average_time =
        total_time /
        static_cast<double>(iterations);


    return {
        total_time,
        average_time
    };
}


// Benchmark 2: Next-token inference
BenchmarkResult benchmarkNextToken(
    Transformer& model,
    const Tensor& token_ids,
    int iterations
) {

    // Warm-up
    for (int i = 0;
         i < 3;
         ++i) {

        model.forwardNextToken(token_ids);
    }


    Timer timer;

    timer.start();

    for (int i = 0;
         i < iterations;
         ++i) {

        model.forwardNextToken(token_ids);
    }

    const double total_time =
        timer.stopMilliseconds();

    const double average_time =
        total_time /
        static_cast<double>(iterations);


    return {
        total_time,
        average_time
    };
}


// Profile 3: Decoder blocks during next-token inference
void profileDecoderBlocksNextToken(
    Transformer& model,
    const Tensor& token_ids,
    int iterations
) {

    // warm up
    for (int i = 0;
         i < 3;
         ++i) {

        model.forwardNextToken(token_ids);
    }


    Profiler profiler;


    // Run next-token inference while collecting profiling data.
    for (int i = 0;
         i < iterations;
         ++i) {

        model.forwardNextToken(
            token_ids,
            &profiler
        );
    }


    // Print the profiling report.
    std::cout
        << "\n========================================\n"
        << "DECODER BLOCK PROFILE\n"
        << "========================================\n";

    profiler.printReport();
}


int main()
{

    // Transformer configuration.

    TransformerConfig config;

    config.vocabulary_size = 10000;
    config.hidden_size = 256;
    config.num_heads = 4;
    config.num_layers = 2;
    config.intermediate_size = 1024;
    config.max_sequence_length = 128;

    Transformer model(config);

    const size_t sequence_length = 32;

    Tensor token_ids({
        sequence_length
    });

    for (size_t i = 0;
         i < sequence_length;
         ++i) {

        token_ids[i] =
            static_cast<float>(
                i % config.vocabulary_size
            );
    }


    constexpr int iterations = 10;


    // Select experiments

    const BenchmarkResult full_result =
        benchmarkFullSequence(
            model,
            token_ids,
            iterations
        );

    const BenchmarkResult next_token_result =
        benchmarkNextToken(
            model,
            token_ids,
            iterations
        );


    // Print benchmark results
    std::cout
        << "\nCPU Transformer Benchmark\n"
        << "==========================\n";

    std::cout
        << "Layers: "
        << config.num_layers
        << '\n';

    std::cout
        << "Hidden size: "
        << config.hidden_size
        << '\n';

    std::cout
        << "Attention heads: "
        << config.num_heads
        << '\n';

    std::cout
        << "Sequence length: "
        << sequence_length
        << '\n';

    std::cout
        << "Iterations: "
        << iterations
        << '\n';


    // Full-sequence results.
    std::cout
        << "\n--- model.forward() ---\n";

    std::cout
        << "Output shape: ["
        << sequence_length
        << ", "
        << config.vocabulary_size
        << "]\n";

    std::cout
        << "Average latency: "
        << full_result.average_time
        << " ms\n";

    std::cout
        << "Throughput: "
        << (
            1000.0 *
            sequence_length /
            full_result.average_time
        )
        << " tokens/s\n";


    // Next-token results.
    std::cout
        << "\n--- model.forwardNextToken() ---\n";

    std::cout
        << "Output shape: [1, "
        << config.vocabulary_size
        << "]\n";

    std::cout
        << "Average latency: "
        << next_token_result.average_time
        << " ms\n";

    std::cout
        << "Generation throughput: "
        << (
            1000.0 /
            next_token_result.average_time
        )
        << " tokens/s\n";


    // Comparison.
    const double speedup =
        full_result.average_time /
        next_token_result.average_time;

    std::cout
        << "\n--- Comparison ---\n";

    std::cout
        << "Speedup: "
        << speedup
        << "x\n";


    // Decoder block profiling
    profileDecoderBlocksNextToken(
        model,
        token_ids,
        iterations
    );


    return 0;
}