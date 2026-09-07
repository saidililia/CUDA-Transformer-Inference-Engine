#include "transformer/Transformer.h"
#include "utils/Timer.h"
#include "utils/Profiler.h"

#include <iostream>

using namespace transformer;

int main()
{
    TransformerConfig config;

    config.vocabulary_size = 10000;
    config.hidden_size = 256;
    config.num_heads = 4;
    config.num_layers = 2;
    config.intermediate_size = 1024;
    config.max_sequence_length = 128;

    Transformer model(config);

    const size_t sequence_length = 32;

    Tensor token_ids({sequence_length});

    // this loop initializes the token_ids tensor with a sequence of integers from 0 to sequence_length - 1.
    for (size_t i = 0;
         i < sequence_length;
         ++i)
    {
        token_ids[i] =
            static_cast<float>(
                i % config.vocabulary_size);
    }

    constexpr int iterations = 10;

    // warm up the model by running a few iterations of both forward passes to ensure that any initial setup or caching is done before the actual benchmarking begins.
    for (int i = 0;
         i < 3;
         ++i)
    {
        model.forward(token_ids);
        model.forwardNextToken(token_ids);
    }

    Profiler full_profiler;
    Profiler next_token_profiler;

    
    // Benchmark 1 : model.forward()
    Timer full_timer;

    full_timer.start();

    for (int i = 0;
         i < iterations;
         ++i)
    {
        model.forward(
            token_ids,
            &full_profiler);
    }

    const double full_total_time =
        full_timer.stopMilliseconds();

    const double full_average_time =
        full_total_time /
        static_cast<double>(iterations);

    

    // Benchmark 2 : model.forwardNextToken()    
    Timer next_token_timer;

    next_token_timer.start();

    for (int i = 0;
         i < iterations;
         ++i)
    {
        model.forwardNextToken(
            token_ids,
            &next_token_profiler);
    }

    const double next_token_total_time =
        next_token_timer.stopMilliseconds();

    const double next_token_average_time =
        next_token_total_time /
        static_cast<double>(iterations);

    

    // Printing results and comparison of the two benchmarks.

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

    

    // Result 1 : model.forward()
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
        << full_average_time
        << " ms\n";

    std::cout
        << "Throughput: "
        << (1000.0 *
            sequence_length /
            full_average_time)
        << " tokens/s\n";

    
    // Result 2 : model.forwardNextToken()
    std::cout
        << "\n--- model.forwardNextToken() ---\n";

    std::cout
        << "Output shape: [1, "
        << config.vocabulary_size
        << "]\n";

    std::cout
        << "Average latency: "
        << next_token_average_time
        << " ms\n";


    // Only one token is actually projected by forwardNextToken().
    std::cout
        << "Throughput: "
        << (1000.0 /
            next_token_average_time)
        << " tokens/s\n";

    
    // Comparison of the two benchmarks.
    const double speedup =
        full_average_time /
        next_token_average_time;

    std::cout
        << "\n--- Comparison ---\n";

    std::cout
        << "Speedup: "
        << speedup
        << "x\n";


    // Print profiling reports for both benchmarks.
    std::cout
        << "\n========================================\n"
        << "FULL-SEQUENCE PROFILE\n"
        << "========================================\n";

    full_profiler.printReport();

    std::cout
        << "\n========================================\n"
        << "NEXT-TOKEN PROFILE\n"
        << "========================================\n";

    next_token_profiler.printReport();

    return 0;
}