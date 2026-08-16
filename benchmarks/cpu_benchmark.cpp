// this code is a simple benchmark for measuring the performance of the CPU transformer model. It initializes a transformer model with a given configuration, creates an example input sequence of token IDs, performs multiple forward passes to measure the average latency and throughput, and then outputs the results.

#include "transformer/Transformer.h"
#include "utils/Timer.h"

#include <iostream>

using namespace transformer;

int main() {

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

    /*
     * Warm-up.
     */

    for (int i = 0;
         i < 3;
         ++i) {

        model.forward(token_ids);
    }

    constexpr int iterations = 10;

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

    std::cout
        << "CPU Transformer Benchmark\n"
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

    std::cout
        << "Average latency: "
        << average_time
        << " ms\n";

    std::cout
        << "Throughput: "
        << (
            1000.0 *
            sequence_length /
            average_time
        )
        << " tokens/s\n";

    return 0;
}