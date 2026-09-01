// this code is a simple example of how to use the CPU transformer model for inference. It initializes a transformer model with a given configuration, creates an example input sequence of token IDs, performs inference to obtain logits, and then selects the highest-logit token for the final position in the sequence. The code also measures and outputs the inference latency.

#include "transformer/Transformer.h"
#include "utils/Logger.h"
#include "utils/Timer.h"

#include <algorithm>
#include <iostream>

using namespace transformer;

int main() {

    TransformerConfig config;

    config.vocabulary_size = 10000;
    config.hidden_size = 256; // embedding dimension
    config.num_heads = 4;
    config.num_layers = 2;
    config.intermediate_size = 1024;
    config.max_sequence_length = 128;

    Logger::info(
        "Initializing CPU transformer..."
    );

    Transformer model(config);

    /*
     * Example input sequence.
     *
     * In the future this will come from
     * a tokenizer.
     */

    Tensor token_ids({
        8
    });

    token_ids[0] = 12;
    token_ids[1] = 42;
    token_ids[2] = 891;
    token_ids[3] = 17;
    token_ids[4] = 302;
    token_ids[5] = 51;
    token_ids[6] = 900;
    token_ids[7] = 4;

    Timer timer;

    timer.start();

    Tensor logits =
        model.forward(token_ids);

    double elapsed =
        timer.stopMilliseconds();

    std::cout
        << "\nInference complete.\n";

    std::cout
        << "Sequence length: "
        << token_ids.dim(0)
        << '\n';

    std::cout
        << "Vocabulary size: "
        << config.vocabulary_size
        << '\n';

    std::cout
        << "Logits shape: ["
        << logits.dim(0)
        << ", "
        << logits.dim(1)
        << "]\n";

    std::cout
        << "Inference latency: "
        << elapsed
        << " ms\n";

    /*
     * Select the highest-logit token
     * for the final position.
     */

    const size_t final_position =
        logits.dim(0) - 1;

    size_t predicted_token = 0;

    float best_logit =
        logits[
            final_position *
            config.vocabulary_size
        ];

    for (size_t token = 1;
         token < config.vocabulary_size;
         ++token) {

        const float value =
            logits[
                final_position *
                config.vocabulary_size
                +
                token
            ];

        if (value > best_logit) {

            best_logit = value;
            predicted_token = token;
        }
    }

    std::cout
        << "Predicted next token: "
        << predicted_token
        << '\n';

    return 0;
}