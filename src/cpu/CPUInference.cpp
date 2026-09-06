#include "transformer/Transformer.h"
#include "utils/Logger.h"
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

    Logger::info(
        "Initializing CPU transformer..."
    );

    Transformer model(config);

    /*
     * Example input sequence.
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


    /*
     * ============================================================
     * Full-sequence inference
     * ============================================================
     *
     * Produces logits for every position:
     *
     * [sequence_length, vocabulary_size]
     *
     * [8, 10000]
     */

    Timer full_timer;

    full_timer.start();

    Tensor full_logits =
        model.forward(token_ids);

    double full_latency =
        full_timer.stopMilliseconds();


    /*
     * Find the predicted next token from
     * the final position.
     */

    const size_t final_position =
        full_logits.dim(0) - 1;

    size_t full_predicted_token = 0;

    float full_best_logit =
        full_logits[
            final_position *
            config.vocabulary_size
        ];

    for (size_t token = 1;
         token < config.vocabulary_size;
         ++token) {

        const float value =
            full_logits[
                final_position *
                config.vocabulary_size +
                token
            ];

        if (value > full_best_logit) {

            full_best_logit = value;
            full_predicted_token = token;
        }
    }


    /*
     * ============================================================
     * Next-token inference
     * ============================================================
     *
     * Produces logits only for the final position:
     *
     * [1, vocabulary_size]
     *
     * [1, 10000]
     */

    Timer next_token_timer;

    next_token_timer.start();

    Tensor next_token_logits =
        model.forwardNextToken(token_ids);

    double next_token_latency =
        next_token_timer.stopMilliseconds();


    /*
     * Find the predicted next token.
     */

    size_t next_token_predicted_token = 0;

    float next_token_best_logit =
        next_token_logits[0];

    for (size_t token = 1;
         token < config.vocabulary_size;
         ++token) {

        const float value =
            next_token_logits[token];

        if (value > next_token_best_logit) {

            next_token_best_logit = value;
            next_token_predicted_token = token;
        }
    }


    /*
     * ============================================================
     * Results
     * ============================================================
     */

    std::cout
        << "\n========================================\n"
        << "CPU Inference Comparison\n"
        << "========================================\n";

    std::cout
        << "\nInput sequence length: "
        << token_ids.dim(0)
        << '\n';

    std::cout
        << "Vocabulary size: "
        << config.vocabulary_size
        << '\n';


    /*
     * Full forward results
     */

    std::cout
        << "\n--- model.forward() ---\n";

    std::cout
        << "Logits shape: ["
        << full_logits.dim(0)
        << ", "
        << full_logits.dim(1)
        << "]\n";

    std::cout
        << "Latency: "
        << full_latency
        << " ms\n";

    std::cout
        << "Predicted next token: "
        << full_predicted_token
        << '\n';


    /*
     * Next-token results
     */

    std::cout
        << "\n--- model.forwardNextToken() ---\n";

    std::cout
        << "Logits shape: ["
        << next_token_logits.dim(0)
        << ", "
        << next_token_logits.dim(1)
        << "]\n";

    std::cout
        << "Latency: "
        << next_token_latency
        << " ms\n";

    std::cout
        << "Predicted next token: "
        << next_token_predicted_token
        << '\n';


    /*
     * ============================================================
     * Correctness comparison
     * ============================================================
     */

    std::cout
        << "\n--- Comparison ---\n";

    std::cout
        << "Predicted token match: "
        << (
            full_predicted_token ==
            next_token_predicted_token
            ? "YES"
            : "NO"
        )
        << '\n';


    /*
     * Calculate speedup.
     */

    if (next_token_latency > 0.0) {

        const double speedup =
            full_latency /
            next_token_latency;

        std::cout
            << "Speedup: "
            << speedup
            << "x\n";
    }


    return 0;
}