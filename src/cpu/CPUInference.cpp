#include "transformer/Transformer.h"
#include "utils/Logger.h"
#include "utils/Timer.h"

#include <iostream>

using namespace transformer;

// Full-sequence inference
void runFullSequenceInference(
    Transformer &model,
    const Tensor &token_ids,
    const TransformerConfig &config)
{

    Timer timer;

    timer.start();

    Tensor logits =
        model.forward(token_ids);

    const double latency =
        timer.stopMilliseconds();

    // Find the predicted next token using the logits corresponding to the final sequence position.

    const size_t final_position =
        logits.dim(0) - 1;

    size_t predicted_token = 0;

    float best_logit =
        logits[final_position *
               config.vocabulary_size];

    for (size_t token = 1;
         token < config.vocabulary_size;
         ++token)
    {

        const float value =
            logits[final_position *
                       config.vocabulary_size +
                   token];

        if (value > best_logit)
        {

            best_logit = value;
            predicted_token = token;
        }
    }

    // Results
    std::cout
        << "\n========================================\n"
        << "Full-Sequence CPU Inference\n"
        << "========================================\n";

    std::cout
        << "Input sequence length: "
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
        << "Latency: "
        << latency
        << " ms\n";

    std::cout
        << "Predicted next token: "
        << predicted_token
        << '\n';
}


// Next-token inference
void runNextTokenInference(
    Transformer &model,
    const Tensor &token_ids,
    const TransformerConfig &config)
{

    Timer timer;

    timer.start();

    Tensor logits =
        model.forwardNextToken(token_ids);

    const double latency =
        timer.stopMilliseconds();

    // Find the token with the highest logit.
    size_t predicted_token = 0;

    float best_logit =
        logits[0];

    for (size_t token = 1;
         token < config.vocabulary_size;
         ++token)
    {

        const float value =
            logits[token];

        if (value > best_logit)
        {

            best_logit = value;
            predicted_token = token;
        }
    }

    // Results
    std::cout
        << "\n========================================\n"
        << "Next-Token CPU Inference\n"
        << "========================================\n";

    std::cout
        << "Input sequence length: "
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
        << "Latency: "
        << latency
        << " ms\n";

    std::cout
        << "Predicted next token: "
        << predicted_token
        << '\n';
}

int main()
{

    TransformerConfig config;

    config.vocabulary_size = 10000;
    config.hidden_size = 256;
    config.num_heads = 4;
    config.num_layers = 2;
    config.intermediate_size = 1024;
    config.max_sequence_length = 128;

    Logger::info(
        "Initializing CPU transformer...");

    Transformer model(config);

    // Example input sequence.
    Tensor token_ids({8});

    token_ids[0] = 12;
    token_ids[1] = 42;
    token_ids[2] = 891;
    token_ids[3] = 17;
    token_ids[4] = 302;
    token_ids[5] = 51;
    token_ids[6] = 900;
    token_ids[7] = 4;

    // Run full-sequence inference.
    runFullSequenceInference(
        model,
        token_ids,
        config);

    // Run optimized next-token inference.
    /*
    runNextTokenInference(
        model,
        token_ids,
        config
    );
    */

    return 0;
}