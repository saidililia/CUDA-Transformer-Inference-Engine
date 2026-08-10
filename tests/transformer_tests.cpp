#include "transformer/Transformer.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace transformer;

int main() {

    TransformerConfig config;

    config.vocabulary_size = 100;
    config.hidden_size = 16;
    config.num_heads = 4;
    config.num_layers = 2;
    config.intermediate_size = 64;
    config.max_sequence_length = 16;

    Transformer model(config);

    Tensor token_ids({4});

    token_ids[0] = 1;
    token_ids[1] = 7;
    token_ids[2] = 13;
    token_ids[3] = 42;

    Tensor logits =
        model.forward(token_ids);

    /*
     * Expected:
     *
     * [sequence_length, vocabulary_size]
     */

    assert(
        logits.dim(0)
        == 4
    );

    assert(
        logits.dim(1)
        == 100
    );

    /*
     * Numerical sanity check.
     */

    for (size_t i = 0;
         i < logits.numel();
         ++i) {

        assert(
            std::isfinite(logits[i])
        );
    }

    /*
     * Verify deterministic inference.
     */

    Tensor logits_second =
        model.forward(token_ids);

    for (size_t i = 0;
         i < logits.numel();
         ++i) {

        assert(
            logits[i]
            ==
            logits_second[i]
        );
    }

    std::cout
        << "Transformer tests passed.\n";

    return 0;
}