// this attention_tests.cpp file contains tests for the Attention layer defined in the include/layers directory. It creates an instance of the Attention layer, performs a forward pass with a sample input tensor, and checks the output dimensions and properties to ensure correctness. If all assertions pass, it prints a success message indicating that the attention tests have passed.

#include "layers/Attention.h"
#include "transformer/TransformerConfig.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace transformer;

int main() {

    const size_t hidden_size = 16;
    const size_t num_heads = 4;
    const size_t sequence_length = 8;

    Attention attention(
        hidden_size,
        num_heads,
        sequence_length
    );

    Tensor input({
        sequence_length,
        hidden_size
    });

    for (size_t i = 0;
         i < input.numel();
         ++i) {

        input[i] =
            std::sin(
                static_cast<float>(i)
            );
    }

    Tensor output =
        attention.forward(input);

    assert(
        output.dim(0)
        == sequence_length
    );

    assert(
        output.dim(1)
        == hidden_size
    );

    for (size_t i = 0;
         i < output.numel();
         ++i) {

        assert(
            std::isfinite(output[i])
        );
    }

    std::cout
        << "Attention tests passed.\n";

    return 0;
}