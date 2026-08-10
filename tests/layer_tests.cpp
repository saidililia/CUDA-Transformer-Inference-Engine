#include "layers/LayerNorm.h"
#include "layers/Softmax.h"
#include "layers/Linear.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace transformer;

int main() {

    /*
     * Linear
     */

    Linear linear(4, 3);

    Tensor input({2, 4});

    input.fill(1.0f);

    Tensor linear_output =
        linear.forward(input);

    assert(linear_output.dim(0) == 2);
    assert(linear_output.dim(1) == 3);

    /*
     * LayerNorm
     */

    LayerNorm norm(4);

    Tensor norm_input({2, 4});

    norm_input[0] = 1.0f;
    norm_input[1] = 2.0f;
    norm_input[2] = 3.0f;
    norm_input[3] = 4.0f;

    Tensor norm_output =
        norm.forward(norm_input);

    assert(norm_output.dim(0) == 2);
    assert(norm_output.dim(1) == 4);

    /*
     * Softmax
     */

    Tensor softmax_input({1, 3});

    softmax_input[0] = 1.0f;
    softmax_input[1] = 2.0f;
    softmax_input[2] = 3.0f;

    Tensor probabilities =
        softmax(softmax_input);

    float sum = 0.0f;

    for (size_t i = 0;
         i < 3;
         ++i) {

        sum += probabilities[i];
    }

    assert(
        std::abs(sum - 1.0f)
        < 1e-5f
    );

    std::cout
        << "Layer tests passed.\n";

    return 0;
}