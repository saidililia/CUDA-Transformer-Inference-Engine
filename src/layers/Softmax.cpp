#include "layers/Softmax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace transformer {

Tensor softmax(const Tensor& input) {

    if (input.ndim() != 2) {
        throw std::runtime_error(
            "Softmax expects a 2D tensor"
        );
    }

    const size_t rows = input.dim(0);
    const size_t columns = input.dim(1);

    Tensor output({
        rows,
        columns
    });

    for (size_t row = 0; row < rows; ++row) {

        float maximum = input[row * columns];

        for (size_t column = 1; column < columns; ++column) {

            maximum = std::max(
                maximum,
                input[row * columns + column]
            );
        }

        float sum = 0.0f;

        for (size_t column = 0; column < columns; ++column) {

            const float value =
                std::exp(
                    input[row * columns + column]
                    - maximum
                );

            output[row * columns + column] = value;

            sum += value;
        }

        for (size_t column = 0; column < columns; ++column) {

            output[row * columns + column] /= sum;
        }
    }

    return output;
}

} // namespace transformer