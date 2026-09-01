// LayerNorm.cpp that performs layer normalization on a 2D tensor. The input tensor is expected to have shape [rows, dimension], where 'rows' is the number of samples and 'dimension' is the number of features. The layer normalization is applied across the 'dimension' axis for each row independently.
#include "layers/LayerNorm.h"
#include <cmath>
#include <stdexcept>

namespace transformer {

LayerNorm::LayerNorm(
    size_t dimension,
    float epsilon
)
    : dimension_(dimension),
      epsilon_(epsilon),
      gamma_({dimension}, 1.0f),
      beta_({dimension}, 0.0f) {}

Tensor LayerNorm::forward(const Tensor& input) const {

    if (input.ndim() != 2) {
        throw std::runtime_error(
            "LayerNorm expects a 2D tensor"
        );
    }

    const size_t rows = input.dim(0);

    if (input.dim(1) != dimension_) {
        throw std::runtime_error(
            "LayerNorm dimension mismatch"
        );
    }

    Tensor output({
        rows,
        dimension_
    });

    // Apply layer normalization for each row independently
    for (size_t row = 0; row < rows; ++row) {

        float mean = 0.0f;

        for (size_t i = 0; i < dimension_; ++i) {
            mean += input[row * dimension_ + i];
        }

        mean /= static_cast<float>(dimension_);

        float variance = 0.0f;

        for (size_t i = 0; i < dimension_; ++i) {

            const float difference =
                input[row * dimension_ + i] - mean;

            variance += difference * difference;
        }

        variance /= static_cast<float>(dimension_);

        const float inverse_std =
            1.0f / std::sqrt(variance + epsilon_);

        for (size_t i = 0; i < dimension_; ++i) {

            const float normalized =
                (input[row * dimension_ + i] - mean)
                * inverse_std;

            output[row * dimension_ + i] =
                gamma_[i] * normalized + beta_[i];
        }
    }

    return output;
}

} // namespace transformer