#include "layers/Linear.h"

#include <stdexcept>

namespace transformer {

Linear::Linear(
    size_t input_features,
    size_t output_features,
    bool use_bias
)
    : input_features_(input_features),
      output_features_(output_features),
      weight_({output_features, input_features}),
      use_bias_(use_bias) {

    Random random(42);

    for (size_t i = 0; i < weight_.numel(); ++i) {
        weight_[i] = random.normal(0.0f, 0.02f);
    }

    if (use_bias_) {
        bias_ = Tensor({output_features}, 0.0f);
    }
}

Tensor Linear::forward(const Tensor& input) const {

    if (input.ndim() != 2) {
        throw std::runtime_error(
            "Linear::forward expects a 2D tensor"
        );
    }

    const size_t rows = input.dim(0);

    if (input.dim(1) != input_features_) {
        throw std::runtime_error(
            "Linear input dimension mismatch"
        );
    }

    Tensor output({rows, output_features_});

    // Perform matrix multiplication: output = input * weight^T + bias
    for (size_t i = 0; i < rows; ++i) {

        for (size_t j = 0; j < output_features_; ++j) {

            float sum = 0.0f;

            for (size_t k = 0; k < input_features_; ++k) {

                sum +=
                    input[i * input_features_ + k] *
                    weight_[j * input_features_ + k];
            }

            if (use_bias_) {
                sum += bias_[j];
            }

            output[i * output_features_ + j] = sum;
        }
    }

    return output;
}

size_t Linear::input_features() const {
    return input_features_;
}

size_t Linear::output_features() const {
    return output_features_;
}

} // namespace transformer