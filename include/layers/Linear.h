#pragma once

#include "tensor/Tensor.h"
#include "utils/Random.h"

namespace transformer {

class Linear {
public:
    Linear() = default;

    Linear(
        size_t input_features,
        size_t output_features,
        bool use_bias = true
    );

    Tensor forward(const Tensor& input) const;

    size_t input_features() const;
    size_t output_features() const;

private:
    size_t input_features_ = 0;
    size_t output_features_ = 0;

    Tensor weight_;
    Tensor bias_;

    bool use_bias_ = true;
};

} // namespace transformer