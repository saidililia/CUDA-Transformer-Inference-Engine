// LayerNorm.h header file for the Layer Normalization layer in the transformer namespace, which normalizes the input across the features dimension to improve training stability and convergence.

#pragma once

#include "tensor/Tensor.h"

namespace transformer {

class LayerNorm {
public:
    LayerNorm(
        size_t dimension,
        float epsilon = 1e-5f
    );

    Tensor forward(const Tensor& input) const;

private:
    size_t dimension_;
    float epsilon_;

    Tensor gamma_;
    Tensor beta_;
};

} // namespace transformer