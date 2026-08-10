#pragma once

#include "layers/Linear.h"
#include "tensor/Tensor.h"

namespace transformer {

class FeedForward {
public:
    FeedForward(
        size_t hidden_size,
        size_t intermediate_size
    );

    Tensor forward(const Tensor& input) const;

private:
    Linear up_projection_;
    Linear down_projection_;
};

} // namespace transformer