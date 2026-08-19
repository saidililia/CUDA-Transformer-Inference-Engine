// FeedForward.h header file for the FeedForward layer in the transformer namespace, which implements a two-layer feedforward neural network with an activation function, typically used in transformer architectures to process the output of the attention mechanism.
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