// Softmax.h header file for the Softmax layer in the transformer namespace, which applies the softmax function to the input tensor to convert logits into probabilities.
#pragma once

#include "tensor/Tensor.h"

namespace transformer {

Tensor softmax(
    const Tensor& input
);

} // namespace transformer