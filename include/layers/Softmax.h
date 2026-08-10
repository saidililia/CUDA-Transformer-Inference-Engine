#pragma once

#include "tensor/Tensor.h"

namespace transformer {

Tensor softmax(
    const Tensor& input
);

} // namespace transformer