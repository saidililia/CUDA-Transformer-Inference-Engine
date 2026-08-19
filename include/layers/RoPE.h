// RoPE.h header file for the Rotary Positional Encoding (RoPE) layer in the transformer namespace, which applies rotary positional encoding to the query and key tensors to incorporate positional information into the attention mechanism.

#pragma once

#include "tensor/Tensor.h"

namespace transformer {

class RoPE {
public:
    RoPE(
        size_t head_dimension,
        size_t max_sequence_length,
        float theta = 10000.0f
    );

    void apply(
        Tensor& query,
        Tensor& key,
        size_t sequence_length
    ) const;

private:
    size_t head_dimension_;
    size_t max_sequence_length_;
    float theta_;
};

} // namespace transformer