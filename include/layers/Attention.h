#pragma once

#include "layers/Linear.h"
#include "layers/RoPE.h"
#include "tensor/Tensor.h"

namespace transformer {

class Attention {
public:
    Attention(
        size_t hidden_size,
        size_t num_heads,
        size_t max_sequence_length
    );

    Tensor forward(const Tensor& input) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dimension_;

    Linear query_projection_;
    Linear key_projection_;
    Linear value_projection_;
    Linear output_projection_;

    RoPE rope_;
};

} // namespace transformer