// represents a single block in the decoder part of a transformer model, which consists of a multi-head self-attention mechanism followed by a feedforward neural network, with layer normalization applied before each sub-layer to stabilize training and improve convergence.

#pragma once

#include "layers/Attention.h"
#include "layers/FeedForward.h"
#include "layers/LayerNorm.h"
#include "tensor/Tensor.h"
#include "transformer/TransformerConfig.h"

namespace transformer {

class DecoderBlock {
public:
    explicit DecoderBlock(
        const TransformerConfig& config
    );

    Tensor forward(const Tensor& input) const;

private:
    LayerNorm attention_norm_;
    Attention attention_;

    LayerNorm feed_forward_norm_;
    FeedForward feed_forward_;
};

} // namespace transformer