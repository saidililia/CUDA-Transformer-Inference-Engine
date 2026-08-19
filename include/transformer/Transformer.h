// complete forward pass of a transformer model, which takes a batch of token IDs as input, passes them through an embedding layer, a series of decoder blocks, applies layer normalization, and finally projects the output to the vocabulary size using a linear layer to produce logits for each token in the sequence.
#pragma once
// include necessary headers for the transformer model
#include "layers/Embedding.h"
#include "layers/LayerNorm.h"
#include "layers/Linear.h"
#include "tensor/Tensor.h"
#include "transformer/DecoderBlock.h"
#include "transformer/TransformerConfig.h"

#include <vector>

namespace transformer {

class Transformer {
public:
    explicit Transformer(
        const TransformerConfig& config
    );

    Tensor forward(
        const Tensor& token_ids
    ) const;

private:
    TransformerConfig config_;

    Embedding embedding_;

    std::vector<DecoderBlock> layers_;

    LayerNorm final_norm_;

    Linear lm_head_;
};

} // namespace transformer