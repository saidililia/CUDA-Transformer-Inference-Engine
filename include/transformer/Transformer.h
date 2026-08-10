#pragma once

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