#include "transformer/Transformer.h"

namespace transformer {

Transformer::Transformer(
    const TransformerConfig& config
)
    : config_(config),
      embedding_(
          config.vocabulary_size,
          config.hidden_size
      ),
      final_norm_(
          config.hidden_size,
          config.layer_norm_epsilon
      ),
      lm_head_(
          config.hidden_size,
          config.vocabulary_size,
          false
      ) {

    layers_.reserve(config.num_layers);

    for (size_t i = 0;
         i < config.num_layers;
         ++i) {

        layers_.emplace_back(config);
    }
}

Tensor Transformer::forward(
    const Tensor& token_ids
) const {

    /*
     * Token IDs
     *     ↓
     * Embedding
     */

    Tensor x =
        embedding_.forward(token_ids);

    /*
     * Transformer blocks
     */

    for (const auto& layer : layers_) {

        x =
            layer.forward(x);
    }

    /*
     * Final LayerNorm
     */

    x =
        final_norm_.forward(x);

    /*
     * Language-model head
     *
     * [sequence, hidden]
     *
     * →
     *
     * [sequence, vocabulary]
     */

    Tensor logits =
        lm_head_.forward(x);

    return logits;
}

} // namespace transformer