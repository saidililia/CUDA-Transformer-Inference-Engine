#include "transformer/Transformer.h"
#include <string>

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

        layers_.emplace_back(config); // create a new DecoderBlock with the given configuration and add it to the layers_ vector
    }
}

Tensor Transformer::forward(
    const Tensor& token_ids,
    Profiler* profiler
) const {

    /*
     * Embedding
     */

    if (profiler) {
        profiler->start("Embedding");
    }

    Tensor x =
        embedding_.forward(token_ids);

    if (profiler) {
        profiler->stop("Embedding");
    }


    /*
     * Transformer blocks
     */

    for (size_t i = 0;
         i < layers_.size();
         ++i) {

        const std::string name =
            "DecoderBlock " +
            std::to_string(i);

        if (profiler) {
            profiler->start(name);
        }

        x = layers_[i].forward(x);

        if (profiler) {
            profiler->stop(name);
        }
    }


    /*
     * Final LayerNorm
     */

    if (profiler) {
        profiler->start("Final LayerNorm");
    }

    x =
        final_norm_.forward(x);

    if (profiler) {
        profiler->stop("Final LayerNorm");
    }


    /*
     * Language-model head
     */

    if (profiler) {
        profiler->start("LM Head");
    }

    Tensor logits =
        lm_head_.forward(x); // project the output of the final layer normalization to the vocabulary size using a linear layer to produce logits for each token in the sequence

    if (profiler) {
        profiler->stop("LM Head");
    }

    return logits;
}

} // namespace transformer