#include "transformer/DecoderBlock.h"

namespace transformer {

DecoderBlock::DecoderBlock(
    const TransformerConfig& config
)
    : attention_norm_(
          config.hidden_size,
          config.layer_norm_epsilon
      ),
      attention_(
          config.hidden_size,
          config.num_heads,
          config.max_sequence_length
      ),
      feed_forward_norm_(
          config.hidden_size,
          config.layer_norm_epsilon
      ),
      feed_forward_(
          config.hidden_size,
          config.intermediate_size
      ) {}

Tensor DecoderBlock::forward(
    const Tensor& input
) const {

    /*
     * Pre-LN Transformer:
     *
     * x = x + Attention(LN(x))
     */

    Tensor normalized = attention_norm_.forward(input); // apply layer normalization to the input tensor before passing it to the attention mechanism

    Tensor attention_output = attention_.forward(normalized);

    Tensor residual_1 = input;

    for (size_t i = 0;
         i < residual_1.numel();
         ++i) {

        residual_1[i] +=
            attention_output[i];
    }

    /*
     * x = x + FFN(LN(x))
     */

    normalized =
        feed_forward_norm_.forward(
            residual_1
        );

    Tensor feed_forward_output =
        feed_forward_.forward(
            normalized
        );

    Tensor output =
        residual_1;

    for (size_t i = 0;
         i < output.numel();
         ++i) {

        output[i] +=
            feed_forward_output[i];
    }

    return output;
}

} // namespace transformer