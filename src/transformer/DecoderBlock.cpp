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
    const Tensor& input,
    Profiler* profiler,
    size_t block_index
) const {

    // Prefix for profiling names to identify the specific decoder block in the profiling report.
    const std::string prefix =
        "DecoderBlock " +
        std::to_string(block_index) +
        " / ";


    // Pre-LN Transformer: x = x + Attention(LN(x))

    if (profiler) {

        profiler->start(
            prefix + "Attention LayerNorm"
        );
    }

    Tensor normalized = attention_norm_.forward(input); // apply layer normalization to the input tensor before passing it to the attention mechanism

    if (profiler) {

        profiler->stop(
            prefix + "Attention LayerNorm"
        );
    }
    
     if (profiler) {

        profiler->start(
            prefix + "Attention"
        );
    }

    Tensor attention_output = attention_.forward(normalized);

    if (profiler) {

        profiler->stop(
            prefix + "Attention"
        );
    }

    
    if (profiler) {

        profiler->start(
            prefix + "Residual 1"
        );
    }

    Tensor residual_1 = input; // create a copy of the input tensor to use as a residual connection

    for (size_t i = 0;
         i < residual_1.numel();
         ++i) {

        residual_1[i] +=
            attention_output[i];
    }

    if (profiler) {

        profiler->stop(
            prefix + "Residual 1"
        );
    }


    /*
     * x = x + FFN(LN(x))
     */

    if (profiler) {

        profiler->start(
            prefix + "FeedForward LayerNorm"
        );
    }

    normalized = feed_forward_norm_.forward( residual_1 ); // apply layer normalization to the output of the attention mechanism before passing it to the feedforward network

    if (profiler) {

        profiler->stop(
            prefix + "FeedForward LayerNorm"
        );
    }
    

    if (profiler) {

        profiler->start(
            prefix + "FeedForward"
        );
    }

    Tensor feed_forward_output = feed_forward_.forward( normalized ); // pass the normalized tensor through the feedforward network, which consists of two linear transformations with a non-linear activation function in between

    if (profiler) {

        profiler->stop(
            prefix + "FeedForward"
        );
    }

    

    
     if (profiler) {

        profiler->start(
            prefix + "Residual 2"
        );
    }

    Tensor output = residual_1;

    for (size_t i = 0;
         i < output.numel();
         ++i) {

        output[i] +=
            feed_forward_output[i];
    }

    if (profiler) {

        profiler->stop(
            prefix + "Residual 2"
        );
    }


    return output;
}

} 