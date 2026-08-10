#include "layers/Attention.h"
#include "layers/Softmax.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace transformer {

Attention::Attention(
    size_t hidden_size,
    size_t num_heads,
    size_t max_sequence_length
)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      head_dimension_(hidden_size / num_heads),
      query_projection_(
          hidden_size,
          hidden_size
      ),
      key_projection_(
          hidden_size,
          hidden_size
      ),
      value_projection_(
          hidden_size,
          hidden_size
      ),
      output_projection_(
          hidden_size,
          hidden_size
      ),
      rope_(
          hidden_size / num_heads,
          max_sequence_length
      ) {

    if (hidden_size % num_heads != 0) {
        throw std::runtime_error(
            "Hidden size must be divisible by number of heads"
        );
    }
}

Tensor Attention::forward(
    const Tensor& input
) const {

    const size_t sequence_length =
        input.dim(0);

    Tensor q = query_projection_.forward(input);
    Tensor k = key_projection_.forward(input);
    Tensor v = value_projection_.forward(input);

    /*
     * Convert:
     *
     * [sequence, hidden]
     *
     * into:
     *
     * [heads, sequence, head_dimension]
     */

    Tensor q_heads({
        num_heads_,
        sequence_length,
        head_dimension_
    });

    Tensor k_heads({
        num_heads_,
        sequence_length,
        head_dimension_
    });

    Tensor v_heads({
        num_heads_,
        sequence_length,
        head_dimension_
    });

    for (size_t position = 0;
         position < sequence_length;
         ++position) {

        for (size_t head = 0;
             head < num_heads_;
             ++head) {

            for (size_t dimension = 0;
                 dimension < head_dimension_;
                 ++dimension) {

                const size_t source =
                    position * hidden_size_
                    +
                    head * head_dimension_
                    +
                    dimension;

                const size_t destination =
                    head *
                    sequence_length *
                    head_dimension_
                    +
                    position *
                    head_dimension_
                    +
                    dimension;

                q_heads[destination] = q[source];
                k_heads[destination] = k[source];
                v_heads[destination] = v[source];
            }
        }
    }

    rope_.apply(
        q_heads,
        k_heads,
        sequence_length
    );

    Tensor attention_output({
        sequence_length,
        hidden_size_
    });

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(head_dimension_)
        );

    for (size_t head = 0;
         head < num_heads_;
         ++head) {

        /*
         * Attention scores:
         *
         * [sequence, sequence]
         */

        Tensor scores({
            sequence_length,
            sequence_length
        });

        for (size_t i = 0;
             i < sequence_length;
             ++i) {

            for (size_t j = 0;
                 j < sequence_length;
                 ++j) {

                /*
                 * Causal masking:
                 *
                 * position i cannot attend
                 * to future position j.
                 */

                if (j > i) {

                    scores[
                        i * sequence_length + j
                    ] =
                        -std::numeric_limits<float>::infinity();

                    continue;
                }

                float dot = 0.0f;

                for (size_t d = 0;
                     d < head_dimension_;
                     ++d) {

                    const size_t q_index =
                        head *
                        sequence_length *
                        head_dimension_
                        +
                        i * head_dimension_
                        +
                        d;

                    const size_t k_index =
                        head *
                        sequence_length *
                        head_dimension_
                        +
                        j * head_dimension_
                        +
                        d;

                    dot +=
                        q_heads[q_index] *
                        k_heads[k_index];
                }

                scores[
                    i * sequence_length + j
                ] = dot * scale;
            }
        }

        Tensor probabilities =
            softmax(scores);

        /*
         * probabilities × V
         */

        for (size_t i = 0;
             i < sequence_length;
             ++i) {

            for (size_t d = 0;
                 d < head_dimension_;
                 ++d) {

                float sum = 0.0f;

                for (size_t j = 0;
                     j < sequence_length;
                     ++j) {

                    const size_t v_index =
                        head *
                        sequence_length *
                        head_dimension_
                        +
                        j * head_dimension_
                        +
                        d;

                    sum +=
                        probabilities[
                            i * sequence_length + j
                        ] *
                        v_heads[v_index];
                }

                const size_t output_index =
                    i * hidden_size_
                    +
                    head * head_dimension_
                    +
                    d;

                attention_output[output_index] =
                    sum;
            }
        }
    }

    return output_projection_.forward(
        attention_output
    );
}

} // namespace transformer