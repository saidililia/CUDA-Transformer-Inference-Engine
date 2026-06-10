# script to generate random weights for a simple transformer model and save them in a binary file
import numpy as np
np.random.seed(42)

vocab_size = 16
d_model = 8
mlp_hidden = 16
n_layers = 2
max_len = 16

def rand(shape):
    return np.random.randn(*shape).astype(np.float32) * 0.02

with open("weights.bin", "wb") as f:

    # 1. Token embeddings
    emb = rand((vocab_size, d_model))
    f.write(emb.tobytes())

    # 2. Positional embeddings
    pos = rand((max_len, d_model))
    f.write(pos.tobytes())

    for _ in range(n_layers):

        # Attention weights
        Wq = rand((d_model, d_model))
        Wk = rand((d_model, d_model))
        Wv = rand((d_model, d_model))
        Wo = rand((d_model, d_model))

        f.write(Wq.tobytes())
        f.write(Wk.tobytes())
        f.write(Wv.tobytes())
        f.write(Wo.tobytes())

        # MLP weights
        W1 = rand((d_model, mlp_hidden))
        W2 = rand((mlp_hidden, d_model))

        f.write(W1.tobytes())
        f.write(W2.tobytes())

        # LayerNorm params
        gamma = np.ones(d_model, dtype=np.float32)
        beta = np.zeros(d_model, dtype=np.float32)

        f.write(gamma.tobytes())
        f.write(beta.tobytes())

print("weights.bin generated successfully")