# script to generate random weights for a simple transformer model and save them in a binary file
import numpy as np
# Every time you run it, you get the SAME weights, which is useful for debugging and testing
np.random.seed(42)

# Number of tokens in the vocabulary that the model understands
vocab_size = 16

# Number of dimensions of each token representation (embedding size)
d_model = 8

# The size of the feedforward expansions layer in the transformer (the hidden layer size in the MLP)
mlp_hidden = 16

# number of transformer layers
n_layers = 2

# maximum sequence length, maximum number of tokens in a sequence 
max_len = 16

# helper function to generate random weights with a small standard deviation
def rand(shape):
    return np.random.randn(*shape).astype(np.float32) * 0.02

with open("weights.bin", "wb") as f:

    # 1. Embedding matrix, a lookup table that converts token IDs into dense vectors of size d_model
    emb = rand((vocab_size, d_model))
    f.write(emb.tobytes())

    # 2. Positional embeddings that encode the position of each token in the sequence, allowing the model to understand the order of tokens     
    pos = rand((max_len, d_model))
    f.write(pos.tobytes())

    # 3. Transformer layers, each layer has attention weights, MLP weights, and LayerNorm parameters, We build 2 identical blocks.
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
        gamma = np.ones(d_model, dtype=np.float32) # scaling
        beta = np.zeros(d_model, dtype=np.float32) # shifting

        f.write(gamma.tobytes())
        f.write(beta.tobytes())

print("weights.bin generated successfully")