## How attention mechanism works in neural networks

# Step 1: Step 1 — Input embeddings X:
# Step 2: Linear projections to obtain Q, K, V using learned weight matrices W_Q, W_K, W_V:
# Step 3: Reshape Q, K and V into heads
# Step 4: Compute attention per head using the formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
# Step 5: Concatenate the outputs of all heads
# Step 6: Apply a final linear projection to obtain the output of the multi-head attention mechanism
# Step 7: Add residual connection and layer normalization to the output of the multi-head attention mechanism

void attention_forward(float* Q, float* K, float* V, float* output) {
    // placeholder
}