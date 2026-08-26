# CUDA Transformer Inference Engine

A lightweight C++ inference engine implementing a 2-layer decoder-only Transformer. The project uses a CPU reference implementation to establish correctness and performance baselines before implementing CUDA kernels for compute-intensive operations.

> **Current status:** CPU reference implementation complete, tested, and benchmarked. CUDA kernel implementation and GPU profiling are in progress.

---

## Current Model

The reference implementation uses the following configuration:

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Layers | 2 |
| Hidden size | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| FFN intermediate size | 1024 |
| Vocabulary size | 10,000 |
| Maximum sequence length | 128 |
| Data type | FP32 |
| Attention | Causal multi-head self-attention |
| Positional encoding | RoPE |
| Activation | GELU |
| Normalization | LayerNorm |

---

## Architecture

The inference pipeline is:

```text
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
Decoder Block 1
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    │
    ▼
Decoder Block 2
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    │
    ▼
Final LayerNorm
    │
    ▼
Language Model Head
    │
    ▼
Output Logits
    │
    ▼
Predicted Token