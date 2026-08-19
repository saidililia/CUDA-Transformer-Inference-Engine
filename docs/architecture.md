# CUDA Transformer Inference Engine

A lightweight C++ inference engine for a 2-layer decoder-only Transformer, developed as a systems-oriented project to study Transformer inference, CUDA kernel design, GPU parallelism, memory behavior, and hardware-aware optimization.

The project follows a **reference-first engineering methodology**: a complete CPU implementation is established and validated before progressively mapping compute-intensive operations to an NVIDIA GPU.

> **Current status:** CPU reference implementation complete, tested, and benchmarked. CUDA implementation and GPU optimization are the next development phase.

---

## Overview

Transformer inference consists of a collection of computational primitives that are well suited to parallel execution, including matrix multiplication, attention, normalization, and element-wise operations.

Rather than relying on a high-level inference framework, this project implements the core Transformer inference pipeline from the ground up in C++.

The primary objectives are to develop practical understanding of:

- Transformer inference internals
- C++ systems programming
- tensor representations and memory layout
- CPU reference implementations
- CUDA programming
- GPU thread/block mapping
- memory access patterns
- parallel reductions
- GPU memory hierarchy
- numerical correctness
- performance profiling
- kernel-level optimization
- end-to-end inference performance

---

## Current Model

The current reference implementation uses a small decoder-only Transformer:

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

The current inference pipeline is:

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
Logits