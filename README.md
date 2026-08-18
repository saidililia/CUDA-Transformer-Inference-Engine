# CUDA Transformer Inference Engine

A lightweight C++ inference engine for a 2-layer decoder-only Transformer, developed as a systems-oriented project to study **GPU acceleration, CUDA kernel design, memory access patterns, and hardware-aware optimization**.

The project follows a reference-first methodology: a complete CPU implementation is established as a correctness and performance baseline before progressively mapping compute-intensive Transformer operations onto NVIDIA GPUs.

> **Current status:** CPU reference implementation complete and validated. CUDA kernel development in progress.

---

## Overview

Modern Transformer inference is dominated by a relatively small set of computational primitives, including matrix multiplication, attention, normalization, and feed-forward projections.

This project implements these operations from the ground up rather than relying on high-level inference frameworks. The goal is to understand and optimize the underlying computation at the level of:

- C++ data structures and tensor layouts
- CPU execution
- GPU thread mapping
- CUDA kernel design
- Global-memory access patterns
- Parallel reduction
- Memory bandwidth
- Arithmetic intensity
- Kernel launch overhead
- Numerical correctness
- End-to-end inference performance

The implementation currently consists of a **2-layer decoder-only Transformer** with a CPU reference backend. CUDA kernels are being introduced incrementally and validated against the CPU implementation.

---

## Architecture

The current model implements the following inference pipeline:

```text
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
┌───────────────────────────────┐
│ Decoder Block 1               │
│                               │
│ LayerNorm                     │
│     ↓                         │
│ Multi-Head Self-Attention     │
│     ↓                         │
│ Residual Connection           │
│     ↓                         │
│ LayerNorm                     │
│     ↓                         │
│ Feed-Forward Network          │
│     ↓                         │
│ Residual Connection           │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ Decoder Block 2               │
│                               │
│ LayerNorm                     │
│     ↓                         │
│ Multi-Head Self-Attention     │
│     ↓                         │
│ Residual Connection           │
│     ↓                         │
│ LayerNorm                     │
│     ↓                         │
│ Feed-Forward Network          │
│     ↓                         │
│ Residual Connection           │
└───────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Language Model Head
    │
    ▼
Logits