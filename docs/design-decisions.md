# Design Decisions

This document records the main engineering decisions made during development and the reasoning behind them.

---

## 1. CPU Reference Before CUDA

### Decision

Implement and validate the complete Transformer forward pass on the CPU before implementing CUDA kernels.

### Rationale

The CPU implementation provides a simpler reference for validating the model computation before introducing GPU-specific complexity.

CUDA adds additional concerns such as:

- parallel execution
- device memory
- host-device transfers
- thread mapping
- synchronization
- kernel launch configuration
- GPU-specific numerical behavior

Developing the CUDA implementation after establishing a working CPU reference makes it easier to isolate model-level errors from GPU implementation errors.

The CPU implementation therefore serves as the correctness reference for the CUDA implementation.

---


## 2. Small Model Configuration

### Decision

Use a 2-layer Transformer with a hidden size of 256 for the initial implementation.

### Rationale

The model is small enough to simplify debugging and correctness testing while still exercising the main components of Transformer inference, including attention, feed-forward layers, normalization, and residual connections.

The initial implementation focuses on the inference pipeline and GPU execution rather than model scale.

A fixed model configuration also makes CPU and CUDA performance comparisons easier to reproduce.

---

### Development Workflow

```text
CPU implementation
        ↓
Correctness tests
        ↓
CPU benchmark
        ↓
CPU profiling
        ↓
CUDA implementation
        ↓
CPU ↔ CUDA validation
        ↓
GPU profiling
        ↓
Kernel optimization