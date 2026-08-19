
```markdown
# Roadmap

This document defines the development roadmap for the CUDA Transformer Inference Engine.

The project is developed incrementally, with correctness and measurement established before optimization.

---

# Phase 1 — CPU Reference Implementation

**Status: Complete**

The first milestone is a complete CPU implementation of a small decoder-only Transformer.

## Objectives

- Establish the Transformer computational graph.
- Implement core operations.
- Create a deterministic reference implementation.
- Build a testable C++ project.
- Establish an initial performance baseline.

## Completed

- [x] Project structure
- [x] CMake build system
- [x] Tensor abstraction
- [x] Shape abstraction
- [x] Random initialization
- [x] Token embedding
- [x] Linear layers
- [x] LayerNorm
- [x] Softmax
- [x] RoPE
- [x] Multi-head causal self-attention
- [x] GELU
- [x] Feed-forward network
- [x] Residual connections
- [x] Decoder block
- [x] Two-layer Transformer
- [x] Final LayerNorm
- [x] Language-model head
- [x] CPU inference executable
- [x] Unit tests
- [x] Integration tests
- [x] CPU benchmark

---

# Phase 2 — CPU Validation and Profiling

**Status: In progress**

The objective is to understand the computational characteristics of the CPU reference implementation before writing CUDA kernels.

## Objectives

- Validate numerical behavior more thoroughly.
- Measure individual operations.
- Identify computational bottlenecks.
- Understand scaling with sequence length.
- Establish operation-level performance baselines.

## Tasks

- [ ] Add operation-level timers.
- [ ] Measure embedding time.
- [ ] Measure LayerNorm time.
- [ ] Measure Q/K/V projection time.
- [ ] Measure RoPE time.
- [ ] Measure QKᵀ time.
- [ ] Measure Softmax time.
- [ ] Measure attention × V time.
- [ ] Measure output projection time.
- [ ] Measure FFN time.
- [ ] Measure final LayerNorm.
- [ ] Measure LM head.
- [ ] Analyze sequence-length scaling.
- [ ] Record profiling results in `docs/profiling.md`.
- [ ] Update `docs/benchmark.md`.

---

# Phase 3 — CUDA Infrastructure

**Status: Planned**

Before implementing Transformer kernels, establish the basic CUDA runtime infrastructure.

## Objectives

- Introduce CUDA compilation through CMake.
- Establish device memory management.
- Create basic CUDA error handling.
- Create CUDA tensor/device representations.
- Establish CPU-to-GPU and GPU-to-CPU transfers.

## Tasks

- [ ] Enable CUDA language in CMake.
- [ ] Add `.cu` source files.
- [ ] Create CUDA utility infrastructure.
- [ ] Implement CUDA error checking.
- [ ] Implement device memory allocation.
- [ ] Implement host/device transfers.
- [ ] Implement CUDA tensor representation.
- [ ] Create basic CUDA kernel test.
- [ ] Validate GPU execution independently.

---

# Phase 4 — First CUDA Kernels

**Status: Planned**

The first CUDA kernels should be simple and highly parallel.

## Candidate operations

- [ ] Element-wise operations
- [ ] Residual addition
- [ ] GELU
- [ ] RoPE

The purpose of this phase is to become comfortable with:

- thread indexing
- grid/block configuration
- device memory
- kernel launches
- synchronization
- CUDA error handling

---

# Phase 5 — CUDA Linear Algebra

**Status: Planned**

Matrix multiplication is expected to be one of the most important GPU workloads.

## Tasks

- [ ] Implement baseline CUDA matrix multiplication.
- [ ] Validate against CPU implementation.
- [ ] Benchmark baseline kernel.
- [ ] Analyze memory access.
- [ ] Introduce tiled matrix multiplication.
- [ ] Investigate shared memory.
- [ ] Tune block dimensions.
- [ ] Analyze register usage.
- [ ] Compare kernel variants.

## Validation

Every implementation should satisfy:

```text
CPU output
    ≈
CUDA output