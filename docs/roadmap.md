
```markdown
# Roadmap

This roadmap tracks the development of the CUDA Transformer Inference Engine from the CPU reference implementation to a profiled and optimized CUDA implementation.

---

# Phase 1 — CPU Reference Implementation

**Status: Complete**

Establish a complete and testable CPU implementation of a small decoder-only Transformer.

## Completed

- [x] Implement tensor abstraction and basic tensor operations
- [x] Implement linear layers and token embeddings
- [x] Implement LayerNorm, Softmax, RoPE, and GELU
- [x] Implement multi-head self-attention
- [x] Implement feed-forward network
- [x] Implement decoder block and 2-layer Transformer
- [x] Implement CPU inference pipeline
- [x] Add tensor, layer, attention, and Transformer tests
- [x] Verify tensor shapes and end-to-end inference
- [x] Establish CPU benchmark and baseline


## Current baseline

Layers: 2
Hidden size: 256
Attention heads: 4
Sequence length: 32
Iterations: 10
Average latency: 193.785 ms
Throughput: 165.132 tokens/s

---

# Phase 2 — CPU Profiling

**Status: Planned**

Profile the CPU reference implementation to identify compute-intensive operations and establish operation-level baselines before CUDA implementation.


## Tasks

- [ ] Add operation-level timing
- [ ] Measure embedding, LayerNorm, Q/K/V projections, and RoPE
- [ ] Measure QKᵀ, Softmax, and attention × V
- [ ] Measure output projection and FFN
- [ ] Measure final LayerNorm and LM head
- [ ] Analyze performance across sequence lengths
- [ ] Record profiling results in `docs/profiling.md`
- [ ] Update `docs/benchmark.md`

---

# Phase 3 — CUDA Implementation

**Status: Planned**

Introduce CUDA execution and progressively move compute-intensive Transformer operations to the GPU.

## Tasks

- [ ] Enable CUDA compilation through CMake
- [ ] Implement CUDA error handling and device memory management
- [ ] Implement host/device data transfers
- [ ] Implement CUDA tensor/device representations
- [ ] Implement and test basic CUDA kernels
- [ ] Implement CUDA element-wise, residual, GELU, and RoPE operations
- [ ] Implement CUDA matrix multiplication
- [ ] Implement CUDA attention operations
- [ ] Implement CUDA feed-forward operations
- [ ] Integrate CUDA kernels into the inference pipeline
- [ ] Validate CUDA outputs against the CPU reference

---

# Phase 4 — GPU Profiling & Optimization

**Status: Planned**

Profile the CUDA implementation, identify bottlenecks, and optimize kernels based on measured performance.

## Tasks

- [ ] Establish CUDA performance baseline
- [ ] Measure kernel execution time and end-to-end latency
- [ ] Measure throughput
- [ ] Analyze memory access behavior
- [ ] Profile GPU execution
- [ ] Identify performance bottlenecks
- [ ] Optimize thread and block configurations
- [ ] Optimize memory access patterns
- [ ] Investigate shared memory and register usage
- [ ] Reduce unnecessary memory transfers
- [ ] Re-profile after each optimization
- [ ] Compare CPU, baseline CUDA, and optimized CUDA performance
- [ ] Document profiling results and optimization decisions

