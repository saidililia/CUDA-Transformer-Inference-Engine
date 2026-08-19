
```markdown
# References

This document collects the technical references used to understand and develop the CUDA Transformer Inference Engine.

References are grouped by topic so that implementation decisions can be traced back to relevant technical literature and documentation.

---

## 1. Transformer Architecture

### Attention Is All You Need

Vaswani et al., 2017.

The foundational Transformer paper introducing the attention-based architecture.

- Paper: https://arxiv.org/abs/1706.03762

Relevant concepts:

- scaled dot-product attention
- multi-head attention
- positional representations
- residual connections
- feed-forward networks

---

## 2. Rotary Positional Embeddings

### RoFormer: Enhanced Transformer with Rotary Position Embedding

Su et al., 2021.

Introduces Rotary Position Embeddings (RoPE), used by the current implementation.

- Paper: https://arxiv.org/abs/2104.09864

Relevant concepts:

- rotary position embeddings
- position-dependent rotations
- applying rotations to query and key representations

---

## 3. CUDA Programming

### NVIDIA CUDA C++ Programming Guide

Official NVIDIA documentation covering CUDA programming concepts.

- Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Relevant topics:

- kernels
- threads
- blocks
- grids
- device memory
- shared memory
- synchronization
- CUDA execution model

---

## 4. CUDA Best Practices

### NVIDIA CUDA C++ Best Practices Guide

Official NVIDIA guidance on CUDA performance optimization.

- Documentation: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

Relevant topics:

- memory coalescing
- memory hierarchy
- occupancy
- synchronization
- performance measurement
- optimization methodology

---

## 5. CUDA Runtime API

### NVIDIA CUDA Runtime API

Reference documentation for CUDA runtime functions.

- Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/

Relevant future APIs include:

- memory allocation
- memory transfers
- kernel launches
- CUDA events
- synchronization

---

## 6. Nsight Systems

### NVIDIA Nsight Systems

System-level performance analysis and visualization tool.

- Documentation: https://docs.nvidia.com/nsight-systems/

Relevant use cases:

- GPU/CPU timeline analysis
- kernel scheduling
- synchronization
- memory transfers
- launch overhead

---

## 7. Nsight Compute

### NVIDIA Nsight Compute

Kernel-level CUDA performance analysis tool.

- Documentation: https://docs.nvidia.com/nsight-compute/

Relevant use cases:

- occupancy
- memory throughput
- instruction analysis
- register usage
- warp execution
- kernel bottleneck analysis

---

## 8. CMake

### CMake Documentation

Used to configure and build the C++ project.

- Documentation: https://cmake.org/documentation/

Relevant concepts:

- project configuration
- C++ targets
- libraries
- executable targets
- compiler configuration

---

## 9. Microsoft Visual C++

### Microsoft C++ Documentation

Documentation for the MSVC compiler and Windows C++ development environment.

- Documentation: https://learn.microsoft.com/cpp/

Relevant concepts:

- MSVC
- C++ compilation
- linker behavior
- Windows development

---

## 10. Performance Engineering

The project follows a general performance-engineering methodology:

```text
Measure
   ↓
Identify bottleneck
   ↓
Understand behavior
   ↓
Form hypothesis
   ↓
Optimize
   ↓
Measure again