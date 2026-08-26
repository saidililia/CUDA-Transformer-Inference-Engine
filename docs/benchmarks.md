
```markdown
# Benchmark

This document describes how performance is measured in the CUDA Transformer Inference Engine, from the CPU reference implementation to the CUDA implementation.

---

## 1. Benchmarking Steps

1. Establish a CPU reference baseline for correctness and performance.
2. Identify compute-intensive operations in the transformer inference pipeline.
3. Implement CUDA kernels by mapping parallel workloads to GPU threads.
4. Measure kernel execution time and overall inference latency.
5. Analyze memory access patterns and GPU execution behavior.
6. Evaluate kernel performance and identify bottlenecks.
7. Optimize kernels based on profiling results.
8. Compare CUDA performance against the CPU baseline using equivalent workloads.

---

## 2. Current Benchmark Configuration

The current benchmark uses:

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Layers | 2 |
| Hidden size | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| FFN intermediate size | 1024 |
| Vocabulary size | 10,000 |
| Sequence length | 32 |
| Data type | FP32 |
| Iterations | 10 |

---

## 3. Current CPU Baseline

The current measured CPU baseline is:

```text
CPU Transformer Benchmark
==========================
Layers: 2
Hidden size: 256
Attention heads: 4
Sequence length: 32
Iterations: 10
Average latency: 193.785 ms
Throughput: 165.132 tokens/s