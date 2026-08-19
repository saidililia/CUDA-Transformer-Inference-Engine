
```markdown
# Benchmark

This document defines the benchmarking methodology for the CUDA Transformer Inference Engine.

The purpose of benchmarking is not simply to produce a single latency number. The goal is to establish reproducible baselines and measure the performance impact of individual implementation and optimization decisions.

---

## 1. Benchmarking Principles

The project follows these principles:

1. Establish a CPU reference baseline first.
2. Separate correctness validation from performance measurement.
3. Warm up the implementation before timing steady-state execution.
4. Use multiple iterations.
5. Report average execution time.
6. Keep model configuration fixed when comparing implementations.
7. Compare equivalent workloads.
8. Record the implementation and hardware configuration used for measurements.

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