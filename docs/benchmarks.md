# Benchmark

This document describes how performance is measured in the CUDA Transformer Inference Engine, from the CPU reference implementation to the CUDA implementation.

## 1. Benchmarking Steps
- Establish a CPU reference baseline for correctness and performance.
- Identify compute-intensive operations in the transformer inference pipeline.
- Implement CUDA kernels by mapping parallel workloads to GPU threads.
- Measure kernel execution time and overall inference latency.
- Analyze memory access patterns and GPU execution behavior.
- Evaluate kernel performance and identify bottlenecks.
- Optimize kernels based on profiling results.
- Compare CUDA performance against the CPU baseline using equivalent workloads.


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

### CPU Transformer Benchmark

| Parameter | Value |
|---|---:|
| Layers | 2 |
| Hidden size | 256 |
| Attention heads | 4 |
| Sequence length | 32 |
| Iterations | 10 |
| Average latency | 193.785 ms |
| Throughput | 165.132 tokens/s |

---

## 4. CPU Inference Optimization — LM Head

The optimized path still processes the complete input sequence through the embedding layer, decoder blocks, and final layer normalization. It then extracts only the hidden state corresponding to the final token and passes that single hidden state through the language-model head.

### Results
#### Full-Sequence Inference

Transformer::forward() produces:

| Parameter | Value |
|---|---:|
| Output shape | [32, 10000] |
| Average latency | 191.415 ms |
| Throughput | 167.176 tokens/s |

#### Next-Token Inference

Transformer::forwardNextToken() produces:

| Parameter | Value |
|---|---:|
| Output shape | [1, 10000] |
| Average latency | 76.7318 ms |
| Next token inference | 13.0324 |

### Performance Comparison

| Metric | forward() | forwardNextToken() |
|---|---|---:|
| Output shape | [32, 10000] | [1, 10000] |
| Average latency | 191.415 ms | 76.7318 ms |
| Relative latency | 100% | 40.1% |
| Speedup | 1.00× | 2.49× |

The optimized path reduced end-to-end latency by approximately 59.9%, while achieving a 2.49× speedup.

## LM Head Work Reduction

- The original language-model head performs a matrix multiplication equivalent to: [32, 256] × [256, 10000]

- For next-token inference, the optimized path performs: [1, 256] × [256, 10000]. Therefore, the number of output rows processed by the LM Head is reduced by a factor of 32.

- The measured LM Head latency confirms this reduction. Where Full forward inference takes 117.135 ms and Next-token inference takes 3.717 ms. This corresponds to approximately a 31.5× reduction in LM Head latency.

- The end-to-end speedup is smaller because the embedding, decoder blocks, and final layer normalization are still executed for the complete sequence.