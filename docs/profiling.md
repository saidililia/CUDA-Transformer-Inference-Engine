# Profiling

This document records profiling experiments, performance measurements, bottleneck analysis, and optimization observations for the CPU and CUDA implementations of the transformer inference engine.

The profiling methodology follows an evidence-driven workflow:

Measure → Identify bottlenecks → Form hypotheses → Optimize → Benchmark → Analyze

The CPU implementation serves as the correctness reference and performance baseline before GPU acceleration is introduced.

## Experiment 1 — High-Level CPU Profiling
### Objective

The first profiling experiment measured the execution time of the major stages of the transformer forward pass.

The goal was to identify where the CPU implementation spends most of its execution time before attempting optimization or CUDA acceleration.

The following components were measured:

```text
Transformer::forward()
│
├── Embedding
├── DecoderBlock 0
├── DecoderBlock 1
├── Final LayerNorm
└── LM Head
```

Profiling measurements were collected across 10 inference iterations following an initial warm-up phase.

### Results
| Operation | Total Time | Calls | Average Time |
|---|---|---|---:|
| Embedding | 0.222 ms | 10 | 0.022 ms |
| DecoderBlock 0 | 370.872 ms | 10 | 37.087 ms |
| DecoderBlock 1 | 363.737 ms | 10 | 36.374 ms |
| Final LayerNorm | 0.416 ms | 10 | 0.042 ms |
| LM Head | 1168.798 ms | 10 | 116.880 ms |

The measured component times closely account for the total end-to-end inference latency.

### Performance Analysis

- The profiling experiment identified the LM Head as the dominant CPU bottleneck.
- The average execution time of the LM Head was 116.880 ms compared with DecoderBlock 0 with 37.087 ms and DecoderBlock 1 with 36.374 ms.
- The LM Head alone consumes approximately 61% of the measured end-to-end inference latency for the current benchmark configuration.
- The two decoder blocks combined consume approximately 39% of the total latency, while embedding and final layer normalization contribute a negligible amount.

### Key Observation

- The first profiling experiment demonstrated an important performance-engineering principle: **Performance bottlenecks should be measured rather than assumed**.
- Although transformer attention is commonly considered computationally expensive, the measured bottleneck for the current model configuration is the final vocabulary projection.
- The profiling results therefore guide the next stage of investigation toward the implementation and computational behavior of the Linear layer used by the LM Head.

### Next Investigation

The next profiling and analysis step will examine the CPU implementation of the:

```text
LM Head
```

## Experiment 2 — LM Head Optimization

The initial CPU profile was collected using a sequence length of 32 and 10 measured iterations. The original implementation projected the hidden state of every sequence position through the vocabulary projection:

```text
[32, 256]
     ↓
  LM Head
     ↓
[32, 10000]
```

For next-token prediction, only the final position is required. The computation was therefore changed to:

```text
[32, 256]
     ↓
extract final hidden state
     ↓
[1, 256]
     ↓
  LM Head
     ↓
[1, 10000]
```

### Post-Optimization Profile
| Operation | Total Time | Calls | Average Time |
|---|---|---|---:|
| Embedding | 0.128 ms | 10 | 0.013 ms |
| DecoderBlock 0 | 364.958 ms | 10 | 36.496 ms |
| DecoderBlock 1 | 364.685 ms | 10 | 36.469 ms |
| Extract Last Hidden State | 0.008 | 10 | 0.001 |
| Final LayerNorm | 0.342 ms | 10 | 0.034 ms |
| LM Head | 37.170 ms | 10 | 3.717 ms | 


- The LM Head latency decreased from 117.135 ms to 3.717 ms, representing an approximately 31.5× reduction.
- The decoder block timings remain approximately unchanged.
- The extraction of the final hidden state adds only approximately 0.001 ms per call and is therefore negligible relative to the transformer computation.

### Bottleneck Shift

- Before the optimization, the LM Head was the dominant runtime component.
- After the optimization, the decoder blocks become the dominant components.
- The next profiling target is therefore the internal operations of DecoderBlock.

## Experiment 3 — DecoderBlock Profiling

### Objective

Following the optimization of the LM Head, the decoder blocks became the dominant component of next-token inference.

The objective of this experiment was therefore to decompose the execution time of each `DecoderBlock` and determine whether the dominant cost originates from the attention mechanism, feed-forward network, or supporting operations such as LayerNorm and residual connections.

The profiled structure was:

```text
DecoderBlock

├── Attention LayerNorm
├── Attention
├── Residual 1
├── FeedForward LayerNorm
├── FeedForward
└── Residual 2
```

Profiling was performed during next-token inference using a sequence length of 32 across 10 measured iterations following an initial warm-up phase.

### Results

| Operation                 | Total Time | Calls | Average Time |
| ------------------------- | ---------: | ----: | -----------: |
| DecoderBlock 0            | 368.416 ms |    10 |    36.842 ms |
| ├── Attention             | 128.411 ms |    10 |    12.841 ms |
| ├── Attention LayerNorm   |   0.339 ms |    10 |     0.034 ms |
| ├── FeedForward           | 238.731 ms |    10 |    23.873 ms |
| ├── FeedForward LayerNorm |   0.376 ms |    10 |     0.038 ms |
| ├── Residual 1            |   0.138 ms |    10 |     0.014 ms |
| └── Residual 2            |   0.143 ms |    10 |     0.014 ms |
| DecoderBlock 1            | 367.778 ms |    10 |    36.778 ms |
| ├── Attention             | 128.456 ms |    10 |    12.846 ms |
| ├── Attention LayerNorm   |   0.375 ms |    10 |     0.038 ms |
| ├── FeedForward           | 237.976 ms |    10 |    23.798 ms |
| ├── FeedForward LayerNorm |   0.346 ms |    10 |     0.035 ms |
| ├── Residual 1            |   0.136 ms |    10 |     0.014 ms |
| └── Residual 2            |   0.241 ms |    10 |     0.024 ms |
| Embedding                 |   0.136 ms |    10 |     0.014 ms |
| Extract Last Hidden State |   0.010 ms |    10 |     0.001 ms |
| Final LayerNorm           |   0.429 ms |    10 |     0.043 ms |
| LM Head                   |  36.596 ms |    10 |     3.660 ms |

### Performance Analysis

The profiling results show that the two decoder blocks have nearly identical execution times, with average latencies of approximately 36.8 ms each.

Within each decoder block, the FeedForward network is the dominant operation:

* FeedForward: approximately **23.8 ms per block**. accounting for approximately 65% of the execution time of each decoder block.
* Attention: approximately **12.8 ms per block**. accounting for approximately 35% of the execution time of each decoder block.
* LayerNorm and residual operations: less than **0.1 ms per operation**

The results indicate that the dominant computational workload is not the supporting tensor operations but the arithmetic performed inside the FeedForward network.

### Bottleneck Identification

The FeedForward network consists primarily of two Linear transformations separated by a GELU activation:

```text
Input [32, 256]

      ↓

Linear
[32, 256] × [256, 1024]

      ↓

GELU

      ↓

Linear
[32, 1024] × [1024, 256]

      ↓

Output [32, 256]
```

The Linear transformations perform matrix multiplication and therefore account for the majority of the arithmetic work within the FeedForward network. This makes the Linear layer the natural target for lower-level parallel optimization.

### Key Observation

- This experiment confirms that the bottleneck has shifted from the LM Head to the decoder blocks following the next-token optimization.
- Within the decoder blocks, the FeedForward network is the dominant component, and its computational workload is primarily composed of matrix multiplications.

The profiling hierarchy therefore establishes the following bottleneck:

```text
Transformer
    ↓
Decoder Blocks
    ↓
FeedForward
    ↓
Linear transformations
    ↓
Matrix multiplication
```

The supporting operations, including LayerNorm, residual connections, and GELU, contribute comparatively little to the measured runtime.

### Next Investigation

The next stage will implement and validate a CUDA-based Linear operation.

The process will follow:

```text
CPU Linear
    ↓
CUDA Linear
    ↓
Numerical correctness validation
    ↓
CPU vs GPU benchmark
    ↓
GPU profiling
    ↓
CUDA optimization
```

The initial CUDA implementation will prioritize correctness and a clear CPU-to-GPU performance comparison before introducing more advanced optimizations.
