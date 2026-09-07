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

Operation	Total Time	Calls	Average Time
Embedding	0.222 ms	10	0.022 ms
DecoderBlock 0	370.872 ms	10	37.087 ms
DecoderBlock 1	363.737 ms	10	36.374 ms
Final LayerNorm	0.416 ms	10	0.042 ms
LM Head	1168.798 ms	10	116.880 ms

The measured component times closely account for the total end-to-end inference latency.

### Performance Analysis

The profiling experiment identified the LM Head as the dominant CPU bottleneck.

The average execution time of the LM Head was:

116.880 ms


Compared with:

DecoderBlock 0: 37.087 ms
DecoderBlock 1: 36.374 ms


The LM Head alone consumes approximately 61% of the measured end-to-end inference latency for the current benchmark configuration.

The two decoder blocks combined consume approximately 39% of the total latency, while embedding and final layer normalization contribute a negligible amount.

Key Observation

The first profiling experiment demonstrated an important performance-engineering principle:

Performance bottlenecks should be measured rather than assumed.

Although transformer attention is commonly considered computationally expensive, the measured bottleneck for the current model configuration is the final vocabulary projection.

The profiling results therefore guide the next stage of investigation toward the implementation and computational behavior of the Linear layer used by the LM Head.

### Next Investigation

The next profiling and analysis step will examine the CPU implementation of the:

LM Head

### Experiment 2 — LM Head Optimization

The initial CPU profile was collected using a sequence length of 32 and 10 measured iterations.

The original implementation projected the hidden state of every sequence position through the vocabulary projection:

[32, 256]
     ↓
  LM Head
     ↓
[32, 10000]


For next-token prediction, only the final position is required.

The computation was therefore changed to:

[32, 256]
     ↓
extract final hidden state
     ↓
[1, 256]
     ↓
  LM Head
     ↓
[1, 10000]

Post-Optimization Profile
Operation	Total (ms)	Calls	Average (ms)
DecoderBlock 0	364.958	10	36.496
DecoderBlock 1	364.685	10	36.469
Embedding	0.128	10	0.013
Extract Last Hidden State	0.008	10	0.001
Final LayerNorm	0.342	10	0.034
LM Head	37.170	10	3.717
Before vs. After
LM Head Latency
Metric	Before	After
Latency	117.135 ms	3.717 ms
Reduction	—	≈31.5×

The LM Head latency decreased from 117.135 ms to 3.717 ms, representing an approximately 31.5× reduction.

Decoder Block Latency
Operation	Before	After
DecoderBlock 0	37.257 ms	36.496 ms
DecoderBlock 1	36.915 ms	36.469 ms

The decoder block timings remain approximately unchanged.

The extraction of the final hidden state adds only approximately 0.001 ms per call and is therefore negligible relative to the transformer computation.

Bottleneck Shift

Before the optimization, the LM Head was the dominant runtime component.

After the optimization, the decoder blocks become the dominant components.

The next profiling target is therefore the internal operations of DecoderBlock.