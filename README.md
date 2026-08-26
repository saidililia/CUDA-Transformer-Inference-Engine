# CUDA Transformer Inference Engine

A lightweight decoder-only Transformer inference engine implemented in C++, with a CPU reference implementation and a CUDA implementation planned.

The project focuses on understanding Transformer inference at the systems level, including tensor operations, GPU parallelism, CUDA kernel design, memory behavior, profiling, and hardware-aware optimization.

---

## Project Status

**Current phase: CPU reference complete — CUDA implementation next**

The CPU implementation currently provides:

- A complete 2-layer decoder-only Transformer inference pipeline
- Custom C++ tensor and neural network components
- Multi-head causal self-attention
- Rotary Position Embeddings (RoPE)
- Feed-forward network with GELU activation
- LayerNorm and residual connections
- Unit tests for tensors, layers, attention, and the Transformer
- A reproducible CPU performance baseline

CUDA implementation, GPU profiling, and kernel optimization are planned next.

---

## Architecture

The current Transformer configuration is:

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Layers | 2 |
| Hidden size | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| FFN intermediate size | 1024 |
| Vocabulary size | 10,000 |
| Maximum sequence length | 128 |
| Data type | FP32 |
| Attention | Causal multi-head self-attention |
| Positional encoding | RoPE |
| Activation | GELU |
| Normalization | LayerNorm |

The inference pipeline is:

```text
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
Decoder Block 1
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    │
    ▼
Decoder Block 2
    │
    ├── LayerNorm
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed-Forward Network
    └── Residual Connection
    │
    ▼
Final LayerNorm
    │
    ▼
Language Model Head
    │
    ▼
Predicted Token
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed architecture.

---

## CPU Baseline

The current CPU benchmark uses a sequence length of 32 and 10 iterations.

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
```

This baseline will be used for comparison with the CUDA implementation.

See [`docs/benchmark.md`](docs/benchmark.md) for the benchmarking methodology and results.

---

## Repository Structure

```text
.
├── include/
│   ├── layers/
│   └── utils/
│
├── src/
│   ├── layers/
│   └── utils/
│
├── tests/
│
├── docs/
│   ├── architecture.md
│   ├── benchmark.md
│   ├── design-decisions.md
│   ├── profiling.md
│   ├── references.md
│   └── roadmap.md
│
├── CMakeLists.txt
└── README.md
```

---

## Build

The project uses CMake and Microsoft Visual Studio on Windows.

### Configure the project

```powershell
cmake -S . -B build
```

### Build the Release configuration

```powershell
cmake --build .\build --config Release
```

---

## Run

### Tests

Run the test executables from the Release build directory:

```powershell
.\build\Release\tensor_tests.exe
.\build\Release\layer_tests.exe
.\build\Release\attention_tests.exe
.\build\Release\transformer_tests.exe
```

### Expected Result

```text
Tensor tests passed.
Layer tests passed.
Attention tests passed.
Transformer tests passed.
```

### CPU Inference

```powershell
.\build\Release\transformer_cpu.exe
```

### CPU Benchmark

```powershell
.\build\Release\cpu_benchmark.exe
```

---

## Development Roadmap

The project is being developed in four main phases:

```text
CPU Reference
      ↓
CPU Profiling
      ↓
CUDA Implementation
      ↓
GPU Profiling & Optimization
```

The CUDA phase will progressively move compute-intensive Transformer operations to the GPU, validate CUDA results against the CPU reference, and profile kernel performance before applying optimizations.

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed development plan.

---

## Documentation

| Document | Description |
|---|---|
| [`architecture.md`](docs/architecture.md) | System architecture and Transformer computation flow |
| [`benchmark.md`](docs/benchmark.md) | Benchmarking methodology and performance results |
| [`design-decisions.md`](docs/design-decisions.md) | Engineering decisions and their rationale |
| [`profiling.md`](docs/profiling.md) | Profiling results and performance analysis |
| [`references.md`](docs/references.md) | Technical references and learning resources |
| [`roadmap.md`](docs/roadmap.md) | Development roadmap |

---

## Goals

The project is intended to provide a practical understanding of Transformer inference from the model computation level down to GPU execution.

The main goals are:

- Build the Transformer inference pipeline from low-level components
- Understand how Transformer operations map to parallel hardware
- Implement custom CUDA kernels
- Analyze memory access and execution behavior
- Profile GPU workloads
- Optimize kernels based on measured bottlenecks
- Establish reproducible CPU and GPU performance comparisons