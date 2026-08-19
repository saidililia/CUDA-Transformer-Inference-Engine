
```markdown
# Design Decisions

This document records important engineering decisions made during development of the CUDA Transformer Inference Engine and explains the reasoning behind them.

The purpose is to preserve the rationale behind the architecture rather than documenting only what the implementation currently does.

---

## 1. CPU Reference Before CUDA

### Decision

Implement the complete Transformer forward pass on the CPU before developing CUDA kernels.

### Rationale

CUDA introduces additional sources of complexity:

- parallel execution
- device memory
- host/device transfers
- thread mapping
- synchronization
- kernel launch configuration
- GPU-specific numerical behavior

Implementing CUDA first would make it harder to distinguish a mathematical/modeling bug from a GPU implementation bug.

The CPU implementation therefore acts as a reference.

### Development model

```text
CPU correctness
      ↓
CPU benchmark
      ↓
CPU profiling
      ↓
CUDA implementation
      ↓
CPU ↔ CUDA validation
      ↓
GPU optimization