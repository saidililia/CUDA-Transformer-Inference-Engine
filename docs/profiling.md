
```markdown
# Profiling

This document describes the profiling methodology for the CUDA Transformer Inference Engine.

Profiling is treated as an engineering activity used to identify bottlenecks and validate optimization hypotheses.

---

## 1. Profiling Philosophy

The project follows:

```text
Measure
   ↓
Identify bottleneck
   ↓
Understand bottleneck
   ↓
Form optimization hypothesis
   ↓
Implement
   ↓
Measure again