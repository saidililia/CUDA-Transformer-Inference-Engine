# CUDA Transformer Inference Engine

A lightweight **2-layer decoder-only Transformer inference engine written from scratch in C++**, with a CPU reference implementation and an incremental CUDA backend focused on understanding GPU execution, memory behavior, and hardware-aware optimization.

The project is being developed as a systems-oriented exploration of **Transformer inference, CUDA kernel programming, GPU parallelism, memory access patterns, profiling, and performance optimization**.

> **Current status:** CPU reference implementation complete, validated, and benchmarked. CUDA backend development is the next major milestone.

---

## Overview

Modern Transformer inference relies heavily on highly optimized GPU kernels and libraries. This project takes a lower-level approach: instead of treating inference as a black box, it implements the computational pipeline explicitly and progressively maps its workloads onto GPU hardware.

The development process follows:

```text
Transformer Mathematics
        ↓
C++ CPU Reference
        ↓
Correctness Validation
        ↓
CPU Benchmarking
        ↓
CPU Profiling
        ↓
CUDA Kernels
        ↓
CPU ↔ GPU Numerical Validation
        ↓
GPU Profiling
        ↓
Hardware-Aware Optimization