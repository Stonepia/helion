---
title: Helion integrations
mapped: 2026-07-22
---

# Integrations

## PyTorch

- Tensor metadata and execution flow through PyTorch APIs.
- `torch.compile` and Inductor integration is exercised by `test/test_torch_compile.py`.
- Runtime device selection is centralized partly in `helion/_testing.py`.

## Accelerator Runtimes

- NVIDIA CUDA and AMD ROCm use the Triton backend through `helion/_compiler/triton/`.
- Intel XPU also uses Triton; CI selects the XPU runtime in `.github/matrix.json`.
- TPU/Pallas, CUTLASS CuTe, TileIR, Metal, and MTIA are optional or specialized paths.
- Hardware metadata and cache keys are represented in `helion/_hardware.py`.

## Continuous Integration

- `.github/workflows/test.yml` installs uv, PyTorch, Helion, and test plugins.
- The XPU matrix entry uses an Intel GPU runner and XPU runtime wheel index.
- XPU CI exports `TRITON_XPU_GEN_NATIVE_CODE=1` before invoking PyTest.
- Normal non-distributed jobs ignore `test/test_examples_dist.py`.

## External Services

- Autotuner modules contain optional remote-cache and LLM integrations.
- Core compiler and functionality tests do not require a database or web service.
- Network access is required only for dependency installation and optional integrations.

## XPU Baseline Constraint

- The requested development baseline uses XPU nightly PyTorch and its coupled Triton.
- No standalone Triton source build is part of the initial local setup.
