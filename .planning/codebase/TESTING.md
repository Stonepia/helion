---
title: Helion testing patterns
mapped: 2026-07-22
---

# Testing

## Framework and Layout

- PyTest discovers tests under `test/`; many classes use unittest APIs.
- `helion/_testing.py` provides `DEVICE`, skip decorators, example checks, and references.
- `DEVICE` prefers XPU when `torch.xpu.is_available()` is true.
- Examples are covered by `test/test_examples.py`.
- Distributed examples are isolated in `test/test_examples_dist.py`.

## CI Command Pattern

- `.github/workflows/test.yml` uses four xdist workers for ordinary accelerator jobs.
- Per-test timeout is 60 seconds with the thread timeout method.
- CI reruns failures twice, but discovery for this effort intentionally does not rerun.
- XPU CI exports `TRITON_XPU_GEN_NATIVE_CODE=1`.

## XPU Discovery Command

- Set `HELION_AUTOTUNE_EFFORT=none` for functionality-only execution.
- Set `HELION_BACKEND=triton` and `TRITON_XPU_GEN_NATIVE_CODE=1`.
- Run all tests except `test/test_examples_dist.py` with `-n4` and `-ra`.
- Emit a JUnit XML report plus a complete terminal log under `xpu-enabling-logs/`.

## Skip Sources

- `helion._testing.skipIfXPU` directly marks known XPU exclusions.
- `helion._testing.skipIfNotCUDA` may hide portability gaps in generic tests.
- Direct CUDA capability/property calls can require device-neutral replacements.
- Runtime skips may also arise from optional dependencies or insufficient device count.

## Diagnosis

- The first baseline is authoritative for skip inventory and deterministic failures.
- Failed tests are rerun individually with `-x -vv -s` and code-generation diagnostics.
- Autotuner-specific failures are also rerun without the global effort override.
- Every XPU-only skip is treated as a defect until a detailed capability analysis proves otherwise.
