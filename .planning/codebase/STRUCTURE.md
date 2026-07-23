---
title: Helion repository structure
mapped: 2026-07-22
---

# Repository Structure

## Product Code

- `helion/` — public package, compiler, runtime, autotuner, and testing helpers.
- `helion/language/` — DSL operations exposed to kernel authors.
- `helion/_compiler/` — tracing, analysis, IR, code generation, and backend registry.
- `helion/_compiler/triton/` — Triton-specific code generation used by XPU.
- `helion/runtime/` — kernel binding, settings, configuration, and launch behavior.
- `helion/autotuner/` — search algorithms, benchmarking, caches, and config generation.

## Validation

- `test/` — PyTest and unittest-compatible test modules.
- `test/*.expected` — golden generated-code expectations.
- `examples/` — runnable kernels, largely exercised by `test/test_examples.py`.
- `benchmarks/` — performance runners, outside the initial functionality-only scope.

## Operations

- `.github/workflows/test.yml` — primary CI installation and test entry point.
- `.github/matrix.json` — runtime, Python, PyTorch, backend, and runner matrix.
- `scripts/` — lint, environment, installation, and remote execution utilities.
- `docs/` — Sphinx documentation.

## Planning and Local Artifacts

- `.planning/` — openGSD project context and phase artifacts.
- `.venv/` — requested local uv environment.
- `xpu-enabling-logs/` — local installation, hardware, PyTest, and report artifacts.

## Naming

- Python modules and functions use snake_case.
- Test modules use `test_<feature>.py`.
- Examples must define a `main()` function.
