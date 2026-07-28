---
title: Helion technology stack
mapped: 2026-07-22
---

# Technology Stack

## Runtime

- Python 3.10+ package configured in `pyproject.toml`.
- PyTorch supplies tensors, devices, compilation integration, and runtime APIs.
- Triton is the primary code-generation backend used for CUDA, ROCm, and XPU.
- Optional compiler backends live under `helion/_compiler/cute/`, `pallas/`, and `metal/`.

## Packaging

- Hatchling and hatch-vcs provide the build backend.
- Development dependencies are declared in `pyproject.toml` under the `dev` extra.
- CI creates `.venv` with uv and installs Helion editable with `uv pip install -e .'[dev]'`.

## Core Modules

- `helion/language/` defines the user-facing DSL.
- `helion/_compiler/` traces, analyzes, lowers, and generates backend programs.
- `helion/runtime/` binds kernels, settings, configs, caches, and launch behavior.
- `helion/autotuner/` generates, benchmarks, and selects configurations.
- `helion/_testing.py` centralizes test devices, skips, and correctness helpers.

## Quality Tooling

- PyTest is the test framework; pytest-timeout is a declared dev dependency.
- CI additionally installs pytest-xdist and pytest-rerunfailures.
- Ruff formatting/linting and Pyrefly type checking are driven through `lint.sh`.
- Tests are expected to run quickly and are organized under `test/test_*.py`.
