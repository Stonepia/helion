---
title: Helion architecture
mapped: 2026-07-22
---

# Architecture

## Overview

Helion is a Python-embedded kernel DSL. Python functions decorated as Helion kernels are
bound to tensor arguments, traced, compiled to a selected backend, and launched on the
input device.

## Main Flow

1. Public decorators and language constructs enter through `helion/__init__.py` and
   `helion/language/`.
2. `helion/runtime/kernel.py` binds arguments, settings, and configuration state.
3. `helion/_compiler/kernel_compiler.py` and related compiler modules build the IR.
4. Backend selection goes through `helion/_compiler/backend_registry.py`.
5. Backend-specific lowering and code generation live in `helion/_compiler/<backend>/`.
6. Generated programs execute and correctness tests compare against PyTorch references.

## Portability Boundary

- The DSL and most compiler logic are intended to be device agnostic.
- `TritonBackend` serves multiple accelerator runtime targets.
- Platform capability queries should be isolated behind compatibility or hardware helpers.
- Direct `torch.cuda` calls in otherwise generic code or tests are portability risks.

## Configuration

- `helion/runtime/settings.py` resolves environment variables and kernel settings.
- `HELION_BACKEND` selects the backend and `HELION_AUTOTUNE_EFFORT` controls search effort.
- `helion/runtime/config.py` represents launch and tiling configuration.

## Testing Boundary

- `helion/_testing.py` selects `DEVICE`, defines backend predicates, and supplies decorators.
- Examples are exercised through `test/test_examples.py`.
- Distributed examples have a separate test module and are excluded from this XPU phase.
