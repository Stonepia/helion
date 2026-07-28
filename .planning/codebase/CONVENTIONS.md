---
title: Helion development conventions
mapped: 2026-07-22
---

# Conventions

## Python

- Use Python 3.10+ type hints and `from __future__ import annotations`.
- Ruff formatting uses double quotes and an 88-character line length.
- Imports are sorted and normally remain at module scope.
- Public Helion usage imports `helion` and `helion.language as hl` separately.

## Tests

- New tests live in `test/` and follow `test_<feature>.py` naming.
- Tensor creation should normally use `helion._testing.DEVICE`.
- Correctness checks commonly compare a Helion kernel against a PyTorch reference.
- `helion._testing.skipIfFn` defers device checks until execution for xdist safety.
- Skip reasons must describe the missing capability or constraint precisely.

## Device Portability

- Generic tests should avoid direct `torch.cuda` calls.
- Capability checks belong in shared compatibility or hardware helpers when possible.
- Tile indexing preserves dimensions and should not be worked around with rank-changing code.
- Kernel code must not use `print()`; diagnostics belong on the host or in logging.

## Change Discipline

- Keep fixes focused on the observed XPU gap.
- Do not introduce defensive `hasattr`, `getattr`, or broad exception handling without need.
- Run targeted tests during iteration and the agreed non-distributed suite for integration.
- Use `./lint.sh fix` before publication when source changes are made.

## Git

- Each XPU root-cause category receives its own worktree and branch.
- Atomic commits are authorized for this effort.
- Publication is restricted to the user's personal fork, never public upstream.
