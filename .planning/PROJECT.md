# Helion XPU Parity

## What This Is

This project removes XPU-specific gaps from Helion's device-agnostic functionality.
It establishes a reproducible XPU nightly environment, inventories every non-distributed
test skip and failure, and enables the affected tests through separately reviewed fixes.

## Core Value

Every device-agnostic Helion test that runs on CUDA must also execute and pass on XPU,
with no unexplained XPU-only skip.

## Requirements

### Validated

- ✓ Helion provides a device-agnostic Python DSL and Triton compiler path — existing.
- ✓ The shared test layer selects XPU through `helion._testing.DEVICE` — existing.
- ✓ CI defines an Intel XPU runtime entry and validates basic XPU compute — existing.

### Active

- [ ] Build a local uv environment with XPU nightly PyTorch and editable Helion.
- [ ] Run the complete non-distributed functionality suite with autotuning disabled.
- [ ] Produce a durable baseline report that classifies all skips and failures.
- [ ] Discuss and approve every root-cause category before implementation begins.
- [ ] Enable all device-agnostic tests on XPU with separate worktrees and fix branches.
- [ ] Verify the integrated suite and publish fixes only to the personal fork.

### Out of Scope

- Distributed examples and distributed test modules — explicitly deferred for this milestone.
- Performance benchmarking or XPU tuning — this milestone validates functionality only.
- Public upstream pull requests — fixes are published to the personal fork first.
- Standalone Triton source builds — XPU nightly PyTorch supplies the coupled Triton package.

## Context

Helion is expected to be device agnostic, so an XPU/CUDA behavioral gap is a defect by
default. Existing XPU skips may reflect test harness assumptions, direct CUDA APIs,
timeouts, compiler/runtime gaps, or upstream limitations. None may be retained without a
specific reproducer and explanation. The expected non-distributed suite duration is
approximately 15–20 minutes on the target XPU machine.

## Constraints

- **Environment**: Create `.venv` in the repository with uv and install XPU nightly PyTorch.
- **Triton**: Use the Triton package installed with the nightly PyTorch wheel.
- **Functionality mode**: Set `HELION_AUTOTUNE_EFFORT=none` for the baseline suite.
- **CI parity**: Set `HELION_BACKEND=triton` and `TRITON_XPU_GEN_NATIVE_CODE=1`.
- **Test scope**: Run all tests except `test/test_examples_dist.py`.
- **Diagnostics**: Write every install, hardware, test, and analysis log under
  `xpu-enabling-logs/`; never use `/tmp` for project logs.
- **Baseline checkpoint**: Stop after the report and discuss results with the user.
- **Fix checkpoint**: Discuss every fix category before creating its worktree or editing code.
- **Isolation**: Give each approved category its own worktree and fix branch.
- **Execution**: Parallelize simple non-overlapping categories; serialize difficult core changes.
- **Git**: Atomic commits and pushes are authorized only for the personal fork.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat XPU-only skips as defects | Helion is device agnostic | — Pending |
| Run functionality with autotuning disabled | Fast coverage is the objective | — Pending |
| Defer distributed tests | Multi-device work is not part of this first milestone | — Pending |
| Use nightly-coupled Triton | Avoid a mismatched standalone Triton build | — Pending |
| Require baseline and per-category discussions | User approval controls scope and approach | — Pending |
| Publish only to the personal fork | Public PRs are intentionally deferred | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Move invalidated requirements to Out of Scope with a reason.
2. Move verified requirements to Validated with a phase reference.
3. Add newly discovered requirements to Active.
4. Record decisions and whether their outcomes need revisiting.
5. Update the project description if the scope changes.

**After the milestone**:
1. Review all active and out-of-scope requirements.
2. Reconfirm that device parity remains the core value.
3. Record the final XPU test inventory and remaining external gaps, if any.

---
*Last updated: 2026-07-22 after initialization*
