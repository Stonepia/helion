---
status: executing
current_phase: 1
current_phase_name: Environment and Baseline
milestone: v1
milestone_name: XPU Functionality Parity
progress:
  total_phases: 4
  completed_phases: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-07-22)

**Core value:** Every device-agnostic Helion test that runs on CUDA also executes and
passes on XPU, with no unexplained XPU-only skip.
**Current focus:** Phase 1 — Environment and Baseline.

## Current Position

- Codebase map: complete.
- Planning contract: approved by the user.
- Environment setup: pending.
- Baseline test run: pending.
- Mandatory next checkpoint: report and discuss the baseline before any fixes.

## Decisions

- Run all non-distributed tests with `HELION_AUTOTUNE_EFFORT=none`.
- Use the Triton package installed by XPU nightly PyTorch.
- Exclude only `test/test_examples_dist.py` from the first milestone.
- Save all logs and reports inside the repository.
- Require a user discussion before every fix category.
- Commit and push only personal-fork fix branches.

## Blockers

(None)

## Next Action

Create `.venv`, install and validate the XPU environment, then run the approved baseline.
