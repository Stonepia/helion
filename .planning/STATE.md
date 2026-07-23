---
status: needs_review
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
- Environment setup: complete; XPU import and matrix-multiplication smoke test passed.
- Baseline test run: complete; 1,643 passed, 1,295 skipped, 12 failed, 1 worker-crash error.
- Baseline report: `.planning/phases/01-environment-and-baseline/01-BASELINE-REPORT.md`.
- Mandatory next checkpoint: user review and category-by-category design discussion before
  any fixes.

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

Review the baseline report with the user and agree on the functionality-mode verification
contract before selecting the first fix category.
