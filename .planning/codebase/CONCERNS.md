---
title: Helion XPU enabling concerns
mapped: 2026-07-22
---

# Concerns

## Known Portability Risks

- `helion/_testing.py` exposes both device-neutral `DEVICE` and CUDA-specific predicates.
- Multiple tests import `skipIfXPU` or `skipIfNotCUDA`.
- Several otherwise generic tests directly call `torch.cuda` capability or property APIs.
- Existing skip reasons mention XPU timeouts, unsupported RNG specialization, and CUDA checks.

## Test-Mode Risk

- The requested baseline globally sets `HELION_AUTOTUNE_EFFORT=none`.
- This makes functionality tests fast but changes behavior of autotuner-specific tests.
- Such failures require a confirming rerun without the override before classification.

## Environment Risk

- XPU nightly PyTorch brings a coupled Triton package that may differ from CI's pinned fork.
- CI currently tests a PyTorch 2.12 XPU matrix entry rather than nightly.
- Environment metadata must be recorded before attributing failures to Helion.

## Execution Risk

- Four xdist workers can contend for XPU memory or compiler resources.
- Timeout failures must be distinguished from unsupported functionality.
- The full non-distributed suite should take roughly 15–20 minutes on the target machine.

## Scope Controls

- Distributed tests are explicitly deferred.
- Benchmarks are outside functionality parity and are not part of the first baseline.
- No implementation begins until the baseline report is reviewed with the user.
- Each fix category requires a separate user discussion before worktree creation.

## Publication Safety

- Fix branches may be committed and pushed only to the personal fork.
- Public upstream branches and pull requests are out of scope.
