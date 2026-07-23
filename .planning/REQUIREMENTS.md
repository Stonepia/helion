# Requirements: Helion XPU Parity

**Defined:** 2026-07-22
**Core Value:** Every device-agnostic Helion test that runs on CUDA also executes and
passes on XPU, with no unexplained XPU-only skip.

## v1 Requirements

### Environment

- [x] **ENV-01**: Developer can activate a repository-local `.venv` containing XPU
  nightly PyTorch, its coupled Triton package, and editable Helion development dependencies.
- [x] **ENV-02**: The environment report proves that XPU is available and can execute a
  basic PyTorch matrix operation before the Helion suite begins.

### Baseline

- [x] **BASE-01**: The complete non-distributed PyTest suite runs with
  `HELION_AUTOTUNE_EFFORT=none`, `HELION_BACKEND=triton`, and
  `TRITON_XPU_GEN_NATIVE_CODE=1`.
- [x] **BASE-02**: The first run records raw terminal output and JUnit XML under
  `xpu-enabling-logs/` without failure reruns hiding the initial result.
- [x] **BASE-03**: A written baseline report lists every skip and failure with its node ID,
  reason, root-cause category, suspected ownership layer, and recommended next step.

### Discussion Gates

- [ ] **DISC-01**: The user receives and discusses the baseline report before any XPU fix
  worktree is created.
- [ ] **DISC-02**: Every root-cause category has an explicit user-approved implementation
  approach before code changes begin.

### Fixes

- [ ] **FIX-01**: Every XPU-only skip in a device-agnostic test is removed or converted into
  an executing XPU test that passes.
- [ ] **FIX-02**: Each approved root-cause category is implemented in an isolated worktree
  and fix branch with atomic commits.
- [ ] **FIX-03**: Simple non-overlapping categories may run in parallel, while difficult
  compiler/runtime/codegen categories run serially.

### Verification and Publication

- [ ] **VER-01**: Targeted tests pass in each fix branch and the integrated complete
  non-distributed functionality suite passes on XPU.
- [ ] **VER-02**: Any remaining gap has a minimal reproducer, owning layer, technical root
  cause, unblock condition, and justification reviewed by the user.
- [ ] **PUB-01**: Verified fix branches are pushed only to the personal fork and target that
  fork's `main` branch; no public upstream PR is opened.

## v2 Requirements

### Distributed XPU

- **DIST-01**: Distributed Helion examples execute and pass across multiple XPU devices.

### Performance

- **PERF-01**: XPU kernels meet separately defined performance and autotuning objectives.

## Out of Scope

| Feature | Reason |
|---------|--------|
| `test/test_examples_dist.py` | Distributed testing is explicitly deferred. |
| Performance benchmarks | This milestone validates functionality rather than tuning. |
| Public upstream PRs | Personal-fork review is required first. |
| Standalone Triton build | Nightly PyTorch supplies the compatible Triton dependency. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| BASE-01 | Phase 1 | Complete |
| BASE-02 | Phase 1 | Complete |
| BASE-03 | Phase 1 | Complete |
| DISC-01 | Phase 2 | Pending |
| DISC-02 | Phase 2 | Pending |
| FIX-01 | Phase 3 | Pending |
| FIX-02 | Phase 3 | Pending |
| FIX-03 | Phase 3 | Pending |
| VER-01 | Phase 4 | Pending |
| VER-02 | Phase 4 | Pending |
| PUB-01 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-22*
*Last updated: 2026-07-22 after Phase 1 baseline collection*
