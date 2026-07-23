# Phase 1 Baseline Report: Helion XPU Functionality Parity

**Date:** 2026-07-22  
**Upstream source:** `origin/main` at `48971c7baf88f1378cdd72df51b99175ff3edb72`  
**Planning head:** `4f9b019fd7360e7084ddde65dc11911252019315`  
**Status:** Awaiting user review; no Helion source fix or fix worktree has been created.

## Executive Summary

The approved first non-distributed XPU run completed in 6 minutes 33 seconds:

| Result | Count |
|--------|------:|
| Passed | 1,643 |
| Skipped | 1,295 |
| Failed | 13 outcomes: 12 failures and 1 worker-crash error |

The run started with 2,941 collected items. JUnit contains 2,951 records because xdist
replaced `gw1` after the RNG worker crash and replayed work.

| Skip population | Count | Interpretation |
|-----------------|------:|----------------|
| XPU parity candidates | 269 | Must be enabled or receive a detailed, user-reviewed explanation. |
| Distributed/CCL | 47 | Explicitly deferred from this milestone. |
| Backend-generic or unrelated | 979 | Pallas/TPU, Metal, CuTe-only, unavailable optional dependencies, or skips expected in the same Triton job on CUDA. |

The 269 count is conservative rather than a claim that the remaining 979 can never hide an
XPU issue. The complete 1,295-row inventory is retained for re-audit.

## Environment Provenance

| Component | Observed value |
|-----------|----------------|
| Python | 3.12.13 |
| uv | 0.10.8 |
| PyTorch | `2.14.0.dev20260722+xpu` |
| Triton | `3.7.2` (`triton-xpu==3.7.2+git5fcc14d9`) |
| Helion | editable `0.0.0` |
| Hardware | 4 Intel Data Center GPU Max 1550 devices; PyTorch exposes 8 logical XPU devices |
| Smoke check | XPU 256x256 PyTorch matrix multiplication passed |

The environment is repository-local at `.venv`. XPU nightly PyTorch installed its matching
Triton package; no standalone Triton build was performed.

## Baseline Command

```bash
LD_LIBRARY_PATH=/home/stonepia/helion/.venv/lib \
HELION_AUTOTUNE_EFFORT=none \
HELION_BACKEND=triton \
TRITON_XPU_GEN_NATIVE_CODE=1 \
.venv/bin/python -m pytest \
  -n4 -ra --timeout=60 --timeout-method=thread \
  --ignore=test/test_examples_dist.py \
  --junitxml=xpu-enabling-logs/07-baseline-junit.xml \
  .
```

This follows the user's functionality-only contract. Failures caused solely by forcing
`HELION_AUTOTUNE_EFFORT=none` are tracked as harness conflicts rather than XPU defects.

## Failure Triage

### A. Functionality-mode harness conflicts: 10

Ten cache/autotuner/config tests failed because the global override prevented the behavior
they explicitly assert. A focused rerun without the override passed all 10 in 9.68 seconds.
The exact node IDs are in `xpu-enabling-logs/09-failures.tsv` and the passing control log is
`xpu-enabling-logs/10-rerun-without-effort-none.log`.

**Suspected owner:** test harness/environment contract.  
**Recommended next step:** keep the main functionality suite at effort `none`, but use a
small control invocation without the override for tests whose purpose is autotuning or effort
configuration. This requires user approval as part of the verification contract.

### B. Reproducible XPU correctness failures: 2

- `test.test_misc.TestMisc::test_torch_topk_in_kernel`
- `test.test_misc.TestMisc::test_torch_topk_smallest`

Both fail serially with effort `none`. The largest-value case has 3/16 mismatches with
maximum absolute difference 2.150169; the smallest-value case has 9/16 mismatches with
maximum absolute difference 0.253067.

**Suspected owner:** Helion lowering/code generation or XPU Triton `topk` semantics.  
**Recommended next step:** compare eager PyTorch, generated Triton, and a direct Triton
reproducer before deciding whether the fix belongs in Helion or upstream Triton.

### C. XPU RNG worker crash: 1

- `test.test_rng.TestRNG::test_rand_like_with_dynamic_tile_sizes`

This killed xdist worker `gw1`, which PyTest replaced. It was not immediately rerun alone
because native-crash diagnosis should use dedicated logs and one GPU/process.

**Suspected owner:** XPU RNG code generation/runtime, potentially Triton-XPU.  
**Recommended next step:** serial reproduction with faulthandler and generated-code capture,
then reduce to the smallest dynamic tile shape.

## XPU Skip Classification

Counts cover all 269 conservative XPU parity candidates and are mutually exclusive at this
reporting level.

| Category | Count | Suspected ownership | Proposed direction |
|----------|------:|---------------------|--------------------|
| Autodiff scan/HOP aborts | 75 | PyTorch scan-HOP/autograd integration and Helion test policy | Reproduce one minimal backward graph; determine whether to remove the class skip, work around locally, or file upstream. |
| Tensor descriptors | 59 | Triton-XPU capability and Helion descriptor gating/lowering | Separate capability-gating skips from accuracy bugs; enable supported shapes first. |
| CUDA assumptions and helpers | 49 | Helion tests, config generation, device-property abstraction | Replace CUDA availability/property assumptions with backend-neutral helpers; retain only genuinely CUDA-semantic tests. |
| Architecture-specific operations/artifacts | 30 | CUDA/PTX/pretuned/float6 semantics | Identify device-agnostic intent versus NVIDIA-specific contracts; add XPU equivalents where semantics exist. |
| RNG/random | 24 | Helion RNG lowering/runtime and Triton-XPU | Reduce crashes/timeouts serially, then enable deterministic and distribution tests in slices. |
| Non-RNG stability/timeouts | 13 | Generated kernels, XPU compiler/runtime, test sizing | Reproduce serially and distinguish compiler failure from xdist contention. |
| Other correctness/codegen gaps | 19 | Helion lowering/codegen and possibly Triton-XPU | Handle as focused groups, beginning with reproducible numerical failures such as `topk`. |
| **Total** | **269** | | |

The detailed candidate list is `xpu-enabling-logs/17-xpu-gap-candidates.tsv`; the complete skip
inventory is `xpu-enabling-logs/16-all-skips.tsv`. Aggregated source-marker and helper-gate
evidence is in logs 12 through 15.

## Proposed Discussion and Execution Order

No item below is approved merely by appearing here.

1. Agree on the functionality-mode verification contract for the 10 autotuning-sensitive
   tests.
2. Discuss CUDA assumptions/helpers, likely the safest and most parallelizable category.
3. Discuss non-RNG stability/timeouts and small correctness groups, starting with `topk`.
4. Discuss RNG, tensor descriptors, and autodiff as difficult serial investigations.
5. Discuss architecture-specific tests individually because parity may require an XPU semantic
   equivalent rather than executing PTX/CUDA artifacts.

After user approval, each category receives its own worktree, branch, targeted test contract,
atomic commits, and personal-fork-only publication. Difficult compiler/runtime investigations
remain serial; simple non-overlapping changes may use parallel subagents.

## Evidence Index

- `xpu-enabling-logs/01-uv-venv.log` through `06-environment-smoke.log`
- `xpu-enabling-logs/07-baseline-pytest.log`
- `xpu-enabling-logs/07-baseline-junit.xml`
- `xpu-enabling-logs/08-skip-reason-summary.txt`
- `xpu-enabling-logs/09-failures.tsv`
- `xpu-enabling-logs/10-rerun-without-effort-none.log`
- `xpu-enabling-logs/11-rerun-topk.log`
- `xpu-enabling-logs/12-source-xpu-cuda-skip-markers.txt` through
  `17-xpu-gap-candidates.tsv`

## Mandatory Checkpoint

Phase 1 evidence collection is complete. Phase 2 cannot create a fix worktree or edit Helion
source until the user reviews this report and explicitly approves the first category approach.
