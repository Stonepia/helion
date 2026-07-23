# Roadmap: Helion XPU Parity

## Milestone v1: XPU Functionality Parity

### Phase 1: Environment and Baseline
**Goal:** Produce a reproducible XPU nightly environment and authoritative non-distributed
test inventory without changing Helion behavior.
**Requirements:** ENV-01, ENV-02, BASE-01, BASE-02, BASE-03
**Success Criteria**:
1. `.venv` imports XPU nightly PyTorch, Triton, and editable Helion.
2. XPU availability and a basic matrix operation are recorded in repository-local logs.
3. The full non-distributed suite completes with the approved functionality-only settings.
4. `BASELINE-REPORT.md` classifies every observed skip and failure.
5. No source-code fix is made during baseline collection.

### Phase 2: Baseline Review and Fix Design
**Goal:** Review the baseline with the user and turn each root-cause category into an
explicitly approved, independently verifiable fix design.
**Requirements:** DISC-01, DISC-02
**Success Criteria**:
1. The user receives the raw artifacts and a concise baseline report.
2. Every XPU-specific skip/failure is assigned to a root-cause category.
3. Each category documents alternatives, recommendation, risk, files, and test command.
4. No worktree is created for a category before the user approves that category.

### Phase 3: Approved Category Fixes
**Goal:** Implement all user-approved XPU parity fixes in isolated branches while keeping
categories reviewable and independently testable.
**Requirements:** FIX-01, FIX-02, FIX-03
**Success Criteria**:
1. Each category uses its own worktree, branch, ownership boundary, and atomic commits.
2. All tests previously skipped for that category execute and pass on XPU.
3. Simple non-overlapping categories may run concurrently without shared-file conflicts.
4. Core compiler/runtime/codegen changes execute serially and receive focused verification.

### Phase 4: Integration and Personal-Fork Review
**Goal:** Prove complete XPU functionality parity and publish only personal-fork PRs.
**Requirements:** VER-01, VER-02, PUB-01
**Success Criteria**:
1. The integrated full non-distributed suite passes with no unexplained XPU-specific skip.
2. Every retained external limitation has the required detailed gap report and user review.
3. Each fix branch is pushed only to the configured personal fork.
4. Personal-fork PRs target personal `main`; no public upstream PR is created.

## Progress

| Phase | Status | Requirements |
|-------|--------|--------------|
| 1. Environment and Baseline | In progress | ENV-01, ENV-02, BASE-01, BASE-02, BASE-03 |
| 2. Baseline Review and Fix Design | Pending | DISC-01, DISC-02 |
| 3. Approved Category Fixes | Pending | FIX-01, FIX-02, FIX-03 |
| 4. Integration and Personal-Fork Review | Pending | VER-01, VER-02, PUB-01 |
