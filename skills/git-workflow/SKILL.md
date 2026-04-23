---
name: git-workflow
description: How the agent must handle git branching, commits, and reporting for every task.
version: 1.0.0
---

# Git Workflow — Mandatory for Every Task

**Always follow this workflow exactly. No exceptions.**

## 1. Start from a clean base

```bash
git checkout main         # or the default branch
git pull origin main      # sync with remote
```

If the repo has no `main`, use `master` or whatever `git remote show origin` reports as the default.

## 2. Create a task branch

Branch name format: `agent/<slug>`

Where `<slug>` is derived from the Linear card title:
- lowercase only
- spaces → hyphens
- strip non-alphanumeric characters except hyphens
- max 50 characters
- Example: `agent/remove-stale-skipifxpu-decorators`

```bash
git checkout -b agent/<slug>
```

## 3. Make your changes

- Keep changes minimal and focused on the task
- Do not refactor unrelated code
- Do not touch files not mentioned in the task unless strictly necessary

## 4. Verify

Run the test suite if one exists:
```bash
python -m pytest              # Python projects
# or whatever the project's test command is
```

If tests require hardware (e.g. XPU device), note "requires hardware" and skip running.

## 5. Commit

```bash
git add -A
git commit -m "<short imperative summary of what was changed>"
```

Commit message rules:
- Start with a verb: "Remove", "Fix", "Add", "Update"
- No period at the end
- Max 72 characters

## 6. Output your summary

At the very end of your response, output EXACTLY this block (no extra text before/after):

```
### Agent Summary
- **What I found:** <one sentence describing root cause or current state>
- **What I changed:** <bullet list of files and what changed in each>
- **Tests:** <passed N/M | failed | not applicable — reason>
- **Branch:** agent/<slug>
- **Open questions / risks:** <any concerns, or "None">
```

The `Branch:` line is critical — it must contain the exact branch name you created.
The dispatcher script parses this line to post the branch to Linear.
If you do not output it, the review comment will be missing.

## Important rules

- Never commit to `main` or `master` directly
- Never push to remote — the approval handler does that
- Never create a pull request — human reviews first
- If you hit a merge conflict, abort and report in "Open questions / risks"
- If tests fail, do NOT commit — report the failure in the summary instead
