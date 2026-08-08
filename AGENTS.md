# AGENTS.md

## Objective

Complete assigned research and engineering objectives end-to-end.
Inspect the repository, plans, tests, reports, and Git history before
making changes.

Continue autonomously through implementation, testing, debugging,
evaluation, documentation, and self-review.

## Local environment

The canonical repository and runtime environment are on macOS Apple
Silicon.

Do not assume CUDA is available.

Before choosing a compute backend, detect the actual environment.
Prefer the repository's existing Python environment and dependency
versions.

Do not change the global Python installation.

## Allowed autonomous actions

You may autonomously:

- read and modify files in this repository
- create local branches and worktrees
- make local commits
- run tests, linters, preprocessing, evaluation, and benchmarks
- inspect logs and repair failures
- create additive reports and derived outputs
- install ordinary project dependencies inside the project environment
- use subagents for implementation, testing, and independent review
- make reasonable engineering choices and document them

Do not stop for routine implementation decisions.

## Human approval gates

Stop and request explicit approval before:

1. git push
2. creating, merging, or closing a pull request
3. publishing a release, package, model, dataset, website, or app
4. deploying to any external environment
5. reading or modifying credentials, tokens, signing certificates,
   API keys, SSH keys, or production secrets
6. initiating paid cloud or GPU resources
7. performing purchases, payments, transfers, or billing changes
8. deleting raw data or irreplaceable experiment outputs
9. rewriting Git history
10. sending external messages in the user's name

## Research data safety

- Never modify raw RHEED or AFM data.
- Derived data must be written to designated processed/results folders.
- Never silently overwrite publication freezes.
- Preserve sample IDs, physical units, provenance, and split metadata.
- Treat growth groups as leakage boundaries.
- Do not fabricate missing measurements.
- Do not report success when required checks fail.

## Git workflow

- Begin from a clean working tree.
- Use a dedicated branch or worktree.
- Make small local commits grouped by purpose.
- Never force-push.
- Stop before any remote push.

## Completion criteria

Before declaring completion:

1. run required tests
2. inspect all relevant logs
3. run leakage and split-integrity checks when applicable
4. inspect the final git diff
5. perform an independent review pass
6. verify no raw data was modified
7. report:
   - files changed
   - commands run
   - test and experiment results
   - branch and commit
   - known limitations
   - unresolved risks
   - actions waiting for user approval
