# Development guide

Keep production behavior centralized in `src/rheed2morph/realtime/`. Scripts
should be thin entry points or explicit audit/visualization tools. New model
experiments belong on a development branch and must not overwrite frozen assets
or results.

Before a change:

1. create a dedicated branch or worktree;
2. record the current end-to-end regression output;
3. identify whether the change affects numerics, serialization, UI only, or
   documentation.

Before a commit:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python scripts/validate_release.py
```

Numerical changes require a new model/version identifier, refreshed bundle and
asset checksums, strict growth-level leakage audit, updated result tables and
model card, and an end-to-end comparison against the previous release. Do not
hide test failures with broad skips or relaxed numerical tolerances.

UI and headless inference must continue to call the same
`RealtimeMorphologyPredictor`. Keep raw-data reads isolated from all derived
output writes, which belong under `artifacts/`.
