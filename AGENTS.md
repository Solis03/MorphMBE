# Release engineering constraints

This branch is the frozen MorphMBE M22 public release.

- Never modify raw RHEED or AFM measurements.
- Treat a complete growth group as the leakage boundary.
- Preserve sample IDs, physical units, rotations, seeds, and split metadata.
- Do not change model numerics without a new model version and refreshed
  end-to-end equivalence evidence.
- Keep generated outputs under `artifacts/`; do not commit caches, raw data,
  checkpoints, or historical experiment archives.
- Run `uv run pytest -q`, `uv run ruff check .`, and
  `uv run python scripts/validate_release.py` before committing.
- Confirm the 6063 frozen fixture and strict outer-LOO fold audit before a
  release.
- Do not claim prospective validation; M22 is retrospective method development.
