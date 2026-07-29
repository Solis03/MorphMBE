from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import pandas as pd

from analysis.rheed_video_afm_story.common import repo_path, write_csv

from .dataset import load_config


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    repository = repo_path(".")
    output = repo_path(config["output_root"]).parent
    report = repo_path(config["report_root"])
    rows = []
    for role, root in (("derived_output", output), ("report", report)):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "artifact_manifest.csv":
                continue
            rows.append(
                {
                    "role": role,
                    "path": str(path.relative_to(repository)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _hash(path),
                }
            )
    reproducibility_files = [
        *sorted(
            (
                repository
                / "analysis"
                / "rheed_manual_vs_auto_selection"
            ).glob("*.py")
        ),
        repository / "configs" / "rheed_manual_vs_auto_selection.json",
        repository / "configs" / "rheed_manual_vs_auto_generation.json",
        repository / "tests" / "test_rheed_manual_vs_auto_selection.py",
    ]
    for path in reproducibility_files:
        rows.append(
            {
                "role": "code_or_config",
                "path": str(path.relative_to(repository)),
                "size_bytes": path.stat().st_size,
                "sha256": _hash(path),
            }
        )
    for role, value in (
        ("standalone_target_parameters", config["standalone_target_parameters"]),
        (
            "standalone_generator_parameters",
            config["standalone_generator_parameters"],
        ),
    ):
        path = Path(value).expanduser()
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _hash(path),
            }
        )
    table = pd.DataFrame(rows).sort_values(["role", "path"])
    write_csv(table, report / "artifact_manifest.csv")
    print(
        f"wrote {len(table)} artifact records; "
        f"{int(table['size_bytes'].sum())} bytes",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_manual_vs_auto_selection.json",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
