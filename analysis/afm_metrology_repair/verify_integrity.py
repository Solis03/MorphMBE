"""Verify that every AFM source used by the metrology audit is unchanged."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(
    audit_path: Path,
    output_root: Path,
) -> dict[str, object]:
    audit = pd.read_csv(audit_path, dtype={"sample_id": str})
    specifications = (
        ("raw_afm_file", "raw_afm_sha256", "raw_afm"),
        (
            "decoded_zsensor_path",
            "decoded_zsensor_sha256",
            "decoded_zsensor",
        ),
    )
    records: list[dict[str, object]] = []
    for path_column, expected_column, source_kind in specifications:
        sources = audit[[path_column, expected_column]].drop_duplicates()
        for row in sources.itertuples(index=False):
            relative = Path(str(row[0]))
            path = relative if relative.is_absolute() else ROOT / relative
            expected = str(row[1])
            exists = path.is_file()
            actual = _digest(path) if exists else ""
            records.append(
                {
                    "source_kind": source_kind,
                    "path": str(relative),
                    "expected_sha256": expected,
                    "current_sha256": actual,
                    "exists": exists,
                    "hash_matches": bool(exists and actual == expected),
                }
            )
    table = pd.DataFrame(records).sort_values(
        ["source_kind", "path"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "source_integrity_after.csv", index=False)
    result = {
        "audit_path": str(audit_path.relative_to(ROOT)),
        "checked_source_count": int(len(table)),
        "raw_afm_source_count": int(
            table["source_kind"].eq("raw_afm").sum()
        ),
        "decoded_zsensor_source_count": int(
            table["source_kind"].eq("decoded_zsensor").sum()
        ),
        "all_sources_exist": bool(table["exists"].all()),
        "all_hashes_match": bool(table["hash_matches"].all()),
        "raw_data_modified": not bool(table["hash_matches"].all()),
    }
    (output_root / "source_integrity_after.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["all_hashes_match"]:
        failures = table.loc[~table["hash_matches"], "path"].tolist()
        raise RuntimeError(f"AFM source integrity failed: {failures[:10]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default="outputs/afm_metrology_line3_v1/afm_scan_audit.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/afm_metrology_line3_v1",
    )
    args = parser.parse_args()
    audit = Path(args.audit)
    output = Path(args.output_root)
    if not audit.is_absolute():
        audit = ROOT / audit
    if not output.is_absolute():
        output = ROOT / output
    print(json.dumps(verify(audit, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
