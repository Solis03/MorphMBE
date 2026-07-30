from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .gwyddion_crosscheck import GwyddionAPI


QUADRANT_ORIGINS = {
    # NanoScope declares "Frame direction: Down".  Gwyddion normalises that
    # acquisition direction on import, whereas the repository decoder retains
    # stored row order.  Consequently the identical physical subfield is
    # vertically mirrored between the two array conventions.
    "q00_top_left": (0, 256),
    "q01_top_right": (256, 256),
    "q10_bottom_left": (0, 0),
    "q11_bottom_right": (256, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check extra-five 1 µm AFM crops in Gwyddion",
    )
    parser.add_argument("--scan-table", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    scans = pd.read_csv(args.scan_table, dtype={"sample_id": str})
    api = GwyddionAPI()
    records: list[dict[str, object]] = []
    for row in scans.to_dict("records"):
        origin = QUADRANT_ORIGINS[str(row["quadrant"])]
        raw_path = Path(str(row["raw_afm_file"]))
        if not raw_path.is_absolute():
            raw_path = args.path_root / raw_path
        values = api.subfield_line_sq_nm(
            raw_path,
            col=origin[0],
            row=origin[1],
            width=256,
            height=256,
        )
        record = {
            "sample_id": str(row["sample_id"]),
            "afm_file_id": str(row["afm_file_id"]),
            "raw_afm_file": str(raw_path.resolve()),
            "quadrant": str(row["quadrant"]),
        }
        for order in range(4):
            pipeline = float(row[f"sq_order_{order}_nm"])
            gwyddion = float(values[order])
            record[f"pipeline_line{order}_sq_nm"] = pipeline
            record[f"gwyddion_line{order}_sq_nm"] = gwyddion
            record[f"absolute_delta_line{order}_nm"] = abs(
                pipeline - gwyddion
            )
        records.append(record)
    output = pd.DataFrame(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    for order in range(4):
        delta = output[f"absolute_delta_line{order}_nm"]
        print(
            f"line-{order}: n={len(delta)}, "
            f"max_abs_delta_nm={delta.max():.12g}, "
            f"mean_abs_delta_nm={delta.mean():.12g}"
        )


if __name__ == "__main__":
    main()
