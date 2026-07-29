#!/usr/bin/env python3
"""Plot every generated AFM result from one real-time UI session."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--sample-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = Path(args.session).resolve()
    timeline = pd.read_csv(session / "prediction_timeline.csv")
    if args.expected_count is not None and len(timeline) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} predictions, found {len(timeline)}"
        )

    maps: list[np.ndarray] = []
    for row in timeline.itertuples(index=False):
        archive = session / str(row.generated_npz)
        with np.load(archive, allow_pickle=False) as payload:
            maps.append(np.asarray(payload["height_nm"], dtype=float))
    if len(maps) != len(timeline):
        raise RuntimeError("One generated AFM map is required per timeline row")

    columns = 4
    rows = int(math.ceil(len(maps) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.4, 3.05 * rows),
        constrained_layout=True,
    )
    flat_axes = np.asarray(axes).reshape(-1)
    absolute_limit = max(float(np.max(np.abs(item))) for item in maps)
    image = None
    for index, (axis, height, row) in enumerate(
        zip(flat_axes, maps, timeline.itertuples(index=False), strict=False)
    ):
        image = axis.imshow(
            height,
            cmap="afmhot",
            vmin=-absolute_limit,
            vmax=absolute_limit,
            extent=(0.0, 1.0, 1.0, 0.0),
            interpolation="nearest",
        )
        axis.set_title(
            f"#{index + 1} · frame {int(row.event_frame)}\n"
            f"Sq {row.predicted_rq_nm:.2f} nm · "
            f"FSMI {row.predicted_fsmi_nm:.2f} nm · "
            f"conf. {100 * row.model_confidence:.0f}%",
            fontsize=9,
        )
    for axis in flat_axes[len(maps) :]:
        axis.set_axis_off()
    if image is None:
        raise RuntimeError("The session contains no generated AFM maps")
    colorbar = figure.colorbar(
        image,
        ax=list(flat_axes[: len(maps)]),
        shrink=0.83,
        pad=0.015,
    )
    colorbar.set_label("Relative height (nm), shared scale")
    figure.supxlabel("x (µm)")
    figure.supylabel("y (µm)")
    sample = str(args.sample_id or "session")
    figure.suptitle(
        f"N{sample}: every detected clear moment triggers generated AFM "
        f"({len(maps)}/{len(maps)})",
        fontsize=15,
        fontweight="bold",
    )
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=240)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
