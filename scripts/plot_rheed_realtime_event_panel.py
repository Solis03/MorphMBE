#!/usr/bin/env python3
"""Plot RHEED event crops beside their generated AFM predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = Path(args.session).resolve()
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    timeline = pd.read_csv(session / "prediction_timeline.csv")
    if args.expected_count is not None and len(timeline) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} predictions, found {len(timeline)}"
        )
    roi = metadata["selector"]["model_input_roi"]
    reader = imageio.get_reader(metadata["source_video"], "ffmpeg")
    crops: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    try:
        for row in timeline.itertuples(index=False):
            frame = np.asarray(reader.get_data(int(row.event_frame)))
            crops.append(
                frame[
                    int(roi["y"]) : int(roi["y"] + roi["height"]),
                    int(roi["x"]) : int(roi["x"] + roi["width"]),
                ]
            )
            with np.load(
                session / str(row.generated_npz),
                allow_pickle=False,
            ) as payload:
                maps.append(np.asarray(payload["height_nm"], dtype=float))
    finally:
        reader.close()

    figure, axes = plt.subplots(
        len(timeline),
        2,
        figsize=(10.0, 3.5 * len(timeline)),
        constrained_layout=True,
        squeeze=False,
    )
    absolute_limit = max(float(np.max(np.abs(item))) for item in maps)
    afm_image = None
    for index, (row, crop, height) in enumerate(
        zip(timeline.itertuples(index=False), crops, maps, strict=True)
    ):
        axes[index, 0].imshow(crop)
        axes[index, 0].set_title(
            f"RHEED clear moment #{index + 1}\nframe {int(row.event_frame)}",
            fontsize=11,
        )
        axes[index, 0].set_axis_off()
        afm_image = axes[index, 1].imshow(
            height,
            cmap="afmhot",
            vmin=-absolute_limit,
            vmax=absolute_limit,
            extent=(0.0, 1.0, 1.0, 0.0),
            interpolation="nearest",
        )
        axes[index, 1].set(
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=(
                f"Generated AFM #{index + 1}\n"
                f"Sq {row.predicted_rq_nm:.2f} nm · "
                f"FSMI {row.predicted_fsmi_nm:.2f} nm\n"
                f"confidence {100 * row.model_confidence:.0f}%"
            ),
        )
        axes[index, 1].title.set_fontsize(10)
    if afm_image is None:
        raise RuntimeError("The session contains no prediction events")
    colorbar = figure.colorbar(
        afm_image,
        ax=list(axes[:, 1]),
        shrink=0.88,
        pad=0.02,
    )
    colorbar.set_label("Relative height (nm), shared scale")
    figure.suptitle(
        f"N{metadata['sample_id']} full-lattice fallback: "
        f"{len(timeline)}/{len(timeline)} events generated",
        fontsize=14,
        fontweight="bold",
    )
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=240)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
