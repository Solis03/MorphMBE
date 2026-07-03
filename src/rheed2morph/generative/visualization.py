"""Image grid helpers for AFM MVP reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_image(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output[..., 0]
    return output


def write_panel_grid(
    path: Path,
    rows: Sequence[Sequence[np.ndarray]],
    column_titles: Sequence[str],
    row_titles: Sequence[str] | None = None,
    cmaps: Sequence[str] | None = None,
) -> None:
    if not rows:
        return
    row_count = len(rows)
    column_count = len(column_titles)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.0 * column_count, 2.8 * row_count),
        dpi=150,
        squeeze=False,
    )
    for row_index, panels in enumerate(rows):
        for column_index in range(column_count):
            axis = axes[row_index, column_index]
            image = _as_image(np.asarray(panels[column_index]))
            cmap = cmaps[column_index] if cmaps is not None and column_index < len(cmaps) else "viridis"
            axis.imshow(image, cmap=cmap)
            axis.axis("off")
            title = column_titles[column_index]
            if row_titles is not None and column_index == 0:
                title = f"{title}\n{row_titles[row_index]}"
            axis.set_title(title, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
