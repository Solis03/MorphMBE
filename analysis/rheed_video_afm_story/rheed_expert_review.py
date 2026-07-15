from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import repo_path, sha256_file, write_csv, write_json


TEMPLATE_COLUMNS = [
    "review_id",
    "streakiness_0_to_4",
    "spottyness_0_to_4",
    "lateral_connection_0_to_4",
    "diffuse_background_0_to_4",
    "pattern_regime",
    "confidence",
    "quality_exclude",
    "notes",
]


def _sheet(frames: np.ndarray, path: Path, title: str) -> None:
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(max(2, n * 1.1), 1.4))
    if n == 1:
        axes = [axes]
    for ax, frame in zip(axes, frames):
        ax.imshow(frame, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_expert_review(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    root = repo_path(config["report_root"]) / "rheed_expert_review"
    contact_root = root / "contact_sheets"
    clip_root = repo_path(config["rheed_clip_root"])
    root.mkdir(parents=True, exist_ok=True)
    contact_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["random_seed"]))
    order = manifest["sample_id"].astype(str).to_numpy()
    rng.shuffle(order)
    mapping, template = [], []
    html_parts = ["<h1>Blinded RHEED Review</h1><p>Only RHEED imagery and neutral review IDs are shown.</p>"]
    for i, sid in enumerate(order):
        review_id = f"R{i:03d}"
        mapping.append({"review_id": review_id, "sample_id": sid})
        template.append({c: "" for c in TEMPLATE_COLUMNS} | {"review_id": review_id})
        imgs = []
        for variant in config["rheed_clip_variants"]:
            frames = np.load(clip_root / variant / f"{sid}.npz")["frames_uint8"]
            path = contact_root / f"{review_id}_{variant}.png"
            _sheet(frames, path, f"{review_id} {variant} raw luminance")
            imgs.append((variant, f"contact_sheets/{path.name}"))
        html_parts.append(f"<section><h2>{review_id}</h2>" + "".join([f"<figure><img src='{src}' width='420'><figcaption>{variant}</figcaption></figure>" for variant, src in imgs]) + "</section>")
    (root / "index.html").write_text("<!doctype html><html><body>" + "\n".join(html_parts) + "</body></html>", encoding="utf-8")
    write_csv(pd.DataFrame(mapping), root / "review_mapping_private.csv")
    write_csv(pd.DataFrame(template), root / "expert_rheed_labels_template.csv")
    label_path = repo_path(config["expert_label_path"])
    status = {"expert_branch_available": False, "expert_label_hash": None, "expert_branch_status": "pending_missing_or_incomplete_labels"}
    if label_path.exists():
        labels = pd.read_csv(label_path)
        complete = set(TEMPLATE_COLUMNS).issubset(labels.columns) and labels[TEMPLATE_COLUMNS[1:-1]].notna().all().all()
        if complete:
            status = {"expert_branch_available": True, "expert_label_hash": sha256_file(label_path), "expert_branch_status": "complete_frozen_before_training"}
    write_json(status, root / "expert_branch_status.json")
    return status
