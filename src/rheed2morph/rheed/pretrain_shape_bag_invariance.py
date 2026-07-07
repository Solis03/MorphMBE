"""One-step smoke interface for future shape-bag invariance pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from rheed2morph.rheed.models.shape_bag_encoder import RHEEDShapeBagEncoder
from rheed2morph.rheed.rheed_shape_bag_dataset import RHEEDShapeBagDataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embedding-dim", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = RHEEDShapeBagDataset(Path(args.manifest))
    item = dataset[0]
    model = RHEEDShapeBagEncoder(
        in_channels=item["frames"].shape[1],
        consensus_channels=item["consensus_maps"].shape[0],
        shape_feature_dim=item["shape_features"].numel(),
        embedding_dim=args.embedding_dim,
    )
    model.train()
    frames = item["frames"].unsqueeze(0)
    jittered = torch.clamp(frames * 0.85 + 0.05, -1.0, 1.0)
    out_a = model(
        frames,
        item["frame_mask"].unsqueeze(0),
        item["frame_weights"].unsqueeze(0),
        item["consensus_maps"].unsqueeze(0),
        item["shape_features"].unsqueeze(0),
    )["sample_embedding"]
    out_b = model(
        jittered,
        item["frame_mask"].unsqueeze(0),
        item["frame_weights"].unsqueeze(0),
        item["consensus_maps"].unsqueeze(0),
        item["shape_features"].unsqueeze(0),
    )["sample_embedding"]
    loss = torch.nn.functional.mse_loss(out_a, out_b)
    loss.backward()
    print(f"one_step_invariance_smoke_loss={float(loss.detach()):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

