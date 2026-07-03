"""Small frame reconstruction model for RHEED SSL pretraining."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rheed2morph.generative.rheed_ssl_augmentations import random_patch_mask


class SmallCNNFrameEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(10, 160),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(160, self.embedding_dim), nn.SiLU())

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.forward_features(x)))


class SmallCNNMAE(nn.Module):
    """A compact masked-frame autoencoder used as an MVP SSL signal."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, embedding_dim: int = 256) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embedding_dim = int(embedding_dim)
        self.encoder = SmallCNNFrameEncoder(embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(160, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.ConvTranspose2d(24, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.6) -> dict[str, torch.Tensor]:
        masked, mask = random_patch_mask(x, self.patch_size, mask_ratio)
        encoded = self.encoder.forward_features(masked)
        recon = torch.sigmoid(self.decoder(encoded))
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss_map = (recon - x).pow(2)
        denom = mask.mean().clamp_min(1e-4)
        loss = (loss_map * mask).mean() / denom
        embedding = self.encoder(x)
        return {"reconstruction": recon, "masked": masked, "mask": mask, "loss": loss, "embedding": embedding}


def build_rheed_mae(
    model: str = "small_cnn_mae",
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 256,
    **_kwargs: Any,
) -> SmallCNNMAE:
    if model not in {"small_cnn_mae", "vit_mae"}:
        raise ValueError(f"Unsupported RHEED MAE model: {model}")
    return SmallCNNMAE(image_size=image_size, patch_size=patch_size, embedding_dim=embed_dim)


def load_mae_encoder_state(frame_encoder: SmallCNNFrameEncoder, checkpoint: str | None, strict: bool = False) -> bool:
    if not checkpoint:
        return False
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    encoder_state = {}
    target_state = frame_encoder.state_dict()
    for key, value in state.items():
        if key.startswith("encoder."):
            target_key = key[len("encoder.") :]
            if strict or (target_key in target_state and tuple(target_state[target_key].shape) == tuple(value.shape)):
                encoder_state[target_key] = value
    if not encoder_state:
        return False
    frame_encoder.load_state_dict(encoder_state, strict=strict)
    return True
