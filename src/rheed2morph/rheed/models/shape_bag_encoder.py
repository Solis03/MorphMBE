"""Neural input encoder for RHEED shape-bag tensors.

This module only defines the encoder interface for later experiments. It does
not train against AFM labels.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RHEEDShapeBagEncoder(nn.Module):
    """Encode variable-length RHEED shape bags into sample embeddings."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        consensus_channels: int = 6,
        shape_feature_dim: int = 0,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        frame_dropout: float = 0.0,
        channel_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frame_dropout = float(frame_dropout)
        self.channel_dropout = float(channel_dropout)
        self.frame_cnn = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, hidden_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.consensus_cnn = nn.Sequential(
            ConvBlock(consensus_channels, 32),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )
        self.shape_feature_dim = int(shape_feature_dim)
        self.shape_mlp = (
            nn.Sequential(nn.LayerNorm(shape_feature_dim), nn.Linear(shape_feature_dim, hidden_dim), nn.SiLU())
            if shape_feature_dim > 0
            else None
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        fusion_dim = hidden_dim * 2 + (hidden_dim if shape_feature_dim > 0 else 0)
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def _apply_channel_dropout(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.training or self.channel_dropout <= 0:
            return frames
        keep = torch.rand(frames.shape[0], 1, frames.shape[2], 1, 1, device=frames.device) >= self.channel_dropout
        return frames * keep.to(frames.dtype)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        frame_weights: torch.Tensor | None = None,
        consensus_maps: torch.Tensor | None = None,
        shape_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames [B,K,C,H,W], got {tuple(frames.shape)}")
        batch, frame_count, channels, height, width = frames.shape
        _ = channels, height, width
        frame_mask = frame_mask.to(frames.dtype)
        if frame_weights is None:
            frame_weights = frame_mask
        else:
            frame_weights = frame_weights.to(frames.dtype) * frame_mask
        if self.training and self.frame_dropout > 0:
            keep = (torch.rand_like(frame_mask) >= self.frame_dropout).to(frames.dtype)
            keep = torch.maximum(keep, (frame_mask.sum(dim=1, keepdim=True) <= 1).to(frames.dtype))
            frame_mask = frame_mask * keep
            frame_weights = frame_weights * keep
        frames = self._apply_channel_dropout(frames)
        flat_frames = frames.reshape(batch * frame_count, *frames.shape[2:])
        frame_embeddings = self.frame_cnn(flat_frames).reshape(batch, frame_count, -1)
        weight_feature = frame_weights.unsqueeze(-1)
        attention_logits = self.attention(torch.cat([frame_embeddings, weight_feature], dim=-1)).squeeze(-1)
        attention_logits = attention_logits.masked_fill(frame_mask <= 0, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = torch.where(frame_mask > 0, attention_weights, torch.zeros_like(attention_weights))
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attention_pooled = torch.sum(frame_embeddings * attention_weights.unsqueeze(-1), dim=1)
        normalized_weights = frame_weights / frame_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_mean = torch.sum(frame_embeddings * normalized_weights.unsqueeze(-1), dim=1)
        if consensus_maps is None:
            consensus_maps = torch.zeros(batch, 6, frames.shape[-2], frames.shape[-1], dtype=frames.dtype, device=frames.device)
        consensus_embedding = self.consensus_cnn(consensus_maps)
        pieces = [attention_pooled + weighted_mean, consensus_embedding]
        if self.shape_mlp is not None:
            if shape_features is None:
                raise ValueError("shape_features is required when shape_feature_dim > 0")
            pieces.append(self.shape_mlp(shape_features.to(frames.dtype)))
        sample_embedding = self.output(torch.cat(pieces, dim=-1))
        return {
            "sample_embedding": sample_embedding,
            "attention_weights": attention_weights,
            "frame_embeddings": frame_embeddings,
        }

