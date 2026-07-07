"""MVP-9 shape-bag morphology predictor."""

from __future__ import annotations

import torch
from torch import nn

from rheed2morph.rheed.models.shape_bag_encoder import ConvBlock


class SmallMapEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameBagBranch(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 96, frame_dropout: float = 0.0, channel_dropout: float = 0.0) -> None:
        super().__init__()
        self.frame_dropout = float(frame_dropout)
        self.channel_dropout = float(channel_dropout)
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 24),
            ConvBlock(24, 48),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attention = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor, frame_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, k, channels, height, width = frames.shape
        mask = frame_mask.to(frames.dtype)
        weights = mask if frame_weights is None else frame_weights.to(frames.dtype) * mask
        if self.training and self.frame_dropout > 0:
            keep = (torch.rand_like(mask) >= self.frame_dropout).to(frames.dtype)
            keep = torch.maximum(keep, (mask.sum(dim=1, keepdim=True) <= 1).to(frames.dtype))
            mask = mask * keep
            weights = weights * keep
        if self.training and self.channel_dropout > 0:
            keep_channels = (torch.rand(batch, 1, channels, 1, 1, device=frames.device) >= self.channel_dropout).to(frames.dtype)
            frames = frames * keep_channels
        emb = self.cnn(frames.reshape(batch * k, channels, height, width)).reshape(batch, k, -1)
        logits = self.attention(torch.cat([emb, weights.unsqueeze(-1)], dim=-1)).squeeze(-1)
        mask_value = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
        logits = logits.masked_fill(mask <= 0, mask_value)
        attn = torch.softmax(logits, dim=1)
        attn = torch.where(mask > 0, attn, torch.zeros_like(attn))
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(emb * attn.unsqueeze(-1), dim=1)
        return pooled, attn, emb


class RHEEDShapeBagMorphologyPredictor(nn.Module):
    """Predict AFM condition descriptors from one RHEED shape-bag sample."""

    def __init__(
        self,
        *,
        target_dim: int,
        prototype_count: int = 0,
        in_channels: int = 6,
        consensus_channels: int = 6,
        stable_feature_dim: int = 0,
        metadata_dim: int = 0,
        hidden_dim: int = 96,
        embedding_dim: int = 192,
        use_frames: bool = True,
        use_consensus: bool = True,
        use_stable_features: bool = True,
        use_metadata: bool = False,
        predict_uncertainty: bool = True,
        frame_dropout: float = 0.0,
        channel_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_frames = bool(use_frames)
        self.use_consensus = bool(use_consensus)
        self.use_stable_features = bool(use_stable_features and stable_feature_dim > 0)
        self.use_metadata = bool(use_metadata and metadata_dim > 0)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.target_dim = int(target_dim)
        self.prototype_count = int(prototype_count)
        branches = 0
        self.frame_branch = FrameBagBranch(in_channels, hidden_dim, frame_dropout, channel_dropout) if self.use_frames else None
        if self.use_frames:
            branches += 1
        self.consensus_branch = SmallMapEncoder(consensus_channels, hidden_dim) if self.use_consensus else None
        if self.use_consensus:
            branches += 1
        self.feature_branch = (
            nn.Sequential(nn.LayerNorm(stable_feature_dim), nn.Linear(stable_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            if self.use_stable_features
            else None
        )
        if self.use_stable_features:
            branches += 1
        self.metadata_branch = (
            nn.Sequential(nn.LayerNorm(metadata_dim), nn.Linear(metadata_dim, hidden_dim), nn.SiLU())
            if self.use_metadata
            else None
        )
        if self.use_metadata:
            branches += 1
        if branches <= 0:
            raise ValueError("At least one input branch must be enabled.")
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * branches, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )
        self.descriptor_mean_head = nn.Linear(embedding_dim, target_dim)
        self.descriptor_logvar_head = nn.Linear(embedding_dim, target_dim) if self.predict_uncertainty else None
        self.prototype_head = nn.Linear(embedding_dim, prototype_count) if prototype_count > 0 else None

    def forward(
        self,
        *,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        frame_weights: torch.Tensor,
        consensus_maps: torch.Tensor,
        stable_shape_features: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        attention: torch.Tensor
        frame_embeddings: torch.Tensor
        if self.frame_branch is not None:
            frame_embedding, attention, frame_embeddings = self.frame_branch(frames, frame_mask, frame_weights)
            pieces.append(frame_embedding)
        else:
            attention = frame_weights / frame_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            frame_embeddings = torch.zeros(frames.shape[0], frames.shape[1], 1, dtype=frames.dtype, device=frames.device)
        if self.consensus_branch is not None:
            pieces.append(self.consensus_branch(consensus_maps))
        if self.feature_branch is not None:
            pieces.append(self.feature_branch(stable_shape_features))
        if self.metadata_branch is not None:
            if metadata is None:
                raise ValueError("metadata tensor is required when metadata branch is enabled.")
            pieces.append(self.metadata_branch(metadata))
        embedding = self.fusion(torch.cat(pieces, dim=-1))
        mean = self.descriptor_mean_head(embedding)
        output = {
            "sample_embedding": embedding,
            "descriptor_mean": mean,
            "attention_weights": attention,
            "frame_embeddings": frame_embeddings,
        }
        if self.descriptor_logvar_head is not None:
            output["descriptor_logvar"] = self.descriptor_logvar_head(embedding).clamp(-7.0, 5.0)
        if self.prototype_head is not None:
            output["prototype_logits"] = self.prototype_head(embedding)
        return output


def attention_entropy(attention: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    attn = attention.clamp_min(1e-8)
    entropy = -(attn * attn.log()).sum(dim=1)
    denom = frame_mask.sum(dim=1).clamp_min(2.0).log()
    return (entropy / denom).mean()


def exposure_consistency_loss(out_a: dict[str, torch.Tensor], out_b: dict[str, torch.Tensor], *, prototype_weight: float = 0.1) -> torch.Tensor:
    loss = torch.nn.functional.mse_loss(out_a["sample_embedding"], out_b["sample_embedding"])
    loss = loss + torch.nn.functional.mse_loss(out_a["descriptor_mean"], out_b["descriptor_mean"])
    if "prototype_logits" in out_a and "prototype_logits" in out_b:
        loss = loss + prototype_weight * torch.nn.functional.mse_loss(out_a["prototype_logits"], out_b["prototype_logits"])
    return loss
