"""Temporal RHEED morphology condition encoder for MVP-6."""

from __future__ import annotations

import torch
import torch.nn as nn

from rheed2morph.generative.models.rheed_mae import SmallCNNFrameEncoder


class AttentionTemporalPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class RheedTemporalMorphologyEncoder(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        handcrafted_dim: int = 0,
        metadata_dim: int = 0,
        prototype_classes: int = 0,
        frame_encoder: str = "small_cnn",
        temporal_pooling: str = "attention",
        visual_embedding_dim: int = 256,
        hidden_dim: int = 256,
        use_visual: bool = True,
        use_handcrafted: bool = True,
        use_metadata: bool = True,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if frame_encoder not in {"small_cnn", "mae_encoder", "resnet18", "resnet50"}:
            raise ValueError(f"Unsupported frame_encoder: {frame_encoder}")
        if frame_encoder in {"resnet18", "resnet50"}:
            frame_encoder = "small_cnn"
        self.descriptor_dim = int(descriptor_dim)
        self.handcrafted_dim = int(handcrafted_dim)
        self.metadata_dim = int(metadata_dim)
        self.prototype_classes = int(prototype_classes)
        self.temporal_pooling = str(temporal_pooling)
        self.use_visual = bool(use_visual)
        self.use_handcrafted = bool(use_handcrafted and handcrafted_dim > 0)
        self.use_metadata = bool(use_metadata and metadata_dim > 0)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.visual_embedding_dim = int(visual_embedding_dim)
        if self.use_visual:
            self.frame_encoder = SmallCNNFrameEncoder(self.visual_embedding_dim)
            if self.temporal_pooling == "attention":
                self.temporal = AttentionTemporalPool(self.visual_embedding_dim)
                visual_out = self.visual_embedding_dim
            elif self.temporal_pooling == "gru":
                self.temporal = nn.GRU(self.visual_embedding_dim, self.visual_embedding_dim // 2, batch_first=True, bidirectional=True)
                visual_out = self.visual_embedding_dim
            elif self.temporal_pooling in {"mean", "max", "final"}:
                self.temporal = nn.Identity()
                visual_out = self.visual_embedding_dim
            elif self.temporal_pooling == "transformer":
                layer = nn.TransformerEncoderLayer(d_model=self.visual_embedding_dim, nhead=4, dim_feedforward=self.visual_embedding_dim * 2, batch_first=True)
                self.temporal = nn.TransformerEncoder(layer, num_layers=1)
                visual_out = self.visual_embedding_dim
            else:
                raise ValueError(f"Unsupported temporal_pooling: {temporal_pooling}")
        else:
            self.frame_encoder = None
            self.temporal = None
            visual_out = 0
        if self.use_handcrafted:
            self.handcrafted_mlp = nn.Sequential(
                nn.Linear(self.handcrafted_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.SiLU(),
            )
            handcrafted_out = hidden_dim // 2
        else:
            self.handcrafted_mlp = None
            handcrafted_out = 0
        if self.use_metadata:
            self.metadata_mlp = nn.Sequential(nn.Linear(self.metadata_dim, hidden_dim // 4), nn.SiLU())
            metadata_out = hidden_dim // 4
        else:
            self.metadata_mlp = None
            metadata_out = 0
        fusion_dim = visual_out + handcrafted_out + metadata_out
        if fusion_dim <= 0:
            fusion_dim = 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.descriptor_head = nn.Linear(hidden_dim, self.descriptor_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.descriptor_dim) if self.predict_uncertainty else None
        self.prototype_head = nn.Linear(hidden_dim, self.prototype_classes) if self.prototype_classes > 0 else None

    def _visual_forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.frame_encoder is None:
            raise ValueError("visual input requested but no frame encoder exists")
        b, t, c, h, w = video.shape
        embeddings = self.frame_encoder(video.reshape(b * t, c, h, w)).reshape(b, t, -1)
        if self.temporal_pooling == "attention":
            return self.temporal(embeddings)
        if self.temporal_pooling == "gru":
            _seq, hidden = self.temporal(embeddings)
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        if self.temporal_pooling == "max":
            return embeddings.max(dim=1).values
        if self.temporal_pooling == "final":
            return embeddings[:, -1]
        if self.temporal_pooling == "transformer":
            encoded = self.temporal(embeddings)
            return encoded.mean(dim=1)
        return embeddings.mean(dim=1)

    def forward(
        self,
        video: torch.Tensor | None = None,
        handcrafted: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        batch = 1
        device = torch.device("cpu")
        if self.use_visual:
            if video is None:
                raise ValueError("video is required when use_visual=True")
            batch = int(video.shape[0])
            device = video.device
            pieces.append(self._visual_forward(video))
        if self.use_handcrafted:
            if handcrafted is None:
                raise ValueError("handcrafted features are required when use_handcrafted=True")
            batch = int(handcrafted.shape[0])
            device = handcrafted.device
            pieces.append(self.handcrafted_mlp(handcrafted))
        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata features are required when use_metadata=True")
            batch = int(metadata.shape[0])
            device = metadata.device
            pieces.append(self.metadata_mlp(metadata))
        fused_input = torch.cat(pieces, dim=1) if pieces else torch.zeros((batch, 1), device=device)
        hidden = self.fusion(fused_input)
        out = {"descriptor": self.descriptor_head(hidden)}
        if self.logvar_head is not None:
            out["log_variance"] = self.logvar_head(hidden).clamp(-8.0, 6.0)
        if self.prototype_head is not None:
            out["prototype_logits"] = self.prototype_head(hidden)
        return out
