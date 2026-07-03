"""RHEED-to-AFM-condition encoder for MVP-2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallFrameCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _torchvision_backbone(name: str, embedding_dim: int) -> nn.Module:
    try:
        import torchvision.models as models
    except Exception as exc:
        raise RuntimeError(f"torchvision is not available for {name}: {exc}") from exc
    if name == "resnet18":
        backbone = models.resnet18(weights=None)
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Linear(in_features, embedding_dim)
    elif name == "resnet50":
        backbone = models.resnet50(weights=None)
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Linear(in_features, embedding_dim)
    else:
        raise ValueError(f"Unsupported visual backbone: {name}")
    first = backbone.conv1
    backbone.conv1 = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
    return backbone


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(frame_embeddings), dim=1)
        return torch.sum(frame_embeddings * weights, dim=1)


class RheedConditionEncoder(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        handcrafted_dim: int,
        metadata_dim: int = 0,
        prototype_classes: int = 0,
        visual_backbone: str = "small_cnn",
        temporal_pooling: str = "attention",
        visual_embedding_dim: int = 128,
        hidden_dim: int = 256,
        use_visual: bool = True,
        use_handcrafted: bool = True,
        use_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.handcrafted_dim = int(handcrafted_dim)
        self.metadata_dim = int(metadata_dim)
        self.prototype_classes = int(prototype_classes)
        self.use_visual = bool(use_visual)
        self.use_handcrafted = bool(use_handcrafted)
        self.use_metadata = bool(use_metadata and metadata_dim > 0)
        self.visual_embedding_dim = int(visual_embedding_dim)
        if self.use_visual:
            if visual_backbone == "small_cnn":
                self.visual_encoder = SmallFrameCNN(visual_embedding_dim)
            elif visual_backbone in {"resnet18", "resnet50"}:
                self.visual_encoder = _torchvision_backbone(visual_backbone, visual_embedding_dim)
            else:
                raise ValueError(f"Unsupported visual_backbone: {visual_backbone}")
            self.temporal_pooling = temporal_pooling
            self.attention_pool = AttentionPool(visual_embedding_dim) if temporal_pooling == "attention" else None
            visual_out_dim = visual_embedding_dim
        else:
            self.visual_encoder = None
            self.temporal_pooling = "none"
            self.attention_pool = None
            visual_out_dim = 0
        if self.use_handcrafted:
            self.handcrafted_mlp = nn.Sequential(
                nn.Linear(handcrafted_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.SiLU(),
            )
            handcrafted_out_dim = hidden_dim // 2
        else:
            self.handcrafted_mlp = None
            handcrafted_out_dim = 0
        if self.use_metadata:
            self.metadata_mlp = nn.Sequential(nn.Linear(metadata_dim, hidden_dim // 4), nn.SiLU())
            metadata_out_dim = hidden_dim // 4
        else:
            self.metadata_mlp = None
            metadata_out_dim = 0
        fusion_dim = visual_out_dim + handcrafted_out_dim + metadata_out_dim
        if fusion_dim <= 0:
            fusion_dim = 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.descriptor_head = nn.Linear(hidden_dim, descriptor_dim)
        self.prototype_head = nn.Linear(hidden_dim, prototype_classes) if prototype_classes > 0 else None

    def _visual_forward(self, video: torch.Tensor) -> torch.Tensor:
        if not self.use_visual or self.visual_encoder is None:
            return torch.empty((video.shape[0], 0), device=video.device, dtype=video.dtype)
        batch, frames, channels, height, width = video.shape
        flat = video.reshape(batch * frames, channels, height, width)
        embeddings = self.visual_encoder(flat).reshape(batch, frames, -1)
        if self.attention_pool is not None:
            return self.attention_pool(embeddings)
        return embeddings.mean(dim=1)

    def forward(
        self,
        video: torch.Tensor | None,
        handcrafted: torch.Tensor | None,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if self.use_visual:
            if video is None:
                raise ValueError("video tensor is required when use_visual=True")
            pieces.append(self._visual_forward(video))
        if self.use_handcrafted:
            if handcrafted is None:
                raise ValueError("handcrafted features are required when use_handcrafted=True")
            pieces.append(self.handcrafted_mlp(handcrafted))
        if self.use_metadata:
            if metadata is None:
                raise ValueError("metadata tensor is required when use_metadata=True")
            pieces.append(self.metadata_mlp(metadata))
        if pieces:
            fused_input = torch.cat(pieces, dim=1)
        else:
            batch = 1
            device = torch.device("cpu")
            if video is not None:
                batch = video.shape[0]
                device = video.device
            elif handcrafted is not None:
                batch = handcrafted.shape[0]
                device = handcrafted.device
            fused_input = torch.zeros((batch, 1), device=device)
        hidden = self.fusion(fused_input)
        output = {"descriptor": self.descriptor_head(hidden)}
        if self.prototype_head is not None:
            output["prototype_logits"] = self.prototype_head(hidden)
        return output


def build_rheed_condition_encoder(**kwargs: object) -> RheedConditionEncoder:
    return RheedConditionEncoder(**kwargs)
