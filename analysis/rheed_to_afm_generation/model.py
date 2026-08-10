from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from analysis.rheed_video_afm_story.afm_autoencoder import ResidualBlock
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_torch


class ConditionalAFMVAE(nn.Module):
    """Compact conditional VAE for unit-Rq AFM morphology maps.

    The approximate posterior q(z | x, c) is used for reconstruction training.
    A learned conditional prior p(z | c) is used for genuine generation.  The
    decoder has no access to a source AFM image at inference.
    """

    def __init__(
        self,
        condition_dim: int,
        latent_dim: int = 16,
        resolution: int = 128,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError("resolution must be 64 or 128")
        self.condition_dim = int(condition_dim)
        self.latent_dim = int(latent_dim)
        self.resolution = int(resolution)
        self.base_channels = int(base_channels)
        c = self.base_channels

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            ResidualBlock(c),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            ResidualBlock(c * 4),
            nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1),
            ResidualBlock(c * 4),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        encoded_dim = c * 4 * 4 * 4
        hidden = max(64, self.latent_dim * 4)
        self.posterior = nn.Sequential(
            nn.Linear(encoded_dim + self.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.latent_dim),
        )
        self.prior = nn.Sequential(
            nn.Linear(self.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, encoded_dim),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                ResidualBlock(c * 4),
                ResidualBlock(c * 4),
                ResidualBlock(c * 2),
                ResidualBlock(c),
            ]
        )
        self.up_convs = nn.ModuleList(
            [
                nn.Conv2d(c * 4, c * 4, 3, padding=1),
                nn.Conv2d(c * 4, c * 2, 3, padding=1),
                nn.Conv2d(c * 2, c, 3, padding=1),
            ]
        )
        decoder_channels = [c * 4, c * 4, c * 2, c]
        self.condition_modulations = nn.ModuleList(
            [
                nn.Linear(self.condition_dim, 2 * channels)
                for channels in decoder_channels
            ]
        )
        for modulation in self.condition_modulations:
            nn.init.zeros_(modulation.weight)
            nn.init.zeros_(modulation.bias)
        self.final = nn.Conv2d(c, 1, 3, padding=1)

    @staticmethod
    def _split_gaussian(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 5.0)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + torch.exp(0.5 * logvar) * noise

    def encode(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.image_encoder(image).flatten(1)
        return self._split_gaussian(
            self.posterior(torch.cat([features, condition], dim=1))
        )

    def conditional_prior(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_gaussian(self.prior(condition))

    def decode_raw(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        c = self.base_channels
        hidden = self.from_latent(torch.cat([latent, condition], dim=1))
        hidden = hidden.view(latent.shape[0], c * 4, 4, 4)
        hidden = self.decoder_blocks[0](hidden)
        hidden = self._modulate(hidden, condition, self.condition_modulations[0])
        for block, convolution, modulation in zip(
            self.decoder_blocks[1:],
            self.up_convs,
            self.condition_modulations[1:],
            strict=False,
        ):
            hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
            hidden = convolution(hidden)
            hidden = block(hidden)
            hidden = self._modulate(hidden, condition, modulation)
        hidden = F.interpolate(
            hidden,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )
        return self.final(hidden)

    @staticmethod
    def _modulate(
        hidden: torch.Tensor,
        condition: torch.Tensor,
        layer: nn.Linear,
    ) -> torch.Tensor:
        scale, shift = layer(condition).chunk(2, dim=1)
        scale = 0.25 * torch.tanh(scale)[:, :, None, None]
        shift = 0.25 * shift[:, :, None, None]
        return hidden * (1.0 + scale) + shift

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return project_unit_rq_torch(self.decode_raw(latent, condition))

    def forward(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        posterior_mean, posterior_logvar = self.encode(image, condition)
        prior_mean, prior_logvar = self.conditional_prior(condition)
        latent = self.reparameterize(posterior_mean, posterior_logvar)
        reconstruction = self.decode(latent, condition)
        return {
            "reconstruction": reconstruction,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "latent": latent,
        }

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        use_prior_mean: bool = False,
    ) -> torch.Tensor:
        prior_mean, prior_logvar = self.conditional_prior(condition)
        latent = (
            prior_mean
            if use_prior_mean
            else self.reparameterize(prior_mean, prior_logvar, generator=generator)
        )
        return self.decode(latent, condition)


def gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) averaged over the batch."""

    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    squared_mean = (posterior_mean - prior_mean).square() * torch.exp(-prior_logvar)
    kl = 0.5 * (prior_logvar - posterior_logvar + variance_ratio + squared_mean - 1.0)
    return kl.sum(dim=1).mean()


def architecture_summary(model: ConditionalAFMVAE) -> dict[str, object]:
    return {
        "class": model.__class__.__name__,
        "condition_dim": model.condition_dim,
        "latent_dim": model.latent_dim,
        "resolution": model.resolution,
        "base_channels": model.base_channels,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "inference_distribution": "learned diagonal Gaussian p(z|condition)",
        "output_constraint": "mean-centered unit-Rq projection",
        "conditioning": "bottleneck concatenation plus four decoder FiLM stages",
        "retrieval_or_source_image_at_inference": False,
    }
