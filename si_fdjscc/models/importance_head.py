from __future__ import annotations

import math

import torch
from torch import nn


class ImportanceHead(nn.Module):
    """Lightweight channel-conditioned distortion-sensitivity predictor.

    The original prototype predicted importance from the latent token alone.
    The revised SDS definition is conditional on the estimated channel, SNR,
    and CSI uncertainty.  ``channel_features`` therefore modulate the same
    token/context representation used by the legacy head.  The conditioner is
    zero-initialised so legacy checkpoints keep their exact behaviour until the
    new branch is trained.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 0,
        temperature: float = 0.5,
        dropout: float = 0.0,
        channel_feature_dim: int = 11,
    ) -> None:
        super().__init__()
        if channels < 1 or temperature <= 0 or channel_feature_dim < 0:
            raise ValueError(
                "channels and temperature must be positive and "
                "channel_feature_dim must be non-negative"
            )
        hidden = hidden_channels or max(channels // 2, 8)
        self.channels = int(channels)
        self.temperature = float(temperature)
        self.channel_feature_dim = int(channel_feature_dim)
        self.token_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        feature_dim = 2 * channels + 1
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if self.channel_feature_dim:
            self.channel_conditioner = nn.Sequential(
                nn.LayerNorm(self.channel_feature_dim),
                nn.Linear(self.channel_feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, feature_dim),
            )
            nn.init.zeros_(self.channel_conditioner[-1].weight)
            nn.init.zeros_(self.channel_conditioner[-1].bias)
        else:
            self.channel_conditioner = None

    def logits(
        self,
        z_tokens: torch.Tensor,
        channel_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_tokens.ndim != 3 or z_tokens.shape[-1] != self.channels:
            raise ValueError(f"z_tokens must have shape [B,M,{self.channels}]")
        context = z_tokens.mean(dim=1)
        context = self.context_norm(context).unsqueeze(1).expand_as(z_tokens)
        token = self.token_norm(z_tokens)
        norm = z_tokens.norm(dim=-1, keepdim=True) / math.sqrt(self.channels)
        features = torch.cat((token, context, norm), dim=-1)
        logits = self.mlp(features).squeeze(-1)
        if self.channel_conditioner is None:
            if channel_features is not None and channel_features.shape[-1] != 0:
                raise ValueError("This ImportanceHead was built without channel features")
            return logits
        if channel_features is None:
            channel_features = torch.zeros(
                z_tokens.shape[0],
                self.channel_feature_dim,
                device=z_tokens.device,
                dtype=z_tokens.dtype,
            )
        if channel_features.shape != (z_tokens.shape[0], self.channel_feature_dim):
            raise ValueError(
                "channel_features must have shape "
                f"[B,{self.channel_feature_dim}]"
            )
        modulation = self.channel_conditioner(
            channel_features.to(device=z_tokens.device, dtype=features.dtype)
        )
        conditional_logits = (
            features * modulation.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(features.shape[-1])
        return logits + conditional_logits

    @staticmethod
    def positive_scores(logits: torch.Tensor) -> torch.Tensor:
        """Return non-negative SDS values used by the log-Smooth-L1 loss."""
        return torch.nn.functional.softplus(logits)

    def forward(
        self,
        z_tokens: torch.Tensor,
        channel_features: torch.Tensor | None = None,
        return_logits: bool = False,
    ):
        logits = self.logits(z_tokens, channel_features=channel_features)
        scores = torch.softmax(logits / self.temperature, dim=1)
        if return_logits:
            return scores, logits
        return scores
