from __future__ import annotations

from typing import Dict, Optional

import torch

from ..losses.reconstruction import ReconstructionLoss, per_sample_mse
from ..models.system import SIFDJSCC
from ..train.sensitivity import channel_conditioned_ablation_target


@torch.no_grad()
def token_ablation_damage(
    model: SIFDJSCC,
    images: torch.Tensor,
    max_tokens: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Measure clean reconstruction damage after zeroing individual tokens."""
    tokens, metadata = model.backbone.encode(images)
    importance = model.importance_head(tokens)
    clean = model.backbone.decode(tokens, metadata)
    clean_loss = per_sample_mse(clean, images)
    n_tokens = tokens.shape[1]
    selected = torch.arange(n_tokens, device=tokens.device)
    if max_tokens is not None and max_tokens < n_tokens:
        selected = torch.linspace(0, n_tokens - 1, max_tokens, device=tokens.device).long()
    damage = []
    for token_index in selected.tolist():
        ablated = tokens.clone()
        ablated[:, token_index] = 0
        reconstruction = model.backbone.decode(ablated, metadata)
        damage.append(per_sample_mse(reconstruction, images) - clean_loss)
    return {
        "importance": importance.index_select(1, selected),
        "feature_norm": tokens.norm(dim=-1).index_select(1, selected),
        "damage": torch.stack(damage, dim=1),
        "token_indices": selected,
    }


@torch.no_grad()
def clean_ablation_damage_at_indices(
    model: SIFDJSCC,
    images: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Non-channel-conditioned perturbation damage at explicit token indices."""
    tokens, metadata = model.backbone.encode(images)
    if token_indices.ndim != 2 or token_indices.shape[0] != tokens.shape[0]:
        raise ValueError("token_indices must have shape [B,K]")
    clean = model.backbone.decode(tokens, metadata)
    clean_loss = per_sample_mse(clean, images)
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    damage = []
    for column in range(token_indices.shape[1]):
        ablated = tokens.clone()
        ablated[batch_indices, token_indices[:, column]] = 0
        reconstruction = model.backbone.decode(ablated, metadata)
        damage.append(
            (per_sample_mse(reconstruction, images) - clean_loss).clamp_min(0.0)
        )
    return torch.stack(damage, dim=1)


def gradient_feature_scores(
    model: SIFDJSCC,
    images: torch.Tensor,
) -> torch.Tensor:
    """Clean-path gradient-times-feature auxiliary sensitivity baseline."""
    with torch.enable_grad():
        encoded, metadata = model.backbone.encode(images)
        tokens = encoded.detach().requires_grad_(True)
        reconstruction = model.backbone.decode(tokens, metadata)
        loss = per_sample_mse(reconstruction, images).sum()
        gradient = torch.autograd.grad(
            loss,
            tokens,
            retain_graph=False,
            create_graph=False,
        )[0]
        return (tokens.float() * gradient.float()).abs().mean(dim=-1).detach()


@torch.no_grad()
def channel_conditioned_token_ablation_damage(
    model: SIFDJSCC,
    images: torch.Tensor,
    snr_db: torch.Tensor,
    channel_cfg,
    mapper_mode: str = "proposed_ufd",
    n_channel_draws: int = 4,
    max_tokens: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Validate predicted SDS against same-channel perturbation damage."""
    _, auxiliary = model(
        images,
        snr_db,
        channel_cfg=channel_cfg,
        mapper_mode=mapper_mode,
        return_aux=True,
        n_channel_draws=n_channel_draws,
        detach_importance_input=True,
    )
    n_tokens = auxiliary["z"].shape[1]
    count = n_tokens if max_tokens is None else min(int(max_tokens), n_tokens)
    damage, observed, indices = channel_conditioned_ablation_target(
        model,
        images,
        snr_db,
        auxiliary,
        ReconstructionLoss(lambda_mse=1.0, lambda_msssim=0.0),
        tokens_per_batch=count,
        attenuation=0.0,
    )
    return {
        "importance": torch.gather(auxiliary["importance_scores"], 1, indices),
        "feature_norm": torch.gather(auxiliary["z"].norm(dim=-1), 1, indices),
        "damage": torch.gather(damage, 1, indices),
        "observed": torch.gather(observed, 1, indices),
        "token_indices": indices,
        "channel_context": auxiliary["channel_context"],
    }


def spearman_importance_damage(importance: torch.Tensor, damage: torch.Tensor) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError as error:
        raise RuntimeError("Spearman validation requires scipy") from error
    correlations = []
    for scores, effects in zip(importance.detach().cpu().numpy(), damage.detach().cpu().numpy()):
        correlation = spearmanr(scores, effects).statistic
        if correlation == correlation:  # exclude NaN from constant arrays
            correlations.append(float(correlation))
    return float(sum(correlations) / len(correlations)) if correlations else float("nan")
