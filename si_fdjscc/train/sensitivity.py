from __future__ import annotations

from typing import Any, Mapping

import torch


def sample_stratified_token_indices(
    scores: torch.Tensor,
    count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample unique high/middle/low-score semantic units per batch item."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B,M]")
    if count < 1:
        raise ValueError("count must be positive")
    batch, n_tokens = scores.shape
    count = min(int(count), n_tokens)
    # Put any remainder into the high- and mid-risk strata first.
    base = count // 3
    allocations = [base + int(count % 3 > 0), base + int(count % 3 > 1), base]
    rows = []
    for sample in range(batch):
        order = torch.argsort(scores[sample], stable=True)
        mid_boundary = int(round(0.50 * n_tokens))
        high_boundary = int(round(0.80 * n_tokens))
        groups = (
            order[high_boundary:],
            order[mid_boundary:high_boundary],
            order[:mid_boundary],
        )
        selected = []
        for group, requested in zip(groups, allocations):
            take = min(requested, int(group.numel()))
            if take:
                permutation = torch.randperm(
                    group.numel(), device=scores.device, generator=generator
                )
                selected.append(group[permutation[:take]])
        chosen = (
            torch.cat(selected)
            if selected
            else torch.empty(0, dtype=torch.long, device=scores.device)
        )
        if chosen.numel() < count:
            available = torch.ones(n_tokens, dtype=torch.bool, device=scores.device)
            available[chosen] = False
            remainder = torch.nonzero(available, as_tuple=False).flatten()
            permutation = torch.randperm(
                remainder.numel(), device=scores.device, generator=generator
            )
            chosen = torch.cat(
                (chosen, remainder[permutation[: count - chosen.numel()]])
            )
        rows.append(chosen[:count])
    return torch.stack(rows, dim=0)


@torch.no_grad()
def channel_conditioned_ablation_target(
    model,
    images: torch.Tensor,
    snr_db: torch.Tensor,
    transmission_aux: Mapping[str, Any],
    reconstruction_loss,
    tokens_per_batch: int,
    attenuation: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure same-channel reconstruction damage for sampled semantic units.

    The clean and perturbed decodes reuse the exact mapping, channel impulse
    response, CSI error, and AWGN seed stored by ``SIFDJSCC.forward``.  The
    returned sensitivity and mask have shape ``[B,M]``; unobserved entries must
    be ignored by the supervision loss.
    """
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("attenuation must be between 0 and 1")
    tokens = transmission_aux["z"].detach()
    if tokens.ndim != 3:
        raise ValueError("transmission_aux['z'] must have shape [B,M,C]")
    scores = transmission_aux.get("importance_scores")
    if scores is None:
        scores = tokens.norm(dim=-1)
    indices = sample_stratified_token_indices(
        scores.detach(), tokens_per_batch
    )
    batch_size, n_tokens, _ = tokens.shape
    draw_count = len(transmission_aux["draws"])
    if draw_count < 1:
        raise ValueError("At least one channel draw is required")

    # Disable codec dropout/stochastic depth for the teacher so the measured
    # delta is caused by the semantic perturbation alone.
    backbone_was_training = model.backbone.training
    model.backbone.eval()
    try:
        clean_reconstructions = model.reconstruct_with_fixed_transmission(
            tokens,
            transmission_aux["backbone_metadata"],
            snr_db,
            transmission_aux,
        ).float()
        repeated_images = images.float().unsqueeze(0).expand_as(
            clean_reconstructions
        )
        clean_losses = reconstruction_loss(
            clean_reconstructions.reshape(-1, *images.shape[1:]),
            repeated_images.reshape(-1, *images.shape[1:]),
            reduction="none",
        ).reshape(draw_count, batch_size)

        perturbed_losses = []
        batch_indices = torch.arange(batch_size, device=tokens.device)
        for column in range(indices.shape[1]):
            perturbed = tokens.clone()
            perturbed[batch_indices, indices[:, column]] *= float(attenuation)
            reconstructions = model.reconstruct_with_fixed_transmission(
                perturbed,
                transmission_aux["backbone_metadata"],
                snr_db,
                transmission_aux,
            ).float()
            losses = reconstruction_loss(
                reconstructions.reshape(-1, *images.shape[1:]),
                repeated_images.reshape(-1, *images.shape[1:]),
                reduction="none",
            ).reshape(draw_count, batch_size)
            perturbed_losses.append(losses)
    finally:
        model.backbone.train(backbone_was_training)

    # Average [D_m^(i)-D_m]+ over the identical M channel draws.
    damage = (
        torch.stack(perturbed_losses, dim=0) - clean_losses.unsqueeze(0)
    ).clamp_min(0.0).mean(dim=1).transpose(0, 1)
    sensitivity = torch.zeros(
        batch_size,
        n_tokens,
        device=tokens.device,
        dtype=damage.dtype,
    )
    observed = torch.zeros(
        batch_size, n_tokens, device=tokens.device, dtype=torch.bool
    )
    sensitivity.scatter_(1, indices, damage)
    observed.scatter_(1, indices, True)
    return sensitivity, observed, indices
