from __future__ import annotations

import math

import torch


def cvar_loss(
    per_draw_losses: torch.Tensor,
    alpha: float = 0.8,
    *,
    tail_fraction: float | None = None,
) -> torch.Tensor:
    """Average the upper tail at CVaR confidence level ``alpha``.

    In the research plan, ``alpha=0.8`` means that the worst 20% of channel
    draws form the empirical tail.  ``tail_fraction`` is an explicit migration
    hook for older callers whose parameter represented the tail size directly.
    """
    if per_draw_losses.ndim < 1 or per_draw_losses.shape[0] < 1:
        raise ValueError("per_draw_losses must have a non-empty draw dimension")
    if tail_fraction is not None:
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0,1]")
        fraction = float(tail_fraction)
    else:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("CVaR confidence alpha must be in [0,1)")
        fraction = 1.0 - float(alpha)
    k = max(1, math.ceil(fraction * per_draw_losses.shape[0]))
    worst = torch.topk(per_draw_losses, k=k, dim=0, largest=True).values
    return worst.mean()


def importance_weighted_latent_loss(
    clean_tokens: torch.Tensor,
    received_tokens: torch.Tensor,
    importance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Importance-weighted latent MSE for one or multiple channel draws."""
    if received_tokens.ndim == clean_tokens.ndim + 1:
        clean_tokens = clean_tokens.unsqueeze(0).expand_as(received_tokens)
        importance = importance.unsqueeze(0).expand(
            received_tokens.shape[0], *importance.shape
        )
    if received_tokens.shape != clean_tokens.shape:
        raise ValueError("clean_tokens and received_tokens shapes are incompatible")
    weights = importance.detach()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    error = (clean_tokens - received_tokens).square().mean(dim=-1)
    return (weights * error).sum(dim=-1).mean()
