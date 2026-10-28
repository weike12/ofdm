from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import nn


def sensitivity_distribution(
    sensitivity: torch.Tensor,
    temperature: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert non-negative SDS values into a scale-invariant distribution.

    ``mask`` is used when only a stratified subset of semantic units was
    perturbed.  Unobserved units receive zero probability and are excluded from
    the normalisation rather than being treated as zero-sensitivity labels.
    """
    if sensitivity.ndim != 2:
        raise ValueError("sensitivity must have shape [B,M]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mask is None:
        valid = torch.ones_like(sensitivity, dtype=torch.bool)
    else:
        if mask.shape != sensitivity.shape:
            raise ValueError("mask must have the same shape as sensitivity")
        valid = mask.to(device=sensitivity.device, dtype=torch.bool)
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("Every sample must contain at least one valid sensitivity")
    values = sensitivity.detach().float().clamp_min(0.0)
    masked_values = torch.where(valid, values, torch.zeros_like(values))
    sample_max = masked_values.amax(dim=1, keepdim=True)
    nonzero_sample = sample_max > 0
    safe_scale = torch.where(nonzero_sample, sample_max, torch.ones_like(sample_max))
    relative = masked_values / safe_scale
    # An all-zero observed subset falls back to a uniform distribution over
    # only the observed units.
    relative = torch.where(
        nonzero_sample,
        relative,
        valid.to(relative.dtype),
    )
    negative_infinity = torch.full_like(relative, -torch.inf)
    target_logits = torch.where(
        valid & (relative > 0),
        torch.log(relative.clamp_min(torch.finfo(relative.dtype).tiny))
        / float(temperature),
        negative_infinity,
    )
    return torch.softmax(target_logits, dim=1).detach()


def distortion_sensitivity_target(
    z_tokens: torch.Tensor,
    clean_image_loss: torch.Tensor,
    temperature: float = 0.5,
    retain_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a scale-invariant gradient-times-activation DSI target.

    ``clean_image_loss`` may be a scalar or one loss per batch item. Per-sample
    losses are summed before differentiation so the target temperature does not
    accidentally depend on batch size. ``create_graph=False`` prevents second
    derivatives.

    Raw gradient-times-activation values can be much smaller than one. Applying
    ``softmax(sensitivity / temperature)`` directly therefore collapses the
    target towards a uniform distribution. Instead, this function treats the
    non-negative sensitivities as unnormalised masses and applies temperature
    in log space::

        target_i = sensitivity_i ** (1 / temperature) / sum_j(...)

    Dividing by the per-sample maximum before taking the logarithm makes the
    computation invariant to the absolute DSI scale. Exact zero scores receive
    zero probability; an all-zero sample falls back to a uniform target.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not z_tokens.requires_grad:
        raise ValueError("z_tokens must require gradients to build a DSI target")
    if clean_image_loss.numel() not in {1, z_tokens.shape[0]}:
        raise ValueError("clean_image_loss must be scalar or have one value per batch item")
    objective = clean_image_loss.sum()
    gradient = torch.autograd.grad(
        objective,
        z_tokens,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    # Always form the target in float32. Under AMP, half-precision products can
    # underflow before their relative differences have a chance to be normalised.
    sensitivity = (
        (z_tokens.float() * gradient.float()).abs().mean(dim=-1).detach()
    )
    target = sensitivity_distribution(sensitivity, temperature=temperature)
    return target, sensitivity


def importance_target_statistics(
    target: torch.Tensor,
    sensitivity: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return scale-aware diagnostics for detecting a collapsed DSI target."""
    if target.ndim != 2 or target.shape != sensitivity.shape:
        raise ValueError("target and sensitivity must have identical shape [B,M]")
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.shape != target.shape:
            raise ValueError("mask must have the same shape as target")
        valid = mask.to(device=target.device, dtype=torch.bool)
    token_count = valid.sum(dim=1).clamp_min(1)
    probabilities = target.detach().float() * valid
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    positive = probabilities > 0
    log_probabilities = torch.zeros_like(probabilities)
    log_probabilities[positive] = torch.log(probabilities[positive])
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    log_count = token_count.float().log()
    normalized_entropy = torch.where(
        token_count > 1,
        entropy / log_count.clamp_min(torch.finfo(log_count.dtype).eps),
        torch.zeros_like(entropy),
    )
    uniform = valid.to(probabilities.dtype) / token_count.unsqueeze(1)
    observed_sensitivity = sensitivity.detach().float()[valid]
    return {
        "dsi_mean": observed_sensitivity.mean(),
        "dsi_max": observed_sensitivity.amax(),
        "target_peak_ratio": (
            probabilities.amax(dim=1) * token_count
        ).mean(),
        "target_entropy_normalized": normalized_entropy.mean(),
        "target_l1_from_uniform": (
            probabilities - uniform
        ).abs().sum(dim=1).mean(),
    }


def pairwise_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int = 1024,
    generator: Optional[torch.Generator] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("logits and target must both have shape [B,M]")
    batch, n_tokens = logits.shape
    if n_tokens < 2 or n_pairs < 1:
        return logits.sum() * 0.0
    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError("mask must have the same shape as logits and target")
        losses = []
        for sample in range(batch):
            valid = torch.nonzero(mask[sample], as_tuple=False).flatten()
            if valid.numel() < 2:
                continue
            left_position = torch.randint(
                valid.numel(),
                (n_pairs,),
                device=logits.device,
                generator=generator,
            )
            offset = torch.randint(
                1,
                valid.numel(),
                (n_pairs,),
                device=logits.device,
                generator=generator,
            )
            right_position = torch.remainder(left_position + offset, valid.numel())
            left = valid[left_position]
            right = valid[right_position]
            logit_delta = logits[sample, left] - logits[sample, right]
            target_delta = target[sample, left] - target[sample, right]
            labels = torch.sign(target_delta)
            informative = labels != 0
            if informative.any():
                losses.append(
                    functional.softplus(
                        -labels[informative] * logit_delta[informative]
                    ).mean()
                )
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0
    left = torch.randint(
        n_tokens, (batch, n_pairs), device=logits.device, generator=generator
    )
    offset = torch.randint(
        1, n_tokens, (batch, n_pairs), device=logits.device, generator=generator
    )
    right = torch.remainder(left + offset, n_tokens)
    logit_delta = torch.gather(logits, 1, left) - torch.gather(logits, 1, right)
    target_delta = torch.gather(target, 1, left) - torch.gather(target, 1, right)
    labels = torch.sign(target_delta)
    informative = labels != 0
    if not informative.any():
        return logits.sum() * 0.0
    return functional.softplus(-labels[informative] * logit_delta[informative]).mean()


class ImportanceSupervisionLoss(nn.Module):
    def __init__(
        self,
        rank_weight: float = 0.1,
        n_pairs: int = 1024,
        prediction_temperature: float = 0.5,
        objective: str = "kl",
        log_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if prediction_temperature <= 0:
            raise ValueError("prediction_temperature must be positive")
        self.rank_weight = float(rank_weight)
        self.n_pairs = int(n_pairs)
        self.prediction_temperature = float(prediction_temperature)
        objective = str(objective).lower()
        if objective not in {"kl", "log_smooth_l1"}:
            raise ValueError("objective must be 'kl' or 'log_smooth_l1'")
        if log_epsilon <= 0:
            raise ValueError("log_epsilon must be positive")
        self.objective = objective
        self.log_epsilon = float(log_epsilon)

    def forward(
        self,
        predicted: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predicted.shape != target.shape or logits.shape != target.shape:
            raise ValueError("predicted, logits, and target must have identical shapes")
        target_float = target.detach().float()
        if target_mask is not None:
            if target_mask.shape != target.shape:
                raise ValueError("target_mask must have the same shape as target")
            target_mask = target_mask.to(device=target.device, dtype=torch.bool)
        rank = pairwise_rank_loss(
            logits.float(),
            target_float,
            n_pairs=self.n_pairs,
            mask=target_mask,
        )
        if self.objective == "kl":
            if target_mask is not None:
                target_float = sensitivity_distribution(
                    target_float,
                    temperature=1.0,
                    mask=target_mask,
                )
            # Work from logits rather than log(clamp(probability)). log_softmax
            # is stable even for very sharp distributions.
            log_predicted = functional.log_softmax(
                logits.float() / self.prediction_temperature,
                dim=1,
            )
            primary = functional.kl_div(
                log_predicted,
                target_float,
                reduction="batchmean",
            )
            components = {"importance_kl": primary}
        else:
            positive_prediction = functional.softplus(logits.float())
            elementwise = functional.smooth_l1_loss(
                torch.log(positive_prediction + self.log_epsilon),
                torch.log(target_float.clamp_min(0.0) + self.log_epsilon),
                reduction="none",
            )
            if target_mask is None:
                primary = elementwise.mean()
            else:
                if not target_mask.any():
                    raise ValueError("target_mask must contain at least one observed unit")
                primary = elementwise[target_mask].mean()
            components = {"importance_smooth_l1": primary}
        total = primary + self.rank_weight * rank
        return total, {**components, "importance_rank": rank}
