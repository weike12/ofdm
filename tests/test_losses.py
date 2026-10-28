import torch

from si_fdjscc.losses import (
    ImportanceSupervisionLoss,
    ReconstructionLoss,
    cvar_loss,
    distortion_sensitivity_target,
    importance_weighted_latent_loss,
    range_violation_loss,
    sensitivity_distribution,
)


def test_cvar_selects_worst_draws() -> None:
    losses = torch.tensor([[1.0, 4.0], [2.0, 3.0], [7.0, 2.0], [0.0, 1.0]])
    assert cvar_loss(losses, alpha=0.8).item() == 5.5
    assert cvar_loss(losses, alpha=0.9).item() == 5.5
    assert cvar_loss(losses, alpha=0.7).item() == 4.0
    assert cvar_loss(losses, tail_fraction=0.25).item() == 5.5


def test_importance_weighted_latent_loss() -> None:
    clean = torch.zeros(1, 2, 1)
    received = torch.tensor([[[1.0], [2.0]]])
    importance = torch.tensor([[0.75, 0.25]])
    loss = importance_weighted_latent_loss(clean, received, importance)
    assert torch.allclose(loss, torch.tensor(1.75))


def test_bounded_reconstruction_loss_and_range_penalty() -> None:
    reconstruction = torch.tensor([[[[-2.0, 2.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])
    bounded = ReconstructionLoss(clamp_prediction=True)
    unbounded = ReconstructionLoss(clamp_prediction=False)
    assert torch.allclose(bounded(reconstruction, target), torch.tensor(0.0))
    assert torch.allclose(unbounded(reconstruction, target), torch.tensor(2.5))
    assert torch.allclose(
        range_violation_loss(reconstruction),
        torch.tensor([2.5]),
    )


def _small_scale_dsi(scale: float) -> torch.Tensor:
    tokens = (
        torch.tensor([1.0, 2.0, 3.0, 4.0])
        .reshape(1, 4, 1)
        .mul(scale)
        .requires_grad_(True)
    )
    per_sample_loss = tokens.square().flatten(1).mean(dim=1)
    target, sensitivity = distortion_sensitivity_target(
        tokens,
        per_sample_loss,
        temperature=0.5,
        retain_graph=False,
    )
    assert sensitivity.max() < 1.0e-8
    return target


def test_dsi_target_preserves_relative_differences_at_tiny_scale() -> None:
    target = _small_scale_dsi(1.0e-6)
    uniform = torch.full_like(target, 0.25)
    assert torch.isfinite(target).all()
    assert torch.allclose(target.sum(dim=1), torch.ones(1))
    assert not torch.allclose(target, uniform)
    assert target.argmax(dim=1).item() == 3


def test_dsi_target_is_invariant_to_absolute_scale() -> None:
    reference = _small_scale_dsi(1.0e-6)
    much_smaller = _small_scale_dsi(1.0e-9)
    assert torch.allclose(reference, much_smaller, rtol=1.0e-5, atol=1.0e-7)


def test_all_zero_dsi_target_has_finite_uniform_fallback() -> None:
    tokens = torch.zeros(1, 4, 2, requires_grad=True)
    loss = (tokens * 0.0).sum()
    target, sensitivity = distortion_sensitivity_target(tokens, loss)
    assert torch.equal(sensitivity, torch.zeros_like(sensitivity))
    assert torch.allclose(target, torch.full_like(target, 0.25))


def test_importance_kl_is_finite_without_probability_clamp() -> None:
    logits = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
    predicted = torch.softmax(logits / 0.5, dim=1)
    target = torch.tensor([[0.2, 0.3, 0.5]])
    criterion = ImportanceSupervisionLoss(
        rank_weight=0.1,
        n_pairs=16,
        prediction_temperature=0.5,
    )
    loss, components = criterion(predicted, logits, target)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["importance_kl"])
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_log_smooth_l1_uses_only_observed_ablation_targets() -> None:
    logits = torch.tensor([[0.1, -0.2, 0.3, 0.0]], requires_grad=True)
    predicted = torch.softmax(logits / 0.5, dim=1)
    target = torch.tensor([[0.0, 0.25, 0.0, 1.0]])
    mask = torch.tensor([[False, True, False, True]])
    criterion = ImportanceSupervisionLoss(
        objective="log_smooth_l1",
        rank_weight=0.1,
        n_pairs=32,
    )
    loss, components = criterion(
        predicted, logits, target, target_mask=mask
    )
    distribution = sensitivity_distribution(target, mask=mask)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["importance_smooth_l1"])
    assert torch.equal(distribution[~mask], torch.zeros_like(distribution[~mask]))
    assert torch.allclose(distribution.sum(dim=1), torch.ones(1))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
