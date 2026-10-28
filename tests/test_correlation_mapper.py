import math

import pytest
import torch

from si_fdjscc.comm import (
    MultipathChannel,
    OFDMFrequencySelectiveChannel,
    SemanticFrequencyMapper,
    SubcarrierPlan,
)
from si_fdjscc.evaluation.metrics import assigned_high_sensitivity_correlation


def _small_plan() -> SubcarrierPlan:
    return SubcarrierPlan(
        n_fft=8,
        occupied_shifted=(-3, -2, -1, 1, 2, 3),
        pilot_shifted=(-2, 2),
        n_bands=1,
    )


def test_exponential_pdp_data_correlation_matches_closed_form() -> None:
    plan = _small_plan()
    channel = MultipathChannel(n_taps=3, tau=1.5)
    observed = channel.data_subcarrier_correlation(plan, dtype=torch.float64)

    carriers = torch.tensor(plan.data_shifted, dtype=torch.float64)
    delta = carriers[:, None] - carriers[None, :]
    taps = torch.arange(channel.n_taps, dtype=torch.float64)
    phase = -2.0 * math.pi * delta[..., None] * taps / plan.n_fft
    expected = (
        torch.exp(torch.complex(torch.zeros_like(phase), phase))
        * channel.pdp.double().to(torch.complex128)
    ).sum(dim=-1).abs()
    expected = expected / channel.pdp.double().sum()

    assert observed.shape == (plan.n_data, plan.n_data)
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(torch.diagonal(observed), torch.ones(plan.n_data, dtype=torch.float64))
    assert torch.allclose(observed, observed.transpose(0, 1))


def test_mapper_rejects_wrong_correlation_shape() -> None:
    plan = _small_plan()
    mapper = SemanticFrequencyMapper(plan)
    symbols = torch.ones(1, 2, dtype=torch.complex64)
    with pytest.raises(ValueError, match="correlation_matrix"):
        mapper.map(
            symbols,
            torch.zeros(2, dtype=torch.long),
            torch.ones(1, 1),
            None,
            correlation_matrix=torch.eye(plan.n_data + 1),
        )


def test_seeded_random_spectral_notch_is_resolved_and_reproducible() -> None:
    channel = OFDMFrequencySelectiveChannel()
    notch = {
        "width_range": [8, 12],
        "random_start": True,
        "attenuation_db_range": [15.0, 25.0],
    }
    first = channel.prepare_state(
        1, 1, torch.device("cpu"), channel_seed=123, spectral_notch=notch
    )
    repeated = channel.prepare_state(
        1, 1, torch.device("cpu"), channel_seed=123, spectral_notch=notch
    )
    different = channel.prepare_state(
        1, 1, torch.device("cpu"), channel_seed=124, spectral_notch=notch
    )

    assert first.spectral_notch == repeated.spectral_notch
    assert first.spectral_notch != different.spectral_notch
    assert torch.equal(first.frequency_gain, repeated.frequency_gain)
    assert 8 <= first.spectral_notch["width"] <= 12
    assert 0 <= first.spectral_notch["start"] <= channel.plan.n_data - first.spectral_notch["width"]
    assert 15.0 <= first.spectral_notch["attenuation_db"] <= 25.0


def test_fixed_spectral_notch_keeps_legacy_defaults() -> None:
    channel = OFDMFrequencySelectiveChannel()
    state = channel.prepare_state(
        1, 1, torch.device("cpu"), channel_seed=1, spectral_notch={"width": 10}
    )
    assert state.spectral_notch["start"] == 19
    assert state.spectral_notch["width"] == 10
    assert state.spectral_notch["attenuation_db"] == 20.0


def test_high_sensitivity_mapping_correlation_metric() -> None:
    grid = torch.tensor([[[0, 1, 2, 3]]])
    token_ids = torch.tensor([0, 0, 1, 1])
    sensitivity = torch.tensor([[2.0, 1.0]])
    correlation = torch.eye(4)
    correlation[0, 1] = correlation[1, 0] = 0.25
    observed = assigned_high_sensitivity_correlation(
        grid, token_ids, sensitivity, correlation, high_quantile=0.8
    )
    assert torch.allclose(observed, torch.tensor([0.25]))
