"""Server-side properties of greedy token-internal energy rearrangement."""

import math

import pytest
import torch

from si_fdjscc.comm import SemanticFrequencyMapper, SubcarrierPlan


def small_plan() -> SubcarrierPlan:
    return SubcarrierPlan(
        n_fft=8, occupied_shifted=(-2, -1, 1, 2), pilot_shifted=(), n_bands=2
    )


def channel(plan: SubcarrierPlan, gains, batch=1):
    h = torch.ones(batch, 2, plan.n_fft, dtype=torch.complex128)
    values = torch.as_tensor(gains, dtype=torch.complex128)
    if values.ndim == 1:
        values = values.unsqueeze(0).expand(2, -1)
    h[:, :, list(plan.data_fft)] = values
    return h


def powers_to_symbols(powers):
    p = torch.as_tensor(powers, dtype=torch.float64)
    return p.sqrt().to(torch.complex128)


def independent_proxy(assignment, symbols, estimated_h, noise, plan):
    mean_h = estimated_h.mean(dim=0)[list(plan.data_fft)]
    bias = (noise / (mean_h.abs().square() + noise)).square()
    terms = [
        float(symbols[source].abs().square() * bias[slot % plan.n_data])
        for slot, source in enumerate(assignment.tolist()) if source >= 0
    ]
    return math.fsum(terms)


def token_slots(assignment, token_ids):
    return {
        int(token): sorted(
            slot for slot, source in enumerate(assignment.tolist())
            if source >= 0 and int(token_ids[source]) == int(token)
        )
        for token in token_ids.unique()
    }


def test_energy_order_reduces_proxy_without_changing_token_slot_sets():
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    ids = torch.tensor([0] * 4 + [1] * 4)
    importance = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    symbols = powers_to_symbols([1, 16, 4, 9, 1, 25, 4, 16])[None]
    h = channel(plan, [2, math.sqrt(3), math.sqrt(2), 1])
    _, greedy = mapper.map(symbols, ids, importance, h, mode="greedy", noise_variance=1.0)
    grid, energy = mapper.map(
        symbols, ids, importance, h, mode="greedy_token_energy", noise_variance=1.0
    )
    initial = greedy.symbol_index_grid[0].flatten()
    final = energy.symbol_index_grid[0].flatten()
    assert token_slots(initial, ids) == token_slots(final, ids)
    assert torch.equal(energy.valid_mask, greedy.valid_mask)
    assert torch.equal(mapper.unmap(grid, energy), symbols)
    assert sorted(final.tolist()) == list(range(8))
    d = energy.energy_diagnostics
    assert d is not None
    before = independent_proxy(initial, symbols[0], h[0], 1.0, plan)
    after = independent_proxy(final, symbols[0], h[0], 1.0, plan)
    assert after < before
    assert d["energy_proxy_initial"].item() == pytest.approx(before)
    assert d["energy_proxy_final"].item() == pytest.approx(after)
    changed = int((initial != final).sum().item())
    assert d["energy_changed_symbol_count"].item() == changed
    assert d["energy_changed_symbol_fraction"].item() == pytest.approx(changed / 8)


@pytest.mark.parametrize("two_dimensional", [False, True])
def test_batch_padding_and_ties_are_deterministic(two_dimensional):
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2], [2, 2, 1, 1, 0, 0, 0]])
    importance = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float64)
    symbols = powers_to_symbols([[4, 1, 9, 1, 16, 4, 25], [9, 1, 4, 16, 1, 4, 25]])
    h = channel(plan, [2, 2, 1, 1], batch=2)
    if two_dimensional:
        h = h[:, 0]
    noise = torch.tensor([1.0, 2.0], dtype=torch.float64)
    _, greedy = mapper.map(symbols, ids, importance, h, mode="greedy", noise_variance=noise)
    grid, energy = mapper.map(
        symbols, ids, importance, h, mode="greedy_token_energy", noise_variance=noise
    )
    _, repeat = mapper.map(
        symbols, ids, importance, h, mode="greedy_token_energy", noise_variance=noise
    )
    assert torch.equal(energy.symbol_index_grid, repeat.symbol_index_grid)
    assert torch.equal(mapper.unmap(grid, energy), symbols)
    assert torch.equal(energy.valid_mask, greedy.valid_mask)
    for sample in range(2):
        old = greedy.symbol_index_grid[sample].flatten()
        new = energy.symbol_index_grid[sample].flatten()
        assert token_slots(old, ids[sample]) == token_slots(new, ids[sample])
        assert sorted(new[new >= 0].tolist()) == list(range(7))
        assert energy.energy_diagnostics["energy_proxy_final"][sample] <= (
            energy.energy_diagnostics["energy_proxy_initial"][sample]
        )


def test_flat_reference_channel_returns_exact_greedy():
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    ids = torch.tensor([0] * 4 + [1] * 4)
    symbols = powers_to_symbols([1, 16, 4, 9, 1, 25, 4, 16])[None]
    importance = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    h = channel(plan, [1, 1, 1, 1])
    _, greedy = mapper.map(symbols, ids, importance, h, mode="greedy", noise_variance=1.0)
    _, energy = mapper.map(
        symbols, ids, importance, h, mode="greedy_token_energy", noise_variance=1.0
    )
    assert torch.equal(energy.symbol_index_grid, greedy.symbol_index_grid)
    assert energy.energy_diagnostics["energy_changed_symbol_count"].item() == 0
    assert energy.energy_diagnostics["energy_proxy_final"].item() == pytest.approx(
        energy.energy_diagnostics["energy_proxy_initial"].item()
    )


def test_complex_temporal_mean_is_used_for_bias():
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    ids = torch.zeros(8, dtype=torch.long)
    symbols = powers_to_symbols([100, 81, 64, 49, 36, 25, 16, 1])[None]
    importance = torch.ones(1, 1, dtype=torch.float64)
    h = channel(plan, [[2, 1, 0.5, 0.25], [-2, 1, 0.5, 0.25]])
    _, greedy = mapper.map(symbols, ids, importance, h, mode="greedy", noise_variance=1.0)
    _, energy = mapper.map(
        symbols, ids, importance, h, mode="greedy_token_energy", noise_variance=1.0
    )
    initial = greedy.symbol_index_grid[0].flatten()
    final = energy.symbol_index_grid[0].flatten()
    assert token_slots(initial, ids) == token_slots(final, ids)
    assert independent_proxy(final, symbols[0], h[0], 1.0, plan) < independent_proxy(
        initial, symbols[0], h[0], 1.0, plan
    )
    # The carrier with large instantaneous power has zero complex mean.
    assert max(abs(symbols[int(final[0])]), abs(symbols[int(final[4])])) < max(
        abs(symbols[int(initial[0])]), abs(symbols[int(initial[4])])
    )


@pytest.mark.parametrize("bad_noise", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_energy_mode_rejects_missing_or_invalid_noise(bad_noise):
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    with pytest.raises(ValueError, match="noise_variance"):
        mapper.map(
            powers_to_symbols([1, 2, 3, 4])[None],
            torch.zeros(4, dtype=torch.long),
            torch.ones(1, 1),
            channel(plan, [2, 1, 1, 1]),
            mode="greedy_token_energy",
            noise_variance=bad_noise,
        )


def test_energy_mode_rejects_missing_csi_and_nonfinite_input():
    plan = small_plan()
    mapper = SemanticFrequencyMapper(plan)
    symbols = powers_to_symbols([1, 2, 3, 4])[None]
    ids = torch.zeros(4, dtype=torch.long)
    scores = torch.ones(1, 1)
    with pytest.raises(ValueError, match="estimated h_hat"):
        mapper.map(symbols, ids, scores, None, mode="greedy_token_energy", noise_variance=1)
    h = channel(plan, [2, 1, 1, 1])
    h[0, 0, 0] = complex(float("nan"), 0)
    with pytest.raises(ValueError, match="h_hat"):
        mapper.map(symbols, ids, scores, h, mode="greedy_token_energy", noise_variance=1)
