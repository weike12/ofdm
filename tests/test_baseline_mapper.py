"""Controls for global permutation and greedy token-internal permutation."""

import pytest
import torch

from si_fdjscc.comm import SemanticFrequencyMapper, SubcarrierPlan


def baseline_inputs():
    plan = SubcarrierPlan(
        n_fft=8, occupied_shifted=(-2, -1, 1, 2), pilot_shifted=(), n_bands=2
    )
    mapper = SemanticFrequencyMapper(plan)
    values = torch.arange(1, 23, dtype=torch.float64).reshape(2, 11)
    symbols = torch.complex(values, -values / 3)
    ids = torch.tensor(
        [[0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2], [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1]]
    )
    scores = torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.1, 0.3]])
    gains = torch.arange(1, 49, dtype=torch.float64).reshape(2, 3, 8)
    h_hat = torch.complex(gains, gains / 4)
    return mapper, symbols, ids, scores, h_hat


def token_slot_sets(assignment, token_ids):
    return {
        int(token): {
            slot for slot, source in enumerate(assignment.tolist())
            if source >= 0 and int(token_ids[source]) == int(token)
        }
        for token in token_ids.unique()
    }


def test_sequential_keeps_source_order_and_ignores_importance_and_csi():
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    grid, meta = mapper.map(symbols, ids, scores, h_hat, mode="sequential")
    _, changed = mapper.map(
        symbols, ids, scores.flip(-1), h_hat.flip(-1), mode="sequential", seed=999
    )
    expected = list(range(11)) + [-1]
    assert meta.symbol_index_grid.reshape(2, -1).tolist() == [expected, expected]
    assert torch.equal(meta.symbol_index_grid, changed.symbol_index_grid)
    assert torch.equal(mapper.unmap(grid, meta), symbols)


def test_global_random_uses_sequential_valid_slots_and_ignores_csi():
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    grid, meta = mapper.map(symbols, ids, scores, h_hat, mode="random", seed=123)
    _, changed = mapper.map(
        symbols, ids, scores.flip(-1), h_hat.flip(-1), mode="random", seed=123
    )
    assert torch.equal(meta.symbol_index_grid, changed.symbol_index_grid)
    for assignment in meta.symbol_index_grid.reshape(2, -1):
        assert assignment[-1].item() == -1
        assert sorted(assignment[:-1].tolist()) == list(range(11))
    assert torch.equal(mapper.unmap(grid, meta), symbols)


def test_token_random_preserves_every_greedy_token_slot_and_padding():
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    _, greedy = mapper.map(symbols, ids, scores, h_hat, mode="greedy")
    grid, randomized = mapper.map(
        symbols, ids, scores, h_hat, mode="greedy_token_random", seed=123
    )
    assert torch.equal(randomized.valid_mask, greedy.valid_mask)
    assert torch.equal(mapper.unmap(grid, randomized), symbols)
    assert randomized.energy_diagnostics is None
    for sample in range(symbols.shape[0]):
        old = greedy.symbol_index_grid[sample].flatten()
        new = randomized.symbol_index_grid[sample].flatten()
        assert token_slot_sets(old, ids[sample]) == token_slot_sets(new, ids[sample])
        assert sorted(new[new >= 0].tolist()) == list(range(11))


@pytest.mark.parametrize("mode", ["random", "greedy_token_random"])
def test_random_baselines_are_seeded_and_independent_of_symbol_energy(mode):
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    _, first = mapper.map(symbols, ids, scores, h_hat, mode=mode, seed=123)
    torch.rand(10)  # Unrelated use of the global RNG must not change the mapping.
    _, repeat = mapper.map(symbols, ids, scores, h_hat, mode=mode, seed=123)
    _, other_seed = mapper.map(symbols, ids, scores, h_hat, mode=mode, seed=124)
    _, other_energy = mapper.map(
        symbols.flip(-1).conj() * 100, ids, scores, h_hat, mode=mode, seed=123
    )
    assert torch.equal(first.symbol_index_grid, repeat.symbol_index_grid)
    assert torch.equal(first.symbol_index_grid, other_energy.symbol_index_grid)
    assert not torch.equal(first.symbol_index_grid, other_seed.symbol_index_grid)


@pytest.mark.parametrize("mode", ["random", "greedy_token_random"])
@pytest.mark.parametrize("seed", [None, 123])
def test_random_baselines_do_not_advance_global_rng(mode, seed):
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    before = torch.random.get_rng_state().clone()
    mapper.map(symbols, ids, scores, h_hat, mode=mode, seed=seed)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_random_modes_use_distinct_streams_even_with_one_token_and_flat_csi():
    mapper, symbols, _, _, h_hat = baseline_inputs()
    ids = torch.zeros(symbols.shape[1], dtype=torch.long)
    scores = torch.ones(2, 1)
    h_hat = torch.ones_like(h_hat)
    _, global_random = mapper.map(symbols, ids, scores, h_hat, mode="random", seed=123)
    _, token_random = mapper.map(
        symbols, ids, scores, h_hat, mode="greedy_token_random", seed=123
    )
    _, greedy = mapper.map(symbols, ids, scores, h_hat, mode="greedy")
    assert not torch.equal(global_random.symbol_index_grid, token_random.symbol_index_grid)
    # Flat CSI gives no energy-proxy gain, but random control must still shuffle.
    assert not torch.equal(greedy.symbol_index_grid, token_random.symbol_index_grid)


def test_token_random_accepts_identity_draw_without_retry(monkeypatch):
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    calls = []

    def identity(size, **kwargs):
        calls.append(size)
        return torch.arange(size)

    monkeypatch.setattr(torch, "randperm", identity)
    _, greedy = mapper.map(symbols, ids, scores, h_hat, mode="greedy")
    _, randomized = mapper.map(
        symbols, ids, scores, h_hat, mode="greedy_token_random", seed=123
    )
    assert torch.equal(greedy.symbol_index_grid, randomized.symbol_index_grid)
    assert len(calls) == 6  # One draw for each token in each batch item.


def test_token_random_rejects_invalid_token_ids_instead_of_dropping_sources():
    mapper, symbols, ids, scores, h_hat = baseline_inputs()
    ids[0, 0] = scores.shape[1]
    with pytest.raises(ValueError, match="outside importance"):
        mapper.map(symbols, ids, scores, h_hat, mode="greedy_token_random", seed=123)


@pytest.mark.parametrize("source", ["oracle", "temporal_mean"])
def test_token_random_protocol_rejects_nonestimated_mapper_csi(source):
    from si_fdjscc.models import ImportanceHead, SIFDJSCC, TinyConvBackbone

    model = SIFDJSCC(
        backbone=TinyConvBackbone(latent_channels=8),
        importance_head=ImportanceHead(channels=8),
    ).eval()
    with pytest.raises(ValueError, match="requires mapper_csi_source='estimated'"):
        model(
            torch.zeros(1),
            torch.tensor([10.0]),
            {"mapper_csi_source": source},
            mapper_mode="greedy_token_random",
        )
