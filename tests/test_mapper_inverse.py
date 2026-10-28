import pytest
import torch

from si_fdjscc.comm import SemanticFrequencyMapper, SubcarrierPlan


@pytest.mark.parametrize(
    "mode",
    [
        "sequential",
        "random",
        "greedy",
        "greedy_token_random",
        "greedy_token_energy",
    ],
)
def test_mapper_is_exact_and_has_no_overhead(mode: str) -> None:
    torch.manual_seed(5)
    batch, n_tokens, channels = 2, 11, 8
    n_symbols = n_tokens * channels // 2
    symbols = torch.complex(torch.randn(batch, n_symbols), torch.randn(batch, n_symbols))
    token_ids = torch.arange(n_tokens).repeat_interleave(channels // 2)
    importance = torch.softmax(torch.randn(batch, n_tokens), dim=1)
    h_hat = torch.complex(torch.randn(batch, 1, 64), torch.randn(batch, 1, 64))
    mapper = SemanticFrequencyMapper(SubcarrierPlan())
    grid, metadata = mapper.map(
        symbols, token_ids, importance, h_hat, mode=mode, seed=123,
        noise_variance=1.0 if mode == "greedy_token_energy" else None,
    )
    restored = mapper.unmap(grid, metadata)
    assert torch.equal(restored, symbols)
    assert int(metadata.valid_mask.sum().item()) == batch * n_symbols
    assert grid.shape[1] * grid.shape[2] >= n_symbols
