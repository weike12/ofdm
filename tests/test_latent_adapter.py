import torch

from si_fdjscc.models import ComplexAdapter


def test_power_normalization_and_inverse() -> None:
    torch.manual_seed(3)
    latent = torch.randn(4, 7, 5)
    adapter = ComplexAdapter(normalize_power=True, restore_scale=True)
    symbols, metadata = adapter(latent)
    power = symbols.abs().square().mean(dim=1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-6, rtol=1e-6)
    restored = adapter.inverse(symbols, metadata)
    assert torch.allclose(restored, latent, atol=1e-6, rtol=1e-6)
    assert metadata.pad_length == 1
    assert metadata.token_ids.numel() == symbols.shape[1]

