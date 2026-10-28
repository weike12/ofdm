import pytest
import torch

from si_fdjscc.train.common import build_system


@pytest.mark.parametrize("source", ["oracle", "temporal_mean"])
def test_energy_mapper_rejects_nonestimated_csi_before_encoding(source):
    model = build_system(
        {
            "model": {"backbone": {"type": "tiny", "latent_channels": 8}},
            "ofdm": {"channel_taps": 2, "n_cp": 8},
        }
    ).eval()
    # An invalid image confirms the source guard runs before the encoder.
    with pytest.raises(ValueError, match="requires mapper_csi_source='estimated'"):
        model(
            torch.zeros(1),
            torch.tensor([10.0]),
            {"mapper_csi_source": source},
            mapper_mode="greedy_token_energy",
        )


def test_energy_mapper_exposes_exact_complex_symbols_for_residual_audit():
    torch.manual_seed(17)
    model = build_system(
        {
            "model": {
                "backbone": {"type": "tiny", "latent_channels": 8},
                "complex_adapter": {"restore_scale": False},
            },
            "ofdm": {"channel_taps": 2, "n_cp": 8},
        }
    ).eval()
    with torch.no_grad():
        _, auxiliary = model(
            torch.rand(1, 3, 16, 16),
            torch.tensor([10.0]),
            {
                "csi_mode": "imperfect",
                "csi_nmse_db": -10,
                "channel_seed": 2026,
                "mapper_csi_source": "estimated",
                "receiver_csi_source": "temporal_mean",
            },
            mapper_mode="greedy_token_energy",
            return_aux=True,
        )
    transmitted = auxiliary["symbols_tx"]
    recovered = auxiliary["draws"][0]["symbols_rx"]
    assert transmitted.shape == recovered.shape
    assert transmitted.ndim == 2 and transmitted.is_complex() and recovered.is_complex()
    assert torch.isfinite(transmitted).all() and torch.isfinite(recovered).all()
