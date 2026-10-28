import pytest
import torch

from si_fdjscc.comm import CSIModel, MMSEEqualizer, OFDMModem


@pytest.mark.parametrize("nmse_db", [-30.0, -20.0, -10.0])
def test_controlled_nmse_matches_target(nmse_db: float) -> None:
    torch.manual_seed(1)
    h = torch.complex(torch.randn(8, 4, 64), torch.randn(8, 4, 64))
    model = CSIModel()
    h_hat = model(h, mode="imperfect", nmse_db=torch.tensor(nmse_db), seed=7)
    measured = (h_hat - h).abs().square().mean() / h.abs().square().mean()
    assert abs(10 * torch.log10(measured).item() - nmse_db) < 0.05


def test_mmse_converges_to_zf_at_zero_noise() -> None:
    h = torch.complex(torch.randn(2, 3, 64), torch.randn(2, 3, 64))
    x = torch.complex(torch.randn(2, 3, 64), torch.randn(2, 3, 64))
    y = h * x
    equalized = MMSEEqualizer()(y, h, torch.zeros(2))
    assert torch.max(torch.abs(equalized - x)).item() < 1e-5


def test_pilot_ls_recovers_flat_channel() -> None:
    modem = OFDMModem()
    data = torch.ones(2, 3, 48, dtype=torch.complex64)
    transmitted = modem.assemble(data)
    flat_h = torch.tensor(2.0 + 1.0j, dtype=torch.complex64)
    received = transmitted * flat_h
    estimate = CSIModel().estimate_from_pilots(
        received, modem.pilot_values, modem.plan
    )
    data_indices = modem.plan.tensor_indices("data", torch.device("cpu"))
    assert torch.max(torch.abs(estimate.index_select(-1, data_indices) - flat_h)).item() < 1e-6
