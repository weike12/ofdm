import pytest
import torch

from si_fdjscc.comm import ChannelState, OFDMFrequencySelectiveChannel, OFDMModem


def random_grid(batch: int = 2, n_symbols: int = 3) -> torch.Tensor:
    return torch.complex(
        torch.randn(batch, n_symbols, 48), torch.randn(batch, n_symbols, 48)
    ).to(torch.complex64)


def test_modem_round_trip() -> None:
    modem = OFDMModem(n_cp=16)
    data = random_grid()
    frequency = modem.assemble(data)
    restored_frequency = modem.demodulate(modem.modulate(frequency))
    restored_data, _ = modem.split(restored_frequency)
    assert torch.max(torch.abs(restored_data - data)).item() < 1e-5


def test_ofdm_channel_identity_path() -> None:
    data = random_grid()
    channel = OFDMFrequencySelectiveChannel()
    batch, n_symbols, _ = data.shape
    h_time = torch.ones(batch, 1, dtype=torch.complex64)
    h_freq = torch.ones(batch, n_symbols, 64, dtype=torch.complex64)
    state = ChannelState(h_time=h_time, h_freq=h_freq, h_hat=h_freq, csi_mode="perfect")
    restored, _ = channel(
        data,
        snr_db=torch.full((batch,), float("inf")),
        channel_state=state,
        channel_seed=9,
    )
    assert torch.max(torch.abs(restored - data)).item() < 1e-5


@pytest.mark.parametrize(
    ("n_taps", "n_cp"),
    [(2, 8), (4, 8), (8, 8), (12, 16)],
)
def test_cp_sufficient_multipath_noiseless(n_taps: int, n_cp: int) -> None:
    torch.manual_seed(11)
    data = random_grid(batch=1, n_symbols=2)
    channel = OFDMFrequencySelectiveChannel(n_taps=n_taps, n_cp=n_cp)
    h_time = channel.channel.sample_impulse_response(
        1, torch.device("cpu"), torch.complex64, seed=77
    )
    h = channel.channel.frequency_response(h_time, 64)[:, None, :].expand(1, 2, 64)
    state = ChannelState(h_time=h_time, h_freq=h, h_hat=h, csi_mode="perfect")
    restored, _ = channel(
        data,
        snr_db=torch.tensor([float("inf")]),
        channel_state=state,
        channel_seed=10,
    )
    assert torch.max(torch.abs(restored - data)).item() < 2e-5


def test_cp_rejects_uncovered_twelve_tap_channel() -> None:
    with pytest.raises(ValueError, match="n_cp"):
        OFDMFrequencySelectiveChannel(n_taps=12, n_cp=8)
