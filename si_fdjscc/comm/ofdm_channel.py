from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .csi import CSIModel
from .equalizer import MMSEEqualizer, ZFEqualizer
from .multipath_channel import MultipathChannel
from .ofdm_modem import OFDMModem
from .subcarrier_index import SubcarrierPlan


RECEIVER_CSI_SOURCES = ("estimated", "temporal_mean", "oracle")


@dataclass
class ChannelState:
    h_time: torch.Tensor  # [B,L]
    h_freq: torch.Tensor  # [B,T,N_fft]
    h_hat: Optional[torch.Tensor]  # unavailable until reception for pilot_ls
    csi_mode: str
    base_h_freq: Optional[torch.Tensor] = None
    frequency_gain: Optional[torch.Tensor] = None
    spectral_notch: Optional[Dict[str, Any]] = None


class OFDMFrequencySelectiveChannel(nn.Module):
    """Differentiable OFDM, multipath, CSI, and equalization pipeline."""

    def __init__(
        self,
        n_fft: int = 64,
        n_cp: int = 16,
        n_taps: int = 8,
        tau: float = 2.0,
        equalizer: str = "mmse",
        plan: Optional[SubcarrierPlan] = None,
    ) -> None:
        super().__init__()
        self.plan = plan or SubcarrierPlan(n_fft=n_fft)
        if self.plan.n_fft != n_fft:
            raise ValueError("plan.n_fft must match n_fft")
        if n_cp < n_taps - 1:
            raise ValueError("n_cp must be at least n_taps-1 for the default CP-sufficient model")
        self.modem = OFDMModem(self.plan, n_cp=n_cp)
        self.channel = MultipathChannel(n_taps=n_taps, tau=tau)
        self.csi = CSIModel()
        equalizer = equalizer.lower()
        if equalizer == "mmse":
            self.equalizer = MMSEEqualizer()
        elif equalizer == "zf":
            self.equalizer = ZFEqualizer()
        else:
            raise ValueError("equalizer must be 'mmse' or 'zf'")
        self.equalizer_name = equalizer

    def data_subcarrier_correlation(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return normalized absolute PDP correlation for the data carriers.

        This is the correlation prior consumed by
        :meth:`SemanticFrequencyMapper.map`; it is deterministic for a fixed
        ``n_fft``, data-carrier plan, and exponential PDP.
        """
        return self.channel.data_subcarrier_correlation(
            self.plan,
            device=device,
            dtype=dtype,
        )

    def prepare_state(
        self,
        batch_size: int,
        n_ofdm_symbols: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex64,
        csi_mode: str = "perfect",
        csi_nmse_db: Optional[torch.Tensor] = None,
        channel_seed: Optional[int] = None,
        spectral_notch: Optional[Mapping[str, Any]] = None,
    ) -> ChannelState:
        h_time = self.channel.sample_impulse_response(
            batch_size, device=device, dtype=dtype, seed=channel_seed
        )
        h_single = self.channel.frequency_response(h_time, self.plan.n_fft)
        base_h_freq = h_single[:, None, :].expand(
            batch_size, n_ofdm_symbols, self.plan.n_fft
        )
        frequency_gain = None
        resolved_notch: Optional[Dict[str, Any]] = None
        h_freq = base_h_freq
        notch_count = 1
        if spectral_notch:
            raw_count = spectral_notch.get("count", 1)
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count not in (1, 2)
            ):
                raise ValueError("spectral_notch.count must be 1 or 2")
            notch_count = raw_count
        if spectral_notch and notch_count == 1:
            notch_generator: Optional[torch.Generator] = None
            if channel_seed is not None:
                notch_generator = torch.Generator(device="cpu")
                notch_generator.manual_seed(int(channel_seed) + 17_381)

            width_range = spectral_notch.get("width_range")
            if width_range is not None:
                if not isinstance(width_range, (list, tuple)) or len(width_range) != 2:
                    raise ValueError("spectral_notch.width_range must be [min, max]")
                width_low, width_high = (int(width_range[0]), int(width_range[1]))
                if (
                    width_low < 1
                    or width_high < width_low
                    or width_high > self.plan.n_data
                ):
                    raise ValueError(
                        "spectral_notch.width_range must contain ordered bounds "
                        "between 1 and n_data"
                    )
                width = int(
                    torch.randint(
                        width_low,
                        width_high + 1,
                        (1,),
                        generator=notch_generator,
                    ).item()
                )
            else:
                width = int(spectral_notch.get("width", 10))
            if width < 1 or width > self.plan.n_data:
                raise ValueError("spectral_notch.width must be between 1 and n_data")

            raw_start = spectral_notch.get("start")
            start_is_random = bool(spectral_notch.get("random_start", False))
            if isinstance(raw_start, str) and raw_start.lower() == "random":
                start_is_random = True
            if start_is_random:
                start = int(
                    torch.randint(
                        0,
                        self.plan.n_data - width + 1,
                        (1,),
                        generator=notch_generator,
                    ).item()
                )
            elif raw_start is None:
                start = (self.plan.n_data - width) // 2
            else:
                start = int(raw_start)
            if start < 0 or start + width > self.plan.n_data:
                raise ValueError("spectral_notch start/width is outside the data carriers")

            attenuation_range = spectral_notch.get("attenuation_db_range")
            if attenuation_range is not None:
                if not isinstance(attenuation_range, (list, tuple)) or len(attenuation_range) != 2:
                    raise ValueError(
                        "spectral_notch.attenuation_db_range must be [min, max]"
                    )
                attenuation_low, attenuation_high = (
                    float(attenuation_range[0]),
                    float(attenuation_range[1]),
                )
                if attenuation_high < attenuation_low:
                    raise ValueError(
                        "spectral_notch.attenuation_db_range bounds must be ordered"
                    )
                draw = float(torch.rand((), generator=notch_generator).item())
                attenuation_db = attenuation_low + draw * (
                    attenuation_high - attenuation_low
                )
            else:
                attenuation_db = float(spectral_notch.get("attenuation_db", 20.0))
            if not torch.isfinite(torch.tensor(attenuation_db)):
                raise ValueError("spectral_notch attenuation must be finite")
            amplitude = 10.0 ** (-attenuation_db / 20.0)
            frequency_gain = torch.ones(
                self.plan.n_fft, device=device, dtype=base_h_freq.real.dtype
            )
            data_indices = self.plan.tensor_indices("data", device)
            frequency_gain[data_indices[start : start + width]] = amplitude
            h_freq = base_h_freq * frequency_gain.view(1, 1, -1)
            resolved_notch = {
                "start": start,
                "width": width,
                "attenuation_db": attenuation_db,
                "amplitude": amplitude,
            }
        if spectral_notch and notch_count == 2:
            notch_generator = None
            if channel_seed is not None:
                notch_generator = torch.Generator(device="cpu")
                notch_generator.manual_seed(int(channel_seed) + 17_381)

            raw_gap = spectral_notch.get("min_gap", 1)
            if isinstance(raw_gap, bool) or not isinstance(raw_gap, int) or raw_gap < 1:
                raise ValueError("spectral_notch.min_gap must be a positive integer")
            min_gap = raw_gap

            width_range = spectral_notch.get("width_range")
            if width_range is not None:
                if not isinstance(width_range, (list, tuple)) or len(width_range) != 2:
                    raise ValueError("spectral_notch.width_range must be [min, max]")
                width_low, width_high = (int(width_range[0]), int(width_range[1]))
                if width_low < 1 or width_high < width_low or width_high > self.plan.n_data:
                    raise ValueError(
                        "spectral_notch.width_range must contain ordered bounds "
                        "between 1 and n_data"
                    )
                width_options = range(width_low, width_high + 1)
            else:
                width = int(spectral_notch.get("width", 10))
                if width < 1 or width > self.plan.n_data:
                    raise ValueError("spectral_notch.width must be between 1 and n_data")
                width_options = (width,)
            fixed_starts = spectral_notch.get("starts")
            fixed_positions = None
            if fixed_starts is not None:
                if not isinstance(fixed_starts, (list, tuple)) or len(fixed_starts) != 2:
                    raise ValueError("spectral_notch.starts must contain two data-carrier positions")
                if spectral_notch.get("start") is not None or spectral_notch.get("random_start", False):
                    raise ValueError("spectral_notch.starts cannot be combined with start/random_start")
                fixed_positions = (int(fixed_starts[0]), int(fixed_starts[1]))
            width_pairs = [
                (left, right)
                for left in width_options
                for right in width_options
                if left + right + min_gap <= self.plan.n_data
                and (
                    fixed_positions is None
                    or (
                        fixed_positions[0] >= 0
                        and fixed_positions[1] >= fixed_positions[0] + left + min_gap
                        and fixed_positions[1] + right <= self.plan.n_data
                    )
                )
            ]
            if not width_pairs:
                if fixed_positions is not None:
                    raise ValueError("Two spectral notches overlap or fall outside data carriers")
                raise ValueError("Two spectral notches and their gap do not fit the data carriers")
            width_index = int(
                torch.randint(len(width_pairs), (1,), generator=notch_generator).item()
            )
            left_width, right_width = width_pairs[width_index]

            if fixed_positions is not None:
                left_start, right_start = fixed_positions
            else:
                raw_start = spectral_notch.get("start")
                if (
                    raw_start is not None
                    and str(raw_start).lower() != "random"
                    and not spectral_notch.get("random_start", False)
                ):
                    raise ValueError(
                        "Two spectral notches require start: random or starts: [left, right]"
                    )
                valid_starts = [
                    (left, right)
                    for left in range(self.plan.n_data - left_width - right_width - min_gap + 1)
                    for right in range(left + left_width + min_gap, self.plan.n_data - right_width + 1)
                ]
                start_index = int(
                    torch.randint(len(valid_starts), (1,), generator=notch_generator).item()
                )
                left_start, right_start = valid_starts[start_index]

            attenuation_range = spectral_notch.get("attenuation_db_range")
            if attenuation_range is not None:
                if not isinstance(attenuation_range, (list, tuple)) or len(attenuation_range) != 2:
                    raise ValueError("spectral_notch.attenuation_db_range must be [min, max]")
                attenuation_low, attenuation_high = (
                    float(attenuation_range[0]), float(attenuation_range[1])
                )
                if attenuation_high < attenuation_low:
                    raise ValueError("spectral_notch.attenuation_db_range bounds must be ordered")
                attenuations = [
                    attenuation_low
                    + float(torch.rand((), generator=notch_generator).item())
                    * (attenuation_high - attenuation_low)
                    for _ in range(2)
                ]
            else:
                attenuations = [float(spectral_notch.get("attenuation_db", 20.0))] * 2
            if not all(torch.isfinite(torch.tensor(value)) for value in attenuations):
                raise ValueError("spectral_notch attenuation must be finite")

            frequency_gain = torch.ones(
                self.plan.n_fft, device=device, dtype=base_h_freq.real.dtype
            )
            data_indices = self.plan.tensor_indices("data", device)
            notches = []
            for start, width, attenuation_db in zip(
                (left_start, right_start), (left_width, right_width), attenuations
            ):
                amplitude = 10.0 ** (-attenuation_db / 20.0)
                frequency_gain[data_indices[start : start + width]] = amplitude
                notches.append(
                    {
                        "start": start,
                        "width": width,
                        "attenuation_db": attenuation_db,
                        "amplitude": amplitude,
                    }
                )
            h_freq = base_h_freq * frequency_gain.view(1, 1, -1)
            resolved_notch = {
                "count": 2,
                "notches": notches,
                "min_gap": min_gap,
                "total_width": left_width + right_width,
            }
        mode = csi_mode.lower()
        if mode == "pilot_ls":
            h_hat = None
        else:
            csi_seed = None if channel_seed is None else int(channel_seed) + 1
            h_hat = self.csi(h_freq, mode=mode, nmse_db=csi_nmse_db, seed=csi_seed)
        return ChannelState(
            h_time=h_time,
            h_freq=h_freq,
            h_hat=h_hat,
            csi_mode=mode,
            base_h_freq=base_h_freq,
            frequency_gain=frequency_gain,
            spectral_notch=resolved_notch,
        )

    def forward(
        self,
        data_grid: torch.Tensor,
        snr_db: torch.Tensor,
        csi_mode: str = "perfect",
        csi_nmse_db: Optional[torch.Tensor] = None,
        channel_state: Optional[ChannelState] = None,
        channel_seed: Optional[int] = None,
        return_waveform: bool = False,
        receiver_csi_source: str = "estimated",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        receiver_csi_source = str(receiver_csi_source).lower()
        if receiver_csi_source not in RECEIVER_CSI_SOURCES:
            raise ValueError(
                "receiver_csi_source must be one of "
                + ", ".join(RECEIVER_CSI_SOURCES)
            )
        if data_grid.ndim != 3 or data_grid.shape[-1] != self.plan.n_data:
            raise ValueError(f"data_grid must have shape [B,T,{self.plan.n_data}]")
        batch, n_ofdm_symbols, _ = data_grid.shape
        if channel_state is None:
            channel_state = self.prepare_state(
                batch,
                n_ofdm_symbols,
                data_grid.device,
                dtype=data_grid.dtype,
                csi_mode=csi_mode,
                csi_nmse_db=csi_nmse_db,
                channel_seed=channel_seed,
            )
        if channel_state.h_freq.shape != (batch, n_ofdm_symbols, self.plan.n_fft):
            raise ValueError("channel_state shape does not match data_grid")

        tx_frequency = self.modem.assemble(data_grid)
        tx_waveform = self.modem.modulate(tx_frequency)
        noise_seed = None if channel_seed is None else int(channel_seed) + 2
        rx_waveform, _, noise_var = self.channel(
            tx_waveform,
            snr_db=snr_db,
            h=channel_state.h_time,
            noise_seed=noise_seed,
        )
        rx_frequency = self.modem.demodulate(rx_waveform)
        if channel_state.frequency_gain is not None:
            base_h = channel_state.base_h_freq
            if base_h is None:
                raise RuntimeError("A notched ChannelState must retain base_h_freq")
            gain = channel_state.frequency_gain.view(1, 1, -1)
            # Alter only the signal transfer function; receiver AWGN remains unchanged.
            rx_frequency = rx_frequency + (gain - 1.0) * base_h * tx_frequency
        if channel_state.csi_mode == "pilot_ls":
            h_hat = self.csi.estimate_from_pilots(
                rx_frequency, self.modem.pilot_values, self.plan
            )
        else:
            if channel_state.h_hat is None:
                raise RuntimeError("Prepared channel state has no H_hat")
            h_hat = channel_state.h_hat

        # Change only the CSI used by the receiver. The raw estimate remains
        # available to the transmitter mapper and SDS context unchanged.
        equalizer_h_hat = h_hat
        if receiver_csi_source == "temporal_mean":
            # Valid for this channel's block-static H and per-symbol CSI noise.
            equalizer_h_hat = h_hat.mean(dim=1, keepdim=True)
        elif receiver_csi_source == "oracle":
            equalizer_h_hat = channel_state.h_freq

        if self.equalizer_name == "mmse":
            equalized_frequency = self.equalizer(
                rx_frequency, equalizer_h_hat, noise_var, symbol_power=1.0
            )
        else:
            equalized_frequency = self.equalizer(rx_frequency, equalizer_h_hat)
        equalized_data, _ = self.modem.split(equalized_frequency)
        auxiliary: Dict[str, Any] = {
            "H": channel_state.h_freq,
            "H_hat": h_hat,
            "equalizer_H_hat": equalizer_h_hat,
            "receiver_csi_source": receiver_csi_source,
            "h_time": channel_state.h_time,
            "noise_var": noise_var,
            "tx_frequency": tx_frequency,
            "rx_frequency": rx_frequency,
            "spectral_notch": channel_state.spectral_notch,
        }
        if return_waveform:
            auxiliary["tx_waveform"] = tx_waveform
            auxiliary["rx_waveform"] = rx_waveform
        return equalized_data, auxiliary
