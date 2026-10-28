from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


def _real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def complex_normal(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    real_dtype = _real_dtype(dtype)
    real = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    return torch.complex(real, imag).to(dtype) / math.sqrt(2.0)


class MultipathChannel(nn.Module):
    """Quasi-static Rayleigh multipath channel and complex AWGN.

    Each OFDM symbol is convolved independently with the same per-image impulse
    response. The discarded convolution tail models a correctly framed receiver;
    when ``n_cp >= n_taps - 1`` this is equivalent to circular convolution over
    the useful FFT interval.
    """

    def __init__(self, n_taps: int = 8, tau: float = 2.0) -> None:
        super().__init__()
        if n_taps < 1:
            raise ValueError("n_taps must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        powers = torch.exp(-torch.arange(n_taps, dtype=torch.float64) / float(tau))
        powers = powers / powers.sum()
        self.n_taps = int(n_taps)
        self.tau = float(tau)
        self.register_buffer("pdp", powers.to(torch.float32))

    def sample_impulse_response(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex64,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        generator = _make_generator(device, seed)
        samples = complex_normal(
            (batch_size, self.n_taps), device=device, dtype=dtype, generator=generator
        )
        return samples * self.pdp.to(device=device, dtype=samples.real.dtype).sqrt()

    @staticmethod
    def frequency_response(h: torch.Tensor, n_fft: int) -> torch.Tensor:
        # With unitary OFDM modulation, the channel multiplier is the unnormalised DFT of h.
        return torch.fft.fft(h, n=n_fft, dim=-1)

    def frequency_correlation_matrix(
        self,
        n_fft: int,
        carrier_indices: Optional[Sequence[int] | torch.Tensor] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return normalized ``|R_H(k_i-k_j)|`` under this channel's PDP.

        For tap powers ``P_l``, the complex frequency correlation is

        ``R_H(delta_k) = sum_l P_l exp(-j 2 pi delta_k l / n_fft)``.

        The returned real matrix is normalized by its diagonal power, so its
        diagonal is one and all entries are in ``[0, 1]`` (up to roundoff).
        ``carrier_indices`` may use shifted or unshifted carrier numbers because
        the expression is periodic in ``n_fft``.
        """
        if isinstance(n_fft, bool) or int(n_fft) != n_fft or n_fft < 1:
            raise ValueError("n_fft must be a positive integer")
        n_fft = int(n_fft)
        target_device = self.pdp.device if device is None else torch.device(device)
        if dtype is None:
            real_dtype = self.pdp.dtype
        elif dtype in {torch.complex64, torch.float32, torch.float16, torch.bfloat16}:
            real_dtype = torch.float32
        elif dtype in {torch.complex128, torch.float64}:
            real_dtype = torch.float64
        else:
            raise TypeError("dtype must be a real or complex floating-point dtype")
        if carrier_indices is None:
            carriers = torch.arange(n_fft, device=target_device, dtype=real_dtype)
        else:
            carriers = torch.as_tensor(carrier_indices, device=target_device)
            if carriers.ndim != 1:
                raise ValueError("carrier_indices must be one-dimensional")
            carriers = carriers.to(real_dtype)

        delta_k = carriers[:, None] - carriers[None, :]
        taps = torch.arange(self.n_taps, device=target_device, dtype=real_dtype)
        phase = (-2.0 * math.pi / n_fft) * delta_k[..., None] * taps
        pdp = self.pdp.to(device=target_device, dtype=real_dtype)
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        phasor = torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(complex_dtype)
        correlation = (phasor * pdp.to(complex_dtype)).sum(dim=-1)
        magnitude = correlation.abs().to(real_dtype)
        diagonal = torch.diagonal(magnitude).clamp_min(torch.finfo(real_dtype).eps)
        normalization = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        return (magnitude / normalization).clamp_(0.0, 1.0)

    def data_subcarrier_correlation(
        self,
        plan: "SubcarrierPlan",
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the normalized absolute correlation of a plan's data carriers."""
        from .subcarrier_index import SubcarrierPlan

        if not isinstance(plan, SubcarrierPlan):
            raise TypeError("plan must be a SubcarrierPlan")
        return self.frequency_correlation_matrix(
            plan.n_fft,
            plan.data_shifted,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _linear_convolution(waveform: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch, n_symbols, n_samples = waveform.shape
        if h.ndim == 2:
            h = h[:, None, :].expand(batch, n_symbols, h.shape[-1])
        if h.shape[:2] != (batch, n_symbols):
            raise ValueError("h must have shape [B,L] or [B,T,L]")
        full_length = n_samples + h.shape[-1] - 1
        fft_length = 1 << (full_length - 1).bit_length()
        waveform_f = torch.fft.fft(waveform, n=fft_length, dim=-1)
        channel_f = torch.fft.fft(h, n=fft_length, dim=-1)
        convolved = torch.fft.ifft(waveform_f * channel_f, n=fft_length, dim=-1)
        return convolved[..., :n_samples]

    def forward(
        self,
        waveform: torch.Tensor,
        snr_db: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        channel_seed: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform.ndim != 3 or not waveform.is_complex():
            raise ValueError("waveform must be a complex tensor with shape [B,T,N]")
        batch = waveform.shape[0]
        if h is None:
            h = self.sample_impulse_response(
                batch, waveform.device, waveform.dtype, seed=channel_seed
            )
        received = self._linear_convolution(waveform, h)

        snr = torch.as_tensor(snr_db, device=waveform.device, dtype=waveform.real.dtype)
        if snr.ndim == 0:
            snr = snr.repeat(batch)
        snr = snr.reshape(batch)
        noise_var = torch.pow(10.0, -snr / 10.0)
        generator = _make_generator(waveform.device, noise_seed)
        noise = complex_normal(
            tuple(received.shape),
            device=waveform.device,
            dtype=waveform.dtype,
            generator=generator,
        )
        received = received + noise * noise_var.sqrt().view(batch, 1, 1)
        return received, h, noise_var
