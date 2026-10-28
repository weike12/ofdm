from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .subcarrier_index import SubcarrierPlan


class OFDMModem(nn.Module):
    """Batched unitary OFDM modem with explicit pilot/data carrier placement."""

    def __init__(
        self,
        plan: Optional[SubcarrierPlan] = None,
        n_cp: int = 16,
        pilot_values: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.plan = plan or SubcarrierPlan()
        if n_cp < 0 or n_cp >= self.plan.n_fft:
            raise ValueError("n_cp must be in [0, n_fft)")
        self.n_cp = int(n_cp)
        if pilot_values is None:
            pilot_values = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.complex64)
        if pilot_values.numel() != self.plan.n_pilot:
            raise ValueError("pilot_values length must match the pilot carrier count")
        self.register_buffer("pilot_values", pilot_values.to(torch.complex64))

    def assemble(self, data_grid: torch.Tensor) -> torch.Tensor:
        if data_grid.ndim != 3 or data_grid.shape[-1] != self.plan.n_data:
            raise ValueError(f"data_grid must have shape [B,T,{self.plan.n_data}]")
        if not data_grid.is_complex():
            raise TypeError("data_grid must be complex")
        batch, n_symbols, _ = data_grid.shape
        grid = torch.zeros(
            batch,
            n_symbols,
            self.plan.n_fft,
            dtype=data_grid.dtype,
            device=data_grid.device,
        )
        data_indices = self.plan.tensor_indices("data", data_grid.device)
        pilot_indices = self.plan.tensor_indices("pilot", data_grid.device)
        grid.index_copy_(2, data_indices, data_grid)
        pilots = self.pilot_values.to(device=data_grid.device, dtype=data_grid.dtype)
        grid[:, :, pilot_indices] = pilots.view(1, 1, -1)
        return grid

    def split(self, frequency_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frequency_grid.ndim != 3 or frequency_grid.shape[-1] != self.plan.n_fft:
            raise ValueError(f"frequency_grid must have shape [B,T,{self.plan.n_fft}]")
        data_indices = self.plan.tensor_indices("data", frequency_grid.device)
        pilot_indices = self.plan.tensor_indices("pilot", frequency_grid.device)
        return (
            frequency_grid.index_select(2, data_indices),
            frequency_grid.index_select(2, pilot_indices),
        )

    def modulate(self, frequency_grid: torch.Tensor) -> torch.Tensor:
        if frequency_grid.shape[-1] != self.plan.n_fft:
            raise ValueError("The last frequency_grid dimension must equal n_fft")
        time = torch.fft.ifft(frequency_grid, n=self.plan.n_fft, dim=-1, norm="ortho")
        if self.n_cp:
            time = torch.cat((time[..., -self.n_cp :], time), dim=-1)
        return time

    def demodulate(self, waveform: torch.Tensor) -> torch.Tensor:
        expected = self.plan.n_fft + self.n_cp
        if waveform.ndim != 3 or waveform.shape[-1] < expected:
            raise ValueError(f"waveform must have at least {expected} time samples per symbol")
        useful = waveform[..., self.n_cp : self.n_cp + self.plan.n_fft]
        return torch.fft.fft(useful, n=self.plan.n_fft, dim=-1, norm="ortho")

    def forward(self, data_grid: torch.Tensor) -> torch.Tensor:
        return self.modulate(self.assemble(data_grid))

