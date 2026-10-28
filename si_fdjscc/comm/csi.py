from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .multipath_channel import _make_generator, complex_normal
from .subcarrier_index import SubcarrierPlan


class CSIModel(nn.Module):
    """Perfect, controlled-NMSE, and pilot-LS channel-state models."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def inject_nmse(
        self,
        h: torch.Tensor,
        nmse_db: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch = h.shape[0]
        nmse = torch.as_tensor(nmse_db, device=h.device, dtype=h.real.dtype)
        if nmse.ndim == 0:
            nmse = nmse.repeat(batch)
        nmse = torch.pow(10.0, nmse.reshape(batch) / 10.0)
        generator = _make_generator(h.device, seed)
        error = complex_normal(tuple(h.shape), h.device, h.dtype, generator)
        reduce_dims = tuple(range(1, h.ndim))
        channel_power = h.abs().square().mean(dim=reduce_dims, keepdim=True)
        error_power = error.abs().square().mean(dim=reduce_dims, keepdim=True)
        target_ratio = nmse.view(batch, *([1] * (h.ndim - 1)))
        scale = torch.sqrt(target_ratio * channel_power / (error_power + self.eps))
        return h + error * scale

    def estimate_from_pilots(
        self,
        received_grid: torch.Tensor,
        pilot_values: torch.Tensor,
        plan: SubcarrierPlan,
    ) -> torch.Tensor:
        pilot_indices = plan.tensor_indices("pilot", received_grid.device)
        received_pilots = received_grid.index_select(-1, pilot_indices)
        pilots = pilot_values.to(device=received_grid.device, dtype=received_grid.dtype)
        h_pilot = received_pilots / pilots.view(1, 1, -1)

        pilot_coordinates = torch.tensor(
            plan.pilot_shifted, device=received_grid.device, dtype=received_grid.real.dtype
        )
        query = torch.arange(
            -(plan.n_fft // 2), plan.n_fft // 2, device=received_grid.device
        ).to(received_grid.real.dtype)
        right = torch.searchsorted(pilot_coordinates, query).clamp(1, plan.n_pilot - 1)
        left = right - 1
        x0 = pilot_coordinates[left]
        x1 = pilot_coordinates[right]
        weight = ((query - x0) / (x1 - x0 + self.eps)).clamp(0.0, 1.0)
        h_shifted = h_pilot[..., left] * (1.0 - weight) + h_pilot[..., right] * weight

        fft_indices = torch.remainder(query.to(torch.long), plan.n_fft)
        h_hat = torch.empty_like(received_grid)
        h_hat.index_copy_(-1, fft_indices, h_shifted)
        return h_hat

    def forward(
        self,
        h: torch.Tensor,
        mode: str = "perfect",
        nmse_db: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        mode = mode.lower()
        if mode == "perfect":
            return h
        if mode in {"imperfect", "controlled_nmse", "nmse"}:
            if nmse_db is None:
                raise ValueError("nmse_db is required for controlled-NMSE CSI")
            return self.inject_nmse(h, nmse_db, seed=seed)
        if mode == "pilot_ls":
            raise ValueError("pilot_ls requires received pilots; call estimate_from_pilots")
        raise ValueError(f"Unsupported CSI mode: {mode}")
