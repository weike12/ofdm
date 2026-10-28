from __future__ import annotations

import torch
from torch import nn


def _expand_batch_parameter(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


class MMSEEqualizer(nn.Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        received: torch.Tensor,
        h_hat: torch.Tensor,
        noise_var: torch.Tensor,
        symbol_power: float = 1.0,
    ) -> torch.Tensor:
        if h_hat.ndim == received.ndim - 1:
            h_hat = h_hat.unsqueeze(-2)
        if received.shape != h_hat.shape:
            h_hat = torch.broadcast_to(h_hat, received.shape)
        noise = _expand_batch_parameter(noise_var.to(received.real.dtype), received)
        denominator = h_hat.abs().square() + noise / float(symbol_power) + self.eps
        return h_hat.conj() * received / denominator


class ZFEqualizer(nn.Module):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, received: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
        if h_hat.ndim == received.ndim - 1:
            h_hat = h_hat.unsqueeze(-2)
        if received.shape != h_hat.shape:
            h_hat = torch.broadcast_to(h_hat, received.shape)
        return received * h_hat.conj() / (h_hat.abs().square() + self.eps)
