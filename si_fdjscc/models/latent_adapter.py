from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ComplexMetadata:
    original_shape: Tuple[int, int, int]
    pad_length: int
    token_ids: torch.Tensor  # [K], token owning the first real component in each pair
    scale: torch.Tensor  # [B,1], retained for exact model-layer inversion


class ComplexAdapter(nn.Module):
    """Pair real latent scalars into power-normalized complex symbols.

    ``restore_scale=True`` makes the adapter exactly invertible by retaining one
    scale value per sample in metadata. This is convenient for controlled neural
    channel experiments. Set it to false when the decoder is trained directly on
    normalized codewords and no scale side information should be assumed.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        normalize_power: bool = True,
        restore_scale: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.normalize_power = bool(normalize_power)
        self.restore_scale = bool(restore_scale)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, ComplexMetadata]:
        if z.ndim != 3 or z.is_complex():
            raise ValueError("z must be a real tensor with shape [B,M,C]")
        batch, n_tokens, channels = z.shape
        flattened = z.reshape(batch, -1)
        pad_length = flattened.shape[1] % 2
        if pad_length:
            flattened = torch.nn.functional.pad(flattened, (0, 1))
        if z.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            real = flattened[:, 0::2].to(torch.float32)
            imag = flattened[:, 1::2].to(torch.float32)
            symbols = torch.complex(real, imag).to(torch.complex64)
        else:
            symbols = torch.complex(flattened[:, 0::2], flattened[:, 1::2]).to(
                torch.complex128
            )
        if self.normalize_power:
            scale = symbols.abs().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
            symbols = symbols / scale
        else:
            scale = torch.ones(
                batch, 1, device=symbols.device, dtype=symbols.real.dtype
            )

        scalar_indices = torch.arange(
            0, flattened.shape[1], 2, device=z.device, dtype=torch.long
        )
        token_ids = torch.clamp(scalar_indices // channels, max=n_tokens - 1)
        metadata = ComplexMetadata(
            original_shape=(batch, n_tokens, channels),
            pad_length=pad_length,
            token_ids=token_ids,
            scale=scale,
        )
        return symbols, metadata

    def inverse(self, symbols_hat: torch.Tensor, metadata: ComplexMetadata) -> torch.Tensor:
        if symbols_hat.ndim != 2 or not symbols_hat.is_complex():
            raise ValueError("symbols_hat must be complex with shape [B,K]")
        batch, n_tokens, channels = metadata.original_shape
        if symbols_hat.shape[0] != batch:
            raise ValueError("Batch size does not match ComplexMetadata")
        restored = symbols_hat
        if self.normalize_power and self.restore_scale:
            restored = restored * metadata.scale.to(
                device=symbols_hat.device, dtype=symbols_hat.real.dtype
            )
        flattened = torch.stack((restored.real, restored.imag), dim=-1).reshape(batch, -1)
        flattened = flattened[:, : n_tokens * channels]
        return flattened.reshape(batch, n_tokens, channels)
