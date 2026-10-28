from __future__ import annotations

import torch
from torch import nn


def per_sample_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (reconstruction - target).square().flatten(1).mean(dim=1)


def range_violation_loss(
    reconstruction: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Penalize decoder values outside ``[0, data_range]``.

    A clamped image loss matches the official SwinJSCC objective, but clamp has
    zero gradient once a value leaves the valid interval. This auxiliary loss
    supplies a gradient that pulls raw decoder outputs back into range.
    """
    values = reconstruction.float()
    below = torch.relu(-values).square()
    above = torch.relu(values - float(data_range)).square()
    return (below + above).flatten(1).mean(dim=1)


def per_sample_psnr(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    mse = per_sample_mse(reconstruction, target)
    return 10.0 * torch.log10((float(data_range) ** 2) / mse.clamp_min(eps))


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_msssim: float = 0.0,
        data_range: float = 1.0,
        clamp_prediction: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_msssim = float(lambda_msssim)
        self.data_range = float(data_range)
        self.clamp_prediction = bool(clamp_prediction)

    @staticmethod
    def _ms_ssim(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            from pytorch_msssim import ms_ssim
        except ImportError as error:
            raise RuntimeError(
                "lambda_msssim > 0 requires the optional pytorch-msssim package"
            ) from error
        return ms_ssim(reconstruction, target, data_range=1.0, size_average=False)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if self.clamp_prediction:
            reconstruction = reconstruction.clamp(0.0, self.data_range)
        losses = self.lambda_mse * per_sample_mse(reconstruction, target)
        if self.lambda_msssim:
            losses = losses + self.lambda_msssim * (
                1.0 - self._ms_ssim(reconstruction, target)
            )
        if reduction == "none":
            return losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        raise ValueError("reduction must be none, mean, or sum")
