from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as functional


def save_importance_heatmap(
    image: torch.Tensor,
    importance: torch.Tensor,
    latent_hw: Tuple[int, int],
    output_path: str,
    alpha: float = 0.45,
) -> None:
    import matplotlib.pyplot as plt

    if image.ndim == 4:
        image = image[0]
    if importance.ndim == 2:
        importance = importance[0]
    heatmap = importance.reshape(1, 1, *latent_hw)
    heatmap = functional.interpolate(
        heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axis.imshow(heatmap.detach().cpu().numpy(), cmap="magma", alpha=alpha)
    axis.axis("off")
    figure.tight_layout(pad=0)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(figure)

