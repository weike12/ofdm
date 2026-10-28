from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..comm.semantic_mapper import PermutationMeta
from ..comm.subcarrier_index import SubcarrierPlan


def plot_mapping_grid(
    metadata: PermutationMeta,
    token_ids: torch.Tensor,
    importance: torch.Tensor,
    output_path: str,
    sample: int = 0,
    plan: Optional[SubcarrierPlan] = None,
) -> None:
    import matplotlib.pyplot as plt

    plan = plan or SubcarrierPlan()
    source = metadata.symbol_index_grid[sample].detach().cpu()
    valid = source >= 0
    source_safe = source.clamp_min(0)
    token_for_slot = token_ids.detach().cpu().index_select(0, source_safe.reshape(-1)).reshape_as(source)
    score = importance[sample].detach().cpu().index_select(0, token_for_slot.reshape(-1)).reshape_as(source)
    score = score.numpy()
    score[~valid.numpy()] = np.nan
    figure, axis = plt.subplots(figsize=(12, max(3, source.shape[0] * 0.5)))
    image = axis.imshow(score, aspect="auto", cmap="viridis")
    for positions in plan.band_data_positions[:-1]:
        axis.axvline(positions[-1] + 0.5, color="white", linewidth=1.2)
    axis.set_xlabel("Frequency-ordered data-carrier position")
    axis.set_ylabel("OFDM symbol")
    axis.set_title(f"Mapper={metadata.mode}: semantic importance per occupied slot")
    figure.colorbar(image, ax=axis, label="importance")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
