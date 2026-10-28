from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import deep_get, load_config
from ..data import build_dataloader
from ..train.common import build_system, load_model_checkpoint, resolve_device, unpack_images
from ..utils import save_config_snapshot, set_seed
from .importance_validation import (
    channel_conditioned_token_ablation_damage,
    clean_ablation_damage_at_indices,
    gradient_feature_scores,
    spearman_importance_damage,
)


def evaluate_importance(config, device_override=None) -> Path:
    evaluation = dict(config.get("importance_evaluation", {}))
    output_dir = Path(str(evaluation.get("output_dir", "runs/eval_importance")))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)
    set_seed(int(evaluation.get("seed", 3000)), deterministic=True)
    device = resolve_device(config, device_override)
    model = build_system(config).to(device).eval()
    checkpoint = evaluation.get("checkpoint") or deep_get(config, "training.init_checkpoint")
    if not checkpoint:
        raise ValueError("importance_evaluation.checkpoint is required")
    load_model_checkpoint(model, str(checkpoint), strict=True)
    loader = build_dataloader(dict(deep_get(config, "data.test", {})), training=False)
    max_batches = int(evaluation.get("max_batches", 100))
    max_tokens = evaluation.get("max_tokens")
    snr_value = float(evaluation.get("snr_db", 10.0))
    csi_value = evaluation.get("csi_nmse_db", -20.0)
    if csi_value is None or str(csi_value).lower() == "perfect":
        csi_mode, csi_nmse_db = "perfect", None
    else:
        csi_mode, csi_nmse_db = "imperfect", float(csi_value)
    channel_draws = int(evaluation.get("channel_draws", 4))
    mapper_mode = str(evaluation.get("mapper_mode", "proposed_ufd"))
    learned_correlations = []
    norm_correlations = []
    random_correlations = []
    gradient_feature_correlations = []
    nonchannel_ablation_correlations = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images, _ = unpack_images(batch, device)
        snr_db = torch.full(
            (images.shape[0],), snr_value, device=device
        )
        result = channel_conditioned_token_ablation_damage(
            model,
            images,
            snr_db,
            channel_cfg={
                "csi_mode": csi_mode,
                "csi_nmse_db": csi_nmse_db,
                "channel_seed": int(evaluation.get("seed", 3000))
                + batch_index * 100_000,
            },
            mapper_mode=mapper_mode,
            n_channel_draws=channel_draws,
            max_tokens=max_tokens,
        )
        gradient_scores = gradient_feature_scores(model, images)
        selected_gradient_scores = torch.gather(
            gradient_scores, 1, result["token_indices"]
        )
        nonchannel_damage = clean_ablation_damage_at_indices(
            model, images, result["token_indices"]
        )
        learned_correlations.append(
            spearman_importance_damage(result["importance"], result["damage"])
        )
        norm_correlations.append(
            spearman_importance_damage(result["feature_norm"], result["damage"])
        )
        random_correlations.append(
            spearman_importance_damage(
                torch.rand_like(result["importance"]), result["damage"]
            )
        )
        gradient_feature_correlations.append(
            spearman_importance_damage(
                selected_gradient_scores, result["damage"]
            )
        )
        nonchannel_ablation_correlations.append(
            spearman_importance_damage(
                nonchannel_damage, result["damage"]
            )
        )
    summary = {
        "batches": len(learned_correlations),
        "snr_db": snr_value,
        "csi_nmse_db": csi_nmse_db,
        "channel_draws": channel_draws,
        "mapper_mode": mapper_mode,
        "importance_damage_spearman_mean": float(np.nanmean(learned_correlations)),
        "feature_norm_damage_spearman_mean": float(np.nanmean(norm_correlations)),
        "random_damage_spearman_mean": float(np.nanmean(random_correlations)),
        "gradient_feature_spearman_mean": float(
            np.nanmean(gradient_feature_correlations)
        ),
        "nonchannel_ablation_spearman_mean": float(
            np.nanmean(nonchannel_ablation_correlations)
        ),
        "importance_damage_spearman_per_batch": learned_correlations,
        "feature_norm_damage_spearman_per_batch": norm_correlations,
        "random_damage_spearman_per_batch": random_correlations,
        "gradient_feature_spearman_per_batch": gradient_feature_correlations,
        "nonchannel_ablation_spearman_per_batch": (
            nonchannel_ablation_correlations
        ),
        "attention_baseline": (
            "unavailable: the configured official SwinJSCC adapter does not "
            "expose token-aligned attention maps"
        ),
    }
    with (output_dir / "importance_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate learned importance against token ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(evaluate_importance(load_config(args.config), args.device))


if __name__ == "__main__":
    main()
