from __future__ import annotations

import argparse
import json

import torch

from ..config import deep_get, load_config
from ..data import build_dataloader
from ..losses import (
    ReconstructionLoss,
    distortion_sensitivity_target,
    importance_target_statistics,
    per_sample_psnr,
)
from ..train.common import (
    build_system,
    load_model_checkpoint,
    resolve_device,
    unpack_images,
)
from ..utils import set_seed


def diagnose(config, checkpoint=None, device_override=None, batches: int = 1) -> dict:
    """Measure DSI scale and target concentration without updating the model."""
    if batches < 1:
        raise ValueError("batches must be at least one")
    set_seed(int(deep_get(config, "experiment.seed", 42)), deterministic=True)
    device = resolve_device(config, device_override)
    model = build_system(config).to(device).eval()
    if checkpoint:
        load_model_checkpoint(model, checkpoint, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data_config = dict(deep_get(config, "data.train", {}))
    data_config["batch_size"] = 1
    data_config["num_workers"] = min(int(data_config.get("num_workers", 2)), 2)
    loader = build_dataloader(data_config, training=False)
    loss_config = dict(config.get("loss", {}))
    criterion = ReconstructionLoss(
        lambda_mse=float(loss_config.get("lambda_mse", 1.0)),
        lambda_msssim=float(loss_config.get("lambda_msssim", 0.0)),
    )
    temperature = float(loss_config.get("target_temperature", 0.5))

    records = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        images, identifiers = unpack_images(batch, device)
        with torch.no_grad():
            frozen_tokens, metadata = model.backbone.encode(images)
        tokens = frozen_tokens.detach().float().requires_grad_(True)
        clean_reconstruction = model.backbone.decode(tokens, metadata)
        clean_loss = criterion(clean_reconstruction, images, reduction="none")
        target, sensitivity = distortion_sensitivity_target(
            tokens,
            clean_loss,
            temperature=temperature,
            retain_graph=False,
        )
        statistics = importance_target_statistics(target, sensitivity)
        record = {
            "batch": batch_index,
            "identifier": list(identifiers) if identifiers is not None else None,
            "clean_loss": float(clean_loss.mean().detach().cpu()),
            "clean_psnr_db": float(
                per_sample_psnr(clean_reconstruction.detach(), images).mean().cpu()
            ),
            **{
                key: float(value.detach().cpu())
                for key, value in statistics.items()
            },
        }
        records.append(record)

    if not records:
        raise RuntimeError("The configured training dataset yielded no batches")
    summary = {
        "batches": len(records),
        "target_temperature": temperature,
        "mean_target_peak_ratio": sum(
            record["target_peak_ratio"] for record in records
        ) / len(records),
        "mean_target_entropy_normalized": sum(
            record["target_entropy_normalized"] for record in records
        ) / len(records),
        "mean_target_l1_from_uniform": sum(
            record["target_l1_from_uniform"] for record in records
        ) / len(records),
        "records": records,
    }
    collapsed = (
        summary["mean_target_peak_ratio"] <= 1.001
        or summary["mean_target_l1_from_uniform"] <= 1.0e-6
    )
    summary["status"] = "COLLAPSED" if collapsed else "NON_UNIFORM"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the scale-invariant DSI target is non-uniform"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Optional internal SDS-UFD-JSCC checkpoint loaded after the backbone checkpoint",
    )
    parser.add_argument("--device")
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    result = diagnose(
        load_config(args.config),
        checkpoint=args.checkpoint,
        device_override=args.device,
        batches=args.batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "COLLAPSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
