from __future__ import annotations

from pathlib import Path

import torch

from ..config import deep_get, load_config
from ..data import build_dataloader
from ..losses import ReconstructionLoss
from ..utils import save_checkpoint, set_seed
from .common import (
    build_system,
    create_amp,
    finish_step,
    load_model_checkpoint,
    optimizer_from_config,
    parse_train_args,
    resolve_device,
    sample_snr,
    setup_experiment,
    unpack_images,
)


def train(config, resume=None, device_override=None) -> Path:
    set_seed(int(deep_get(config, "experiment.seed", 42)))
    device = resolve_device(config, device_override)
    model = build_system(config).to(device)
    parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = optimizer_from_config(parameters, dict(config.get("optimizer", {})))
    initial = resume or deep_get(config, "training.init_checkpoint")
    step = load_model_checkpoint(model, initial, optimizer if resume else None, strict=True)
    loader = build_dataloader(dict(deep_get(config, "data.train", {})), training=True)
    loss_config = dict(config.get("loss", {}))
    criterion = ReconstructionLoss(
        lambda_mse=float(loss_config.get("lambda_mse", 1.0)),
        lambda_msssim=float(loss_config.get("lambda_msssim", 0.0)),
    )
    output_dir, logger = setup_experiment(config)
    training = dict(config.get("training", {}))
    max_steps = int(training.get("max_steps", len(loader) * int(training.get("epochs", 1))))
    save_every = int(training.get("save_every", 1000))
    log_every = int(training.get("log_every", 50))
    use_amp, scaler = create_amp(bool(deep_get(config, "experiment.amp", True)))
    channel_train = dict(config.get("channel_train", {}))
    grad_clip = float(deep_get(config, "optimizer.grad_clip", 1.0))
    model.train()

    while step < max_steps:
        for batch in loader:
            images, _ = unpack_images(batch, device)
            snr_db = sample_snr(images.shape[0], channel_train, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                reconstruction = model(
                    images,
                    snr_db,
                    channel_cfg={"csi_mode": "perfect"},
                    mapper_mode="sequential",
                )
                loss = criterion(reconstruction, images)
            finish_step(loss, optimizer, scaler, parameters, grad_clip)
            step += 1
            if step % log_every == 0 or step == 1:
                logger.log({"step": step, "loss": loss, "snr_db": snr_db.mean()})
            if step % save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_{step:08d}.pt",
                    model,
                    optimizer,
                    step,
                    config,
                    stage="A",
                )
            if step >= max_steps:
                break
    final_path = output_dir / "stage_a_final.pt"
    save_checkpoint(final_path, model, optimizer, step, config, stage="A")
    return final_path


def main() -> None:
    args = parse_train_args("Stage A: SwinJSCC-OFDM sequential baseline")
    path = train(load_config(args.config), resume=args.resume, device_override=args.device)
    print(path)


if __name__ == "__main__":
    main()
