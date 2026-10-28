from __future__ import annotations

from pathlib import Path

import torch

from ..config import deep_get, load_config
from ..data import build_dataloader
from ..losses import (
    ImportanceSupervisionLoss,
    ReconstructionLoss,
    importance_target_statistics,
    sensitivity_distribution,
)
from ..utils import save_checkpoint, set_seed
from .common import (
    build_system,
    create_amp,
    finish_step,
    load_model_checkpoint,
    optimizer_from_config,
    parse_train_args,
    resolve_device,
    sample_csi_config,
    sample_snr,
    setup_experiment,
    unpack_images,
)
from .sensitivity import channel_conditioned_ablation_target


def train(config, resume=None, device_override=None) -> Path:
    set_seed(int(deep_get(config, "experiment.seed", 42)))
    device = resolve_device(config, device_override)
    model = build_system(config).to(device)
    initial = resume or deep_get(config, "training.init_checkpoint")
    if not initial:
        raise ValueError("Stage B requires training.init_checkpoint or --resume")
    load_model_checkpoint(model, initial, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.importance_head.parameters():
        parameter.requires_grad_(True)
    parameters = list(model.importance_head.parameters())
    optimizer = optimizer_from_config(parameters, dict(config.get("optimizer", {})))
    step = load_model_checkpoint(model, resume, optimizer, strict=True) if resume else 0

    loader = build_dataloader(dict(deep_get(config, "data.train", {})), training=True)
    loss_config = dict(config.get("loss", {}))
    reconstruction_loss = ReconstructionLoss(
        lambda_mse=float(loss_config.get("lambda_mse", 1.0)),
        lambda_msssim=float(loss_config.get("lambda_msssim", 0.0)),
    )
    importance_loss = ImportanceSupervisionLoss(
        rank_weight=float(loss_config.get("lambda_rank", 0.1)),
        n_pairs=int(loss_config.get("rank_pairs", 1024)),
        prediction_temperature=float(
            deep_get(config, "model.importance_head.temperature", 0.5)
        ),
        objective=str(loss_config.get("importance_objective", "log_smooth_l1")),
        log_epsilon=float(loss_config.get("sensitivity_epsilon", 1.0e-8)),
    )
    output_dir, logger = setup_experiment(config)
    training = dict(config.get("training", {}))
    max_steps = int(training.get("max_steps", len(loader) * int(training.get("epochs", 1))))
    save_every = int(training.get("save_every", 1000))
    log_every = int(training.get("log_every", 50))
    use_amp, scaler = create_amp(bool(deep_get(config, "experiment.amp", True)))
    grad_clip = float(deep_get(config, "optimizer.grad_clip", 1.0))
    target_temperature = float(loss_config.get("target_temperature", 0.5))
    channel_train = dict(config.get("channel_train", {}))
    label_settings = dict(config.get("sensitivity_label", {}))
    label_draws = int(
        label_settings.get(
            "channel_draws", channel_train.get("n_channel_draws", 4)
        )
    )
    tokens_per_batch = int(label_settings.get("tokens_per_batch", 12))
    attenuation = float(label_settings.get("attenuation", 0.0))
    mapper_mode = str(label_settings.get("mapper_mode", "sequential"))
    model.backbone.eval()
    model.importance_head.train()

    while step < max_steps:
        for batch in loader:
            images, _ = unpack_images(batch, device)
            snr_db = sample_snr(images.shape[0], channel_train, device)
            channel_cfg = sample_csi_config(channel_train)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, auxiliary = model(
                    images,
                    snr_db,
                    channel_cfg=channel_cfg,
                    mapper_mode=mapper_mode,
                    return_aux=True,
                    n_channel_draws=label_draws,
                    detach_importance_input=True,
                )
                sensitivity, target_mask, sampled_indices = (
                    channel_conditioned_ablation_target(
                        model,
                        images,
                        snr_db,
                        auxiliary,
                        reconstruction_loss,
                        tokens_per_batch=tokens_per_batch,
                        attenuation=attenuation,
                    )
                )
                target_distribution = sensitivity_distribution(
                    sensitivity,
                    temperature=target_temperature,
                    mask=target_mask,
                )
            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted, logits = model.importance_head(
                    auxiliary["z"].detach(),
                    channel_features=auxiliary["channel_context"].detach(),
                    return_logits=True,
                )
                loss, components = importance_loss(
                    predicted,
                    logits,
                    sensitivity,
                    target_mask=target_mask,
                )
                target_statistics = importance_target_statistics(
                    target_distribution,
                    sensitivity,
                    mask=target_mask,
                )
            finish_step(loss, optimizer, scaler, parameters, grad_clip)
            step += 1
            if step % log_every == 0 or step == 1:
                logger.log(
                    {
                        "step": step,
                        "loss": loss,
                        "label_method": "channel_conditioned_ablation",
                        "label_channel_draws": label_draws,
                        "label_tokens_per_sample": sampled_indices.shape[1],
                        "snr_db": snr_db.mean(),
                        "csi_mode": channel_cfg["csi_mode"],
                        "csi_nmse_db": channel_cfg.get("csi_nmse_db"),
                        "spectral_notch": auxiliary["draws"][0].get(
                            "spectral_notch"
                        ),
                        **components,
                        **target_statistics,
                    }
                )
            if step % save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_{step:08d}.pt",
                    model,
                    optimizer,
                    step,
                    config,
                    stage="B",
                )
            if step >= max_steps:
                break
    final_path = output_dir / "stage_b_final.pt"
    save_checkpoint(final_path, model, optimizer, step, config, stage="B")
    return final_path


def main() -> None:
    args = parse_train_args("Stage B: channel-conditioned SDS supervision")
    path = train(load_config(args.config), resume=args.resume, device_override=args.device)
    print(path)


if __name__ == "__main__":
    main()
