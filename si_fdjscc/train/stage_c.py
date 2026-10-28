from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import deep_get, load_config
from ..data import build_dataloader
from ..losses import (
    ImportanceSupervisionLoss,
    ReconstructionLoss,
    cvar_loss,
    distortion_sensitivity_target,
    importance_target_statistics,
    importance_weighted_latent_loss,
    range_violation_loss,
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
from .stage_c_stability import AmpOverflowGuard, CurriculumSchedule, DivergenceGuard
from .sensitivity import channel_conditioned_ablation_target


TRAINING_REVISION = "sds_ufd_stage_c_v4"


def _stage_c_optimizer(model, config: Mapping[str, Any]):
    optimizer_config = dict(config.get("optimizer", {}))
    default_lr = float(optimizer_config.get("lr", 1.0e-6))
    backbone_lr = float(optimizer_config.get("backbone_lr", default_lr))
    importance_lr = float(optimizer_config.get("importance_lr", default_lr))
    other_lr = float(optimizer_config.get("other_lr", backbone_lr))

    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    importance_parameters = [
        parameter
        for parameter in model.importance_head.parameters()
        if parameter.requires_grad
    ]
    grouped_ids = {
        id(parameter) for parameter in backbone_parameters + importance_parameters
    }
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in grouped_ids
    ]
    groups = []
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": backbone_lr, "group_name": "backbone"}
        )
    if importance_parameters:
        groups.append(
            {
                "params": importance_parameters,
                "lr": importance_lr,
                "group_name": "importance_head",
            }
        )
    if other_parameters:
        groups.append({"params": other_parameters, "lr": other_lr, "group_name": "other"})
    parameters = backbone_parameters + importance_parameters + other_parameters
    if not parameters:
        raise RuntimeError("Stage C has no trainable parameters")
    return optimizer_from_config(groups, optimizer_config), parameters


def _optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        f"lr_{group.get('group_name', index)}": float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _save_divergence_stop(
    output_dir: Path,
    logger,
    model,
    optimizer,
    scaler,
    step: int,
    config: Mapping[str, Any],
    reason: str,
    diagnostics: Mapping[str, Any],
) -> None:
    path = output_dir / f"diverged_step_{step:08d}.pt"
    logger.log(
        {
            "step": step,
            "event": "automatic_divergence_stop",
            "reason": reason,
            **diagnostics,
        }
    )
    save_checkpoint(
        path,
        model,
        optimizer,
        step,
        config,
        stage="C_v2_DIVERGED",
        training_revision=TRAINING_REVISION,
        divergence_reason=reason,
        amp_scaler=scaler.state_dict(),
    )
    raise RuntimeError(
        f"Stage C stopped automatically at step {step}: {reason}. "
        f"Diagnostic checkpoint: {path}"
    )


def _validate_v2_resume(path: str) -> bool:
    payload = torch.load(str(Path(path).expanduser()), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("stage") != "C_v2":
        stage = payload.get("stage") if isinstance(payload, Mapping) else None
        raise ValueError(
            "Stage C v2 may resume only from a healthy C_v2 checkpoint; "
            f"got stage={stage!r}. Restart from the Stage B checkpoint instead."
        )
    return payload.get("training_revision") == TRAINING_REVISION


def train(config, resume=None, device_override=None) -> Path:
    version = str(deep_get(config, "training.version", "legacy"))
    if version != "v2":
        raise ValueError(
            "The legacy Stage C configuration is disabled because it can diverge. "
            "Use configs/div2k_stageC_robust_v2.yaml."
        )
    resume_state_is_compatible = _validate_v2_resume(resume) if resume else False
    set_seed(int(deep_get(config, "experiment.seed", 42)))
    device = resolve_device(config, device_override)
    model = build_system(config).to(device)
    initial = resume or deep_get(config, "training.init_checkpoint")
    if not initial:
        raise ValueError("Stage C requires training.init_checkpoint or --resume")
    if not resume:
        load_model_checkpoint(model, initial, strict=True)
    optimizer, parameters = _stage_c_optimizer(model, config)
    amp_settings = dict(deep_get(config, "experiment.amp_settings", {}))
    use_amp, scaler = create_amp(
        bool(deep_get(config, "experiment.amp", True)),
        init_scale=float(amp_settings.get("init_scale", 1.0)),
        growth_interval=int(amp_settings.get("growth_interval", 10000)),
    )
    step = (
        load_model_checkpoint(
            model,
            resume,
            optimizer if resume_state_is_compatible else None,
            scaler=scaler if resume_state_is_compatible else None,
            strict=True,
        )
        if resume
        else 0
    )
    loader = build_dataloader(dict(deep_get(config, "data.train", {})), training=True)

    loss_config = dict(config.get("loss", {}))
    reconstruction_loss = ReconstructionLoss(
        lambda_mse=float(loss_config.get("lambda_mse", 1.0)),
        lambda_msssim=float(loss_config.get("lambda_msssim", 0.0)),
        clamp_prediction=bool(loss_config.get("clamp_reconstruction", True)),
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
    lambda_imp = float(loss_config.get("lambda_imp", 0.10))
    lambda_latent = float(loss_config.get("lambda_lat", 0.05))
    lambda_tail = float(loss_config.get("lambda_tail", 0.20))
    lambda_anchor = float(loss_config.get("lambda_anchor", 1.0))
    lambda_range = float(loss_config.get("lambda_range", 0.05))
    cvar_alpha = float(loss_config.get("cvar_alpha", 0.8))
    minimum_cvar_draws = int(loss_config.get("minimum_cvar_draws", 4))
    target_temperature = float(loss_config.get("target_temperature", 0.5))
    sensitivity_settings = dict(config.get("sensitivity_label", {}))
    sensitivity_method = str(
        sensitivity_settings.get("method", "hybrid")
    ).lower()
    if sensitivity_method not in {"gradient", "ablation", "hybrid"}:
        raise ValueError(
            "sensitivity_label.method must be gradient, ablation, or hybrid"
        )
    ablation_every = int(sensitivity_settings.get("ablation_every", 4))
    if ablation_every < 1:
        raise ValueError("sensitivity_label.ablation_every must be positive")
    ablation_tokens = int(sensitivity_settings.get("tokens_per_batch", 8))
    ablation_attenuation = float(sensitivity_settings.get("attenuation", 0.0))
    gradient_auxiliary_scale = float(
        sensitivity_settings.get("gradient_auxiliary_scale", 0.10)
    )
    if not 0.0 <= gradient_auxiliary_scale <= 1.0:
        raise ValueError(
            "sensitivity_label.gradient_auxiliary_scale must be in [0,1]"
        )

    output_dir, logger = setup_experiment(config)
    if resume:
        logger.log(
            {
                "step": step,
                "event": "resume",
                "checkpoint": str(Path(resume).expanduser()),
                "optimizer_and_amp_state_restored": resume_state_is_compatible,
                "training_revision": TRAINING_REVISION,
            }
        )
    training = dict(config.get("training", {}))
    channel_train = dict(config.get("channel_train", {}))
    curriculum = CurriculumSchedule(channel_train, config.get("curriculum"))
    divergence_guard = DivergenceGuard(config.get("divergence_guard"))
    amp_overflow_guard = AmpOverflowGuard(config.get("amp_overflow_guard"))
    max_steps = int(training.get("max_steps", len(loader) * int(training.get("epochs", 1))))
    mapper_mode = str(deep_get(config, "mapper.mode", "proposed_ufd"))
    save_every = int(training.get("save_every", 500))
    log_every = int(training.get("log_every", 20))
    grad_clip = float(deep_get(config, "optimizer.grad_clip", 0.5))
    model.train()

    while step < max_steps:
        for batch in loader:
            next_step = step + 1
            phase = curriculum.at(next_step)
            images, _ = unpack_images(batch, device)
            snr_db = sample_snr(images.shape[0], phase.channel, device)
            channel_cfg = sample_csi_config(phase.channel)
            if (
                float(phase.loss_scales["tail"]) > 0.0
                and phase.n_channel_draws < minimum_cvar_draws
            ):
                raise ValueError(
                    "CVaR tail loss requires at least "
                    f"{minimum_cvar_draws} channel draws; phase {phase.name!r} "
                    f"has {phase.n_channel_draws}"
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, auxiliary = model(
                    images,
                    snr_db,
                    channel_cfg=channel_cfg,
                    mapper_mode=mapper_mode,
                    return_aux=True,
                    n_channel_draws=phase.n_channel_draws,
                    detach_importance_input=True,
                )

            use_ablation_target = sensitivity_method == "ablation" or (
                sensitivity_method == "hybrid" and next_step % ablation_every == 0
            )
            ablation_sensitivity = None
            ablation_mask = None
            sampled_indices = None
            if use_ablation_target:
                with torch.cuda.amp.autocast(enabled=False):
                    (
                        ablation_sensitivity,
                        ablation_mask,
                        sampled_indices,
                    ) = channel_conditioned_ablation_target(
                        model,
                        images,
                        snr_db,
                        auxiliary,
                        reconstruction_loss,
                        tokens_per_batch=ablation_tokens,
                        attenuation=ablation_attenuation,
                    )

            # Decoder outputs may be FP16, but MSE/MS-SSIM and the DSI teacher
            # are numerically sensitive near a high-quality reconstruction.
            # Keep the expensive codec forward under AMP and form every loss in
            # FP32; gradients still flow through the casts to the codec.
            with torch.cuda.amp.autocast(enabled=False):
                raw_reconstructions = auxiliary["reconstructions"].float()
                draw_count, batch_size = raw_reconstructions.shape[:2]
                float_images = images.float()
                repeated_target = float_images.unsqueeze(0).expand_as(
                    raw_reconstructions
                )
                draw_losses = reconstruction_loss(
                    raw_reconstructions.reshape(-1, *images.shape[1:]),
                    repeated_target.reshape(-1, *images.shape[1:]),
                    reduction="none",
                ).reshape(draw_count, batch_size)
                image_loss = draw_losses.mean()
                tail_loss = cvar_loss(draw_losses, alpha=cvar_alpha)

                channel_range_loss = range_violation_loss(
                    raw_reconstructions.reshape(-1, *images.shape[1:])
                ).mean()

                # One shared FP32 clean decode supplies both the reconstruction
                # anchor and the detached DSI teacher. This preserves the anchor
                # gradient to the backbone without an extra decoder invocation.
                dsi_tokens = auxiliary["z"].float()
                clean_reconstruction = model.backbone.decode(
                    dsi_tokens, auxiliary["backbone_metadata"]
                )
                clean_losses = reconstruction_loss(
                    clean_reconstruction, float_images, reduction="none"
                )
                anchor_loss = clean_losses.mean()
                if ablation_sensitivity is None:
                    importance_target, sensitivity = distortion_sensitivity_target(
                        dsi_tokens,
                        clean_losses,
                        temperature=target_temperature,
                        retain_graph=True,
                    )
                    target_mask = None
                    label_method = (
                        "gradient_baseline"
                        if sensitivity_method == "gradient"
                        else "gradient_auxiliary"
                    )
                else:
                    sensitivity = ablation_sensitivity.float()
                    target_mask = ablation_mask
                    importance_target = sensitivity_distribution(
                        sensitivity,
                        temperature=target_temperature,
                        mask=target_mask,
                    )
                    label_method = "channel_conditioned_ablation"
                clean_range_loss = range_violation_loss(clean_reconstruction).mean()
                reconstruction_range_loss = 0.5 * (
                    channel_range_loss + clean_range_loss
                )
                imp_loss, imp_components = importance_loss(
                    auxiliary["importance"],
                    auxiliary["importance_logits"],
                    sensitivity,
                    target_mask=target_mask,
                )
                target_statistics = importance_target_statistics(
                    importance_target,
                    sensitivity,
                    mask=target_mask,
                )
                received_tokens = torch.stack(
                    [draw["z_hat"] for draw in auxiliary["draws"]], dim=0
                ).float()
                latent_loss = importance_weighted_latent_loss(
                    auxiliary["z"].float(),
                    received_tokens,
                    auxiliary["importance"].float(),
                )
                effective_lambda_imp = (
                    lambda_imp * phase.loss_scales["importance"]
                    * (
                        1.0
                        if use_ablation_target or sensitivity_method == "gradient"
                        else gradient_auxiliary_scale
                    )
                )
                effective_lambda_latent = (
                    lambda_latent * phase.loss_scales["latent"]
                )
                effective_lambda_tail = lambda_tail * phase.loss_scales["tail"]
                total = (
                    image_loss
                    + lambda_anchor * anchor_loss
                    + lambda_range * reconstruction_range_loss
                    + effective_lambda_imp * imp_loss
                    + effective_lambda_latent * latent_loss
                    + effective_lambda_tail * tail_loss
                )

            raw_min = float(raw_reconstructions.detach().float().amin().cpu())
            raw_max = float(raw_reconstructions.detach().float().amax().cpu())
            raw_mean = float(raw_reconstructions.detach().float().mean().cpu())
            clean_raw_min = float(clean_reconstruction.detach().float().amin().cpu())
            clean_raw_max = float(clean_reconstruction.detach().float().amax().cpu())
            clean_raw_mean = float(clean_reconstruction.detach().float().mean().cpu())
            if not bool(torch.isfinite(total.detach()).item()):
                _save_divergence_stop(
                    output_dir,
                    logger,
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    "non-finite total loss before backward",
                    {
                        "image_loss": image_loss,
                        "raw_reconstruction_min": raw_min,
                        "raw_reconstruction_max": raw_max,
                    },
                )

            step_result = finish_step(
                total, optimizer, scaler, parameters, grad_clip
            )
            grad_norm = step_result.grad_norm
            step = next_step
            diagnostics = {
                "phase": phase.name,
                "n_channel_draws": phase.n_channel_draws,
                "sensitivity_label_method": label_method,
                "sensitivity_tokens_observed": (
                    sampled_indices.shape[1]
                    if sampled_indices is not None
                    else sensitivity.shape[1]
                ),
                "loss": total,
                "image_loss": image_loss,
                "tail_loss": tail_loss,
                "anchor_loss": anchor_loss,
                "range_loss": reconstruction_range_loss,
                "latent_loss": latent_loss,
                **imp_components,
                **target_statistics,
                "raw_reconstruction_min": raw_min,
                "raw_reconstruction_max": raw_max,
                "raw_reconstruction_mean": raw_mean,
                "clean_reconstruction_min": clean_raw_min,
                "clean_reconstruction_max": clean_raw_max,
                "clean_reconstruction_mean": clean_raw_mean,
                "grad_norm": grad_norm,
                "optimizer_step_skipped": step_result.optimizer_step_skipped,
                "amp_scale_before": step_result.amp_scale_before,
                "amp_scale_after": step_result.amp_scale_after,
                "effective_lambda_imp": effective_lambda_imp,
                "effective_lambda_latent": effective_lambda_latent,
                "effective_lambda_tail": effective_lambda_tail,
                "snr_db": snr_db.mean(),
                "csi_mode": channel_cfg["csi_mode"],
                "csi_nmse_db": channel_cfg.get("csi_nmse_db"),
                "spectral_notch": auxiliary["draws"][0].get("spectral_notch"),
                **_optimizer_lrs(optimizer),
            }
            divergence_reason = divergence_guard.update(
                step,
                float(image_loss.detach().cpu()),
                raw_min,
                raw_max,
                anchor_loss=float(anchor_loss.detach().cpu()),
                phase=phase.name,
            )
            overflow_reason = amp_overflow_guard.update(
                step_result.optimizer_step_skipped
            )
            reason = divergence_reason or overflow_reason
            diagnostics.update(divergence_guard.metrics())
            diagnostics.update(amp_overflow_guard.metrics())
            if (
                step % log_every == 0
                or step == 1
                or reason is not None
                or step_result.optimizer_step_skipped
            ):
                logger.log({"step": step, **diagnostics})
            if reason is not None:
                _save_divergence_stop(
                    output_dir,
                    logger,
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    reason,
                    diagnostics,
                )
            if step % save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_{step:08d}.pt",
                    model,
                    optimizer,
                    step,
                    config,
                    stage="C_v2",
                    training_revision=TRAINING_REVISION,
                    curriculum_phase=phase.name,
                    amp_scaler=scaler.state_dict(),
                )
            if step >= max_steps:
                break
    final_path = output_dir / "stage_c_final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        step,
        config,
        stage="C_v2",
        training_revision=TRAINING_REVISION,
        amp_scaler=scaler.state_dict(),
    )
    return final_path


def main() -> None:
    args = parse_train_args("Stage C v2: stabilized curriculum robust fine-tuning")
    path = train(load_config(args.config), resume=args.resume, device_override=args.device)
    print(path)


if __name__ == "__main__":
    main()
