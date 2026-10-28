from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from ..comm import OFDMFrequencySelectiveChannel, SemanticFrequencyMapper, SubcarrierPlan
from ..config import deep_get
from ..models import ComplexAdapter, ImportanceHead, SIFDJSCC, build_backbone
from ..utils import JsonlLogger, save_config_snapshot


@dataclass(frozen=True)
class OptimizationStepResult:
    grad_norm: torch.Tensor
    optimizer_step_skipped: bool
    amp_scale_before: float
    amp_scale_after: float


def parse_train_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    return parser.parse_args()


def resolve_device(config: Mapping[str, Any], override: Optional[str] = None) -> torch.device:
    requested = override or str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_system(config: Mapping[str, Any]) -> SIFDJSCC:
    model_config = dict(config.get("model", {}))
    backbone = build_backbone(dict(model_config.get("backbone", {})))
    importance_config = dict(model_config.get("importance_head", {}))
    importance_head = ImportanceHead(
        channels=backbone.latent_channels,
        hidden_channels=int(importance_config.get("hidden_channels", 0)),
        temperature=float(importance_config.get("temperature", 0.5)),
        dropout=float(importance_config.get("dropout", 0.0)),
        channel_feature_dim=int(importance_config.get("channel_feature_dim", 11)),
    )
    ofdm_config = dict(config.get("ofdm", {}))
    pilot_carriers = ofdm_config.get("pilot_carriers")
    occupied_carriers = ofdm_config.get("occupied_carriers")
    occupied_count = ofdm_config.get(
        "occupied_carrier_count", ofdm_config.get("n_occupied")
    )
    plan = SubcarrierPlan(
        n_fft=int(ofdm_config.get("n_fft", 64)),
        occupied_shifted=(
            tuple(int(value) for value in occupied_carriers)
            if occupied_carriers is not None
            else None
        ),
        pilot_shifted=(
            tuple(int(value) for value in pilot_carriers)
            if pilot_carriers is not None
            else None
        ),
        n_bands=int(deep_get(config, "mapper.frequency_bands", 4)),
        n_occupied=int(occupied_count) if occupied_count is not None else None,
    )
    channel = OFDMFrequencySelectiveChannel(
        n_fft=plan.n_fft,
        n_cp=int(ofdm_config.get("n_cp", 16)),
        n_taps=int(ofdm_config.get("channel_taps", 8)),
        tau=float(ofdm_config.get("tau", 2.0)),
        equalizer=str(ofdm_config.get("equalizer", "mmse")),
        plan=plan,
    )
    mapper_config = dict(config.get("mapper", {}))
    mapper = SemanticFrequencyMapper(
        plan=plan,
        gamma=float(mapper_config.get("gamma", 1.0)),
        high_quantile=float(mapper_config.get("high_quantile", 0.80)),
        mid_quantile=float(mapper_config.get("mid_quantile", 0.50)),
        quality_weight=float(mapper_config.get("quality_weight", 1.0)),
        correlation_weight=float(mapper_config.get("correlation_weight", 1.0)),
        importance_correlation_gain=float(
            mapper_config.get("importance_correlation_gain", 1.0)
        ),
        diversity_levels=tuple(
            int(value)
            for value in mapper_config.get(
                "diversity_levels",
                mapper_config.get("dispersion_orders", [1, 2, 4]),
            )
        ),
        risk_band_threshold=float(mapper_config.get("risk_band_threshold", 0.25)),
        risk_min_bands=mapper_config.get("risk_min_bands", 2),
        swap_error_budget=mapper_config.get("swap_error_budget", 0.005),
        swap_max_swaps=mapper_config.get("swap_max_swaps", 128),
        swap_candidates_per_symbol=mapper_config.get("swap_candidates_per_symbol", 8),
    )
    adapter_config = dict(model_config.get("complex_adapter", {}))
    adapter = ComplexAdapter(
        normalize_power=bool(adapter_config.get("normalize_power", True)),
        restore_scale=bool(adapter_config.get("restore_scale", True)),
    )
    return SIFDJSCC(backbone, importance_head, adapter, mapper, channel)


def unpack_images(batch: Any, device: torch.device) -> tuple[torch.Tensor, Any]:
    if isinstance(batch, (tuple, list)):
        images = batch[0]
        identifiers = batch[1] if len(batch) > 1 else None
    else:
        images, identifiers = batch, None
    return images.to(device, non_blocking=True), identifiers


def sample_snr(
    batch_size: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    values = config.get("snr_db_range", [-5.0, 20.0])
    if isinstance(values, (int, float)):
        return torch.full((batch_size,), float(values), device=device)
    low, high = float(values[0]), float(values[1])
    return torch.empty(batch_size, device=device).uniform_(low, high)


def sample_csi_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    modes = list(config.get("csi_modes", ["perfect"]))
    if not modes:
        raise ValueError("csi_modes must contain at least one mode")
    index = int(torch.randint(len(modes), (1,)).item())
    mode = str(modes[index])
    result: Dict[str, Any] = {"csi_mode": mode}
    if mode in {"imperfect", "controlled_nmse", "nmse"}:
        choices = list(config.get("csi_nmse_db_choices", [-30, -20, -10]))
        if not choices:
            raise ValueError("csi_nmse_db_choices must not be empty")
        result["csi_nmse_db"] = float(choices[int(torch.randint(len(choices), (1,)).item())])
    notch = config.get("spectral_notch")
    notch_probability = float(config.get("spectral_notch_probability", 0.0))
    if not 0.0 <= notch_probability <= 1.0:
        raise ValueError("spectral_notch_probability must be in [0,1]")
    if notch is not None and float(torch.rand(()).item()) < notch_probability:
        if not isinstance(notch, Mapping):
            raise TypeError("spectral_notch must be a mapping")
        result["spectral_notch"] = dict(notch)
    return result


def load_model_checkpoint(
    model: torch.nn.Module,
    path: Optional[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    strict: bool = True,
) -> int:
    if not path:
        return 0
    checkpoint_path = Path(str(path)).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Internal model checkpoint must be a mapping produced by save_checkpoint"
        )
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model entry must be a state_dict mapping")
    if strict:
        incompatible = model.load_state_dict(state, strict=False)
        # SDS-UFD adds a zero-initialised channel conditioner to the legacy
        # ImportanceHead.  Old A/B/C checkpoints are safe to migrate because
        # missing conditioner weights reproduce the old predictions exactly.
        allowed_missing_prefixes = (
            "importance_head.channel_conditioner.",
        )
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is not strictly compatible with the model: "
                f"missing={disallowed_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        loaded_count = len(state)
        if incompatible.missing_keys:
            print(
                "Initialised new channel-conditioned sensitivity parameters "
                "while migrating a legacy checkpoint: "
                + ", ".join(incompatible.missing_keys),
                flush=True,
            )
    else:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no parameters compatible with the model"
            )
        model.load_state_dict(compatible, strict=False)
        loaded_count = len(compatible)
    print(
        f"Loaded model checkpoint: {checkpoint_path} "
        f"({loaded_count} tensors, step={int(payload.get('step', 0))})",
        flush=True,
    )
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("amp_scaler") is not None:
        scaler.load_state_dict(payload["amp_scaler"])
    return int(payload.get("step", 0))


def setup_experiment(config: Mapping[str, Any]):
    output_dir = Path(str(deep_get(config, "experiment.output_dir", "runs/default")))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)
    return output_dir, JsonlLogger(output_dir / "train.jsonl")


def optimizer_from_config(parameters, config: Mapping[str, Any]):
    name = str(config.get("name", "AdamW")).lower()
    kwargs = {
        "lr": float(config.get("lr", 1e-4)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
    }
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def create_amp(
    enabled: bool,
    init_scale: float = 65536.0,
    growth_interval: int = 2000,
):
    use_amp = bool(enabled and torch.cuda.is_available())
    return use_amp, torch.cuda.amp.GradScaler(
        enabled=use_amp,
        init_scale=float(init_scale),
        growth_interval=int(growth_interval),
    )


def finish_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    parameters,
    grad_clip: float,
) -> OptimizationStepResult:
    parameter_list = list(parameters)
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameter_list
        if parameter.grad is not None
    ]
    if gradient_norms:
        grad_norm = torch.stack(gradient_norms).norm(2)
    else:
        grad_norm = loss.detach().new_zeros(())
    finite_gradients = bool(torch.isfinite(grad_norm).item())
    if finite_gradients and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(parameter_list, grad_clip)
    # GradScaler deliberately skips optimizer.step when unscale_ observed an
    # overflow. With AMP disabled, explicitly skip a non-finite update.
    if scaler.is_enabled() or finite_gradients:
        scaler.step(optimizer)
    scaler.update()
    return OptimizationStepResult(
        grad_norm=grad_norm.detach(),
        optimizer_step_skipped=not finite_gradients,
        amp_scale_before=scale_before,
        amp_scale_after=float(scaler.get_scale()),
    )
