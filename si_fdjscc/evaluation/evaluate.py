from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..comm.multipath_channel import MultipathChannel
from ..comm.ofdm_channel import RECEIVER_CSI_SOURCES
from ..config import deep_get, load_config
from ..data import build_dataloader
from ..models.system import MAPPER_CSI_SOURCES
from ..utils import save_config_snapshot, set_seed
from ..train.common import build_system, load_model_checkpoint, resolve_device, unpack_images
from .metrics import (
    MetricAccumulator,
    assigned_high_sensitivity_correlation,
    assigned_mapping_risk_metrics,
    energy_diagnostic_metrics,
    symbol_residual_metrics,
    swap_diagnostic_metrics,
)


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _identifiers(raw, batch_size: int, offset: int):
    if raw is None:
        return [str(offset + index) for index in range(batch_size)]
    if isinstance(raw, torch.Tensor):
        return [str(value.item()) for value in raw]
    return [str(value) for value in raw]


def _csi_scenarios(values):
    for value in _as_list(values):
        if value is None or str(value).lower() == "perfect":
            yield "perfect", None, "perfect"
        else:
            numeric = float(value)
            yield "imperfect", numeric, f"nmse_{numeric:g}db"


def _notch_metadata(resolved_notch: Mapping[str, Any]) -> dict[str, Any]:
    """Record every attenuated data-carrier interval in a stable CSV schema."""
    if "notches" in resolved_notch:
        notches = list(resolved_notch["notches"])
    elif int(resolved_notch.get("width", 0)) > 0:
        notches = [resolved_notch]
    else:
        notches = []
    if len(notches) > 2:
        raise ValueError("Evaluation CSV currently supports at most two spectral notches")
    intervals = [
        {
            "start": int(notch["start"]),
            "width": int(notch["width"]),
            "attenuation_db": float(notch["attenuation_db"]),
        }
        for notch in sorted(notches, key=lambda notch: int(notch["start"]))
    ]
    result: dict[str, Any] = {
        "notch_count": len(intervals),
        "notch_total_width": sum(notch["width"] for notch in intervals),
        "notches_json": json.dumps(intervals, sort_keys=True, separators=(",", ":")),
        # These legacy fields describe only a single notch, never a merged interval.
        "notch_start": intervals[0]["start"] if len(intervals) == 1 else -1,
        "notch_width": intervals[0]["width"] if len(intervals) == 1 else 0,
        "notch_attenuation_db": intervals[0]["attenuation_db"] if len(intervals) == 1 else 0.0,
    }
    for index in range(2):
        interval = intervals[index] if index < len(intervals) else None
        prefix = f"notch_{index + 1}_"
        result[prefix + "start"] = interval["start"] if interval else -1
        result[prefix + "width"] = interval["width"] if interval else 0
        result[prefix + "attenuation_db"] = interval["attenuation_db"] if interval else 0.0
    return result


@torch.no_grad()
def evaluate(config: Mapping[str, Any], device_override=None) -> Path:
    evaluation = dict(config.get("evaluation", {}))
    mapper_csi_source = str(evaluation.get("mapper_csi_source", "estimated")).lower()
    if mapper_csi_source not in MAPPER_CSI_SOURCES:
        raise ValueError(
            "evaluation.mapper_csi_source must be one of "
            + ", ".join(MAPPER_CSI_SOURCES)
        )
    mapper_modes = _as_list(evaluation.get("mapper_modes", ["greedy", "greedy_token_energy"]))
    if (
        any(str(mode).lower() in {"greedy_token_energy", "greedy_token_random"}
            for mode in mapper_modes)
        and mapper_csi_source != "estimated"
    ):
        raise ValueError(
            "Token-internal mapping requires evaluation.mapper_csi_source='estimated' "
            "to retain the original per-symbol CSI for its greedy starting assignment"
        )
    receiver_csi_source = str(evaluation.get("receiver_csi_source", "estimated")).lower()
    if receiver_csi_source not in RECEIVER_CSI_SOURCES:
        raise ValueError(
            "evaluation.receiver_csi_source must be one of "
            + ", ".join(RECEIVER_CSI_SOURCES)
        )
    for enabled, package, name in (
        (evaluation.get("compute_msssim", True), "pytorch_msssim", "MS-SSIM"),
        (evaluation.get("compute_lpips", True), "lpips", "LPIPS"),
    ):
        if enabled and importlib.util.find_spec(package) is None:
            raise RuntimeError(f"{name} evaluation requires the {package} package")
    output_dir = Path(str(evaluation.get("output_dir", "runs/evaluation")))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)
    seed = int(evaluation.get("seed", 1000))
    set_seed(seed, deterministic=bool(evaluation.get("deterministic", True)))
    device = resolve_device(config, device_override)
    model = build_system(config).to(device)
    checkpoint = evaluation.get("checkpoint") or deep_get(config, "training.init_checkpoint")
    if not checkpoint:
        raise ValueError("evaluation.checkpoint is required")
    load_model_checkpoint(model, str(checkpoint), strict=True)
    model.eval()
    loader = build_dataloader(dict(deep_get(config, "data.test", {})), training=False)
    data_indices = model.channel.plan.tensor_indices("data", device)
    collect_energy_metrics = bool(evaluation.get("compute_energy_metrics", False))
    collect_symbol_residual_metrics = bool(
        evaluation.get("compute_symbol_residual_metrics", False)
    )

    snr_values = [float(value) for value in _as_list(evaluation.get("snr_db", [10]))]
    csi_values = evaluation.get("csi_nmse_db", ["perfect"])
    selectivity_values = _as_list(
        evaluation.get("selectivity", {"n_taps": 8, "tau": 2.0})
    )
    notch_values = _as_list(evaluation.get("spectral_notches", None))
    n_realizations = int(evaluation.get("channel_realizations", 20))
    if n_realizations < 1:
        raise ValueError("channel_realizations must be positive")
    draw_chunk_size = int(
        evaluation.get("draw_chunk_size", min(n_realizations, 4))
    )
    if draw_chunk_size < 1:
        raise ValueError("draw_chunk_size must be positive")
    progress_every_images = int(evaluation.get("progress_every_images", 25))
    all_rows = []
    summaries = []
    lpips_model = None

    for mapper_mode, snr_db, csi, selectivity, spectral_notch in itertools.product(
        mapper_modes,
        snr_values,
        list(_csi_scenarios(csi_values)),
        selectivity_values,
        notch_values,
    ):
        csi_mode, nmse_db, csi_label = csi
        print(
            f"[evaluation] mapper={mapper_mode} CSI={csi_label} "
            f"SNR={snr_db:g} notch={spectral_notch is not None}",
            flush=True,
        )
        n_taps = int(selectivity.get("n_taps", 8))
        tau = float(selectivity.get("tau", 2.0))
        if model.channel.modem.n_cp < n_taps - 1:
            raise ValueError(
                f"Evaluation scenario n_taps={n_taps} requires CP >= {n_taps - 1}, "
                f"but the configured CP is {model.channel.modem.n_cp}"
            )
        model.channel.channel = MultipathChannel(n_taps=n_taps, tau=tau).to(device)
        accumulator = MetricAccumulator(
            outage_threshold=float(evaluation.get("outage_threshold", 25.0)),
            compute_msssim=bool(evaluation.get("compute_msssim", True)),
            compute_lpips=bool(evaluation.get("compute_lpips", True)),
            device=device,
            lpips_model=lpips_model,
        )
        image_offset = 0
        for batch_index, batch in enumerate(loader):
            images, raw_identifiers = unpack_images(batch, device)
            identifiers = _identifiers(raw_identifiers, images.shape[0], image_offset)
            image_offset += images.shape[0]
            channel_seed = seed + batch_index * 1_000_000
            correlation_matrix = model.channel.data_subcarrier_correlation(
                device=device, dtype=images.dtype
            )
            snr_tensor = torch.full(
                (images.shape[0],), snr_db, device=device
            )
            for chunk_start in range(0, n_realizations, draw_chunk_size):
                chunk_count = min(draw_chunk_size, n_realizations - chunk_start)
                output, model_aux = model(
                    images,
                    snr_tensor,
                    channel_cfg={
                        "csi_mode": csi_mode,
                        "csi_nmse_db": nmse_db,
                        "channel_seed": channel_seed + chunk_start * 10_000,
                        "spectral_notch": spectral_notch,
                        "mapper_csi_source": mapper_csi_source,
                        "receiver_csi_source": receiver_csi_source,
                    },
                    mapper_mode=str(mapper_mode),
                    return_aux=True,
                    n_channel_draws=chunk_count,
                )
                if output.ndim == images.ndim:
                    output = output.unsqueeze(0)
                for local_draw in range(output.shape[0]):
                    draw = chunk_start + local_draw
                    draw_aux = model_aux["draws"][local_draw]
                    resolved_notch = draw_aux.get("spectral_notch") or {}
                    mapping_correlation = assigned_high_sensitivity_correlation(
                        draw_aux["permutation"].symbol_index_grid,
                        model_aux["complex_metadata"].token_ids,
                        draw_aux["importance_scores"],
                        correlation_matrix,
                        high_quantile=float(
                            deep_get(config, "mapper.high_quantile", 0.80)
                        ),
                    )
                    mapping_risk_metrics = {}
                    if bool(evaluation.get("compute_mapping_risk_metrics", False)):
                        permutation = draw_aux["permutation"]
                        mapping_risk_metrics = assigned_mapping_risk_metrics(
                            permutation.symbol_index_grid,
                            model_aux["complex_metadata"].token_ids,
                            draw_aux["importance"],
                            model.channel.plan.band_data_positions,
                            spectral_notch=resolved_notch,
                            high_quantile=float(deep_get(config, "mapper.high_quantile", 0.80)),
                            mid_quantile=float(deep_get(config, "mapper.mid_quantile", 0.50)),
                            risk_eligible_bands=getattr(permutation, "risk_eligible_bands", None),
                            band_reference_quality=getattr(permutation, "band_reference_quality", None),
                            risk_band_scores=getattr(permutation, "risk_band_scores", None),
                        )
                    swap_metrics = {}
                    if bool(evaluation.get("compute_swap_metrics", False)):
                        swap_metadata = getattr(draw_aux["permutation"], "swap_diagnostics", None)
                        if str(mapper_mode).lower() == "proposed_ufd_swap" and swap_metadata is None:
                            raise ValueError("proposed_ufd_swap did not provide swap_diagnostics")
                        swap_metrics = swap_diagnostic_metrics(
                            swap_metadata, images.shape[0], device
                        )
                    energy_metrics = {}
                    if collect_energy_metrics:
                        energy_metadata = getattr(draw_aux["permutation"], "energy_diagnostics", None)
                        if str(mapper_mode).lower() == "greedy_token_energy" and energy_metadata is None:
                            raise ValueError("greedy_token_energy did not provide energy_diagnostics")
                        energy_metrics = energy_diagnostic_metrics(
                            energy_metadata, images.shape[0], device
                        )
                    residual_metrics = {}
                    if collect_symbol_residual_metrics:
                        mapper_channel = draw_aux["H_hat"]
                        if mapper_csi_source == "oracle":
                            mapper_channel = draw_aux["H"]
                        elif mapper_csi_source == "temporal_mean":
                            mapper_channel = mapper_channel.mean(dim=1, keepdim=True)
                        residual_metrics = symbol_residual_metrics(
                            model_aux["symbols_tx"],
                            draw_aux["symbols_rx"],
                            model_aux["complex_metadata"].token_ids,
                            draw_aux["importance"],
                            draw_aux["permutation"].symbol_index_grid,
                            mapper_channel,
                            draw_aux["equalizer_H_hat"],
                            draw_aux["H"],
                            draw_aux["noise_var"],
                            data_indices,
                        )
                    accumulator.add(
                        output[local_draw],
                        images,
                        identifiers,
                        metadata={
                            "mapper": mapper_mode,
                            "mapper_csi_source": mapper_csi_source,
                            "receiver_csi_source": receiver_csi_source,
                            "snr_db": snr_db,
                            "csi": csi_label,
                            "n_taps": n_taps,
                            "tau": tau,
                            "channel_draw": draw,
                            "semantic_cbr": model_aux["semantic_cbr"],
                            "padded_data_ratio": model_aux["padded_data_ratio"],
                            "pilot_overhead_ratio": model_aux[
                                "pilot_overhead_ratio"
                            ],
                            "guard_dc_overhead_ratio": model_aux[
                                "guard_dc_overhead_ratio"
                            ],
                            "cp_overhead_ratio": model_aux[
                                "cp_overhead_ratio"
                            ],
                            "air_interface_ratio": model_aux["air_interface_ratio"],
                            "high_sensitivity_carrier_correlation": mapping_correlation,
                            **_notch_metadata(resolved_notch),
                            **mapping_risk_metrics,
                            **swap_metrics,
                            **energy_metrics,
                            **residual_metrics,
                        },
                    )
            if progress_every_images > 0 and (
                image_offset % progress_every_images < images.shape[0]
                or batch_index + 1 == len(loader)
            ):
                print(
                    f"[evaluation] mapper={mapper_mode} CSI={csi_label} "
                    f"notch={spectral_notch is not None} images={image_offset} "
                    f"samples={len(accumulator.rows)}",
                    flush=True,
                )
        summary = accumulator.summary()
        lpips_model = accumulator.lpips_model
        summary.update(
            {
                "mapper": mapper_mode,
                "mapper_csi_source": mapper_csi_source,
                "receiver_csi_source": receiver_csi_source,
                "snr_db": snr_db,
                "csi": csi_label,
                "n_taps": n_taps,
                "tau": tau,
                "spectral_notch": spectral_notch,
            }
        )
        summaries.append(summary)
        all_rows.extend(accumulator.rows)

    with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SDS-UFD-JSCC sweeps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--mapper-csi-source", choices=MAPPER_CSI_SOURCES)
    parser.add_argument("--receiver-csi-source", choices=RECEIVER_CSI_SOURCES)
    parser.add_argument("--output-dir")
    parser.add_argument("--mapper-modes", nargs="+", metavar="MODE")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = dict(config.get("evaluation", {}))
    diagnostic_requested = (
        args.mapper_csi_source is not None
        or args.receiver_csi_source is not None
        or str(evaluation.get("mapper_csi_source", "estimated")).lower() != "estimated"
        or str(evaluation.get("receiver_csi_source", "estimated")).lower() != "estimated"
    )
    if diagnostic_requested:
        if args.output_dir is None:
            parser.error("a non-default CSI source requires --output-dir to protect the base run")
        base_output = Path(str(evaluation.get("output_dir", "runs/evaluation"))).resolve()
        target_output = Path(args.output_dir).resolve()
        if target_output == base_output:
            parser.error("--output-dir must differ from the base config's output directory")
        if target_output.exists() and (
            not target_output.is_dir() or any(target_output.iterdir())
        ):
            parser.error("--output-dir must be a new or empty directory for a CSI diagnostic")
    if args.mapper_csi_source is not None:
        evaluation["mapper_csi_source"] = args.mapper_csi_source
    if args.receiver_csi_source is not None:
        evaluation["receiver_csi_source"] = args.receiver_csi_source
    if args.output_dir is not None:
        evaluation["output_dir"] = args.output_dir
    if args.mapper_modes is not None:
        evaluation["mapper_modes"] = args.mapper_modes
    config["evaluation"] = evaluation
    print(evaluate(config, device_override=args.device))


if __name__ == "__main__":
    main()
