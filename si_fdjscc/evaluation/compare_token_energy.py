"""Audit paired greedy/energy or the full five-method baseline evaluation.

This reads saved CSV/JSON/YAML only. Confidence intervals resample images,
keeping each image's channel draws together.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import yaml

CONDITIONS = (("nmse_-10db", False), ("nmse_-10db", True))
DOUBLE_NOTCH_CONDITIONS = tuple(
    (f"nmse_{nmse}db", notched)
    for nmse in (-10, -5) for notched in (False, True)
)
SNR_VALUES = (10, 5, 0, -5, -10)
SNR_NO_NOTCH_CONDITIONS = tuple((snr, "nmse_-10db", False) for snr in SNR_VALUES)
ENERGY_MODE = "greedy_token_energy"
TOKEN_RANDOM_MODE = "greedy_token_random"
PAIR_MODES = ("greedy", ENERGY_MODE)
BASELINE_MODES = ("sequential", "random", "greedy", TOKEN_RANDOM_MODE, ENERGY_MODE)
PAIR_FIELDS = (
    "snr_db", "n_taps", "tau", "notch_start", "notch_width",
    "notch_attenuation_db", "semantic_cbr", "padded_data_ratio",
    "pilot_overhead_ratio", "guard_dc_overhead_ratio", "cp_overhead_ratio",
    "air_interface_ratio",
)
DOUBLE_NOTCH_PAIR_FIELDS = (
    "notch_count", "notch_total_width",
    "notch_1_start", "notch_1_width", "notch_1_attenuation_db",
    "notch_2_start", "notch_2_width", "notch_2_attenuation_db",
)


def _quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _protocol(config):
    evaluation = config["evaluation"]
    mapper = config["mapper"]
    mapper_signature = {key: mapper.get(key) for key in (
        "frequency_bands", "diversity_levels", "high_quantile", "mid_quantile",
        "quality_weight", "correlation_weight", "importance_correlation_gain",
    )}
    mapper_signature["gamma"] = mapper.get("gamma", 1.0)
    evaluation_signature = {key: evaluation.get(key) for key in (
        "checkpoint", "seed", "channel_realizations", "draw_chunk_size",
        "outage_threshold", "deterministic", "snr_db", "csi_nmse_db",
        "selectivity", "spectral_notches",
    )}
    evaluation_signature["mapper_csi_source"] = evaluation.get("mapper_csi_source", "estimated")
    evaluation_signature["receiver_csi_source"] = evaluation.get("receiver_csi_source", "estimated")
    return {
        "data": config["data"]["test"], "model": config["model"],
        "ofdm": config["ofdm"],
        "mapper": mapper_signature,
        "evaluation": evaluation_signature,
    }


def _condition_parts(condition):
    """Keep legacy CSI/notch keys while including SNR in sweep group keys."""
    if len(condition) == 3:
        return condition
    csi, notched = condition
    return 10.0, csi, notched


def _condition_key(snr, csi, notched, conditions):
    if len(conditions[0]) == 3:
        return float(snr), str(csi), bool(notched)
    return str(csi), bool(notched)


def _audit_double_notch_sample(row, notched):
    """Verify the realized geometry rather than trusting the scenario label."""
    key = (row.get("mapper"), row.get("csi"), row.get("image_id"), row.get("channel_draw"))
    count = _number(row, "notch_count", key)
    if not count.is_integer() or count != (2 if notched else 0):
        raise ValueError(f"{key}: double-notch scenario requires exactly two or zero notches")
    try:
        intervals = json.loads(row["notches_json"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key}: invalid notches_json") from error
    if not isinstance(intervals, list) or len(intervals) != count:
        raise ValueError(f"{key}: notch count disagrees with notches_json")
    if (
        _number(row, "notch_start", key) != -1
        or _number(row, "notch_width", key) != 0
        or _number(row, "notch_attenuation_db", key) != 0
    ):
        raise ValueError(f"{key}: legacy single-notch fields must be empty")
    total_width = 0
    previous_end = None
    for index, interval in enumerate(intervals, 1):
        if not isinstance(interval, dict):
            raise ValueError(f"{key}: invalid notch interval")
        start_value = _number(interval, "start", key)
        width_value = _number(interval, "width", key)
        if not start_value.is_integer() or not width_value.is_integer():
            raise ValueError(f"{key}: notch start and width must be integers")
        start = int(start_value)
        width = int(width_value)
        attenuation = _number(interval, "attenuation_db", key)
        if not 0 <= start < 48 or not 8 <= width <= 12 or start + width > 48:
            raise ValueError(f"{key}: notch lies outside the frozen data-carrier geometry")
        if not 15 <= attenuation <= 25:
            raise ValueError(f"{key}: notch attenuation differs from the frozen protocol")
        if previous_end is not None and start - previous_end < 4:
            raise ValueError(f"{key}: two notches overlap or violate the four-carrier gap")
        previous_end = start + width
        total_width += width
        for suffix, expected in (("start", start), ("width", width),
                                 ("attenuation_db", attenuation)):
            field = f"notch_{index}_{suffix}"
            if abs(_number(row, field, key) - expected) > 1e-7:
                raise ValueError(f"{key}: {field} disagrees with notches_json")
    if _number(row, "notch_total_width", key) != total_width:
        raise ValueError(f"{key}: notch_total_width disagrees with notches_json")
    for index in range(len(intervals) + 1, 3):
        for suffix, expected in (("start", -1), ("width", 0), ("attenuation_db", 0)):
            field = f"notch_{index}_{suffix}"
            if _number(row, field, key) != expected:
                raise ValueError(f"{key}: {field} must be empty for an absent notch")


def _load_run(directory, mappers, expected_count, expected_draws, conditions=CONDITIONS):
    directory = Path(directory)
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    configured_modes = config["evaluation"].get("mapper_modes", ())
    if set(configured_modes) != set(mappers) or len(configured_modes) != len(mappers):
        raise ValueError(f"{directory}: mapper mode list differs from the requested suite")
    summaries = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected = {(mode, condition) for mode in mappers for condition in conditions}
    found = {}
    for row in summaries:
        condition = _condition_key(
            row["snr_db"], row["csi"], row["spectral_notch"] is not None, conditions
        )
        key = (str(row["mapper"]), condition)
        if key not in expected:
            raise ValueError(f"{directory}: unexpected summary {key}")
        if key in found:
            raise ValueError(f"{directory}: duplicate summary {key}")
        snr, _, _ = _condition_parts(condition)
        if (float(row["snr_db"]), int(row["n_taps"]), float(row["tau"])) != (float(snr), 8, 2.0):
            raise ValueError(f"{directory}: unexpected SNR/selectivity for {key}")
        if conditions == DOUBLE_NOTCH_CONDITIONS:
            expected_notch = config["evaluation"]["spectral_notches"][int(condition[1])]
            if row["spectral_notch"] != expected_notch:
                raise ValueError(f"{directory}: summary notch specification differs for {key}")
        found[key] = {"summary": row, "samples": {}}
    if set(found) != expected:
        raise ValueError(f"{directory}: missing required summaries {expected - set(found)}")
    with (directory / "samples.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            notched = (int(row["notch_count"]) > 0 if "notch_count" in row
                       and row["notch_count"] not in (None, "")
                       else int(row["notch_width"]) > 0)
            key = (row["mapper"], _condition_key(row["snr_db"], row["csi"], notched, conditions))
            if key not in expected:
                raise ValueError(f"{directory}: unexpected sample group {key}")
            if conditions in (DOUBLE_NOTCH_CONDITIONS, SNR_NO_NOTCH_CONDITIONS):
                _audit_double_notch_sample(row, notched)
            if conditions == SNR_NO_NOTCH_CONDITIONS:
                for field, value in row.items():
                    if field.startswith("mapping_") and field.endswith("notch_fraction"):
                        if _number(row, field, key) != 0.0:
                            raise ValueError(f"{directory}: no-notch sample has nonzero {field}")
            summary = found[key]["summary"]
            for field in ("snr_db", "n_taps", "tau"):
                if _number(row, field, key) != _number(summary, field, key):
                    raise ValueError(f"{directory}: sample {field} disagrees with summary for {key}")
            if row.get("mapper_csi_source") != "estimated" or row.get("receiver_csi_source") != "temporal_mean":
                raise ValueError(f"{directory}: CSI policy differs from the frozen protocol")
            pair_key = (str(row["image_id"]), int(row["channel_draw"]))
            _number(row, "psnr", pair_key)
            if pair_key in found[key]["samples"]:
                raise ValueError(f"{directory}: duplicate sample {key}/{pair_key}")
            found[key]["samples"][pair_key] = row
    for key, item in found.items():
        rows = item["samples"]
        summary = item["summary"]
        if len(rows) != expected_count or int(summary["count"]) != expected_count:
            raise ValueError(f"{directory}: {key} requires {expected_count} samples, found {len(rows)}")
        by_image = defaultdict(set)
        for image_id, draw in rows:
            by_image[image_id].add(draw)
        if any(draws != set(range(expected_draws)) for draws in by_image.values()):
            raise ValueError(f"{directory}: {key} has missing/nonuniform image draws")
        values = [float(row["psnr"]) for row in rows.values()]
        threshold = _number(summary, "outage_threshold", key)
        recalculated = {
            "psnr_mean": fmean(values), "psnr_p5": _quantile(values, 0.05),
            "outage_probability": fmean(float(value < threshold) for value in values),
        }
        for field, value in recalculated.items():
            if abs(value - _number(summary, field, key)) > 1e-6:
                raise ValueError(f"{directory}: {key} summary {field} disagrees with CSV")
        for metric in ("ms_ssim", "lpips"):
            measured = []
            for pair_key, row in rows.items():
                value = _number(row, metric, pair_key)
                if metric == "ms_ssim" and not 0.0 <= value <= 1.0:
                    raise ValueError(f"{directory}: {key} MS-SSIM outside [0, 1]")
                if metric == "lpips" and value < 0:
                    raise ValueError(f"{directory}: {key} negative LPIPS")
                measured.append(value)
            if abs(fmean(measured) - _number(summary, f"{metric}_mean", key)) > 1e-6:
                raise ValueError(f"{directory}: {key} summary {metric} disagrees with CSV")
        if summary.get("mapper_csi_source") != "estimated" or summary.get("receiver_csi_source") != "temporal_mean":
            raise ValueError(f"{directory}: summary CSI policy differs")
        if threshold != 25.0:
            raise ValueError(f"{directory}: outage threshold must remain 25 dB")
    return config, found


def _check_pairing(left, right, description, ignore_snr=False):
    if set(left) != set(right):
        raise ValueError(f"{description}: image/draw keys differ")
    for key in left:
        if ("notch_count" in left[key]) != ("notch_count" in right[key]):
            raise ValueError(f"{description}: {key} notch schema differs")
        fields = PAIR_FIELDS + (
            DOUBLE_NOTCH_PAIR_FIELDS if "notch_count" in left[key] else ()
        )
        for field in fields:
            if ignore_snr and field == "snr_db":
                continue
            a, b = float(left[key][field]), float(right[key][field])
            if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-7:
                raise ValueError(f"{description}: {key} differs in {field}")
        if "notch_count" in left[key] and (
            left[key].get("notches_json") != right[key].get("notches_json")
        ):
            raise ValueError(f"{description}: {key} differs in notches_json")


def _cluster_interval(values, resamples=10000, seed=2000):
    by_image = defaultdict(list)
    for (image_id, _draw), value in sorted(values.items()):
        by_image[image_id].append(value)
    if len(by_image) < 2:
        raise ValueError("Image-bootstrap inference requires at least two images")
    if len({len(group) for group in by_image.values()}) != 1:
        raise ValueError("Image-bootstrap requires the same draw count for every image")
    image_means = [fmean(group) for group in by_image.values()]
    rng = random.Random(seed)
    boot = [fmean(rng.choices(image_means, k=len(image_means))) for _ in range(resamples)]
    return {
        "mean": fmean(values.values()), "ci95_low": _quantile(boot, 0.025),
        "ci95_high": _quantile(boot, 0.975), "images": len(image_means),
        "images_with_positive_mean": sum(value > 0 for value in image_means),
    }


def _comparison(left, right, description, resamples):
    a, b = left["samples"], right["samples"]
    _check_pairing(a, b, description)
    psnr = {key: float(a[key]["psnr"]) - float(b[key]["psnr"]) for key in a}
    outage = {
        key: float(float(a[key]["psnr"]) < 25.0) - float(float(b[key]["psnr"]) < 25.0)
        for key in a
    }
    return {
        "description": description,
        "psnr_delta_db": _cluster_interval(psnr, resamples),
        "outage_delta": _cluster_interval(outage, resamples),
        "metric_deltas": {
            metric: _paired_numeric(left, right, metric, description, resamples)
            for metric in ("ms_ssim", "lpips")
        },
        "residual_deltas": {
            field: _paired_numeric(left, right, field, description, resamples)
            for field in ACTUAL_RESIDUAL_FIELDS
        },
    }




ENERGY_FIELDS = (
    "energy_proxy_initial", "energy_proxy_final",
    "energy_changed_symbol_count", "energy_changed_symbol_fraction",
)
RESIDUAL_FIELDS = (
    "residual_symbol_mse", "residual_sds_weighted_token_mse",
    "residual_estimated_energy_proxy", "residual_mapper_csi_nmse",
    "residual_receiver_csi_nmse",
)
ACTUAL_RESIDUAL_FIELDS = RESIDUAL_FIELDS[:2]


def _number(row, field, key):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key}: missing or invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"{key}: non-finite {field}")
    return value


def _audit_protocol(
    config, expected_modes, expected_draws, dataset="div2k", protocol="single_notch"
):
    """Reject a changed evaluation protocol before comparing methods."""
    test = config["data"]["test"]
    evaluation = config["evaluation"]
    required = {"batch_size": 1}
    if dataset == "div2k":
        required.update({"name": "DIV2K", "crop_size": 256,
                         "shuffle": False, "drop_last": False})
    elif dataset == "kodak":
        required.update({"name": "Kodak"})
        if test.get("crop_size") is not None:
            raise ValueError("Kodak must use full-resolution images")
    else:
        raise ValueError(f"Unsupported comparison dataset: {dataset}")
    if not test.get("root") or any(test.get(key) != value for key, value in required.items()):
        raise ValueError(f"{dataset} test data protocol changed")
    if not evaluation.get("checkpoint"):
        raise ValueError("Evaluation checkpoint is missing")
    modes = evaluation.get("mapper_modes", ())
    if set(modes) != set(expected_modes) or len(modes) != len(expected_modes):
        raise ValueError("Mapper mode list differs from the predeclared jobs")
    if protocol == "single_notch":
        snr_values = [10]
        csi_values = [-10]
        notch_spec = {
            "start": "random", "width_range": [8, 12],
            "attenuation_db_range": [15, 25],
        }
        notch_values = [None, notch_spec]
    elif protocol == "double_notch":
        snr_values = [10]
        csi_values = [-10, -5]
        notch_spec = {
            "count": 2, "min_gap": 4, "start": "random",
            "width_range": [8, 12], "attenuation_db_range": [15, 25],
        }
        notch_values = [None, notch_spec]
    elif protocol == "snr_sweep":
        snr_values = list(SNR_VALUES)
        csi_values = [-10]
        notch_values = [None]
    else:
        raise ValueError(f"Unknown comparison protocol: {protocol}")
    frozen = {
        "seed": 2000 if dataset == "div2k" else 3000,
        "snr_db": snr_values, "csi_nmse_db": csi_values,
        "selectivity": [{"n_taps": 8, "tau": 2.0}],
        "spectral_notches": notch_values,
        "channel_realizations": expected_draws,
        "draw_chunk_size": 4 if dataset == "div2k" else 1,
        "outage_threshold": 25.0, "deterministic": True,
        "mapper_csi_source": "estimated", "receiver_csi_source": "temporal_mean",
        "compute_msssim": True, "compute_lpips": True,
        "compute_mapping_risk_metrics": True,
        "compute_energy_metrics": True, "compute_symbol_residual_metrics": True,
    }
    for key, value in frozen.items():
        if evaluation.get(key) != value:
            raise ValueError(f"Frozen evaluation setting differs: {key}")


def _audit_energy(item):
    """Check the energy proxy for every image and draw, not only its mean."""
    changed = []
    reductions = []
    for key, row in item["samples"].items():
        initial, final, count, fraction = (
            _number(row, field, key) for field in ENERGY_FIELDS
        )
        symbols = _number(row, "mapping_symbol_count", key)
        tolerance = 1e-8 * max(1.0, abs(initial))
        if initial < 0 or final < 0 or final > initial + tolerance:
            raise ValueError(f"{key}: energy proxy increased")
        if not symbols.is_integer() or symbols <= 0:
            raise ValueError(f"{key}: invalid symbol count")
        if not count.is_integer() or not 0 <= count <= symbols:
            raise ValueError(f"{key}: invalid changed-symbol count")
        if not 0 <= fraction <= 1 or abs(fraction - count / symbols) > 1e-9:
            raise ValueError(f"{key}: changed-symbol fraction disagrees with count")
        if count == 0 and abs(final - initial) > tolerance:
            raise ValueError(f"{key}: proxy changed without any moved symbol")
        residual_proxy = _number(row, "residual_estimated_energy_proxy", key)
        if abs(final - residual_proxy) > 1e-5 * max(1.0, final):
            raise ValueError(f"{key}: reported estimated residual proxy disagrees with mapper")
        changed.append(count)
        reductions.append(initial - final)
    return {
        "samples": len(changed),
        "samples_with_change": sum(count > 0 for count in changed),
        "changed_symbol_count_mean": fmean(changed),
        "proxy_reduction_mean": fmean(reductions),
        "proxy_reduction_min": min(reductions),
    }


def _audit_baseline_energy(item, label):
    for key, row in item["samples"].items():
        for field in ENERGY_FIELDS:
            if field not in row:
                raise ValueError(f"{label}/{key}: missing energy schema field {field}")
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label}/{key}: invalid {field}") from error
            if not math.isnan(value):
                raise ValueError(f"{label}/{key}: baseline mode must have NaN {field}")


def _audit_residual(item):
    rows = item["samples"]
    for key, row in rows.items():
        for field in RESIDUAL_FIELDS:
            value = _number(row, field, key)
            if value < 0:
                raise ValueError(f"{key}: negative {field}")
    return {
        field + "_mean": fmean(float(row[field]) for row in rows.values())
        for field in RESIDUAL_FIELDS
    }


def _audit_mapping_preserved(energy_rows, greedy_rows, description):
    _check_pairing(energy_rows, greedy_rows, description)
    first = next(iter(energy_rows.values()))
    fields = sorted(
        field for field in first
        if field.startswith("mapping_") or field == "high_sensitivity_carrier_correlation"
    )
    if not fields:
        raise ValueError(f"{description}: mapping diagnostics missing")
    for key, energy in energy_rows.items():
        greedy = greedy_rows[key]
        for field in fields:
            try:
                a, b = float(energy[field]), float(greedy[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{description}/{key}: invalid {field}") from error
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-9:
                raise ValueError(f"{description}/{key}: mapping changed in {field}")
    return {"description": description, "fields_checked": fields, "samples": len(energy_rows)}


def _audit_greedy_proxy_alignment(energy_rows, greedy_rows, description):
    _check_pairing(energy_rows, greedy_rows, description)
    max_difference = 0.0
    for key, energy in energy_rows.items():
        initial = _number(energy, "energy_proxy_initial", key)
        greedy_proxy = _number(greedy_rows[key], "residual_estimated_energy_proxy", key)
        difference = abs(initial - greedy_proxy)
        max_difference = max(max_difference, difference)
        if difference > 1e-5 * max(1.0, initial):
            raise ValueError(f"{description}/{key}: initial proxy differs from greedy")
    return {"description": description, "max_abs_proxy_difference": max_difference}


def _audit_csi_identical(left, right, description, ignore_snr=False):
    _check_pairing(left, right, description, ignore_snr=ignore_snr)
    for key in left:
        for field in ("residual_mapper_csi_nmse", "residual_receiver_csi_nmse"):
            a, b = _number(left[key], field, key), _number(right[key], field, key)
            if abs(a - b) > 1e-7 * max(1.0, a, b):
                raise ValueError(f"{description}/{key}: CSI diagnostic differs in {field}")


def _audit_unchanged_psnr(energy_rows, greedy_rows, description):
    _check_pairing(energy_rows, greedy_rows, description)
    differences = [
        abs(_number(row, "psnr", key) - _number(greedy_rows[key], "psnr", key))
        for key, row in energy_rows.items()
        if _number(row, "energy_changed_symbol_count", key) == 0
    ]
    maximum = max(differences, default=0.0)
    if maximum > 1e-3:
        raise ValueError(f"{description}: unchanged mapping differs from greedy >0.001 dB")
    return {"description": description, "unchanged_samples": len(differences), "max_abs_psnr_delta_db": maximum}


def _paired_numeric(left, right, field, description, resamples):
    a, b = left["samples"], right["samples"]
    _check_pairing(a, b, description)
    deltas = {key: _number(a[key], field, key) - _number(b[key], field, key) for key in a}
    return _cluster_interval(deltas, resamples)


def _diagnostics(item):
    rows = tuple(item["samples"].values())
    fields = ENERGY_FIELDS + RESIDUAL_FIELDS
    result = {}
    for field in fields:
        values = []
        for row in rows:
            if field in row and row[field] not in (None, ""):
                value = float(row[field])
                if math.isfinite(value):
                    values.append(value)
        result[field] = fmean(values) if values else None
    return result


def _comparison_pairs(modes):
    """Predeclared contrasts, always expressed as left minus right."""
    if tuple(modes) == PAIR_MODES:
        return ((ENERGY_MODE, "greedy"),)
    return (
        *((ENERGY_MODE, baseline) for baseline in BASELINE_MODES if baseline != ENERGY_MODE),
        ("greedy", "sequential"), ("greedy", "random"),
        (TOKEN_RANDOM_MODE, "greedy"),
    )


def _build_report(config, runs, modes, dataset, resamples, conditions=CONDITIONS):
    snr_sweep = conditions == SNR_NO_NOTCH_CONDITIONS
    report = {
        "dataset": dataset, "suite": "pair" if tuple(modes) == PAIR_MODES else "baselines",
        "protocol": _protocol(config), "results": [], "audits": [],
        "comparisons": [], "target_status": None,
        "target_status_by_condition": {},
        "target_status_comparison": {
            "left": ENERGY_MODE, "right": "greedy", "notch": not snr_sweep,
        },
        "note": (
            "Image-cluster intervals; paired channel draws are shared. "
            "All differences are left minus right; higher PSNR/MS-SSIM and lower "
            "LPIPS/outage/residual are favorable. Intervals are individual 95% "
            "intervals without multiple-comparison adjustment."
        ),
    }
    if snr_sweep:
        reference_condition = conditions[0]
        for mode in modes:
            reference = runs[(mode, reference_condition)]["samples"]
            for condition in conditions[1:]:
                rows = runs[(mode, condition)]["samples"]
                description = f"{mode} shared images/channel CSI across SNR/{condition}"
                _audit_csi_identical(rows, reference, description, ignore_snr=True)
                report["audits"].append({
                    "description": description, "samples": len(rows),
                })
    for condition in conditions:
        snr, csi, notch = _condition_parts(condition)
        energy = runs[(ENERGY_MODE, condition)]
        greedy = runs[("greedy", condition)]
        for mode in modes:
            item = runs[(mode, condition)]
            _check_pairing(item["samples"], greedy["samples"], f"{mode} paired/{condition}")
            _audit_csi_identical(
                item["samples"], greedy["samples"], f"{mode} shared channel/{condition}"
            )
            residual = _audit_residual(item)
            if mode == ENERGY_MODE:
                report["audits"].append({
                    "mode": mode, "condition": condition, **_audit_energy(item),
                })
            else:
                _audit_baseline_energy(item, mode)
            if mode in (ENERGY_MODE, TOKEN_RANDOM_MODE):
                report["audits"].append(_audit_mapping_preserved(
                    item["samples"], greedy["samples"],
                    f"{mode} token-carrier aggregates/{condition}",
                ))
            report["results"].append({
                "mode": mode, "snr_db": snr, "csi": csi, "notch": notch,
                "summary": item["summary"], "diagnostics": _diagnostics(item),
            })
            summary = item["summary"]
            print(
                f"{mode}: SNR={snr:g} dB, CSI={csi}, notch={notch}, "
                f"PSNR={summary['psnr_mean']:.5f} dB, "
                f"MS-SSIM={summary['ms_ssim_mean']:.5f}, "
                f"LPIPS={summary['lpips_mean']:.5f}, "
                f"outage={100 * summary['outage_probability']:.2f}%, "
                f"residual={json.dumps(residual)}"
            )
        report["audits"].extend((
            _audit_greedy_proxy_alignment(
                energy["samples"], greedy["samples"], f"greedy energy proxy/{condition}"
            ),
            _audit_unchanged_psnr(
                energy["samples"], greedy["samples"], f"unchanged mapping/{condition}"
            ),
        ))
        energy_vs_greedy = None
        for left_mode, right_mode in _comparison_pairs(modes):
            description = (
                "energy minus greedy" if (left_mode, right_mode) == (ENERGY_MODE, "greedy")
                else f"{left_mode} minus {right_mode}"
            )
            comparison = _comparison(
                runs[(left_mode, condition)], runs[(right_mode, condition)],
                description, resamples,
            )
            report["comparisons"].append({
                "snr_db": snr, "csi": csi, "notch": notch,
                "left": left_mode, "right": right_mode,
                "paired": comparison,
            })
            psnr = comparison["psnr_delta_db"]
            msssim = comparison["metric_deltas"]["ms_ssim"]
            lpips = comparison["metric_deltas"]["lpips"]
            print(
                f"  {description}: PSNR {psnr['mean']:+.5f} dB "
                f"[{psnr['ci95_low']:+.5f}, {psnr['ci95_high']:+.5f}], "
                f"MS-SSIM {msssim['mean']:+.5f} "
                f"[{msssim['ci95_low']:+.5f}, {msssim['ci95_high']:+.5f}], "
                f"LPIPS {lpips['mean']:+.5f} "
                f"[{lpips['ci95_low']:+.5f}, {lpips['ci95_high']:+.5f}]"
            )
            if (left_mode, right_mode) == (ENERGY_MODE, "greedy"):
                energy_vs_greedy = comparison
        if notch or snr_sweep:
            active = any(
                float(row["energy_changed_symbol_count"]) > 0
                for row in energy["samples"].values()
            )
            psnr = energy_vs_greedy["psnr_delta_db"]
            outage = energy_vs_greedy["outage_delta"]
            if not active:
                status = "no_rearrangement_no_effect"
            elif psnr["ci95_low"] > 0:
                status = (
                    ("kodak_psnr_gain" if dataset == "kodak"
                     else "exploratory_candidate_for_holdout") if outage["mean"] <= 0
                    else "psnr_gain_with_outage_tradeoff"
                )
            elif psnr["ci95_high"] < 0:
                status = "not_better_than_greedy"
            else:
                status = "inconclusive_psnr_difference"
            condition_label = f"snr_{snr:g}db" if snr_sweep else csi
            report["target_status_by_condition"][condition_label] = status
            if not snr_sweep and len({_condition_parts(item)[1] for item in conditions}) == 1:
                report["target_status"] = status
    return report


def _write_summary_csv(report, path):
    """One row per method and condition, ready for an experimental result table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "count", "psnr_mean", "psnr_p5", "ms_ssim_mean", "lpips_mean", "outage_probability",
    )
    diagnostic_fields = RESIDUAL_FIELDS + ENERGY_FIELDS
    fields = ("dataset", "mode", "snr_db", "csi", "notch", "notch_count") + summary_fields + diagnostic_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            writer.writerow({
                "dataset": report["dataset"],
                "snr_db": result["snr_db"],
                **{field: result[field] for field in ("mode", "csi", "notch")},
                "notch_count": (
                    int(result["summary"]["spectral_notch"].get("count", 1))
                    if result["notch"] else 0
                ),
                **{field: result["summary"][field] for field in summary_fields},
                **{field: result["diagnostics"][field] for field in diagnostic_fields},
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=("pair", "baselines"), default="pair")
    parser.add_argument("--protocol", choices=("single_notch", "double_notch", "snr_sweep"),
                        default="single_notch")
    parser.add_argument("--dataset", choices=("div2k", "kodak"), default="div2k")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--expected-draws", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.expected_count < 2 or args.expected_draws < 1 or args.expected_count % args.expected_draws:
        parser.error("expected-count must contain complete image/draw groups")
    if args.resamples < 1000:
        parser.error("resamples must be at least 1000")

    modes = PAIR_MODES if args.suite == "pair" else BASELINE_MODES
    conditions = {
        "single_notch": CONDITIONS,
        "double_notch": DOUBLE_NOTCH_CONDITIONS,
        "snr_sweep": SNR_NO_NOTCH_CONDITIONS,
    }[args.protocol]
    config, runs = _load_run(
        args.run_dir, modes, args.expected_count, args.expected_draws, conditions
    )
    _audit_protocol(config, modes, args.expected_draws, args.dataset, args.protocol)
    report = _build_report(config, runs, modes, args.dataset, args.resamples, conditions)
    print("Confidence intervals resample images with all their channel draws.")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    if args.output_csv:
        _write_summary_csv(report, args.output_csv)


if __name__ == "__main__":
    main()
