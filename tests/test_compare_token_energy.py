import csv
import json
from statistics import fmean

import pytest
import yaml
from pathlib import Path

from si_fdjscc.evaluation.compare_token_energy import (
    ACTUAL_RESIDUAL_FIELDS,
    BASELINE_MODES,
    CONDITIONS,
    DOUBLE_NOTCH_CONDITIONS,
    ENERGY_FIELDS,
    ENERGY_MODE,
    PAIR_MODES,
    PAIR_FIELDS,
    _audit_csi_identical,
    _audit_energy,
    _audit_greedy_proxy_alignment,
    _audit_baseline_energy,
    _audit_mapping_preserved,
    _audit_residual,
    _audit_unchanged_psnr,
    _audit_protocol,
    _build_report,
    _load_run,
    _paired_numeric,
    main,
)


def _row():
    return {
        **{field: "0" for field in PAIR_FIELDS},
        "mapping_symbol_count": "10", "mapping_high_notch_fraction": "0.2",
        "high_sensitivity_carrier_correlation": "0.4",
        "energy_proxy_initial": "10", "energy_proxy_final": "9",
        "energy_changed_symbol_count": "2", "energy_changed_symbol_fraction": "0.2",
        "residual_symbol_mse": "0.25", "residual_sds_weighted_token_mse": "0.30",
        "residual_estimated_energy_proxy": "9",
        "residual_mapper_csi_nmse": "0.1", "residual_receiver_csi_nmse": "0.04",
    }


def test_energy_audit_checks_every_draw_and_count_fraction():
    rows = {("a", 0): _row(), ("b", 0): _row()}
    rows[("b", 0)]["energy_proxy_final"] = "10.1"
    rows[("b", 0)]["residual_estimated_energy_proxy"] = "10.1"
    with pytest.raises(ValueError, match="energy proxy increased"):
        _audit_energy({"samples": rows})
    rows[("b", 0)]["energy_proxy_final"] = "9"
    rows[("b", 0)]["residual_estimated_energy_proxy"] = "9"
    rows[("b", 0)]["energy_changed_symbol_fraction"] = "0.1"
    with pytest.raises(ValueError, match="fraction disagrees"):
        _audit_energy({"samples": rows})


def test_energy_residual_proxy_and_greedy_start_must_agree():
    row = _row()
    assert _audit_energy({"samples": {("a", 0): row}})["samples_with_change"] == 1
    row["residual_estimated_energy_proxy"] = "8"
    with pytest.raises(ValueError, match="disagrees with mapper"):
        _audit_energy({"samples": {("a", 0): row}})
    row["residual_estimated_energy_proxy"] = "9"
    greedy = _row()
    greedy["residual_estimated_energy_proxy"] = "10"
    _audit_greedy_proxy_alignment({("a", 0): row}, {("a", 0): greedy}, "start")
    greedy["residual_estimated_energy_proxy"] = "11"
    with pytest.raises(ValueError, match="differs from greedy"):
        _audit_greedy_proxy_alignment({("a", 0): row}, {("a", 0): greedy}, "start")


def test_mapping_and_csi_invariants_detect_wrong_pairing():
    energy, greedy = _row(), _row()
    _audit_mapping_preserved({("a", 0): energy}, {("a", 0): greedy}, "mapping")
    _audit_csi_identical({("a", 0): energy}, {("a", 0): greedy}, "channel")
    greedy["mapping_high_notch_fraction"] = "0.3"
    with pytest.raises(ValueError, match="mapping changed"):
        _audit_mapping_preserved({("a", 0): energy}, {("a", 0): greedy}, "mapping")
    greedy["mapping_high_notch_fraction"] = "0.2"
    greedy["residual_receiver_csi_nmse"] = "0.06"
    with pytest.raises(ValueError, match="CSI diagnostic differs"):
        _audit_csi_identical({("a", 0): energy}, {("a", 0): greedy}, "channel")


def test_baseline_energy_nan_and_actual_residual_fields_are_required():
    row = _row()
    _audit_residual({"samples": {("a", 0): row}})
    row["residual_symbol_mse"] = "nan"
    with pytest.raises(ValueError, match="non-finite"):
        _audit_residual({"samples": {("a", 0): row}})
    row = _row()
    for field in (
        "energy_proxy_initial", "energy_proxy_final",
        "energy_changed_symbol_count", "energy_changed_symbol_fraction",
    ):
        row[field] = "nan"
    _audit_baseline_energy({"samples": {("a", 0): row}}, "greedy")
    row["energy_proxy_final"] = "1"
    with pytest.raises(ValueError, match="baseline mode must have NaN"):
        _audit_baseline_energy({"samples": {("a", 0): row}}, "greedy")
    row["energy_proxy_final"] = "inf"
    with pytest.raises(ValueError, match="baseline mode must have NaN"):
        _audit_baseline_energy({"samples": {("a", 0): row}}, "greedy")


def test_unchanged_energy_mapping_must_match_greedy_psnr():
    energy, greedy = _row(), _row()
    energy.update({
        "energy_changed_symbol_count": "0", "energy_changed_symbol_fraction": "0",
        "energy_proxy_final": "10", "residual_estimated_energy_proxy": "10",
        "psnr": "28.0",
    })
    greedy["psnr"] = "27.9"
    with pytest.raises(ValueError, match="unchanged mapping differs"):
        _audit_unchanged_psnr({("a", 0): energy}, {("a", 0): greedy}, "no change")


def test_perceptual_metrics_use_paired_image_differences():
    energy = {("a", 0): {**_row(), "ms_ssim": "0.9", "lpips": "0.1"},
              ("b", 0): {**_row(), "ms_ssim": "0.8", "lpips": "0.2"}}
    greedy = {("a", 0): {**_row(), "ms_ssim": "0.8", "lpips": "0.2"},
              ("b", 0): {**_row(), "ms_ssim": "0.7", "lpips": "0.3"}}
    left, right = {"samples": energy}, {"samples": greedy}
    assert _paired_numeric(left, right, "ms_ssim", "paired", 1000)["mean"] == pytest.approx(0.1)
    assert _paired_numeric(left, right, "lpips", "paired", 1000)["mean"] == pytest.approx(-0.1)
    del greedy[("b", 0)]["lpips"]
    with pytest.raises(ValueError, match="missing or invalid lpips"):
        _paired_numeric(left, right, "lpips", "paired", 1000)


def test_kodak_validation_uses_matched_full_image_protocol():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "eval_kodak.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modes = config["evaluation"]["mapper_modes"]
    _audit_protocol(config, modes, 20, "kodak")
    config["data"]["test"]["crop_size"] = 256
    with pytest.raises(ValueError, match="full-resolution"):
        _audit_protocol(config, modes, 20, "kodak")


def _write_complete_run(directory, modes=BASELINE_MODES):
    """Two independent images in each condition, with paired mapper draws."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "eval_token_energy.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluation"].update({"mapper_modes": list(modes), "channel_realizations": 1})
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    rows, summaries = [], []
    bonuses = {"sequential": 0, "random": 0.2, "greedy": 0.5,
               "greedy_token_random": 0.4, ENERGY_MODE: 0.8}
    for csi, notch in CONDITIONS:
        for mode in modes:
            samples = []
            for image_index, image in enumerate(("a", "b")):
                bonus = bonuses[mode]
                row = {
                    **_row(), "mapper": mode, "csi": csi, "image_id": image,
                    "channel_draw": 0, "mapper_csi_source": "estimated",
                    "receiver_csi_source": "temporal_mean", "snr_db": 10,
                    "n_taps": 8, "tau": 2, "notch_start": 5 if notch else 0,
                    "notch_width": 8 if notch else 0,
                    "notch_attenuation_db": 20 if notch else 0,
                    "psnr": 27 + image_index + bonus,
                    "ms_ssim": 0.8 + 0.01 * bonus,
                    "lpips": 0.2 - 0.01 * bonus,
                    "residual_symbol_mse": 0.4 - 0.02 * bonus,
                    "residual_sds_weighted_token_mse": 0.3 - 0.01 * bonus,
                }
                if mode != ENERGY_MODE:
                    row.update({field: "nan" for field in ENERGY_FIELDS})
                    row["residual_estimated_energy_proxy"] = 10 if mode == "greedy" else 11
                if mode in ("sequential", "random"):
                    # Global baselines need not preserve greedy's token-slot allocation.
                    row["mapping_high_notch_fraction"] = "0.6"
                samples.append(row)
            rows.extend(samples)
            psnrs = [float(row["psnr"]) for row in samples]
            summaries.append({
                "mapper": mode, "csi": csi, "spectral_notch": {"width": 8} if notch else None,
                "snr_db": 10, "n_taps": 8, "tau": 2, "count": 2,
                "outage_threshold": 25, "psnr_mean": fmean(psnrs),
                "psnr_p5": min(psnrs) + 0.05 * (max(psnrs) - min(psnrs)),
                "outage_probability": 0,
                "ms_ssim_mean": fmean(float(row["ms_ssim"]) for row in samples),
                "lpips_mean": fmean(float(row["lpips"]) for row in samples),
                "mapper_csi_source": "estimated", "receiver_csi_source": "temporal_mean",
            })
    with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    return config


@pytest.mark.parametrize("suite,modes,result_count,comparison_count", [
    ("pair", PAIR_MODES, 4, 2), ("baselines", BASELINE_MODES, 10, 14),
])
def test_cli_reports_complete_pair_and_five_method_suites(
    tmp_path, monkeypatch, suite, modes, result_count, comparison_count,
):
    _write_complete_run(tmp_path, modes)
    output_json, output_csv = tmp_path / "comparison.json", tmp_path / "table.csv"
    monkeypatch.setattr("sys.argv", [
        "compare_token_energy", "--run-dir", str(tmp_path), "--suite", suite,
        "--expected-count", "2", "--expected-draws", "1", "--resamples", "1000",
        "--output-json", str(output_json), "--output-csv", str(output_csv),
    ])
    main()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(report["results"]) == result_count
    assert len(report["comparisons"]) == comparison_count
    assert report["suite"] == suite
    for comparison in report["comparisons"]:
        paired = comparison["paired"]
        assert set(paired["metric_deltas"]) == {"ms_ssim", "lpips"}
        assert set(paired["residual_deltas"]) == set(ACTUAL_RESIDUAL_FIELDS)
        assert paired["psnr_delta_db"]["images"] == 2
    if suite == "baselines":
        assert {
            (comparison["left"], comparison["right"])
            for comparison in report["comparisons"]
        } == {
            *((ENERGY_MODE, baseline) for baseline in BASELINE_MODES if baseline != ENERGY_MODE),
            ("greedy", "sequential"), ("greedy", "random"),
            ("greedy_token_random", "greedy"),
        }
    with output_csv.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == result_count
    assert {row["mode"] for row in table} == set(modes)


def test_five_method_report_checks_token_random_mapping_and_all_csi(tmp_path):
    _write_complete_run(tmp_path)
    config, runs = _load_run(tmp_path, BASELINE_MODES, 2, 1)
    condition = CONDITIONS[0]
    token_random = runs[("greedy_token_random", condition)]["samples"][("a", 0)]
    token_random["mapping_high_notch_fraction"] = "0.3"
    with pytest.raises(ValueError, match="mapping changed"):
        _build_report(config, runs, BASELINE_MODES, "div2k", 1000)
    token_random["mapping_high_notch_fraction"] = "0.2"
    sequential = runs[("sequential", condition)]["samples"][("a", 0)]
    sequential["residual_mapper_csi_nmse"] = "0.9"
    with pytest.raises(ValueError, match="CSI diagnostic differs"):
        _build_report(config, runs, BASELINE_MODES, "div2k", 1000)


@pytest.mark.parametrize("mode", BASELINE_MODES[:-1])
def test_five_method_report_requires_nan_energy_metadata_for_every_baseline(tmp_path, mode):
    _write_complete_run(tmp_path)
    config, runs = _load_run(tmp_path, BASELINE_MODES, 2, 1)
    runs[(mode, CONDITIONS[0])]["samples"][("a", 0)]["energy_proxy_final"] = "0"
    with pytest.raises(ValueError, match="baseline mode must have NaN"):
        _build_report(config, runs, BASELINE_MODES, "div2k", 1000)


def test_load_run_rejects_wrong_modes_missing_samples_and_nonfinite_summary(tmp_path):
    _write_complete_run(tmp_path)
    with pytest.raises(ValueError, match="mapper mode list"):
        _load_run(tmp_path, PAIR_MODES, 2, 1)
    samples_path = tmp_path / "samples.csv"
    lines = samples_path.read_text(encoding="utf-8").splitlines()
    samples_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires 2 samples"):
        _load_run(tmp_path, BASELINE_MODES, 2, 1)
    _write_complete_run(tmp_path)
    summary_path = tmp_path / "summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    summaries[0]["psnr_mean"] = "nan"
    summary_path.write_text(json.dumps(summaries), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite psnr_mean"):
        _load_run(tmp_path, BASELINE_MODES, 2, 1)


def _write_double_notch_run(directory):
    _write_complete_run(directory)
    config_path = directory / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    notch_spec = {
        "count": 2, "min_gap": 4, "start": "random",
        "width_range": [8, 12], "attenuation_db_range": [15, 25],
    }
    config["evaluation"]["csi_nmse_db"] = [-10, -5]
    config["evaluation"]["spectral_notches"] = [None, notch_spec]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with (directory / "samples.csv").open(encoding="utf-8", newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    original_summaries = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows, summaries = [], []
    intervals = [
        {"start": 4, "width": 10, "attenuation_db": 20.0},
        {"start": 28, "width": 10, "attenuation_db": 22.0},
    ]
    for csi, notched in DOUBLE_NOTCH_CONDITIONS:
        for original in original_rows:
            if (int(original["notch_width"]) > 0) != notched:
                continue
            row = dict(original)
            row["csi"] = csi
            row.update({
                "notch_count": 2 if notched else 0,
                "notch_total_width": 20 if notched else 0,
                "notches_json": json.dumps(intervals if notched else [],
                                           sort_keys=True, separators=(",", ":")),
                "notch_start": -1, "notch_width": 0,
                "notch_attenuation_db": 0,
            })
            for index in (1, 2):
                interval = intervals[index - 1] if notched else None
                row.update({
                    f"notch_{index}_start": interval["start"] if interval else -1,
                    f"notch_{index}_width": interval["width"] if interval else 0,
                    f"notch_{index}_attenuation_db": (
                        interval["attenuation_db"] if interval else 0
                    ),
                })
            rows.append(row)
        for original in original_summaries:
            if (original["spectral_notch"] is not None) != notched:
                continue
            summaries.append({
                **original, "csi": csi,
                "spectral_notch": notch_spec if notched else None,
            })
    with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")


def test_double_notch_cli_reports_all_csi_and_notch_conditions(tmp_path, monkeypatch):
    _write_double_notch_run(tmp_path)
    output_json, output_csv = tmp_path / "comparison.json", tmp_path / "table.csv"
    monkeypatch.setattr("sys.argv", [
        "compare_token_energy", "--run-dir", str(tmp_path),
        "--protocol", "double_notch", "--suite", "baselines",
        "--expected-count", "2", "--expected-draws", "1", "--resamples", "1000",
        "--output-json", str(output_json), "--output-csv", str(output_csv),
    ])
    main()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(report["results"]) == 20
    assert len(report["comparisons"]) == 28
    assert set(report["target_status_by_condition"]) == {"nmse_-10db", "nmse_-5db"}
    with output_csv.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 20
    assert {(row["csi"], int(row["notch_count"])) for row in table} == {
        ("nmse_-10db", 0), ("nmse_-10db", 2),
        ("nmse_-5db", 0), ("nmse_-5db", 2),
    }


def test_double_notch_report_rejects_bad_realized_geometry(tmp_path):
    _write_double_notch_run(tmp_path)
    samples_path = tmp_path / "samples.csv"
    with samples_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["notch_count"] == "2")
    row["notch_2_start"] = "15"
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="notch_2_start disagrees"):
        _load_run(tmp_path, BASELINE_MODES, 2, 1, DOUBLE_NOTCH_CONDITIONS)
