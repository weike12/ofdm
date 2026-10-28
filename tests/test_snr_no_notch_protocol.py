"""The five-SNR sweep must retain every mode without merging its draws."""

import csv
import json
from pathlib import Path
from statistics import fmean

import pytest
import yaml

from si_fdjscc.evaluation.compare_token_energy import (
    BASELINE_MODES,
    DOUBLE_NOTCH_PAIR_FIELDS,
    ENERGY_FIELDS,
    ENERGY_MODE,
    PAIR_FIELDS,
    SNR_NO_NOTCH_CONDITIONS,
    _audit_protocol,
    _build_report,
    _load_run,
    main,
)


SNR_VALUES = (10, 5, 0, -5, -10)


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_snr_run(directory):
    """Two images share the same image/draw IDs at every SNR and mode."""
    config_path = Path(__file__).resolve().parents[1] / "configs/eval_snr_no_notch_div2k.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluation"].update({
        "snr_db": list(SNR_VALUES),
        "csi_nmse_db": [-10],
        "spectral_notches": [None],
        "mapper_modes": list(BASELINE_MODES),
        "channel_realizations": 1,
    })
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    bonuses = {
        "sequential": 0.0, "random": 0.2, "greedy": 0.5,
        "greedy_token_random": 0.4, ENERGY_MODE: 0.8,
    }
    rows, summaries = [], []
    for snr in SNR_VALUES:
        for mode in BASELINE_MODES:
            samples = []
            for image_index, image in enumerate(("a", "b")):
                bonus = bonuses[mode]
                row = {
                    **{field: 0 for field in PAIR_FIELDS + DOUBLE_NOTCH_PAIR_FIELDS},
                    "mapper": mode, "csi": "nmse_-10db", "image_id": image,
                    "channel_draw": 0, "snr_db": snr, "n_taps": 8, "tau": 2.0,
                    "mapper_csi_source": "estimated", "receiver_csi_source": "temporal_mean",
                    "notch_start": -1, "notches_json": "[]",
                    "notch_1_start": -1, "notch_2_start": -1,
                    "mapping_symbol_count": 10, "mapping_high_notch_fraction": 0,
                    "mapping_notch_fraction": 0,
                    "high_sensitivity_carrier_correlation": 0.4,
                    "psnr": 23 + 0.2 * snr + image_index + bonus,
                    "ms_ssim": 0.6 + 0.01 * snr + 0.01 * bonus,
                    "lpips": 0.3 - 0.005 * snr - 0.01 * bonus,
                    "energy_proxy_initial": 10, "energy_proxy_final": 9,
                    "energy_changed_symbol_count": 2, "energy_changed_symbol_fraction": 0.2,
                    "residual_symbol_mse": 0.4 - 0.02 * bonus,
                    "residual_sds_weighted_token_mse": 0.3 - 0.01 * bonus,
                    "residual_estimated_energy_proxy": 9,
                    "residual_mapper_csi_nmse": 0.1,
                    "residual_receiver_csi_nmse": 0.04,
                }
                if mode != ENERGY_MODE:
                    row.update({field: "nan" for field in ENERGY_FIELDS})
                    row["residual_estimated_energy_proxy"] = 10 if mode == "greedy" else 11
                samples.append(row)
            psnrs = [sample["psnr"] for sample in samples]
            summaries.append({
                "mapper": mode, "csi": "nmse_-10db", "spectral_notch": None,
                "snr_db": snr, "n_taps": 8, "tau": 2.0, "count": len(samples),
                "outage_threshold": 25.0, "psnr_mean": fmean(psnrs),
                "psnr_p5": min(psnrs) + 0.05 * (max(psnrs) - min(psnrs)),
                "outage_probability": fmean(float(value < 25.0) for value in psnrs),
                "ms_ssim_mean": fmean(sample["ms_ssim"] for sample in samples),
                "lpips_mean": fmean(sample["lpips"] for sample in samples),
                "mapper_csi_source": "estimated", "receiver_csi_source": "temporal_mean",
            })
            rows.extend(samples)
    _write_rows(directory / "samples.csv", rows)
    (directory / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    return config


def test_snr_cli_retains_25_groups_and_correct_outage(tmp_path, monkeypatch, capsys):
    _write_snr_run(tmp_path)
    output_json, output_csv = tmp_path / "comparison.json", tmp_path / "summary.csv"
    monkeypatch.setattr("sys.argv", [
        "compare_token_energy", "--run-dir", str(tmp_path),
        "--suite", "baselines", "--protocol", "snr_sweep",
        "--expected-count", "2", "--expected-draws", "1", "--resamples", "1000",
        "--output-json", str(output_json), "--output-csv", str(output_csv),
    ])
    main()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(report["results"]) == 25
    assert len(report["comparisons"]) == 35
    assert {comparison["snr_db"] for comparison in report["comparisons"]} == set(SNR_VALUES)
    table = _read_rows(output_csv)
    assert [float(row["snr_db"]) for row in table] == [
        snr for snr in SNR_VALUES for _ in BASELINE_MODES
    ]
    assert {(row["csi"], row["notch"], row["notch_count"]) for row in table} == {
        ("nmse_-10db", "False", "0")
    }
    sequential = {float(row["snr_db"]): row for row in table if row["mode"] == "sequential"}
    # A PSNR exactly equal to 25 dB is not an outage.
    assert float(sequential[5]["outage_probability"]) == 0.5
    assert float(sequential[0]["outage_probability"]) == 1.0
    from scripts.format_snr_no_notch_results import print_table
    capsys.readouterr()
    print_table(output_csv)
    text = capsys.readouterr().out
    assert text.count("===== DIV2K | SNR") == 5
    section_positions = [text.index(f"SNR {snr} dB") for snr in SNR_VALUES]
    assert section_positions == sorted(section_positions)
    assert "50.000" in text and "100.000" in text


@pytest.mark.parametrize("field,value", [
    ("snr_db", [10, 5, 0, -5]),
    ("csi_nmse_db", [-10, -5]),
    ("spectral_notches", [None, {"width": 10}]),
])
def test_snr_protocol_rejects_changed_requested_conditions(tmp_path, field, value):
    config = _write_snr_run(tmp_path)
    _audit_protocol(config, BASELINE_MODES, 1, "div2k", "snr_sweep")
    config["evaluation"][field] = value
    with pytest.raises(ValueError, match=field):
        _audit_protocol(config, BASELINE_MODES, 1, "div2k", "snr_sweep")


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra_snr", "extra_csi", "notch"])
def test_snr_loader_rejects_incomplete_or_unexpected_summary_groups(tmp_path, change):
    _write_snr_run(tmp_path)
    path = tmp_path / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "extra_snr":
        rows[0]["snr_db"] = -15
    elif change == "extra_csi":
        rows[0]["csi"] = "nmse_-5db"
    else:
        rows[0]["spectral_notch"] = {"width": 10}
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_run(tmp_path, BASELINE_MODES, 2, 1, SNR_NO_NOTCH_CONDITIONS)


@pytest.mark.parametrize("change", ["sample_snr", "sample_csi", "sample_notch", "notch_metadata"])
def test_snr_loader_rejects_sample_condition_mismatches(tmp_path, change):
    _write_snr_run(tmp_path)
    path = tmp_path / "samples.csv"
    rows = _read_rows(path)
    if change == "sample_snr":
        rows[0]["snr_db"] = "7"
    elif change == "sample_csi":
        rows[0]["csi"] = "nmse_-5db"
    elif change == "sample_notch":
        rows[0]["notch_count"] = "1"
    else:
        # A false scenario label must not hide an applied notch.
        rows[0]["notch_total_width"] = "10"
    _write_rows(path, rows)
    with pytest.raises(ValueError):
        _load_run(tmp_path, BASELINE_MODES, 2, 1, SNR_NO_NOTCH_CONDITIONS)


@pytest.mark.parametrize("dataset", ["div2k", "kodak"])
def test_published_snr_configs_match_the_requested_protocol(dataset):
    path = Path(__file__).resolve().parents[1] / f"configs/eval_snr_no_notch_{dataset}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["evaluation"]["snr_db"] == list(SNR_VALUES)
    assert config["evaluation"]["csi_nmse_db"] == [-10]
    assert config["evaluation"]["spectral_notches"] == [None]
    _audit_protocol(config, BASELINE_MODES, 20, dataset, "snr_sweep")


@pytest.mark.parametrize("change,error", [
    ("image", "image/draw keys differ"),
    ("csi", "CSI diagnostic differs"),
])
@pytest.mark.parametrize("snr", [5, -5, -10])
def test_snr_report_rejects_changed_images_or_channel_csi_between_snrs(tmp_path, change, error, snr):
    _write_snr_run(tmp_path)
    config, runs = _load_run(tmp_path, BASELINE_MODES, 2, 1, SNR_NO_NOTCH_CONDITIONS)
    rows = runs[("sequential", (snr, "nmse_-10db", False))]["samples"]
    if change == "image":
        row = rows.pop(("a", 0))
        row["image_id"] = "different_image"
        rows[("different_image", 0)] = row
    else:
        rows[("a", 0)]["residual_mapper_csi_nmse"] = "0.2"
    with pytest.raises(ValueError, match=error):
        _build_report(config, runs, BASELINE_MODES, "div2k", 1000, SNR_NO_NOTCH_CONDITIONS)
