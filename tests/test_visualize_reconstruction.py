"""Qualitative exports preserve evaluation identity and original image pixels."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

from si_fdjscc.tools import visualize_reconstruction as visualization


@pytest.fixture
def kodak_config(tmp_path):
    root = tmp_path / "kodak"
    root.mkdir()
    width, height = 96, 64
    y, x = np.indices((height, width))
    for number in range(1, 25):
        # A nested file makes the full sorted index differ from image number - 1.
        folder = root / "00-first" if number == 7 else root
        folder.mkdir(exist_ok=True)
        pixels = np.stack((x * 2, y * 3, (x + y + number) % 256), axis=-1)
        Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(
            folder / f"kodim{number:02d}.png"
        )
    return {
        "device": "cpu",
        "data": {"test": {"name": "Kodak", "root": str(root), "batch_size": 1}},
        "model": {"backbone": {"type": "tiny", "latent_channels": 8}},
        "ofdm": {"channel_taps": 8, "tau": 2.0},
        "mapper": {"mode": "greedy"},
        "evaluation": {
            "checkpoint": str(tmp_path / "missing.pt"),
            "seed": 3000,
            "deterministic": True,
            "snr_db": [10],
            "csi_nmse_db": [-10],
            "selectivity": [{"n_taps": 8, "tau": 2.0}],
            "spectral_notches": [
                None,
                {"start": "random", "width_range": [8, 12],
                 "attenuation_db_range": [15, 25]},
            ],
            "channel_realizations": 20,
            "draw_chunk_size": 1,
            "mapper_csi_source": "estimated",
            "receiver_csi_source": "temporal_mean",
        },
    }


def test_full_dataset_index_and_seed_do_not_depend_on_selected_order(kodak_config):
    ids = ["kodim01", "kodim07", "kodim04"]
    first = visualization._make_selection(kodak_config, ids, {}, draw=2)
    second = visualization._make_selection(kodak_config, ids[::-1], {}, draw=2)
    by_id = {item["image_id"]: item for item in first["images"]}
    assert by_id == {item["image_id"]: item for item in second["images"]}
    root = Path(kodak_config["data"]["test"]["root"])
    full_paths = sorted(root.rglob("*.png"))
    for item in first["images"]:
        expected_index = next(index for index, path in enumerate(full_paths)
                              if path.stem == item["image_id"])
        assert item["dataset_index"] == expected_index
        assert item["channel_seed"] == 3000 + expected_index * 1_000_000 + 2 * 10_000
    assert by_id["kodim07"]["dataset_index"] == 0
    assert by_id["kodim01"]["dataset_index"] != 0


@pytest.mark.parametrize("roi", [
    (0, 0, 1), (0, 0, 1, 1, 1), (float("nan"), 0, 1, 1),
    (0, 0, float("inf"), 1), (-0.01, 0, 1, 1), (0, 0, 1.01, 1),
    (0, -0.01, 1, 1), (0, 0, 1, 1.01), (0.5, 0, 0.5, 1),
    (0.6, 0, 0.5, 1), (0, 0.5, 1, 0.5),
])
def test_invalid_and_out_of_bounds_roi_rejected(roi):
    with pytest.raises(ValueError, match="ROI"):
        visualization._roi_pixels(roi, (96, 64))


def test_fractional_roi_uses_enclosing_pixels_without_exceeding_image():
    assert visualization._roi_pixels((0, 0, 1, 1), (96, 64)) == (0, 0, 96, 64)
    assert visualization._roi_pixels((0.1, 0.2, 0.9, 0.8), (11, 7)) == (1, 1, 10, 6)


@pytest.mark.parametrize("field,value,error", [
    ("sha256", "0" * 64, "sha256"),
    ("size", [97, 64], "size"),
    ("dataset_index", 23, "dataset_index"),
    ("channel_seed", 123, "channel_seed"),
])
def test_saved_selection_rejects_changed_image_metadata(kodak_config, field, value, error):
    selection = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    selection["images"][0][field] = value
    with pytest.raises(ValueError, match=error):
        visualization._verify_selection(kodak_config, selection)


def test_saved_selection_rejects_changed_source_content_and_dimensions(kodak_config):
    selection = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    root = Path(kodak_config["data"]["test"]["root"])
    path = root / selection["images"][0]["path"]
    with Image.open(path) as original:
        changed = np.array(original)
    changed[0, 0] = [255, 255, 255]
    Image.fromarray(changed).save(path)
    with pytest.raises(ValueError, match="sha256"):
        visualization._verify_selection(kodak_config, selection)
    resized = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    Image.new("RGB", (97, 64)).save(path)
    with pytest.raises(ValueError, match="Original selection changed"):
        visualization._verify_selection(kodak_config, resized)


def test_saved_selection_rejects_dataset_order_and_protocol_changes(kodak_config):
    selection = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    selection["dataset_order"].reverse()
    with pytest.raises(ValueError, match="ordering changed"):
        visualization._verify_selection(kodak_config, selection)
    selection = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    config = deepcopy(kodak_config)
    config["evaluation"]["seed"] += 1
    with pytest.raises(ValueError, match="Config protocol changed"):
        visualization._verify_selection(config, selection)


def test_saved_selection_rejects_replaced_checkpoint(kodak_config):
    checkpoint = Path(kodak_config["evaluation"]["checkpoint"])
    checkpoint.write_bytes(b"frozen checkpoint bytes")
    selection = visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        visualization._verify_selection(kodak_config, selection)


def test_rgb_export_has_fixed_scale_and_crop_matches_full_png(tmp_path):
    values = np.array([
        [[-0.1, 0, 0.5], [0.1, 0.2, 0.3], [1, 1.1, 0.75]],
        [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
    ], dtype=np.float32)
    image = visualization._rgb_png(values)
    assert image.mode == "RGB"
    expected = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    assert np.array_equal(np.array(image), expected)
    # A constant gray field must stay gray rather than become per-panel normalized.
    gray = np.array(visualization._rgb_png(np.full((2, 3, 3), 0.25)))
    assert np.all(gray == 64)
    full_path, crop_path = tmp_path / "full.png", tmp_path / "crop.png"
    bounds = (1, 0, 3, 2)
    image.save(full_path)
    image.crop(bounds).save(crop_path)
    with Image.open(full_path) as full, Image.open(crop_path) as crop:
        assert np.array_equal(np.array(crop), np.array(full)[0:2, 1:3])


def test_original_exports_and_montages_preserve_native_pixels(kodak_config, tmp_path):
    selection = visualization._make_selection(
        kodak_config, ["kodim01", "kodim07", "kodim04"], {}, 0
    )
    output = tmp_path / "outputs"
    output.mkdir()
    visualization._write_originals(kodak_config, selection, output)
    item = selection["images"][0]
    root = Path(kodak_config["data"]["test"]["root"])
    with Image.open(root / item["path"]) as source:
        original = source.convert("RGB")
    with Image.open(output / "originals" / "kodim01.png") as saved:
        assert saved.size == original.size == (96, 64)
        assert np.array_equal(np.array(saved), np.array(original))
    with Image.open(output / "crops" / "kodim01_original.png") as saved_crop:
        assert np.array_equal(np.array(saved_crop), np.array(original.crop(item["roi_pixels"])))
    images = [original.copy() for _ in range(4)]
    native_arrays = [np.array(image).copy() for image in images]
    rows = [{"psnr": 30.123, "ms_ssim": 0.92345, "lpips": 0.0789} for _ in range(3)]
    panel = output / "comparison.png"
    visualization._comparison_panel(images, item, "no_notch", rows, panel)
    visualization._overview([(item, images, rows)], "no_notch", output)
    for image, pixels in zip(images, native_arrays):
        assert image.size == original.size
        assert np.array_equal(np.array(image), pixels)
    with Image.open(panel) as saved_panel:
        assert saved_panel.size == (2048, 880)
    with Image.open(output / "overview_no_notch.png") as overview:
        assert overview.size == (1536, 474)


def test_prepare_only_cli_works_without_checkpoint_or_torch(kodak_config, tmp_path):
    config_path, output = tmp_path / "config.yaml", tmp_path / "preview"
    config_path.write_text(yaml.safe_dump(kodak_config), encoding="utf-8")
    # Fail even if a future change accidentally makes the preview import torch.
    script = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('prepare-only must not import torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
sys.argv = ['visualize_reconstruction', *sys.argv[1:]]
runpy.run_module('si_fdjscc.tools.visualize_reconstruction', run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, "--config", str(config_path),
         "--output-dir", str(output), "--prepare-only", "--images",
         "kodim04", "kodim01", "kodim07"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    saved = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert [item["image_id"] for item in saved["images"]] == ["kodim04", "kodim01", "kodim07"]
    assert saved["checkpoint_sha256"] is None
    assert (output / "selection_preview.png").is_file()
    assert len(list((output / "originals").glob("*.png"))) == 3
    assert not (output / "reconstructions").exists()


@pytest.mark.parametrize("setting,value", [
    ("batch_size", 2), ("shuffle", True), ("drop_last", True), ("crop_size", 32),
])
def test_selection_rejects_preprocessing_that_breaks_evaluation_pairing(
    kodak_config, setting, value
):
    kodak_config["data"]["test"][setting] = value
    with pytest.raises(ValueError):
        visualization._make_selection(kodak_config, ["kodim01"], {}, 0)
