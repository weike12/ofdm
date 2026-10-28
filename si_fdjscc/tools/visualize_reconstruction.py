"""Export paired Kodak reconstructions and identical full-image/ROI panels."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

if TYPE_CHECKING:
    import torch

METHODS = ("greedy", "greedy_token_random", "greedy_token_energy")
LABELS = ("Original", "Greedy", "Greedy + token random", "Greedy + Token Energy")
DEFAULT_IMAGES = ("kodim01", "kodim07", "kodim04")
# Candidate boxes only: inspect originals with --prepare-only before inference.
DEFAULT_ROI = (0.30, 0.25, 0.70, 0.65)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roi_pixels(roi, size):
    values = tuple(float(value) for value in roi)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("ROI requires four finite normalized values: x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    width, height = size
    return (math.floor(x0 * width), math.floor(y0 * height),
            math.ceil(x1 * width), math.ceil(y1 * height))


def _draw_seed(base_seed: int, dataset_index: int, draw: int) -> int:
    if dataset_index < 0 or draw < 0:
        raise ValueError("Dataset index and channel draw must be non-negative")
    return int(base_seed) + dataset_index * 1_000_000 + draw * 10_000


def _protocol(config: Mapping[str, Any]):
    test = config["data"]["test"]
    evaluation = config["evaluation"]
    if str(test.get("name", "")).lower() != "kodak" or test.get("crop_size") is not None:
        raise ValueError("Use Kodak full-resolution images without crop_size")
    if int(test.get("batch_size", 1)) != 1 or test.get("shuffle", False) or test.get("drop_last", False):
        raise ValueError("Matching evaluation seeds requires batch_size=1, shuffle=false, drop_last=false")
    if int(evaluation.get("draw_chunk_size", 1)) != 1:
        raise ValueError("Use the Kodak config with draw_chunk_size=1")
    for key in ("snr_db", "csi_nmse_db", "selectivity"):
        if not isinstance(evaluation.get(key), list) or len(evaluation[key]) != 1:
            raise ValueError(f"Qualitative export requires exactly one evaluation.{key} value")
    if evaluation.get("mapper_csi_source", "estimated") != "estimated":
        raise ValueError("Token-internal methods require estimated mapper CSI")
    notches = evaluation.get("spectral_notches", [])
    if len(notches) != 2 or notches[0] is not None or not isinstance(notches[1], Mapping):
        raise ValueError("Expected no-notch and random-notch scenarios, in that order")
    if notches[1].get("random_start", False) and str(notches[1].get("start", "")).lower() != "random":
        raise ValueError("The second scenario must use a random spectral notch")
    return {
        "model": config["model"], "ofdm": config["ofdm"], "mapper": config["mapper"],
        "evaluation": {key: evaluation.get(key) for key in (
            "checkpoint", "seed", "snr_db", "csi_nmse_db", "selectivity",
            "spectral_notches", "mapper_csi_source", "receiver_csi_source",
            "channel_realizations", "draw_chunk_size", "deterministic",
        )},
    }


def _dataset_paths(root: Path):
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    expected = {f"kodim{number:02d}" for number in range(1, 25)}
    if len(paths) != 24 or {path.stem for path in paths} != expected:
        raise ValueError(f"Expected exactly kodim01..kodim24 under {root}; found {len(paths)} images")
    return paths


def _make_selection(config, image_ids, overrides, draw):
    protocol = _protocol(config)
    if not 0 <= draw < int(config["evaluation"].get("channel_realizations", 20)):
        raise ValueError("Draw must be inside the configured channel_realizations range")
    if len(set(image_ids)) != len(image_ids) or not image_ids:
        raise ValueError("Select distinct Kodak image IDs")
    if set(overrides) - set(image_ids):
        raise ValueError("ROI override names must belong to the selected images")
    root = Path(config["data"]["test"]["root"]).expanduser()
    paths = _dataset_paths(root)
    lookup = {path.stem: (index, path) for index, path in enumerate(paths)}
    images = []
    for image_id in image_ids:
        if image_id not in lookup:
            raise ValueError(f"Unknown Kodak image: {image_id}")
        index, path = lookup[image_id]
        with Image.open(path) as image:
            size = image.size
        roi = list(overrides.get(image_id, DEFAULT_ROI))
        images.append({
            "image_id": image_id, "path": path.relative_to(root).as_posix(),
            "dataset_index": index, "size": list(size), "sha256": _sha256(path),
            "roi": roi, "roi_pixels": list(_roi_pixels(roi, size)),
            "channel_seed": _draw_seed(int(config["evaluation"]["seed"]), index, draw),
        })
    checkpoint = Path(config["evaluation"]["checkpoint"]).expanduser()
    return {
        "schema_version": 1, "channel_draw": draw, "methods": list(METHODS),
        "roi_note": "Candidate boxes: confirm on original preview before inspecting reconstructions.",
        "protocol": protocol, "dataset_order": [p.relative_to(root).as_posix() for p in paths],
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "images": images,
    }


def _verify_selection(config, selection):
    if selection.get("schema_version") != 1 or selection.get("methods") != list(METHODS):
        raise ValueError("Unsupported selection schema or method list")
    if selection.get("protocol") != _protocol(config):
        raise ValueError("Config protocol changed after preparing the original selection")
    ids = [item["image_id"] for item in selection["images"]]
    overrides = {item["image_id"]: item["roi"] for item in selection["images"]}
    current = _make_selection(config, ids, overrides, int(selection["channel_draw"]))
    if current["dataset_order"] != selection.get("dataset_order"):
        raise ValueError("Full Kodak ordering changed; evaluation seeds would differ")
    for old, new in zip(selection["images"], current["images"]):
        for key in ("sha256", "size", "dataset_index", "path", "channel_seed"):
            if old.get(key) != new[key]:
                raise ValueError(f"Original selection changed: {new['image_id']}/{key}")
    if selection.get("checkpoint_sha256") and current["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preparing the original selection")
    return current


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _paste_image(canvas, image, box, roi=None, nearest=False):
    x, y, width, height = box
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    thumb = ImageOps.contain(image, (width, height), method=resample)
    left, top = x + (width - thumb.width) // 2, y + (height - thumb.height) // 2
    canvas.paste(thumb, (left, top))
    if roi is not None:
        x0, y0, x1, y1 = _roi_pixels(roi, image.size)
        sx, sy = thumb.width / image.width, thumb.height / image.height
        ImageDraw.Draw(canvas).rectangle(
            (left + x0 * sx, top + y0 * sy, left + (x1 - 1) * sx, top + (y1 - 1) * sy),
            outline="red", width=3,
        )


def _rgb_png(array):
    """Use a fixed [0,1] display range; never normalize each method separately."""
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB")


def _write_originals(config, selection, output):
    root = Path(config["data"]["test"]["root"]).expanduser()
    (output / "originals").mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    preview = Image.new("RGB", (400 * len(selection["images"]), 470), "white")
    draw = ImageDraw.Draw(preview)
    for column, item in enumerate(selection["images"]):
        with Image.open(root / item["path"]) as handle:
            image = handle.convert("RGB")
        image.save(output / "originals" / f"{item['image_id']}.png")
        image.crop(tuple(item["roi_pixels"])).save(output / "crops" / f"{item['image_id']}_original.png")
        draw.text((column * 400 + 8, 8), f"{item['image_id']}  {image.width}x{image.height}", fill="black", font=_font(20))
        _paste_image(preview, image, (column * 400 + 8, 42, 384, 384), item["roi"])
        draw.text((column * 400 + 8, 435), f"ROI: {tuple(item['roi_pixels'])}", fill="black", font=_font(15))
    preview.save(output / "selection_preview.png")
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")


def _comparison_panel(images, item, scene, rows, output):
    cell_width, full_height, zoom_height = 512, 384, 320
    panel = Image.new("RGB", (4 * cell_width, 880), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), f"{item['image_id']} | {scene} | channel seed {item['channel_seed']} | full-image metrics", fill="black", font=_font(22))
    for column, (image, label) in enumerate(zip(images, LABELS)):
        x = column * cell_width
        draw.text((x + 10, 46), label, fill="black", font=_font(20))
        _paste_image(panel, image, (x + 8, 80, cell_width - 16, full_height), item["roi"])
        if column:
            row = rows[column - 1]
            text = f"PSNR {row['psnr']:.3f} dB | MS-SSIM {row['ms_ssim']:.5f} | LPIPS {row['lpips']:.5f}"
        else:
            text = "Same red ROI for all four columns"
        draw.text((x + 8, 478), text, fill="black", font=_font(14))
        crop = image.crop(tuple(item["roi_pixels"]))
        _paste_image(panel, crop, (x + 8, 520, cell_width - 16, zoom_height), nearest=True)
    draw.text((10, 852), "Bottom row: identical ROI, enlarged with nearest-neighbor; native-resolution PNG crops are also saved.", fill="black", font=_font(17))
    panel.save(output)


def _overview(entries, scene, output):
    sheet = Image.new("RGB", (4 * 384, 430 * len(entries) + 44), "white")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(LABELS):
        draw.text((column * 384 + 8, 8), label, fill="black", font=_font(18))
    for row_index, (item, images, rows) in enumerate(entries):
        top = 44 + row_index * 430
        for column, image in enumerate(images):
            _paste_image(sheet, image, (column * 384 + 6, top + 25, 372, 350), item["roi"])
            caption = item["image_id"] if column == 0 else f"{rows[column-1]['psnr']:.2f} dB | {rows[column-1]['ms_ssim']:.4f} | {rows[column-1]['lpips']:.4f}"
            draw.text((column * 384 + 8, top + 385), caption, fill="black", font=_font(16))
    sheet.save(output / f"overview_{scene}.png")


def save_reconstruction_comparison(images: Sequence[torch.Tensor], labels: Sequence[str], output_path: str) -> None:
    """Keep the original helper API for callers supplying tensor images."""
    import matplotlib.pyplot as plt
    if len(images) != len(labels) or not images:
        raise ValueError("images and labels must have equal non-zero lengths")
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for axis, tensor, label in zip(axes[0], images, labels):
        tensor = tensor[0] if tensor.ndim == 4 else tensor
        axis.imshow(tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_reconstructions(config, selection, output, device_override):
    import torch
    from ..evaluation.metrics import MetricAccumulator
    from ..train.common import build_system, load_model_checkpoint, resolve_device
    from ..utils import save_config_snapshot, set_seed

    evaluation = config["evaluation"]
    checkpoint = Path(evaluation["checkpoint"]).expanduser()
    set_seed(int(evaluation["seed"]), deterministic=bool(evaluation.get("deterministic", True)))
    device = resolve_device(config, device_override)
    model = build_system(config).to(device)
    load_model_checkpoint(model, str(checkpoint), strict=True)
    model.eval()
    snapshot = copy.deepcopy(config)
    snapshot["evaluation"]["mapper_modes"] = list(METHODS)
    snapshot["evaluation"]["output_dir"] = str(output)
    snapshot["visualization"] = {
        "images": [item["image_id"] for item in selection["images"]],
        "channel_draw": selection["channel_draw"], "full_image_inference": True,
    }
    save_config_snapshot(snapshot, output)
    accumulator = MetricAccumulator(compute_msssim=True, compute_lpips=True, device=device)
    root = Path(config["data"]["test"]["root"]).expanduser()
    scenarios = tuple(zip(("no_notch", "notch"), evaluation["spectral_notches"]))
    snr_db = float(evaluation["snr_db"][0])
    nmse_db = evaluation["csi_nmse_db"][0]
    perfect = nmse_db is None or str(nmse_db).lower() == "perfect"
    selectivity = evaluation["selectivity"][0]
    if (int(selectivity["n_taps"]), float(selectivity["tau"])) != (model.channel.channel.n_taps, model.channel.channel.tau):
        from ..comm.multipath_channel import MultipathChannel
        model.channel.channel = MultipathChannel(n_taps=int(selectivity["n_taps"]), tau=float(selectivity["tau"])).to(device)
    if model.channel.modem.n_cp < int(selectivity["n_taps"]) - 1:
        raise ValueError("CP is too short for configured channel taps")
    manifest = {
        "status": "running", "selection": selection,
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics_domain": "clipped float32 full image before uint8 PNG quantization",
        "torch_version": torch.__version__, "device": str(device), "records": [],
        "source_sha256": {
            path.as_posix(): _sha256(path)
            for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for folder in ("reconstructions", "comparisons"):
        (output / folder).mkdir(exist_ok=True)
    overviews = {scene: [] for scene, _ in scenarios}
    fields = ("image_id", "mapper", "scenario", "channel_draw", "channel_seed", "dataset_index", "psnr", "ms_ssim", "lpips")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selection["images"]:
            with Image.open(root / item["path"]) as source:
                original = source.convert("RGB")
            array = np.array(original, dtype=np.uint8, copy=True)
            target = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
            target = target.to(dtype=torch.float32).div(255).unsqueeze(0).to(device)
            for scene, notch in scenarios:
                images, rows, reference = [original], [], None
                record = {"image_id": item["image_id"], "scenario": scene, "channel_seed": item["channel_seed"], "methods": []}
                for mode in METHODS:
                    print(f"[visualization] {item['image_id']} {scene} {mode} seed={item['channel_seed']}", flush=True)
                    with torch.no_grad():
                        reconstruction, auxiliary = model(
                            target, torch.tensor([snr_db], device=device),
                            channel_cfg={"csi_mode": "perfect" if perfect else "imperfect",
                                         "csi_nmse_db": None if perfect else float(nmse_db),
                                         "channel_seed": item["channel_seed"], "spectral_notch": notch,
                                         "mapper_csi_source": "estimated",
                                         "receiver_csi_source": evaluation.get("receiver_csi_source", "estimated")},
                            mapper_mode=mode, return_aux=True, n_channel_draws=1,
                        )
                        draw_aux = auxiliary["draws"][0]
                        signature = {key: draw_aux[key].detach().cpu() for key in ("H", "H_hat", "noise_var", "importance")}
                        signature["symbols_tx"] = auxiliary["symbols_tx"].detach().cpu()
                        permutation = draw_aux["permutation"].symbol_index_grid.detach().cpu()
                        token_ids = auxiliary["complex_metadata"].token_ids.detach().cpu()
                        token_slots = token_ids[permutation.clamp_min(0)]
                        token_slots = torch.where(permutation >= 0, token_slots, -torch.ones_like(token_slots))
                        resolved = draw_aux.get("spectral_notch")
                        if reference is None:
                            reference = signature
                            reference["token_slots"] = token_slots
                            record["resolved_notch"] = resolved
                        elif any(not torch.equal(signature[key], reference[key]) for key in signature) or resolved != record["resolved_notch"]:
                            raise RuntimeError("Paired channel/CSI/source symbols differ across methods")
                        if not torch.equal(token_slots, reference["token_slots"]):
                            raise RuntimeError("Token-internal permutation changed greedy token slots")
                        if not bool(torch.isfinite(reconstruction).all()):
                            raise RuntimeError("Non-finite reconstruction")
                        if reconstruction.shape != target.shape:
                            raise RuntimeError("Reconstruction must preserve full original image dimensions")
                        accumulator.add(reconstruction, target, [item["image_id"]], metadata={
                            "mapper": mode, "scenario": scene, "channel_draw": selection["channel_draw"],
                            "channel_seed": item["channel_seed"], "dataset_index": item["dataset_index"],
                        })
                        row = accumulator.rows[-1]
                        rgb = _rgb_png(reconstruction[0].detach().float().cpu().permute(1, 2, 0).numpy())
                    full_folder, crop_folder = output / "reconstructions" / scene, output / "crops" / scene
                    full_folder.mkdir(exist_ok=True); crop_folder.mkdir(exist_ok=True)
                    filename = f"{item['image_id']}_{mode}.png"
                    rgb.save(full_folder / filename)
                    rgb.crop(tuple(item["roi_pixels"])).save(crop_folder / filename)
                    writer.writerow(row); handle.flush()
                    images.append(rgb); rows.append(row)
                    record["methods"].append({**row, "token_slots_match_greedy": True,
                        "permutation_sha256": hashlib.sha256(permutation.numpy().tobytes()).hexdigest()})
                    del reconstruction, auxiliary, draw_aux, signature
                _comparison_panel(images, item, scene, rows, output / "comparisons" / f"{item['image_id']}_{scene}.png")
                overviews[scene].append((item, images, rows))
                manifest["records"].append(record)
                manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            del target
    for scene, entries in overviews.items():
        _overview(entries, scene, output)
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed: {output} | {len(accumulator.rows)} paired reconstructions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_kodak.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images", nargs="+", metavar="KODIM")
    parser.add_argument("--roi", action="append", default=[], metavar="KODIM=x0,y0,x1,y1")
    parser.add_argument("--draw", type=int, help="Preselected draw, default 0; never select by reconstruction score")
    parser.add_argument("--prepare-only", action="store_true", help="Export original previews/ROI selection without model inference")
    parser.add_argument("--selection", help="Reuse and verify selection.json prepared before inspecting reconstruction outcomes")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.selection:
        if args.images or args.roi or args.draw is not None:
            parser.error("--selection already fixes images, ROI and draw")
        selection = _verify_selection(config, json.loads(Path(args.selection).read_text(encoding="utf-8")))
    else:
        overrides = {}
        for spec in args.roi:
            name, separator, coordinates = spec.partition("=")
            name = Path(name).stem
            if not separator or name in overrides:
                parser.error("Use each ROI once: --roi kodim01=x0,y0,x1,y1")
            try:
                overrides[name] = [float(value) for value in coordinates.split(",")]
            except ValueError:
                parser.error("ROI coordinates must be normalized numbers")
        selection = _make_selection(config, [Path(name).stem for name in (args.images or DEFAULT_IMAGES)], overrides,
                                    0 if args.draw is None else args.draw)
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be new or empty")
    if not args.prepare_only and not Path(config["evaluation"]["checkpoint"]).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint missing: {config['evaluation']['checkpoint']}")
    output.mkdir(parents=True, exist_ok=True)
    _write_originals(config, selection, output)
    if args.prepare_only:
        print(f"Original preview: {output / 'selection_preview.png'}", flush=True)
        print(f"Fixed selection: {output / 'selection.json'}", flush=True)
        return
    _run_reconstructions(config, selection, output, args.device)


if __name__ == "__main__":
    main()
