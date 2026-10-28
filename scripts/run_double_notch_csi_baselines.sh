#!/usr/bin/env bash
# Evaluate five paired mappers at CSI NMSE -10/-5 dB and zero/two notches.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
DATASET="${DATASET:-div2k}"
case "$DATASET" in
    div2k) CONFIG=configs/eval_double_notch_div2k.yaml; EXPECTED_IMAGES=100 ;;
    kodak) CONFIG=configs/eval_double_notch_kodak.yaml; EXPECTED_IMAGES=24 ;;
    *) echo "DATASET must be div2k or kodak" >&2; exit 2 ;;
esac
MODES=(sequential random greedy greedy_token_random greedy_token_energy)

mkdir -p runs
RUN_DIR="$(mktemp -d "$REPO_ROOT/runs/${DATASET}_double_notch_csi_baselines_XXXXXXXX")"
run_started="$(date +%s)"
printf '%s\n' "$RUN_DIR" | tee "runs/${DATASET}_double_notch_csi_baselines_latest.txt"
printf 'running\n' > "$RUN_DIR/status.txt"
trap 'run_rc=$?; printf "%s\n" "$run_rc" > "$RUN_DIR/run.exitcode"; printf "%s\n" "$(($(date +%s) - run_started))" > "$RUN_DIR/run.seconds"; if [[ "$run_rc" -ne 0 ]]; then printf "failed\n" > "$RUN_DIR/status.txt"; echo "Failed; inspect logs in $RUN_DIR" >&2; fi' EXIT
cp "$CONFIG" "$RUN_DIR/base_config.yaml"

run_stage() {
    local stage="$1"
    shift
    local -a stage_codes
    local stage_rc
    local stage_started="$(date +%s)"
    set +e
    "$@" 2>&1 | tee "$RUN_DIR/$stage.log"
    stage_codes=("${PIPESTATUS[@]}")
    set -e
    stage_rc="${stage_codes[0]}"
    if [[ "${stage_codes[1]}" -ne 0 && "$stage_rc" -eq 0 ]]; then stage_rc=1; fi
    printf '%s\n' "$stage_rc" > "$RUN_DIR/$stage.exitcode"
    printf '%s\n' "$(($(date +%s) - stage_started))" > "$RUN_DIR/$stage.seconds"
    return "$stage_rc"
}

run_stage preflight env CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$CONFIG" "$EXPECTED_IMAGES" "$RUN_DIR" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

if sys.version_info < (3, 9):
    raise RuntimeError("Use the project environment with Python >= 3.9")
import torch
import torchvision
import timm
import pytorch_msssim
import lpips
import yaml

config_path, expected_images, output_dir = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
evaluation = config["evaluation"]
notches = evaluation["spectral_notches"]
if evaluation["csi_nmse_db"] != [-10, -5]:
    raise RuntimeError("CSI NMSE sweep must be [-10, -5] dB")
if len(notches) != 2 or notches[0] is not None or notches[1] != {
    "count": 2, "min_gap": 4, "start": "random", "width_range": [8, 12],
    "attenuation_db_range": [15, 25],
}:
    raise RuntimeError("Expected one no-notch and one two-notch scenario")
data_root = Path(config["data"]["test"]["root"]).expanduser()
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
paths = sorted(p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
if len(paths) != expected_images:
    raise RuntimeError(f"Expected {expected_images} images under {data_root}; found {len(paths)}")
if len({p.stem for p in paths}) != len(paths):
    raise RuntimeError("Image stems must be unique because CSV pairing uses image_id")
checkpoint = Path(evaluation["checkpoint"])
if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
if not torch.cuda.is_available():
    raise RuntimeError("Selected CUDA device is unavailable")
# Load standard pretrained AlexNet/LPIPS weights before starting the long job.
lpips.LPIPS(net="alex").eval()
versions = {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "timm", "pytorch-msssim", "lpips", "PyYAML")}
versions.update(python=platform.python_version(), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), images=len(paths), dataset_root=str(data_root))
(output_dir / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
files = [config_path, checkpoint, Path("scripts/run_double_notch_csi_baselines.sh"),
         Path("scripts/format_double_notch_results.py")]
files.extend(sorted(Path("si_fdjscc").rglob("*.py")))
with (output_dir / "sha256.txt").open("w", encoding="utf-8") as handle:
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        handle.write(f"{digest.hexdigest()}  {path.as_posix()}\n")
print(json.dumps(versions, indent=2), flush=True)
PY

run_stage tests env CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q tests

run_stage evaluation env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m si_fdjscc.evaluation.evaluate \
    --config "$CONFIG" --mapper-modes "${MODES[@]}" --output-dir "$RUN_DIR/evaluation"

run_stage comparison "$PY" -u -m si_fdjscc.evaluation.compare_token_energy \
    --run-dir "$RUN_DIR/evaluation" --dataset "$DATASET" --suite baselines \
    --protocol double_notch \
    --expected-count "$((EXPECTED_IMAGES * 20))" --expected-draws 20 --resamples 10000 \
    --output-json "$RUN_DIR/comparison.json" --output-csv "$RUN_DIR/summary.csv"

run_stage table "$PY" -u scripts/format_double_notch_results.py "$RUN_DIR/summary.csv"
cp "$RUN_DIR/table.log" "$RUN_DIR/table.txt"

printf 'completed\n' > "$RUN_DIR/status.txt"
echo "Finished. Result directory: $RUN_DIR"
