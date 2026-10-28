#!/usr/bin/env bash
# Evaluate five paired mappers at five SNRs, CSI NMSE -10 dB, no notch.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
DATASET="${DATASET:-div2k}"
case "$DATASET" in
    div2k) CONFIG=configs/eval_snr_no_notch_div2k.yaml; EXPECTED_IMAGES=100 ;;
    kodak) CONFIG=configs/eval_snr_no_notch_kodak.yaml; EXPECTED_IMAGES=24 ;;
    *) echo "DATASET must be div2k or kodak" >&2; exit 2 ;;
esac
MODES=(sequential random greedy greedy_token_random greedy_token_energy)
SNR_VALUES=(10 5 0 -5 -10)

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python executable unavailable: $PY; activate the project environment or set PYTHON_BIN." >&2
    exit 2
fi

mkdir -p runs
RUN_DIR="$(mktemp -d "$REPO_ROOT/runs/${DATASET}_snr_no_notch_baselines_XXXXXXXX")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ofdm_snr_no_notch_XXXXXXXX")"
run_started="$(date +%s)"
run_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
active_stage="initialization"
completed=false
touch "$WORK_DIR/stages.tsv"
printf '%s\n' "$RUN_DIR" | tee "runs/${DATASET}_snr_no_notch_baselines_latest.txt"

# Consolidate lifecycle information in one file; stage logs are temporary.
write_metadata() {
    local status="$1" exit_code="$2"
    "$PY" - "$RUN_DIR" "$WORK_DIR/stages.tsv" "$status" "$exit_code" \
        "$run_started" "$run_started_utc" "$active_stage" "$DATASET" "$GPU" "$CONFIG" <<'PY'
import csv
import datetime
import json
import sys
import time
from pathlib import Path

(run_dir, stages_path, status, exit_code, started, started_utc,
 current_stage, dataset, gpu, config_path) = sys.argv[1:]
run_dir = Path(run_dir)
stages = []
with Path(stages_path).open(encoding="utf-8", newline="") as handle:
    for row in csv.reader(handle, delimiter="\t"):
        name, code, seconds = row
        stages.append({"name": name, "exit_code": int(code), "seconds": int(seconds)})
metadata = {
    "status": status,
    "exit_code": None if exit_code == "" else int(exit_code),
    "started_at_utc": started_utc,
    "finished_at_utc": (datetime.datetime.now(datetime.timezone.utc).isoformat()
                        if status != "running" else None),
    "seconds": int(time.time()) - int(started),
    "current_stage": current_stage if status == "running" else None,
    "failed_stage": current_stage if status == "failed" else None,
    "dataset": dataset,
    "gpu": gpu,
    "source_config": config_path,
    "protocol": {"snr_db": [10, 5, 0, -5, -10], "csi_nmse_db": [-10],
                 "spectral_notches": [None], "channel_realizations": 20,
                 "modes": ["sequential", "random", "greedy", "greedy_token_random",
                           "greedy_token_energy"]},
    "stages": stages,
    "retained_files": sorted(path.relative_to(run_dir).as_posix()
                             for path in run_dir.rglob("*")
                             if path.is_file() and path.name != "run.json.tmp"),
}
if "run.json" not in metadata["retained_files"]:
    metadata["retained_files"].append("run.json")
    metadata["retained_files"].sort()
temporary_path = run_dir / "run.json.tmp"
temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
temporary_path.replace(run_dir / "run.json")
PY
}

finish_run() {
    local run_rc=$?
    trap - EXIT INT TERM
    set +e
    if [[ "$run_rc" -eq 0 && "$completed" == true ]]; then
        write_metadata completed 0
        metadata_rc=$?
        if [[ "$metadata_rc" -ne 0 ]]; then run_rc="$metadata_rc"; fi
    else
        if [[ "$run_rc" -eq 0 ]]; then run_rc=1; fi
    fi
    if [[ "$run_rc" -ne 0 ]]; then
        {
            printf 'Run failed at stage: %s\nExit code: %s\n\n' "$active_stage" "$run_rc"
            for log in "$WORK_DIR"/*.log; do
                [[ -f "$log" ]] || continue
                printf '\n===== %s =====\n' "$(basename "$log")"
                cat "$log"
            done
        } > "$RUN_DIR/failure.log"
        write_metadata failed "$run_rc"
        echo "Failed at $active_stage. Inspect $RUN_DIR/failure.log and run.json" >&2
    fi
    # Only this mktemp directory is removed; previous results are never pruned.
    if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then rm -rf -- "$WORK_DIR"; fi
    exit "$run_rc"
}
trap finish_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_metadata running ""

run_stage() {
    local stage="$1"
    shift
    local -a stage_codes
    local stage_rc
    local stage_started="$(date +%s)"
    active_stage="$stage"
    write_metadata running ""
    printf '\n[%s] Starting %s\n' "$(date -u +%H:%M:%S)" "$stage"
    set +e
    "$@" 2>&1 | tee "$WORK_DIR/$stage.log"
    stage_codes=("${PIPESTATUS[@]}")
    set -e
    stage_rc="${stage_codes[0]}"
    if [[ "${stage_codes[1]}" -ne 0 && "$stage_rc" -eq 0 ]]; then stage_rc=1; fi
    local stage_seconds="$(($(date +%s) - stage_started))"
    printf '%s\t%s\t%s\n' "$stage" "$stage_rc" "$stage_seconds" >> "$WORK_DIR/stages.tsv"
    write_metadata running ""
    printf '[%s] Finished %s: exit=%s, seconds=%s\n' "$(date -u +%H:%M:%S)" "$stage" "$stage_rc" "$stage_seconds"
    return "$stage_rc"
}

run_stage preflight env CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$CONFIG" "$EXPECTED_IMAGES" "$RUN_DIR" "${SNR_VALUES[@]}" <<'PY'
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
snr_values = [int(value) for value in sys.argv[4:]]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
evaluation = config["evaluation"]
if evaluation["snr_db"] != snr_values:
    raise RuntimeError(f"SNR sweep must be {snr_values} dB")
if evaluation["csi_nmse_db"] != [-10]:
    raise RuntimeError("Original CSI NMSE must remain [-10] dB")
if evaluation["spectral_notches"] != [None]:
    raise RuntimeError("Only the no-notch scenario is allowed")
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
files = [config_path, checkpoint, Path("scripts/run_snr_no_notch_baselines.sh"),
         Path("scripts/format_snr_no_notch_results.py")]
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

run_stage tests env CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tests

run_stage evaluation env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m si_fdjscc.evaluation.evaluate \
    --config "$CONFIG" --mapper-modes "${MODES[@]}" --output-dir "$RUN_DIR/evaluation"

run_stage comparison "$PY" -u -m si_fdjscc.evaluation.compare_token_energy \
    --run-dir "$RUN_DIR/evaluation" --dataset "$DATASET" --suite baselines \
    --protocol snr_sweep \
    --expected-count "$((EXPECTED_IMAGES * 20))" --expected-draws 20 --resamples 10000 \
    --output-json "$RUN_DIR/comparison.json" --output-csv "$RUN_DIR/summary.csv"

run_stage table "$PY" -u scripts/format_snr_no_notch_results.py "$RUN_DIR/summary.csv"
cp "$WORK_DIR/table.log" "$RUN_DIR/table.txt"

completed=true
echo "Finished. Result directory: $RUN_DIR"
