"""Compact one explicitly selected, successful no-notch SNR run.

The default is a dry run. Only the named legacy bookkeeping files can be
removed; scientific outputs and unknown entries are always retained.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import yaml


STAGES = ("preflight", "tests", "evaluation", "comparison", "table")
LEGACY_FILES = ("base_config.yaml", "status.txt") + tuple(
    f"{stage}.{suffix}"
    for stage in (*STAGES, "run")
    for suffix in ("log", "exitcode", "seconds")
)
REQUIRED_FILES = (
    "table.txt",
    "summary.csv",
    "comparison.json",
    "environment.json",
    "sha256.txt",
    "evaluation/config.yaml",
    "evaluation/samples.csv",
    "evaluation/summary.json",
)
SNR_SETS = ({10, 5, 0}, {10, 5, 0, -5, -10})


def _checked_path(path: Path, *, directory: bool = False) -> None:
    """Reject links, reparse points, and unexpected filesystem objects."""
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Required path is missing: {path}") from exc
    reparse = getattr(info, "st_file_attributes", 0) & 0x400
    if stat.S_ISLNK(info.st_mode) or reparse:
        raise ValueError(f"Refusing symbolic link or reparse point: {path}")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"Expected a normal {kind}: {path}")


def _exists(path: Path) -> bool:
    # lexists also notices dangling symbolic links, which must not be ignored.
    return os.path.lexists(path)


def _number(path: Path, *, integer: bool = False) -> int | float:
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = int(text) if integer else float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric metadata: {path}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid numeric metadata: {path}")
    return value


def _protocol(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / "evaluation/config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("evaluation"), dict):
        raise ValueError("evaluation/config.yaml must contain an evaluation mapping")
    evaluation = config["evaluation"]
    snrs = evaluation.get("snr_db")
    if (
        not isinstance(snrs, list)
        or not all(type(value) in (int, float) and math.isfinite(value) for value in snrs)
        or len(set(snrs)) != len(snrs)
        or set(snrs) not in SNR_SETS
    ):
        raise ValueError("SNR list must contain exactly [10, 5, 0] or [10, 5, 0, -5, -10]")
    if evaluation.get("csi_nmse_db") != [-10]:
        raise ValueError("Only CSI NMSE [-10] runs can be compacted")
    if evaluation.get("spectral_notches") != [None]:
        raise ValueError("Only spectral_notches: [null] runs can be compacted")
    protocol = {
        key: copy.deepcopy(evaluation[key])
        for key in ("snr_db", "csi_nmse_db", "spectral_notches", "channel_realizations")
        if key in evaluation
    }
    if "mapper_modes" in evaluation:
        protocol["modes"] = copy.deepcopy(evaluation["mapper_modes"])
    return protocol


def plan_compaction(run_dir: Path) -> dict:
    """Validate a completed run and describe the only files eligible for removal."""
    run_dir = Path(run_dir).expanduser().absolute()
    _checked_path(run_dir, directory=True)
    run_dir = run_dir.resolve(strict=True)
    _checked_path(run_dir / "evaluation", directory=True)
    for relative in REQUIRED_FILES:
        path = run_dir / relative
        _checked_path(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Required scientific output is empty: {path}")
    protocol = _protocol(run_dir)

    legacy = []
    for relative in LEGACY_FILES:
        path = run_dir / relative
        if _exists(path):
            _checked_path(path)
            legacy.append(relative)

    metadata_path = run_dir / "run.json"
    metadata = None
    if _exists(metadata_path):
        _checked_path(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("status") != "completed"
            or type(metadata.get("exit_code")) is not int
            or metadata["exit_code"] != 0
        ):
            raise ValueError("run.json must report status completed and exit_code 0")
        stages = metadata.get("stages", [])
        if not isinstance(stages, list):
            raise ValueError("run.json stages must be a list")
        names = []
        for stage in stages:
            if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
                raise ValueError("run.json contains malformed stage metadata")
            names.append(stage["name"])
            if "exit_code" in stage and (
                type(stage["exit_code"]) is not int or stage["exit_code"] != 0
            ):
                raise ValueError("run.json contains an unsuccessful stage")
        if len(set(names)) != len(names):
            raise ValueError("run.json contains duplicate stage metadata")

    if "status.txt" in legacy:
        status = (run_dir / "status.txt").read_text(encoding="utf-8").strip()
        if status != "completed":
            raise ValueError(f"Refusing a run with status {status!r}")
    if metadata is None and not {"status.txt", "run.exitcode"}.issubset(legacy):
        raise ValueError("A legacy run requires status.txt completed and run.exitcode 0")

    codes, seconds = {}, {}
    for stage in (*STAGES, "run"):
        exit_path, seconds_path = run_dir / f"{stage}.exitcode", run_dir / f"{stage}.seconds"
        if exit_path.name in legacy:
            codes[stage] = _number(exit_path, integer=True)
            if codes[stage] != 0:
                raise ValueError(f"Refusing unsuccessful stage: {stage}")
        if seconds_path.name in legacy:
            seconds[stage] = _number(seconds_path)

    metadata = copy.deepcopy(metadata) if metadata is not None else {}
    metadata.update(status="completed", exit_code=0)
    if "run" in seconds:
        metadata["seconds"] = seconds["run"]
    if "protocol" not in metadata:
        metadata["protocol"] = protocol
    stages = {stage["name"]: stage for stage in metadata.get("stages", [])}
    for stage in STAGES:
        if stage in codes or stage in seconds:
            entry = stages.setdefault(stage, {"name": stage})
            if stage in codes:
                entry["exit_code"] = codes[stage]
            if stage in seconds:
                entry["seconds"] = seconds[stage]
    metadata["stages"] = list(stages.values())

    known_root = {Path(relative).parts[0] for relative in REQUIRED_FILES} | set(LEGACY_FILES) | {"run.json"}
    unknown = [path.name for path in run_dir.iterdir() if path.name not in known_root]
    expected_evaluation = {"config.yaml", "samples.csv", "summary.json"}
    unknown.extend(
        f"evaluation/{path.name}"
        for path in (run_dir / "evaluation").iterdir()
        if path.name not in expected_evaluation
    )
    metadata["retained_files"] = sorted((*REQUIRED_FILES, "run.json"))
    metadata["preserved_unknown_entries"] = sorted(unknown)
    return {
        "run_dir": str(run_dir),
        "remove_files": sorted(legacy),
        "retained_files": sorted((*REQUIRED_FILES, "run.json")),
        "preserved_unknown_entries": sorted(unknown),
        "metadata": metadata,
    }


def compact_run(run_dir: Path, *, apply: bool = False) -> dict:
    """Write consolidated metadata before removing explicit legacy root files."""
    plan = plan_compaction(run_dir)
    if not apply:
        return plan
    root = Path(plan["run_dir"])
    if not plan["remove_files"] and (root / "run.json").is_file():
        return plan

    metadata = plan["metadata"]
    metadata["compaction"] = {
        "applied_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "removed_legacy_files": plan["remove_files"],
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".run.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, root / "run.json")
        temporary_path = None
        for relative in plan["remove_files"]:
            target = root / relative
            # Each deletion is from the literal allowlist, within this checked root.
            _checked_path(target)
            target.unlink()
    finally:
        if temporary_path is not None and _exists(temporary_path):
            temporary_path.unlink()
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Explicit path to one completed SNR run directory")
    parser.add_argument("--apply", action="store_true", help="Apply the displayed removal plan; default is dry-run")
    args = parser.parse_args()
    try:
        plan = compact_run(args.run_dir, apply=args.apply)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Refused to compact: {exc}\n")
    print(f"{'Applied' if args.apply else 'Dry run (no files changed)'}: {plan['run_dir']}")
    print("Legacy root files eligible for removal:")
    for relative in plan["remove_files"]:
        print(f"  {relative}")
    if not plan["remove_files"]:
        print("  (none; already compact)")
    print("Retained scientific outputs and metadata:")
    for relative in plan["retained_files"]:
        print(f"  {relative}")
    for relative in plan["preserved_unknown_entries"]:
        print(f"WARNING: preserved unknown entry: {relative}")


if __name__ == "__main__":
    main()
