"""Compaction retains scientific evidence and refuses unsuccessful runs."""

import json
from pathlib import Path

import pytest
import yaml

from scripts.compact_snr_run_outputs import LEGACY_FILES, REQUIRED_FILES, compact_run


def _write_run(directory: Path, snrs=None):
    (directory / "evaluation").mkdir()
    for relative in REQUIRED_FILES:
        (directory / relative).write_text("scientific evidence\n", encoding="utf-8")
    config = {
        "evaluation": {
            "snr_db": snrs if snrs is not None else [10, 5, 0],
            "csi_nmse_db": [-10], "spectral_notches": [None],
        },
    }
    (directory / "evaluation/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for stage in ("preflight", "tests", "evaluation", "comparison", "table", "run"):
        (directory / f"{stage}.log").write_text("temporary log\n", encoding="utf-8")
        (directory / f"{stage}.exitcode").write_text("0\n", encoding="utf-8")
        (directory / f"{stage}.seconds").write_text("12\n", encoding="utf-8")
    (directory / "base_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (directory / "status.txt").write_text("completed\n", encoding="utf-8")
    return config


@pytest.mark.parametrize("snrs", [[10, 5, 0], [10, 5, 0, -5, -10]])
def test_dry_run_and_apply_preserve_all_scientific_files(tmp_path, snrs):
    _write_run(tmp_path, snrs)
    scientific = {relative: (tmp_path / relative).read_bytes() for relative in REQUIRED_FILES}
    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    plan = compact_run(tmp_path)
    assert set(plan["remove_files"]) == set(LEGACY_FILES)
    assert not (tmp_path / "run.json").exists()
    assert before == {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    compact_run(tmp_path, apply=True)
    assert all(not (tmp_path / relative).exists() for relative in LEGACY_FILES)
    assert all((tmp_path / relative).read_bytes() == data for relative, data in scientific.items())
    metadata = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed" and metadata["exit_code"] == 0
    assert metadata["seconds"] == 12
    assert metadata["protocol"]["snr_db"] == snrs
    assert len(metadata["stages"]) == 5
    assert all(stage["exit_code"] == 0 and stage["seconds"] == 12 for stage in metadata["stages"])
    assert set(metadata["compaction"]["removed_legacy_files"]) == set(LEGACY_FILES)


def test_unknown_files_and_directories_are_preserved_and_reported(tmp_path):
    _write_run(tmp_path)
    for relative in ("notes.txt", "evaluation/debug.txt", "raw/images/original.png"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep me", encoding="utf-8")
    plan = compact_run(tmp_path, apply=True)
    assert plan["preserved_unknown_entries"] == ["evaluation/debug.txt", "notes.txt", "raw"]
    for relative in ("notes.txt", "evaluation/debug.txt", "raw/images/original.png"):
        assert (tmp_path / relative).read_text(encoding="utf-8") == "keep me"


def test_apply_is_idempotent(tmp_path):
    _write_run(tmp_path)
    compact_run(tmp_path, apply=True)
    metadata = (tmp_path / "run.json").read_bytes()
    plan = compact_run(tmp_path, apply=True)
    assert not plan["remove_files"]
    assert (tmp_path / "run.json").read_bytes() == metadata


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_missing_scientific_output_refuses_all_changes(tmp_path, relative):
    _write_run(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(ValueError, match="missing"):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "status.txt").exists()
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("relative,value", [
    ("status.txt", "running"), ("status.txt", "failed"),
    ("run.exitcode", "1"), ("evaluation.exitcode", "1"),
    ("tests.exitcode", "bad"), ("run.seconds", "nan"),
])
def test_failed_running_or_malformed_legacy_runs_are_not_changed(tmp_path, relative, value):
    _write_run(tmp_path)
    (tmp_path / relative).write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / relative).read_text(encoding="utf-8") == value
    assert (tmp_path / "preflight.log").exists()
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("relative", ["status.txt", "run.exitcode"])
def test_legacy_completion_markers_are_required(tmp_path, relative):
    _write_run(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(ValueError, match="requires"):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "preflight.log").exists()


@pytest.mark.parametrize("field,value", [
    ("snr_db", [10, 5, 0, 0]), ("snr_db", [10, 5, 0, -5]),
    ("snr_db", [10, 5, False]), ("csi_nmse_db", [-10, -5]),
    ("spectral_notches", [None, {"width": 10}]),
])
def test_wrong_protocol_refuses_all_changes(tmp_path, field, value):
    config = _write_run(tmp_path)
    config["evaluation"][field] = value
    (tmp_path / "evaluation/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "status.txt").exists()


@pytest.mark.parametrize("status,code", [("running", None), ("failed", 1), ("completed", 1)])
def test_unsuccessful_run_json_is_not_overridden_by_legacy_status(tmp_path, status, code):
    _write_run(tmp_path)
    value = {"status": status, "exit_code": code}
    (tmp_path / "run.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="run.json"):
        compact_run(tmp_path, apply=True)
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == value
    assert (tmp_path / "status.txt").exists()


def test_successful_existing_run_json_is_accepted_without_legacy_markers(tmp_path):
    _write_run(tmp_path)
    for relative in LEGACY_FILES:
        (tmp_path / relative).unlink()
    metadata = {"status": "completed", "exit_code": 0, "dataset": "div2k", "stages": []}
    (tmp_path / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert compact_run(tmp_path, apply=True)["remove_files"] == []
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == metadata


def test_metadata_write_failure_keeps_legacy_files(tmp_path, monkeypatch):
    _write_run(tmp_path)
    def fail_replace(*args):
        raise OSError("simulated metadata failure")
    monkeypatch.setattr("scripts.compact_snr_run_outputs.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        compact_run(tmp_path, apply=True)
    assert all((tmp_path / relative).exists() for relative in LEGACY_FILES)
    assert not list(tmp_path.glob(".run.*.tmp"))


def test_metadata_is_durable_before_first_legacy_deletion(tmp_path, monkeypatch):
    _write_run(tmp_path)
    original_unlink = Path.unlink
    removed = []
    def check_unlink(path, *args, **kwargs):
        metadata = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "completed" and metadata["exit_code"] == 0
        assert path.name in LEGACY_FILES
        removed.append(path.name)
        original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", check_unlink)
    compact_run(tmp_path, apply=True)
    assert set(removed) == set(LEGACY_FILES)


def test_empty_scientific_output_prevents_cleanup(tmp_path):
    _write_run(tmp_path)
    (tmp_path / "evaluation/samples.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "status.txt").exists()


@pytest.mark.parametrize("stages", [None, [{"exit_code": 0}], [{"name": "evaluation", "exit_code": 1}]])
def test_malformed_existing_stage_metadata_prevents_cleanup(tmp_path, stages):
    _write_run(tmp_path)
    value = {"status": "completed", "exit_code": 0, "stages": stages}
    (tmp_path / "run.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="run.json"):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "status.txt").exists()


def test_managed_directory_cannot_be_deleted_as_a_legacy_file(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "tests.log"
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        compact_run(tmp_path, apply=True)
    assert path.is_dir()
    assert (tmp_path / "status.txt").exists()


def test_symbolic_link_legacy_file_is_rejected(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "tests.log"
    path.unlink()
    try:
        path.symlink_to(tmp_path / "table.txt")
    except OSError:
        pytest.skip("Symbolic links require permission on this Windows host")
    with pytest.raises(ValueError, match="link"):
        compact_run(tmp_path, apply=True)
    assert (tmp_path / "table.txt").read_text(encoding="utf-8") == "scientific evidence\n"


def test_cli_defaults_to_dry_run_and_warns_for_unknown(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path)
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    from scripts.compact_snr_run_outputs import main
    monkeypatch.setattr("sys.argv", ["compact_snr_run_outputs", str(tmp_path)])
    main()
    output = capsys.readouterr().out
    assert "Dry run" in output
    assert "WARNING: preserved unknown entry: notes.txt" in output
    assert (tmp_path / "status.txt").exists()
    assert not (tmp_path / "run.json").exists()
