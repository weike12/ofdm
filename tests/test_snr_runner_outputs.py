"""Exercise output retention and failed-stage reporting without a GPU job."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


def _shell_path(path):
    text = Path(path).resolve().as_posix()
    if os.name == "nt" and len(text) > 2 and text[1] == ":":
        return "/" + text[0].lower() + text[2:]
    return text


def _fixture(tmp_path, failed_stage=None):
    bash = shutil.which("bash")
    if bash is None and Path("D:/Git/bin/bash.exe").is_file():
        bash = "D:/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("Bash is required for the runner lifecycle check")
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "configs").mkdir()
    temporary = tmp_path / "stage_temp"
    temporary.mkdir()
    runner = project / "scripts/run_snr_no_notch_baselines.sh"
    source = Path(__file__).resolve().parents[1] / "scripts/run_snr_no_notch_baselines.sh"
    shutil.copyfile(source, runner)
    (project / "configs/eval_snr_no_notch_div2k.yaml").write_text("{}\n", encoding="utf-8")
    fake = project / "fake_python.py"
    fake.write_text('''import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
def fail(stage):
    if os.environ.get("FAIL_STAGE") == stage:
        print("intentional failure in " + stage, flush=True)
        raise SystemExit(23)

if args[0] == "-" and args[1].endswith(".yaml"):
    sys.stdin.read()
    fail("preflight")
    root = Path(args[3])
    (root / "environment.json").write_text("{}")
    (root / "sha256.txt").write_text("mock hashes")
    print("mock preflight")
elif args[0] == "-":
    script = sys.stdin.read()
    sys.argv = args
    exec(compile(script, "metadata", "exec"), {"__name__": "__main__"})
elif "pytest" in args:
    fail("tests")
    print("mock CPU tests")
elif "si_fdjscc.evaluation.evaluate" in args:
    root = Path(args[args.index("--output-dir") + 1])
    root.mkdir()
    (root / "config.yaml").write_text("mock executed config")
    fail("evaluation")
    (root / "samples.csv").write_text("mock raw samples")
    (root / "summary.json").write_text("[]")
    print("mock evaluation")
elif "si_fdjscc.evaluation.compare_token_energy" in args:
    fail("comparison")
    Path(args[args.index("--output-json") + 1]).write_text("{}")
    Path(args[args.index("--output-csv") + 1]).write_text("mock summary")
    print("mock paired comparison")
elif "scripts/format_snr_no_notch_results.py" in args:
    fail("table")
    print("mock five-SNR result table")
else:
    raise SystemExit("Unexpected mock Python command: " + repr(args))
''', encoding="utf-8")
    wrapper = project / "fake-python"
    wrapper.write_text(
        "#!/usr/bin/env bash\nexec " + shlex.quote(_shell_path(sys.executable)) + " "
        + shlex.quote(fake.as_posix()) + ' "$@"\n', encoding="utf-8"
    )
    wrapper.chmod(0o755)
    environment = dict(os.environ, PYTHON_BIN=_shell_path(wrapper),
                       DATASET="div2k", GPU="0", TMPDIR=_shell_path(temporary))
    if failed_stage:
        environment["FAIL_STAGE"] = failed_stage
    else:
        environment.pop("FAIL_STAGE", None)
    result = subprocess.run(
        [bash, _shell_path(runner)], cwd=project, env=environment,
        text=True, capture_output=True, timeout=60,
    )
    runs = list((project / "runs").glob("div2k_snr_no_notch_baselines_*/run.json"))
    assert len(runs) == 1, result.stdout + result.stderr
    return result, runs[0].parent, temporary


def test_success_keeps_only_scientific_results_and_consolidated_metadata(tmp_path):
    result, root, temporary = _fixture(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        "run.json", "table.txt", "summary.csv", "comparison.json",
        "environment.json", "sha256.txt", "evaluation/config.yaml",
        "evaluation/samples.csv", "evaluation/summary.json",
    }
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert actual == expected
    metadata = json.loads((root / "run.json").read_text())
    assert metadata["status"] == "completed" and metadata["exit_code"] == 0
    assert metadata["failed_stage"] is None
    assert [stage["name"] for stage in metadata["stages"]] == [
        "preflight", "tests", "evaluation", "comparison", "table"
    ]
    assert all(stage["exit_code"] == 0 for stage in metadata["stages"])
    assert set(metadata["retained_files"]) == expected
    assert list(temporary.iterdir()) == []


@pytest.mark.parametrize("stage", ["preflight", "evaluation"])
def test_failure_keeps_one_diagnostic_log_and_records_the_failed_stage(tmp_path, stage):
    result, root, temporary = _fixture(tmp_path, failed_stage=stage)
    assert result.returncode == 23, result.stdout + result.stderr
    metadata = json.loads((root / "run.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 23 and metadata["failed_stage"] == stage
    assert metadata["stages"][-1]["name"] == stage
    assert metadata["stages"][-1]["exit_code"] == 23
    assert "intentional failure in " + stage in (root / "failure.log").read_text()
    assert [path.name for path in root.glob("*.log")] == ["failure.log"]
    assert not list(root.glob("*.exitcode")) and not list(root.glob("*.seconds"))
    if stage == "evaluation":
        assert (root / "evaluation/config.yaml").is_file()
    assert list(temporary.iterdir()) == []
