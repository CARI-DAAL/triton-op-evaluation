import os
import sys
import time
from pathlib import Path

from triton_evaluation.engine import msprof_deep, msprof_op

FAKE = Path(__file__).parent / "fixtures" / "fake_msprof.py"


def _environment(tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("MSPROF_BIN", str(FAKE))
    monkeypatch.setenv("FAKE_MSPROF_LOG", str(log))
    return dict(os.environ), log


def test_op_backend_with_fake_executable(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "complete"
    assert result["target_kernel"] == "add_kernel_hash"
    assert "--kernel-name=add_kernel" in log.read_text()


def test_op_backend_nonzero_exit_is_failed_case(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_OP", "1")
    env["FAKE_MSPROF_FAIL_OP"] = "1"
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "exit=9" in result["error"]


def test_op_backend_pass_timeout_is_failed_case(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "1")
    monkeypatch.setenv("FAKE_MSPROF_SLEEP_SEC", "2")
    env.update({"MSPROF_TIMEOUT_SEC": "1", "FAKE_MSPROF_SLEEP_SEC": "2"})
    result = msprof_op.profile_op_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "op"),
        launch_count=None,
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "per-process timeout" in result["error"]


def test_deep_backend_with_fake_executable(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 8
    assert result["target_kernel"] == "add_kernel_hash"
    assert result["bound"]["name"] == "MTE2 BOUND"
    assert result["per_core"]["core_ids"] == [0, 1]


def test_deep_capture_only_selected_groups_and_skips_per_core(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"], output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path), env=env, deadline=time.monotonic() + 60,
        metric_groups=["Memory", "L2Cache"], include_per_core=False,
    )
    commands = log.read_text().splitlines()
    assert len(commands) == 2
    assert "--aic-metrics=Memory" in commands[0]
    assert "--aic-metrics=L2Cache" in commands[1]
    assert all("--aic-mode=task-based" in command for command in commands)
    assert result["status"] == "complete"
    assert result["bound"] is None
    assert result["per_core"] is None
    assert result["missing_metrics"] == []
    assert result["warnings"] == []
    assert result["raw_metrics"]["_capture_requested"] == {
        "metric_groups": ["Memory", "L2Cache"], "include_per_core": False,
    }
    assert set(result["raw_metrics"]["_merge_evidence"]) == {"Memory", "L2Cache"}


def test_missing_deep_metric_reports_only_requested_group(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    env["FAKE_MSPROF_FAIL_METRIC"] = "Memory"
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"], output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path), env=env, deadline=time.monotonic() + 60,
        metric_groups=["Memory", "L2Cache"], include_per_core=False,
    )
    assert len(log.read_text().splitlines()) == 2
    assert result["status"] == "partial"
    assert result["missing_metrics"] == ["Memory"]
    assert result["bound"] is None


def test_deep_metric_failure_continues_as_partial(tmp_path, monkeypatch):
    env, log = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_METRIC", "Memory")
    env["FAKE_MSPROF_FAIL_METRIC"] = "Memory"
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    assert len(log.read_text().splitlines()) == 8
    assert result["status"] == "partial"
    assert "Memory" in result["missing_metrics"]
    assert result["target_kernel"] == "add_kernel_hash"


def test_deep_sample_failure_keeps_task_metrics(tmp_path, monkeypatch):
    env, _ = _environment(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MSPROF_FAIL_SAMPLE", "1")
    env["FAKE_MSPROF_FAIL_SAMPLE"] = "1"
    result = msprof_deep.profile_deep_case(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=["add_kernel"],
        output_dir=str(tmp_path / "deep"),
        cwd=str(tmp_path),
        env=env,
        deadline=time.monotonic() + 60,
    )
    assert result["status"] == "partial"
    assert result["target_kernel"] == "add_kernel_hash"
    assert result["per_core"] is None
