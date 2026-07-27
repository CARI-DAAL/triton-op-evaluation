import json
import sys
from pathlib import Path

import pytest

from triton_evaluation.engine import profile_runner as profile


def _config(tmp_path, **overrides):
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "import triton\n"
        "@triton.jit\n"
        "def sample_kernel(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    values = {
        "op_name": "sample",
        "verify_dir": str(tmp_path),
        "triton_impl_name": "triton_ascend_impl",
        "profile_mode": "op",
        "case_idx": None,
        "launch_count": None,
        "timeout": 60,
        "output": str(tmp_path / "profile_result.json"),
        "engine_dir": str(tmp_path),
        "python_bin": sys.executable,
    }
    values.update(overrides)
    return profile.ProfileConfig(**values)


def _complete(case_idx):
    return {
        "status": "complete",
        "target_kernel": f"kernel_{case_idx}",
        "target_selection_metric": "OpBasicInfo",
        "kernels": [],
        "raw_metrics": {},
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": [],
        "warnings": [],
        "error": None,
    }


def test_default_profiles_all_cases_and_snapshots_each(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": [{"shape": [16]}]},
        {"case_idx": 1, "input_desc": [{"shape": [32]}]},
    ]
    snapshots = []
    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=lambda **kwargs: _complete(int(kwargs["app_cmd"][-1])),
        deep_backend=None,
        snapshot=lambda path, payload: snapshots.append(
            json.loads(json.dumps(payload))
        ),
    )
    assert exit_code == 0
    assert [item["case_idx"] for item in result["per_case_results"]] == [0, 1]
    assert result["selected_cases"] == 2
    assert result["completed_cases"] == 2
    assert len(snapshots) == 3
    assert (
        result["completed_cases"]
        + result["partial_cases"]
        + result["failed_cases"]
        + len(result["remaining_case_indices"])
        == result["selected_cases"]
    )


def test_single_case_selection_is_zero_based(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]
    seen = []

    def backend(**kwargs):
        seen.append(kwargs["app_cmd"])
        return _complete(1)

    result, exit_code = profile.run_profile(
        _config(tmp_path, case_idx=1),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["requested_case_idx"] == 1
    assert result["selected_cases"] == 1
    assert seen[0][-2:] == ["--case-idx", "1"]


def test_case_failure_does_not_stop_full_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]

    def backend(**kwargs):
        index = int(kwargs["app_cmd"][-1])
        if index == 0:
            value = _complete(index)
            value.update(
                {
                    "status": "failed",
                    "target_kernel": None,
                    "error": "kernel crash",
                }
            )
            return value
        return _complete(index)

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["failed_cases"] == 1
    assert result["completed_cases"] == 1


def test_deadline_returns_timeout_with_remaining_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]

    def backend(**kwargs):
        raise TimeoutError("metric timed out")

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        deep_backend=None,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 124
    assert result["timed_out"] is True
    assert result["remaining_case_indices"] == [0, 1]
    assert (
        result["completed_cases"]
        + result["partial_cases"]
        + result["failed_cases"]
        + len(result["remaining_case_indices"])
        == result["selected_cases"]
    )


def test_timeout_preserves_completed_cases_and_latest_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    discovered = [
        {"case_idx": 0, "input_desc": []},
        {"case_idx": 1, "input_desc": []},
    ]
    snapshots = []

    def backend(**kwargs):
        index = int(kwargs["app_cmd"][-1])
        if index == 1:
            raise TimeoutError("profile job deadline exceeded")
        return _complete(index)

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: discovered,
        op_backend=backend,
        snapshot=lambda path, payload: snapshots.append(
            json.loads(json.dumps(payload))
        ),
    )
    assert exit_code == 124
    assert result["completed_cases"] == 1
    assert result["remaining_case_indices"] == [1]
    assert snapshots[-1] == result


def test_out_of_range_case_is_input_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    with pytest.raises(IndexError, match="out of range"):
        profile.run_profile(
            _config(tmp_path, case_idx=4),
            discover=lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
            op_backend=lambda **kwargs: _complete(0),
            deep_backend=None,
            snapshot=lambda path, payload: None,
        )


def test_invalid_pass_timeout_fails_before_case_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPROF_TIMEOUT_SEC", "0")
    with pytest.raises(ValueError, match="positive integer"):
        profile.run_profile(
            _config(tmp_path),
            discover=lambda **kwargs: pytest.fail("discovery must not run"),
            op_backend=lambda **kwargs: _complete(0),
            snapshot=lambda path, payload: None,
        )


def test_deep_mode_rejects_launch_count_before_discovery(tmp_path):
    with pytest.raises(ValueError, match="only valid in op mode"):
        profile.run_profile(
            _config(tmp_path, profile_mode="deep", launch_count=2),
            discover=lambda **kwargs: pytest.fail("discovery must not run"),
            snapshot=lambda path, payload: None,
        )


def test_case_capture_directories_are_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "write_runner", lambda directory: "runner.py")
    capture_dirs = []

    def backend(**kwargs):
        capture_dirs.append(Path(kwargs["output_dir"]))
        return _complete(0)

    result, exit_code = profile.run_profile(
        _config(tmp_path),
        discover=lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
        op_backend=backend,
        snapshot=lambda path, payload: None,
    )
    assert exit_code == 0
    assert result["completed_cases"] == 1
    assert capture_dirs and all(not path.exists() for path in capture_dirs)


def test_atomic_snapshot_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "profile_result.json"
    profile.atomic_write_json(str(output), {"selected_cases": 1})
    assert json.loads(output.read_text()) == {"selected_cases": 1}
    assert not (tmp_path / ".profile_result.json.tmp").exists()


def test_main_writes_input_error_and_returns_two(tmp_path, monkeypatch):
    config = _config(tmp_path, case_idx=9)
    monkeypatch.setattr(profile, "_config_from_args", lambda: config)
    monkeypatch.setattr(
        profile,
        "discover_cases",
        lambda **kwargs: [{"case_idx": 0, "input_desc": []}],
    )
    assert profile.main() == 2
    payload = json.loads(Path(config.output).read_text())
    assert payload["error_type"] == "IndexError"
