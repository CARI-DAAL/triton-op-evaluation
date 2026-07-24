import csv
import sys
from pathlib import Path

from triton_evaluation.engine import msprof_op

KERNEL_SOURCE = """
import triton

@triton.jit
def first_kernel(x):
    return x

@triton.jit()
def second_kernel(x):
    return x
"""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_op_dir(root: Path, name: str, duration: float, include_l2: bool = True):
    op_dir = root / "OPPROF_fake" / name / "0"
    _write_csv(
        op_dir / "OpBasicInfo.csv",
        ["Op Name", "Op Type", "Task Duration(us)", "Block Dim"],
        [
            {
                "Op Name": name,
                "Op Type": "vector",
                "Task Duration(us)": duration,
                "Block Dim": 8,
            }
        ],
    )
    _write_csv(
        op_dir / "PipeUtilization.csv",
        ["aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"],
        [
            {
                "aiv_vec_ratio": 0.6,
                "aiv_mte2_ratio": 0.3,
                "aiv_mte3_ratio": 0.1,
            }
        ],
    )
    _write_csv(
        op_dir / "Memory.csv",
        ["GM_to_UB_bw_usage_rate(%)", "UB_to_GM_bw_usage_rate(%)"],
        [{"GM_to_UB_bw_usage_rate(%)": 45, "UB_to_GM_bw_usage_rate(%)": 20}],
    )
    if include_l2:
        _write_csv(
            op_dir / "L2Cache.csv",
            ["aiv_total_hit_rate(%)"],
            [{"aiv_total_hit_rate(%)": 91}],
        )


def test_extract_triton_kernel_names_is_ordered_and_unique():
    assert msprof_op.extract_triton_kernel_names(KERNEL_SOURCE) == [
        "first_kernel",
        "second_kernel",
    ]


def test_parse_op_output_selects_slowest_matching_custom_kernel(tmp_path):
    _write_op_dir(tmp_path, "first_kernel_hash", 3.0)
    _write_op_dir(tmp_path, "second_kernel_hash", 8.0, include_l2=False)
    _write_op_dir(tmp_path, "unrelated_torch_op", 99.0)
    parsed = msprof_op.parse_op_output(
        str(tmp_path),
        ["first_kernel", "second_kernel"],
    )
    assert parsed["target_kernel"] == "second_kernel_hash"
    assert [kernel["name"] for kernel in parsed["kernels"]] == [
        "second_kernel_hash",
        "first_kernel_hash",
    ]
    assert parsed["status"] == "partial"
    assert "L2Cache.csv" in " ".join(parsed["warnings"])
    assert parsed["missing_metrics"] == ["L2Cache.csv"]


def test_missing_custom_kernel_is_failed(tmp_path):
    _write_op_dir(tmp_path, "unrelated_torch_op", 99.0)
    parsed = msprof_op.parse_op_output(str(tmp_path), ["first_kernel"])
    assert parsed["status"] == "failed"
    assert parsed["target_kernel"] is None


def test_run_command_timeout_terminates_the_process(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    outcome = msprof_op.run_command(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env={},
        timeout=1,
    )
    assert outcome.timed_out is True
    assert outcome.exit_code == -1


def test_malformed_profiler_directory_does_not_hide_good_kernel(tmp_path):
    _write_op_dir(tmp_path, "first_kernel_hash", 3.0)
    bad = tmp_path / "OPPROF_fake" / "broken" / "0"
    bad.mkdir(parents=True)
    (bad / "OpBasicInfo.csv").write_bytes(b"\xff\xfe")
    parsed = msprof_op.parse_op_output(str(tmp_path), ["first_kernel"])
    assert parsed["status"] == "partial"
    assert parsed["target_kernel"] == "first_kernel_hash"
    assert "cannot parse" in " ".join(parsed["warnings"])


def test_profile_op_case_builds_equals_joined_cli(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *, cwd, env, timeout):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        _write_op_dir(Path(cmd[2].split("=", 1)[1]), "first_kernel_hash", 4.0)
        return msprof_op.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_op, "run_command", fake_run)
    result = msprof_op.profile_op_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["first_kernel"],
        output_dir=str(tmp_path / "capture"),
        launch_count=3,
        cwd=str(tmp_path),
        env={},
        deadline=msprof_op.time.monotonic() + 30,
    )
    assert seen["cmd"][:4] == [
        "msprof",
        "op",
        f"--output={tmp_path / 'capture'}",
        "--launch-count=3",
    ]
    assert "--kernel-name=first_kernel" in seen["cmd"]
    assert result["status"] == "complete"


def test_profile_op_case_pass_timeout_fails_only_that_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        msprof_op,
        "run_command",
        lambda *args, **kwargs: msprof_op.CommandOutcome(
            -1,
            "",
            "",
            timed_out=True,
        ),
    )
    result = msprof_op.profile_op_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["first_kernel"],
        output_dir=str(tmp_path / "capture"),
        launch_count=None,
        cwd=str(tmp_path),
        env={},
        deadline=msprof_op.time.monotonic() + 30,
    )
    assert result["status"] == "failed"
    assert "per-process timeout" in result["error"]
