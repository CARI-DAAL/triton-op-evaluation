import csv
from pathlib import Path

from triton_evaluation.engine import msprof_deep


def _summary(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_metric_merge_keeps_one_complete_op_name(tmp_path):
    pipe = tmp_path / "PipeUtilization"
    memory = tmp_path / "Memory"
    _summary(
        pipe / "op_summary.csv",
        [
            {
                "Op Name": "kernel_a_hash",
                "Task Duration(us)": 9,
                "aiv_mte2_ratio": 0.4,
            },
            {
                "Op Name": "kernel_b_hash",
                "Task Duration(us)": 5,
                "aiv_mte2_ratio": 0.9,
            },
        ],
    )
    _summary(
        memory / "op_summary.csv",
        [
            {
                "Op Name": "kernel_a_hash",
                "Task Duration(us)": 4,
                "GM_to_UB_bw_usage_rate(%)": 31,
            },
            {
                "Op Name": "kernel_b_hash",
                "Task Duration(us)": 20,
                "GM_to_UB_bw_usage_rate(%)": 99,
            },
        ],
    )
    merged = msprof_deep.select_and_merge_metrics(
        {"PipeUtilization": str(pipe), "Memory": str(memory)},
        ["kernel_a", "kernel_b"],
    )
    assert merged["target_kernel"] == "kernel_a_hash"
    assert merged["raw_metrics"]["GM_to_UB_bw_usage_rate(%)"] == "31"
    assert merged["evidence"]["Memory"]["op_name"] == "kernel_a_hash"


def test_missing_metric_stays_missing_not_zero(tmp_path):
    pipe = tmp_path / "PipeUtilization"
    _summary(
        pipe / "op_summary.csv",
        [
            {
                "Op Name": "kernel_a_hash",
                "Task Duration(us)": 9,
                "aiv_mte2_ratio": 0.4,
            }
        ],
    )
    merged = msprof_deep.select_and_merge_metrics(
        {"PipeUtilization": str(pipe)},
        ["kernel_a"],
    )
    assert "Memory" in merged["missing_metrics"]
    assert "GM_to_UB_bw_usage_rate(%)" not in merged["raw_metrics"]


def test_deep_capture_runs_three_warmups_then_seven_plus_sample(
    tmp_path,
    monkeypatch,
):
    commands = []

    def fake_run(cmd, *, cwd, env, timeout):
        commands.append(cmd)
        return msprof_deep.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_deep, "run_command", fake_run)
    monkeypatch.setattr(
        msprof_deep,
        "select_and_merge_metrics",
        lambda dirs, names: {
            "target_kernel": "kernel_a_hash",
            "target_selection_metric": "PipeUtilization",
            "raw_metrics": {},
            "missing_metrics": [],
            "evidence": {},
            "warnings": [],
            "error": None,
        },
    )
    result = msprof_deep.profile_deep_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["kernel_a"],
        output_dir=str(tmp_path),
        cwd=str(tmp_path),
        env={},
        deadline=msprof_deep.time.monotonic() + 60,
    )
    assert commands[:3] == [["python", "runner.py", "--case-idx", "0"]] * 3
    profiled = commands[3:]
    assert len(profiled) == 8
    assert [
        next(arg for arg in cmd if arg.startswith("--aic-metrics="))
        for cmd in profiled[:7]
    ] == [
        f"--aic-metrics={metric}" for metric in msprof_deep.METRIC_GROUPS
    ]
    assert all("--aic-mode=task-based" in cmd for cmd in profiled[:7])
    assert "--aic-mode=sample-based" in profiled[-1]
    assert result["target_kernel"] == "kernel_a_hash"


def test_deep_metric_pass_timeout_is_partial_not_job_timeout(tmp_path, monkeypatch):
    calls = 0

    def fake_run(cmd, *, cwd, env, timeout):
        nonlocal calls
        calls += 1
        if calls == 4:
            return msprof_deep.CommandOutcome(-1, "", "", timed_out=True)
        return msprof_deep.CommandOutcome(0, "", "")

    monkeypatch.setattr(msprof_deep, "run_command", fake_run)
    monkeypatch.setattr(
        msprof_deep,
        "select_and_merge_metrics",
        lambda dirs, names: {
            "target_kernel": "kernel_a_hash",
            "target_selection_metric": "ArithmeticUtilization",
            "raw_metrics": {},
            "missing_metrics": ["PipeUtilization"],
            "evidence": {},
            "warnings": [],
            "error": None,
        },
    )
    result = msprof_deep.profile_deep_case(
        app_cmd=["python", "runner.py", "--case-idx", "0"],
        kernel_names=["kernel_a"],
        output_dir=str(tmp_path),
        cwd=str(tmp_path),
        env={},
        deadline=msprof_deep.time.monotonic() + 60,
    )
    assert calls == 11
    assert result["status"] == "partial"
    assert "per-process timeout" in " ".join(result["warnings"])
