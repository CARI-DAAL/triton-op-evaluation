import csv
import sqlite3
from pathlib import Path

import pytest

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
        lambda dirs, names, groups: {
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
        lambda dirs, names, groups: {
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


@pytest.mark.parametrize("groups", [[], ["Unknown"], ["Memory", "Memory"]])
def test_invalid_metric_groups_fail_before_starting_processes(tmp_path, monkeypatch, groups):
    monkeypatch.setattr(
        msprof_deep, "run_command", lambda *args, **kwargs: pytest.fail("unexpected process"),
    )
    with pytest.raises(ValueError, match="metric_groups"):
        msprof_deep.profile_deep_case(
            app_cmd=["python", "runner.py"], kernel_names=["kernel_a"],
            output_dir=str(tmp_path), cwd=str(tmp_path), env={},
            deadline=msprof_deep.time.monotonic() + 60, metric_groups=groups,
        )


def test_selected_deep_capture_warmup_failure_only_marks_requested_metrics(tmp_path, monkeypatch):
    calls = []

    def failed_warmup(cmd, **kwargs):
        calls.append(cmd)
        return msprof_deep.CommandOutcome(9, "", "failed")

    monkeypatch.setattr(msprof_deep, "run_command", failed_warmup)
    result = msprof_deep.profile_deep_case(
        app_cmd=["python", "runner.py"], kernel_names=["kernel_a"],
        output_dir=str(tmp_path), cwd=str(tmp_path), env={},
        deadline=msprof_deep.time.monotonic() + 60,
        metric_groups=["Memory"], include_per_core=False,
    )
    assert len(calls) == 1
    assert result["status"] == "failed"
    assert result["missing_metrics"] == ["Memory"]
    assert result["raw_metrics"]["_capture_requested"] == {
        "metric_groups": ["Memory"], "include_per_core": False,
    }


def _base_ratios():
    return {
        "aiv_mte2_ratio": "0.10",
        "aic_cube_ratio": "0.10",
        "pushq_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "scalarldst_ratio": "0.10",
    }


def test_bound_priority_and_exact_thresholds():
    raw = _base_ratios()
    raw.update(
        {
            "aiv_mte2_ratio": "0.81",
            "aic_cube_ratio": "0.95",
            "pushq_ratio": "0.90",
        }
    )
    bound = msprof_deep.classify_bound(raw)
    assert bound["name"] == "MTE2 BOUND"
    assert bound["rule"] == "MTE2 > 0.80"


def test_vec_uses_pushq_not_aiv_vec_ratio():
    raw = _base_ratios()
    raw.update({"aiv_vec_ratio": "0.99", "pushq_ratio": "0.20"})
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"


def test_missing_prior_metric_makes_bound_unknown():
    raw = _base_ratios()
    raw.pop("aiv_mte2_ratio")
    raw["aic_cube_ratio"] = "0.95"
    assert msprof_deep.classify_bound(raw) is None
    assert "PipeUtilization.MTE2" in msprof_deep.missing_canonical_pipe_fields(raw)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("aic_cube_ratio", "CUBE BOUND"),
        ("pushq_ratio", "VEC BOUND"),
        ("aic_fixpipe_ratio", "FIXP BOUND"),
        ("aiv_mte3_ratio", "MTE3 BOUND"),
        ("aiv_scalar_ratio", "SCALAR BOUND"),
        ("scalarldst_ratio", "SCALAR BOUND"),
    ],
)
def test_each_bound_rule_is_reachable(field, expected):
    raw = _base_ratios()
    raw[field] = "0.81"
    assert msprof_deep.classify_bound(raw)["name"] == expected


@pytest.mark.parametrize(
    "field",
    [
        "pushq_ratio",
        "aic_fixpipe_ratio",
        "aiv_mte3_ratio",
        "aiv_scalar_ratio",
        "scalarldst_ratio",
    ],
)
def test_eighty_percent_is_not_above_threshold(field):
    raw = _base_ratios()
    raw[field] = "0.80"
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"


@pytest.mark.parametrize("field", ["aiv_mte2_ratio", "aic_cube_ratio"])
def test_seventy_percent_is_not_above_largest_threshold(field):
    raw = _base_ratios()
    raw[field] = "0.70"
    assert msprof_deep.classify_bound(raw)["name"] == "NO BOUND"


def test_v200_row_classifies_without_pushq_or_scalarldst():
    # 910B (V200 vector core) op_summary never exports pushq/scalarldst
    # columns; the vector signal arrives as aiv_vec_ratio and scalar loads
    # are folded into aiv_scalar_ratio.
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_mac_ratio": "0.10",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
        "aiv_vec_ratio": "0.85",
    }
    bound = msprof_deep.classify_bound(raw)
    assert bound is not None
    assert bound["name"] == "VEC BOUND"
    assert msprof_deep.missing_canonical_pipe_fields(raw) == []


def test_scalar_bound_tolerates_missing_scalarldst():
    raw = _base_ratios()
    raw.pop("scalarldst_ratio")
    raw["aiv_scalar_ratio"] = "0.81"
    assert msprof_deep.classify_bound(raw)["name"] == "SCALAR BOUND"


def test_scalar_and_scalarldst_both_missing_still_blocks():
    raw = _base_ratios()
    raw.pop("scalarldst_ratio")
    raw.pop("aiv_scalar_ratio")
    assert msprof_deep.classify_bound(raw) is None
    missing = msprof_deep.missing_canonical_pipe_fields(raw)
    assert any("SCALAR" in field for field in missing)


def test_missing_vector_signal_on_v200_still_blocks():
    raw = {
        "aiv_mte2_ratio": "0.10",
        "aic_mac_ratio": "0.95",
        "aic_fixpipe_ratio": "0.10",
        "aiv_mte3_ratio": "0.10",
        "aiv_scalar_ratio": "0.10",
    }
    assert msprof_deep.classify_bound(raw) is None
    assert "PipeUtilization.PUSHQ" in msprof_deep.missing_canonical_pipe_fields(raw)


def test_per_core_reports_observed_ids_and_profile_run_scope(tmp_path):
    db_dir = tmp_path / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True)
    database = db_dir / "aicore.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE AICoreOriginalData(coreid INTEGER, task_cyc INTEGER)"
    )
    connection.executemany(
        "INSERT INTO AICoreOriginalData VALUES (?, ?)",
        [(2, 100), (2, 50), (9, 200)],
    )
    connection.commit()
    connection.close()
    result = msprof_deep.per_core_balance(
        {"aicore_time(us)": "20"},
        str(tmp_path),
    )
    assert result["scope"] == "profile_run"
    assert result["core_ids"] == [2, 9]
    assert result["active_cores"] == 2
    assert "%" not in result["verdict"]
    assert result["spread_ratio"] == 0.25


def test_per_core_missing_database_is_none(tmp_path):
    assert (
        msprof_deep.per_core_balance(
            {"aicore_time(us)": "20"},
            str(tmp_path),
        )
        is None
    )


def test_per_core_reads_pinned_card_device_dir(tmp_path):
    # Workers pin ASCEND_RT_VISIBLE_DEVICES to a nonzero card and msprof
    # then writes sqlite data under device_<card>, never device_0.
    db_dir = tmp_path / "PROF_fake" / "device_7" / "sqlite"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "aicore.db")
    connection.execute(
        "CREATE TABLE AICoreOriginalData(coreid INTEGER, task_cyc INTEGER)"
    )
    connection.executemany(
        "INSERT INTO AICoreOriginalData VALUES (?, ?)",
        [(0, 1600), (1, 1700)],
    )
    connection.commit()
    connection.close()
    result = msprof_deep.per_core_balance(
        {"aicore_time(us)": "20"},
        str(tmp_path / "PROF_fake"),
    )
    assert result is not None
    assert result["core_ids"] == [0, 1]
    assert result["active_cores"] == 2


def test_per_core_unknown_schema_is_none(tmp_path):
    db_dir = tmp_path / "PROF_fake" / "device_0" / "sqlite"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "aicore.db")
    connection.execute("CREATE TABLE DifferentData(core INTEGER, cycles INTEGER)")
    connection.commit()
    connection.close()
    assert (
        msprof_deep.per_core_balance(
            {"aicore_time(us)": "20"},
            str(tmp_path),
        )
        is None
    )
