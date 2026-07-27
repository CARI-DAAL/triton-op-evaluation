#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Raw msprof deep capture and diagnosis.

Source flow: cannbot-skills/ops/ops-profiling/scripts/msprof_profile_run.sh
and scripts/msprof_perf_summary.py, with rules from references/msprof-guide.md.
Missing data is explicit and rows are merged only after one complete target
Op Name is selected.
"""

from __future__ import annotations

import csv
import glob
import os
import sqlite3
import statistics
import time
from typing import Any

try:
    from .msprof_op import CommandOutcome, remaining_pass_timeout, run_command
except ImportError:  # direct execution through engine/profile_runner.py
    from msprof_op import CommandOutcome, remaining_pass_timeout, run_command


METRIC_GROUPS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)

PIPE_ALIASES = {
    "MTE2": ("aic_mte2_ratio", "aiv_mte2_ratio", "mte2_ratio"),
    "CUBE": ("aic_cube_ratio", "aic_mac_ratio", "cube_ratio"),
    "PUSHQ": ("pushq_ratio", "aiv_pushq_ratio"),
    "FIXP": ("aic_fixpipe_ratio", "fixpipe_ratio"),
    "MTE3": ("aic_mte3_ratio", "aiv_mte3_ratio", "mte3_ratio"),
    "SCALAR": ("aic_scalar_ratio", "aiv_scalar_ratio", "scalar_ratio"),
    "SCALARLDST": (
        "scalarldst_ratio",
        "aic_scalarldst_ratio",
        "aiv_scalarldst_ratio",
    ),
}


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_file(directory: str) -> str | None:
    candidates = sorted(
        glob.glob(
            os.path.join(directory, "**", "op_summary_*.csv"),
            recursive=True,
        )
    )
    if not candidates:
        candidates = sorted(
            glob.glob(
                os.path.join(directory, "**", "op_summary.csv"),
                recursive=True,
            )
        )
    return candidates[-1] if candidates else None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _custom_name(name: str, kernel_names: list[str]) -> bool:
    return any(name == kernel or name.startswith(kernel) for kernel in kernel_names)


def _pick_longest(rows: list[dict[str, str]]) -> dict[str, str] | None:
    timed = [
        (duration, row)
        for row in rows
        if (duration := _float(row.get("Task Duration(us)"))) is not None
    ]
    if not timed:
        return None
    return max(timed, key=lambda item: item[0])[1]


def select_and_merge_metrics(
    metric_dirs: dict[str, str],
    kernel_names: list[str],
) -> dict[str, Any]:
    rows_by_metric: dict[str, list[dict[str, str]]] = {}
    missing = [metric for metric in METRIC_GROUPS if metric not in metric_dirs]
    parse_warnings: list[str] = []
    for metric, directory in metric_dirs.items():
        summary = _summary_file(directory)
        if summary is None:
            missing.append(metric)
            continue
        try:
            rows_by_metric[metric] = _read_rows(summary)
        except (OSError, UnicodeError, csv.Error) as exc:
            missing.append(metric)
            parse_warnings.append(
                f"{metric}: cannot parse {summary}: {type(exc).__name__}: {exc}"
            )

    target_metric = None
    target_row = None
    for metric in METRIC_GROUPS:
        candidates = [
            row
            for row in rows_by_metric.get(metric, [])
            if _custom_name((row.get("Op Name") or "").strip(), kernel_names)
        ]
        chosen = _pick_longest(candidates)
        if chosen is not None:
            target_metric = metric
            target_row = chosen
            break
    if target_row is None:
        return {
            "target_kernel": None,
            "target_selection_metric": None,
            "raw_metrics": {},
            "missing_metrics": sorted(set(missing)),
            "evidence": {},
            "warnings": parse_warnings,
            "error": "no matching custom Triton kernel in deep op_summary",
        }

    target_name = (target_row.get("Op Name") or "").strip()
    merged: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for metric in METRIC_GROUPS:
        rows = [
            row
            for row in rows_by_metric.get(metric, [])
            if (row.get("Op Name") or "").strip() == target_name
        ]
        chosen = _pick_longest(rows)
        if chosen is None:
            missing.append(metric)
            continue
        evidence[metric] = {
            "op_name": target_name,
            "candidate_rows": len(rows),
            "selected_duration_us": _float(chosen.get("Task Duration(us)")),
        }
        for key, value in chosen.items():
            if key and value not in {None, "", "NA", "N/A"} and key not in merged:
                merged[key] = value
    return {
        "target_kernel": target_name,
        "target_selection_metric": target_metric,
        "raw_metrics": merged,
        "missing_metrics": sorted(set(missing)),
        "evidence": evidence,
        "warnings": parse_warnings,
        "error": None,
    }


def _max_alias(raw: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    values = [_float(raw.get(alias)) for alias in aliases]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def canonical_pipe_ratios(raw: dict[str, Any]) -> dict[str, float | None]:
    return {
        name: _max_alias(raw, aliases)
        for name, aliases in PIPE_ALIASES.items()
    }


def missing_canonical_pipe_fields(raw: dict[str, Any]) -> list[str]:
    return [
        f"PipeUtilization.{name}"
        for name, value in canonical_pipe_ratios(raw).items()
        if value is None
    ]


def classify_bound(raw: dict[str, Any]) -> dict[str, Any] | None:
    ratios = canonical_pipe_ratios(raw)
    comparison = [ratios[name] for name in PIPE_ALIASES]

    def largest(name: str) -> bool | None:
        if any(value is None for value in comparison):
            return None
        value = ratios[name]
        return value == max(comparison) and value > 0.70

    rules: list[tuple[str, str, bool | None]] = [
        (
            "MTE2 BOUND",
            "MTE2 > 0.80",
            None if ratios["MTE2"] is None else ratios["MTE2"] > 0.80,
        ),
        ("MTE2 BOUND", "MTE2 is largest and > 0.70", largest("MTE2")),
        (
            "CUBE BOUND",
            "CUBE > 0.80",
            None if ratios["CUBE"] is None else ratios["CUBE"] > 0.80,
        ),
        ("CUBE BOUND", "CUBE is largest and > 0.70", largest("CUBE")),
        (
            "VEC BOUND",
            "PUSHQ > 0.80",
            None if ratios["PUSHQ"] is None else ratios["PUSHQ"] > 0.80,
        ),
        (
            "FIXP BOUND",
            "FIXP > 0.80",
            None if ratios["FIXP"] is None else ratios["FIXP"] > 0.80,
        ),
        (
            "MTE3 BOUND",
            "MTE3 > 0.80",
            None if ratios["MTE3"] is None else ratios["MTE3"] > 0.80,
        ),
        (
            "SCALAR BOUND",
            "SCALAR > 0.80 or SCALARLDST > 0.80",
            (
                None
                if ratios["SCALAR"] is None or ratios["SCALARLDST"] is None
                else (
                    ratios["SCALAR"] > 0.80
                    or ratios["SCALARLDST"] > 0.80
                )
            ),
        ),
    ]
    for name, rule, decision in rules:
        if decision is None:
            return None
        if decision:
            return {
                "name": name,
                "ratios": ratios,
                "rule": rule,
                "evidence": dict(raw),
            }
    return {
        "name": "NO BOUND",
        "ratios": ratios,
        "rule": "no rule matched",
        "evidence": dict(raw),
    }


def load_per_core_cycles(sample_dir: str) -> list[tuple[int, int]]:
    candidates = sorted(
        glob.glob(
            os.path.join(
                sample_dir,
                "**",
                "device_0",
                "sqlite",
                "aicore.db",
            ),
            recursive=True,
        )
    )
    if not candidates:
        return []
    connection = None
    try:
        connection = sqlite3.connect(candidates[-1])
        rows = list(
            connection.execute(
                "SELECT coreid, SUM(task_cyc) "
                "FROM AICoreOriginalData WHERE task_cyc > 0 "
                "GROUP BY coreid ORDER BY coreid"
            )
        )
    except sqlite3.Error:
        return []
    finally:
        if connection is not None:
            connection.close()
    return [
        (int(core_id), int(cycles))
        for core_id, cycles in rows
        if core_id is not None and cycles is not None
    ]


def per_core_balance(
    raw: dict[str, Any],
    sample_dir: str,
) -> dict[str, Any] | None:
    rows = load_per_core_cycles(sample_dir)
    aicore_time = _float(raw.get("aicore_time(us)"))
    if not rows or aicore_time is None or aicore_time <= 0:
        return None
    max_cycles = max(cycles for _, cycles in rows)
    if max_cycles <= 0:
        return None
    ns_per_cycle = aicore_time * 1000.0 / max_cycles
    per_core = [
        {
            "core_id": core_id,
            "time_us": cycles * ns_per_cycle / 1000.0,
        }
        for core_id, cycles in rows
    ]
    times = [entry["time_us"] for entry in per_core]
    maximum = max(times)
    minimum = min(times)
    spread = (maximum - minimum) / maximum if maximum else 0.0
    if spread < 0.10:
        verdict = "balanced"
    elif spread < 0.30:
        verdict = "warning"
    else:
        verdict = "severely_imbalanced"
    return {
        "scope": "profile_run",
        "warning": (
            "SQLite rows are run-wide and are not claimed as target-only."
        ),
        "core_ids": [entry["core_id"] for entry in per_core],
        "active_cores": len(per_core),
        "frequency_ghz": 1.0 / ns_per_cycle,
        "min_us": minimum,
        "avg_us": statistics.mean(times),
        "max_us": maximum,
        "spread_ratio": spread,
        "balanced_under_10pct": spread < 0.10,
        "verdict": verdict,
        "per_core": per_core,
    }


_RECOMMENDATIONS = {
    "MTE2 BOUND": (
        "Reduce GM loads, improve contiguous access, and evaluate buffering overlap."
    ),
    "CUBE BOUND": (
        "Increase useful Cube work per launch and validate tiling with measured latency."
    ),
    "VEC BOUND": "Reduce vector issue pressure or fuse redundant vector passes.",
    "FIXP BOUND": "Check output alignment and FixPipe writeback configuration.",
    "MTE3 BOUND": (
        "Reduce or coalesce GM stores and avoid unnecessary intermediate outputs."
    ),
    "SCALAR BOUND": (
        "Reduce scalar address/control work and shorten serial dependency chains."
    ),
}


def recommendations_for(bound: dict[str, Any] | None) -> list[dict[str, Any]]:
    if bound is None or bound["name"] == "NO BOUND":
        return []
    return [
        {
            "bound": bound["name"],
            "evidence": {
                "ratios": bound["ratios"],
                "rule": bound["rule"],
            },
            "recommendation": _RECOMMENDATIONS[bound["name"]],
            "source": "ops-profiling/references/optimization_quickref.md",
        }
    ]


def _task_command(
    msprof_bin: str,
    output: str,
    metric: str,
    app_cmd: list[str],
) -> list[str]:
    return [
        msprof_bin,
        f"--output={output}",
        "--ai-core=on",
        "--aic-mode=task-based",
        f"--aic-metrics={metric}",
        "--task-time=on",
        "--ascendcl=on",
        *app_cmd,
    ]


def _failed_capture(error: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "status": "failed",
        "target_kernel": None,
        "target_selection_metric": None,
        "kernels": [],
        "raw_metrics": {},
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": list(METRIC_GROUPS),
        "warnings": warnings,
        "error": error,
    }


def profile_deep_case(
    *,
    app_cmd: list[str],
    kernel_names: list[str],
    output_dir: str,
    cwd: str,
    env: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    msprof_bin = os.environ.get("MSPROF_BIN", "msprof")
    warnings: list[str] = []
    for warmup_index in range(3):
        outcome = run_command(
            app_cmd,
            cwd=cwd,
            env=env,
            timeout=remaining_pass_timeout(deadline),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("profile job deadline exceeded")
        if outcome.timed_out:
            return _failed_capture(
                (
                    f"deep warmup {warmup_index + 1} exceeded "
                    "its per-process timeout"
                ),
                warnings,
            )
        if outcome.exit_code != 0:
            return _failed_capture(
                f"warmup exit={outcome.exit_code}; "
                f"stderr-tail={outcome.stderr[-1000:]}",
                warnings,
            )

    metric_dirs: dict[str, str] = {}
    for metric in METRIC_GROUPS:
        metric_dir = os.path.join(output_dir, f"PROF_{metric}")
        os.makedirs(metric_dir, exist_ok=True)
        outcome = run_command(
            _task_command(msprof_bin, metric_dir, metric, app_cmd),
            cwd=cwd,
            env=env,
            timeout=remaining_pass_timeout(deadline),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("profile job deadline exceeded")
        if outcome.timed_out:
            warnings.append(f"{metric}: exceeded its per-process timeout")
        elif outcome.exit_code == 0:
            metric_dirs[metric] = metric_dir
        else:
            warnings.append(
                f"{metric}: exit={outcome.exit_code}; "
                f"stderr-tail={outcome.stderr[-1000:]}"
            )

    sample_dir = os.path.join(output_dir, "PROF_Sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_command = [
        msprof_bin,
        f"--output={sample_dir}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
        "--aic-mode=sample-based",
        "--aic-freq=100",
        "--task-time=on",
        "--ascendcl=on",
        *app_cmd,
    ]
    sample = run_command(
        sample_command,
        cwd=cwd,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    sample_ok = False
    if sample.timed_out:
        warnings.append("sample-based: exceeded its per-process timeout")
    elif sample.exit_code == 0:
        sample_ok = True
    else:
        warnings.append(
            f"sample-based: exit={sample.exit_code}; "
            f"stderr-tail={sample.stderr[-1000:]}"
        )

    merged = select_and_merge_metrics(metric_dirs, kernel_names)
    warnings.extend(merged["warnings"])
    missing_metrics = sorted(
        set(
            merged["missing_metrics"]
            + missing_canonical_pipe_fields(merged["raw_metrics"])
        )
    )
    bound = classify_bound(merged["raw_metrics"])
    per_core = (
        per_core_balance(merged["raw_metrics"], sample_dir)
        if sample_ok
        else None
    )
    if bound is None and merged["target_kernel"] is not None:
        warnings.append(
            "insufficient canonical pipe fields for Bound classification"
        )
    if sample_ok and per_core is None:
        warnings.append(
            "sample-based output could not produce reliable per-core data"
        )
    failed = merged["target_kernel"] is None
    partial = bool(warnings or missing_metrics)
    raw_metrics = dict(merged["raw_metrics"])
    raw_metrics["_merge_evidence"] = merged["evidence"]
    return {
        "status": "failed" if failed else ("partial" if partial else "complete"),
        "target_kernel": merged["target_kernel"],
        "target_selection_metric": merged["target_selection_metric"],
        "kernels": [],
        "raw_metrics": raw_metrics,
        "bound": bound,
        "per_core": per_core,
        "recommendations": recommendations_for(bound),
        "missing_metrics": missing_metrics,
        "warnings": warnings,
        "error": merged["error"],
    }
