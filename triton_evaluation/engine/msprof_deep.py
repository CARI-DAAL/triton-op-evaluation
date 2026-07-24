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
import time
from typing import Any

try:
    from .msprof_op import CommandOutcome, remaining_pass_timeout, run_command
except ImportError:  # direct execution through engine/profile.py
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
    failed = merged["target_kernel"] is None
    partial = bool(warnings or merged["missing_metrics"])
    return {
        "status": "failed" if failed else ("partial" if partial else "complete"),
        "target_kernel": merged["target_kernel"],
        "target_selection_metric": merged["target_selection_metric"],
        "kernels": [],
        "raw_metrics": merged["raw_metrics"],
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": merged["missing_metrics"],
        "warnings": warnings,
        "error": merged["error"],
        "_sample_dir": sample_dir if sample_ok else None,
        "_merge_evidence": merged["evidence"],
    }
