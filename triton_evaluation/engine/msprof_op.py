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

"""Service-adapted msprof-op capture.

Source flow: cannbot-skills/ops/ops-profiling/references/msprof-op-guide.md.
The service adaptation adds a bounded process runner, Triton kernel-name
filtering, recursive OpBasicInfo discovery, and structured JSON-ready output.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


_JIT_DEF_RE = re.compile(
    r"^[ \t]*@(?:triton\.)?jit\b[^\n]*\n"
    r"(?:^[ \t]*@[^\n]*\n)*"
    r"^[ \t]*def\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)


@dataclass
class CommandOutcome:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_command(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> CommandOutcome:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return CommandOutcome(process.returncode, stdout or "", stderr or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return CommandOutcome(-1, stdout or "", stderr or "", timed_out=True)


def remaining_pass_timeout(deadline: float) -> int:
    raw = os.environ.get("MSPROF_TIMEOUT_SEC", "300")
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer") from exc
    if configured <= 0:
        raise ValueError("MSPROF_TIMEOUT_SEC must be a positive integer")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("profile job deadline exceeded")
    return max(1, min(configured, math.ceil(remaining)))


def extract_triton_kernel_names(source: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _JIT_DEF_RE.finditer(source)))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column_mean(rows: list[dict[str, str]], name: str) -> float | None:
    values = [_to_float(row.get(name)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _matches_custom(name: str, kernel_names: list[str]) -> bool:
    return any(name == kernel or name.startswith(kernel) for kernel in kernel_names)


def _parse_kernel_dir(op_dir: str) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    missing_metrics: list[str] = []
    basic = _read_rows(os.path.join(op_dir, "OpBasicInfo.csv"))
    if not basic:
        return None, ["missing OpBasicInfo.csv"]
    row = basic[0]
    name = (row.get("Op Name") or "").strip()
    if not name:
        return None, ["OpBasicInfo.csv has no Op Name"]

    pipe_path = os.path.join(op_dir, "PipeUtilization.csv")
    memory_path = os.path.join(op_dir, "Memory.csv")
    l2_path = os.path.join(op_dir, "L2Cache.csv")
    for path in (pipe_path, memory_path, l2_path):
        if not os.path.isfile(path):
            missing = os.path.basename(path)
            missing_metrics.append(missing)
            warnings.append(f"{name}: missing {missing}")

    pipe = _read_rows(pipe_path)
    memory = _read_rows(memory_path)
    l2 = _read_rows(l2_path)
    metrics = {
        "task_duration_us": _to_float(row.get("Task Duration(us)")),
        "block_dim": _to_float(row.get("Block Dim")),
        "aic_cube_ratio": _column_mean(pipe, "aic_cube_ratio"),
        "aic_mte2_ratio": _column_mean(pipe, "aic_mte2_ratio"),
        "aic_fixpipe_ratio": _column_mean(pipe, "aic_fixpipe_ratio"),
        "aiv_vec_ratio": _column_mean(pipe, "aiv_vec_ratio"),
        "aiv_mte2_ratio": _column_mean(pipe, "aiv_mte2_ratio"),
        "aiv_mte3_ratio": _column_mean(pipe, "aiv_mte3_ratio"),
        "gm_to_l1_usage": _column_mean(memory, "GM_to_L1_bw_usage_rate(%)"),
        "l0c_to_gm_usage": _column_mean(memory, "L0C_to_GM_bw_usage_rate(%)"),
        "gm_to_ub_usage": _column_mean(memory, "GM_to_UB_bw_usage_rate(%)"),
        "ub_to_gm_usage": _column_mean(memory, "UB_to_GM_bw_usage_rate(%)"),
        "aic_l2_hit": _column_mean(l2, "aic_total_hit_rate(%)"),
        "aiv_l2_hit": _column_mean(l2, "aiv_total_hit_rate(%)"),
    }
    if metrics["task_duration_us"] is None:
        missing_metrics.append("OpBasicInfo.Task Duration(us)")
        warnings.append(f"{name}: Task Duration(us) is missing or invalid")
    return {
        "name": name,
        "op_type": (row.get("Op Type") or "").strip(),
        "duration_us": metrics["task_duration_us"],
        "metrics": metrics,
        "missing_metrics": missing_metrics,
    }, warnings


def _op_recommendations(target: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = target["metrics"]
    compute = metrics.get("aiv_vec_ratio")
    load = metrics.get("aiv_mte2_ratio")
    recommendations: list[dict[str, Any]] = []
    if compute is not None and load is not None and load > compute:
        recommendations.append(
            {
                "bound": "op-mode memory-pipe dominant",
                "evidence": {
                    "aiv_mte2_ratio": load,
                    "aiv_vec_ratio": compute,
                },
                "recommendation": (
                    "Reduce GM round trips or fuse adjacent elementwise work."
                ),
                "source": "msprof-op-guide.md",
            }
        )
    if load is not None and load < 0.60:
        recommendations.append(
            {
                "bound": "op-mode low MTE2 activity",
                "evidence": {"aiv_mte2_ratio": load},
                "recommendation": (
                    "Evaluate multibuffer=True and re-profile; keep it only if "
                    "measured latency improves."
                ),
                "source": "msprof-op-guide.md",
            }
        )
    return recommendations


def _failed_result(error: str, *, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "target_kernel": None,
        "target_selection_metric": "OpBasicInfo",
        "kernels": [],
        "raw_metrics": {},
        "bound": None,
        "per_core": None,
        "recommendations": [],
        "missing_metrics": [],
        "warnings": warnings or [],
        "error": error,
    }


def parse_op_output(output_dir: str, kernel_names: list[str]) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    warnings: list[str] = []
    basic_files = sorted(
        glob.glob(
            os.path.join(output_dir, "**", "OpBasicInfo.csv"),
            recursive=True,
        )
    )
    for basic_file in basic_files:
        try:
            parsed, local_warnings = _parse_kernel_dir(os.path.dirname(basic_file))
        except (OSError, UnicodeError, csv.Error) as exc:
            warnings.append(
                f"{basic_file}: cannot parse profiler CSV: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if parsed is not None and _matches_custom(parsed["name"], kernel_names):
            kernels.append(parsed)
            warnings.extend(local_warnings)
    kernels.sort(
        key=lambda item: (
            item["duration_us"] is not None,
            item["duration_us"] or 0.0,
        ),
        reverse=True,
    )
    selectable = [item for item in kernels if item["duration_us"] is not None]
    if not selectable:
        result = _failed_result(
            "no matching custom Triton kernel with a valid duration was parsed",
            warnings=warnings,
        )
        result["kernels"] = kernels
        return result
    target = selectable[0]
    return {
        "status": "partial" if warnings else "complete",
        "target_kernel": target["name"],
        "target_selection_metric": "OpBasicInfo",
        "kernels": kernels,
        "raw_metrics": target["metrics"],
        "bound": None,
        "per_core": None,
        "recommendations": _op_recommendations(target),
        "missing_metrics": target["missing_metrics"],
        "warnings": warnings,
        "error": None,
    }


def profile_op_case(
    *,
    app_cmd: list[str],
    kernel_names: list[str],
    output_dir: str,
    launch_count: int | None,
    cwd: str,
    env: dict[str, str],
    deadline: float,
) -> dict[str, Any]:
    msprof_bin = os.environ.get("MSPROF_BIN", "msprof")
    command = [msprof_bin, "op", f"--output={output_dir}"]
    if launch_count is not None:
        command.append(f"--launch-count={launch_count}")
    command.append(f"--kernel-name={'|'.join(kernel_names)}")
    command.extend(app_cmd)
    outcome = run_command(
        command,
        cwd=cwd,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    if outcome.timed_out:
        return _failed_result("msprof op exceeded its per-process timeout")
    if outcome.exit_code != 0:
        return _failed_result(
            f"msprof op exit={outcome.exit_code}; "
            f"stderr-tail={outcome.stderr[-1000:]}"
        )
    return parse_op_output(output_dir, kernel_names)
