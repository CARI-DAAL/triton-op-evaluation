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

"""Profile CLI and multi-case orchestrator.

This module remains stdlib-only. Torch/NPU imports live in the generated
runner source and execute in a child process pinned by the service worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .msprof_deep import profile_deep_case
    from .msprof_op import (
        extract_triton_kernel_names,
        profile_op_case,
        remaining_pass_timeout,
        run_command,
    )
    from ._common_utils import set_process_title as _set_process_title
except ImportError:  # direct execution: python engine/profile_runner.py
    from msprof_deep import profile_deep_case
    from msprof_op import (
        extract_triton_kernel_names,
        profile_op_case,
        remaining_pass_timeout,
        run_command,
    )
    from _common_utils import set_process_title as _set_process_title


def build_runner_source() -> str:
    return textwrap.dedent(
        r'''
        import argparse
        import importlib.util
        import json
        import os
        import sys


        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load module: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            return module


        def seed(torch):
            torch.manual_seed(0)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(0)


        def build_input_groups(torch_module, torch):
            """构造参考实现的输入组；CPU 缺 kernel 时切到 NPU 上下文重试。

            部分 reference 用「NPU 有 kernel、CPU 无 kernel」的算子/精度组合声明
            输入（例如 torch.randn(shape, dtype=torch.int32)：torch_npu 在 NPU 上
            实现了整型 normal，CPU 侧抛 NotImplementedError）。沙箱阶段的验证以
            NPU 为环境设备运行，这类 reference 可正常构造输入，因此首次构造失败
            时改用 NPU 默认设备重试，而不是让整个评测失败。本函数随生成的 runner
            源码一起下发，故与 engine 包内的同名工具各自独立。
            """
            try:
                return torch_module.get_input_groups()
            except NotImplementedError:
                print("input construction unavailable on CPU; retrying on NPU")
                torch.set_default_device("npu")
                return torch_module.get_input_groups()


        class OfficialGroups:
            def __init__(self, reference, torch):
                self.reference = reference
                self.torch = torch
                self.total = reference.get_case_count()
                if type(self.total) is not int or self.total < 1:
                    raise RuntimeError("official task must contain cases")

            def __len__(self):
                return self.total

            def __getitem__(self, index):
                if type(index) is not int or not 0 <= index < self.total:
                    raise IndexError("official case index out of range")
                seed(self.torch)
                group = self.reference.get_input_group(index)
                if not isinstance(group, (list, tuple)):
                    raise TypeError("official case must be a list or tuple of arguments")
                return list(group)

            def __iter__(self):
                for index in range(self.total):
                    yield self[index]


        class NormalizedGroups:
            """Normalize one case on access without retaining device workloads."""
            def __init__(self, groups):
                self.groups = groups

            def __len__(self):
                return len(self.groups)

            def __getitem__(self, index):
                group = self.groups[index]
                if not isinstance(group, (list, tuple)):
                    raise TypeError(f"case {index} must be a list or tuple of arguments")
                return list(group)


        def resolve_groups(reference, torch):
            seed(torch)
            if (hasattr(reference, "_ADAPTER_VERSION")
                    and hasattr(reference, "get_input_group")
                    and hasattr(reference, "get_case_count")):
                return OfficialGroups(reference, torch)
            if hasattr(reference, "get_input_groups"):
                groups = build_input_groups(reference, torch)
                if not hasattr(groups, "__len__") or not hasattr(groups, "__getitem__"):
                    groups = list(groups)
                if len(groups) == 0:
                    raise RuntimeError("get_input_groups() returned no cases")
            elif hasattr(reference, "get_inputs"):
                groups = [reference.get_inputs()]
            else:
                raise RuntimeError(
                    "reference defines neither get_input_groups nor get_inputs"
                )
            return NormalizedGroups(groups)


        def to_npu(value, torch):
            if isinstance(value, torch.Tensor):
                return value.to("npu:0")
            if isinstance(value, list):
                return [to_npu(item, torch) for item in value]
            if isinstance(value, tuple):
                return tuple(to_npu(item, torch) for item in value)
            if isinstance(value, dict):
                return {key: to_npu(item, torch) for key, item in value.items()}
            return value


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--engine-dir", required=True)
            parser.add_argument("--verify-dir", required=True)
            parser.add_argument("--op-name", required=True)
            parser.add_argument("--triton-impl-name", required=True)
            parser.add_argument("--implementation", choices=("comparison", "torch", "triton"), default="comparison")
            mode = parser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--describe-cases", action="store_true")
            mode.add_argument("--case-idx", type=int)
            args = parser.parse_args()

            sys.path.insert(0, args.engine_dir)
            sys.path.insert(0, args.verify_dir)
            import torch
            from _common_utils import describe_input, enforce_conv_fp32
            from _common_utils import ensure_npu_ambient_device

            # 执行环境不变量：与沙箱一致以 NPU 为环境设备（见 _common_utils）。
            ensure_npu_ambient_device(torch)

            reference = load_module(
                f"{args.op_name}_torch",
                os.path.join(args.verify_dir, f"{args.op_name}_torch.py"),
            )
            groups = resolve_groups(reference, torch)
            if args.describe_cases:
                print(json.dumps({
                    "total_cases": len(groups),
                    "cases": [
                        # Index access: enumerate would keep the previous case
                        # alive while the next one is materialized.
                        {"case_idx": index, "input_desc": describe_input(groups[index])}
                        for index in range(len(groups))
                    ],
                }))
                return

            index = args.case_idx
            if index is None or index < 0 or index >= len(groups):
                raise IndexError(
                    f"case_idx {index} out of range for {len(groups)} cases"
                )
            enforce_conv_fp32()
            if args.implementation == "torch":
                model_type = reference.Model
            else:
                implementation = load_module(
                    f"{args.op_name}_{args.triton_impl_name}",
                    os.path.join(
                        args.verify_dir,
                        f"{args.op_name}_{args.triton_impl_name}.py",
                    ),
                )
                model_type = implementation.ModelNew
            seed(torch)
            model = model_type(*reference.get_init_inputs()).to("npu:0")
            from _common_utils import invoke_model, prepare_candidate_inputs
            raw_inputs = groups[index]
            model_inputs = raw_inputs if args.implementation == "torch" else prepare_candidate_inputs(getattr(reference, "Model", None), raw_inputs)
            inputs = [to_npu(value, torch) for value in model_inputs]
            with torch.no_grad():
                invoke_model(model, inputs)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            print(json.dumps({
                "case_idx": index,
                "input_desc": describe_input(raw_inputs),
            }))


        if __name__ == "__main__":
            main()
        '''
    )


def write_runner(directory: str) -> str:
    path = Path(directory) / "profile_runner.py"
    path.write_text(build_runner_source(), encoding="utf-8")
    return str(path)


def _last_json_object(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("runner produced no JSON object")


def remaining_discovery_timeout(deadline: float) -> int:
    """Per-process budget for case discovery.

    Discovery materializes every input group on the host to describe it, so a
    large official corpus costs minutes of CPU — it is not an msprof capture
    pass and must not die against MSPROF_TIMEOUT_SEC (300 s default). The
    engine deadline still binds: the effective timeout is the smaller of the
    configured discovery budget and the time left on the job.
    """
    raw = os.environ.get("MSPROF_DISCOVERY_TIMEOUT_SEC", "900")
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError("MSPROF_DISCOVERY_TIMEOUT_SEC must be a positive integer") from exc
    if configured <= 0:
        raise ValueError("MSPROF_DISCOVERY_TIMEOUT_SEC must be a positive integer")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("profile job deadline exceeded")
    return max(1, min(configured, math.ceil(remaining)))


def _validated_discovery_payload(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("discovery payload is not an object")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("discovery payload has no cases")
    total = payload.get("total_cases")
    if total != len(cases):
        raise ValueError("discovery payload case count mismatch")
    return cases


def discover_cases(
    *,
    runner_path: str,
    engine_dir: str,
    verify_dir: str,
    op_name: str,
    impl_name: str,
    python_bin: str,
    env: dict[str, str],
    deadline: float,
    implementation: str = "comparison",
) -> list[dict]:
    # The evaluator starts one engine invocation per profiled case; without a
    # cache each of them would re-run discovery (minutes of host-side tensor
    # generation on a 51-case official corpus). op_name embeds the request
    # digest, so the cache file cannot leak across different requests, and
    # verify_dir is a fresh per-evaluation work directory.
    cache_path = Path(verify_dir) / f"{op_name}_cases.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return _validated_discovery_payload(cached)
    except (OSError, ValueError):
        pass  # Missing or corrupt cache: fall through and re-discover.
    command = [
        python_bin,
        runner_path,
        "--engine-dir",
        engine_dir,
        "--verify-dir",
        verify_dir,
        "--op-name",
        op_name,
        "--triton-impl-name",
        impl_name,
        "--implementation",
        implementation,
        "--describe-cases",
    ]
    outcome = run_command(
        command,
        cwd=verify_dir,
        env=env,
        timeout=remaining_discovery_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    if outcome.timed_out:
        raise RuntimeError("case discovery exceeded its per-process timeout")
    if outcome.exit_code != 0:
        raise RuntimeError(f"case discovery failed: {outcome.stderr[-1000:]}")
    payload = _last_json_object(outcome.stdout)
    try:
        cases = _validated_discovery_payload(payload)
    except ValueError:
        raise RuntimeError("case discovery returned no cases") from None
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, cache_path)
    return cases


@dataclass
class ProfileConfig:
    op_name: str
    verify_dir: str
    triton_impl_name: str
    profile_mode: str
    case_idx: int | None
    launch_count: int | None
    timeout: int
    output: str
    engine_dir: str
    python_bin: str
    implementation: str = "comparison"


def validate_profile_config(config: ProfileConfig) -> None:
    if config.implementation not in {"comparison", "torch", "triton"}:
        raise ValueError("invalid implementation")
    if config.profile_mode not in {"op", "deep"}:
        raise ValueError("profile_mode must be 'op' or 'deep'")
    if config.case_idx is not None and config.case_idx < 0:
        raise ValueError("case_idx must be non-negative")
    if config.launch_count is not None and not 1 <= config.launch_count <= 100:
        raise ValueError("launch_count must be between 1 and 100")
    if config.profile_mode == "deep" and config.launch_count is not None:
        raise ValueError("launch_count is only valid in op mode")
    if not 10 <= config.timeout <= 86400:
        raise ValueError("timeout must be between 10 and 86400 seconds")


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def select_case_indices(cases: list[dict], requested: int | None) -> list[int]:
    if requested is None:
        return [int(case["case_idx"]) for case in cases]
    if requested < 0 or requested >= len(cases):
        raise IndexError(f"case_idx {requested} out of range for {len(cases)} cases")
    return [requested]


def _empty_result(
    config: ProfileConfig, total_cases: int, selected: list[int]
) -> dict[str, Any]:
    return {
        "profile_mode": config.profile_mode,
        "requested_case_idx": config.case_idx,
        "total_cases": total_cases,
        "selected_cases": len(selected),
        "completed_cases": 0,
        "partial_cases": 0,
        "failed_cases": 0,
        "remaining_case_indices": list(selected),
        "timed_out": False,
        "summary_text": "",
        "warnings": [],
        "wallclock_sec": 0.0,
        "per_case_results": [],
    }


def _refresh_counts(result: dict[str, Any]) -> None:
    statuses = [case["status"] for case in result["per_case_results"]]
    result["completed_cases"] = statuses.count("complete")
    result["partial_cases"] = statuses.count("partial")
    result["failed_cases"] = statuses.count("failed")
    result["summary_text"] = (
        f"profile mode={result['profile_mode']}: "
        f"complete={result['completed_cases']}, "
        f"partial={result['partial_cases']}, "
        f"failed={result['failed_cases']}, "
        f"remaining={len(result['remaining_case_indices'])}"
    )


def run_profile(
    config: ProfileConfig,
    *,
    discover: Callable[..., list[dict]] | None = None,
    op_backend: Callable[..., dict] | None = None,
    deep_backend: Callable[..., dict] | None = None,
    snapshot: Callable[[str, dict], None] = atomic_write_json,
) -> tuple[dict[str, Any], int]:
    validate_profile_config(config)
    started = time.monotonic()
    deadline = started + config.timeout
    remaining_pass_timeout(deadline)
    discover_fn = discover if discover is not None else discover_cases
    op_backend_fn = op_backend if op_backend is not None else profile_op_case
    deep_backend_fn = (
        deep_backend if deep_backend is not None else profile_deep_case
    )
    runner_path = write_runner(config.verify_dir)
    env = dict(os.environ)
    cases = discover_fn(
        runner_path=runner_path,
        engine_dir=config.engine_dir,
        verify_dir=config.verify_dir,
        op_name=config.op_name,
        impl_name=config.triton_impl_name,
        python_bin=config.python_bin,
        env=env,
        deadline=deadline,
        implementation=config.implementation,
    )
    selected = select_case_indices(cases, config.case_idx)
    result = _empty_result(config, len(cases), selected)
    snapshot(config.output, result)

    kernel_names = None
    if config.implementation != "torch":
        kernel_path = Path(config.verify_dir) / (
            f"{config.op_name}_{config.triton_impl_name}.py"
        )
        kernel_names = extract_triton_kernel_names(kernel_path.read_text(encoding="utf-8"))
        if not kernel_names:
            raise RuntimeError("kernel source contains no @triton.jit/@jit function")

    by_index = {int(case["case_idx"]): case for case in cases}
    for index in selected:
        case_started = time.monotonic()
        app_cmd = [
            config.python_bin,
            runner_path,
            "--engine-dir",
            config.engine_dir,
            "--verify-dir",
            config.verify_dir,
            "--op-name",
            config.op_name,
            "--triton-impl-name",
            config.triton_impl_name,
            "--implementation",
            config.implementation,
            "--case-idx",
            str(index),
        ]
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"profile_case_{index}_",
                dir=config.verify_dir,
            ) as capture_dir:
                if config.profile_mode == "op":
                    case_result = op_backend_fn(
                        app_cmd=app_cmd,
                        kernel_names=kernel_names,
                        output_dir=capture_dir,
                        launch_count=config.launch_count,
                        cwd=config.verify_dir,
                        env=env,
                        deadline=deadline,
                    )
                else:
                    case_result = deep_backend_fn(
                        app_cmd=app_cmd,
                        kernel_names=kernel_names,
                        output_dir=capture_dir,
                        cwd=config.verify_dir,
                        env=env,
                        deadline=deadline,
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("profile job deadline exceeded")
        except TimeoutError as exc:
            result["timed_out"] = True
            result["warnings"].append(str(exc))
            result["wallclock_sec"] = time.monotonic() - started
            _refresh_counts(result)
            snapshot(config.output, result)
            return result, 124
        except Exception as exc:
            case_result = {
                "status": "failed",
                "target_kernel": None,
                "target_selection_metric": None,
                "kernels": [],
                "raw_metrics": {},
                "bound": None,
                "per_core": None,
                "recommendations": [],
                "missing_metrics": [],
                "warnings": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        case_result = dict(case_result)
        case_result["case_idx"] = index
        case_result["input_desc"] = by_index[index]["input_desc"]
        case_result["msprof_wallclock_sec"] = time.monotonic() - case_started
        result["per_case_results"].append(case_result)
        result["remaining_case_indices"].remove(index)
        result["wallclock_sec"] = time.monotonic() - started
        _refresh_counts(result)
        snapshot(config.output, result)

    return result, 0


def _config_from_args() -> ProfileConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_name", required=True)
    parser.add_argument("--verify_dir", required=True)
    parser.add_argument("--triton_impl_name", required=True)
    parser.add_argument("--implementation", choices=("comparison", "torch", "triton"), default="comparison")
    parser.add_argument("--profile-mode", choices=("op", "deep"), required=True)
    parser.add_argument("--case-idx", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return ProfileConfig(
        op_name=args.op_name,
        verify_dir=os.path.abspath(args.verify_dir),
        triton_impl_name=args.triton_impl_name,
        profile_mode=args.profile_mode,
        implementation=args.implementation,
        case_idx=args.case_idx,
        launch_count=args.launch_count,
        timeout=args.timeout,
        output=os.path.abspath(args.output),
        engine_dir=os.path.dirname(os.path.abspath(__file__)),
        python_bin=os.environ.get("MSPROF_PROFILE_PYTHON") or sys.executable,
    )


def main() -> int:
    config = _config_from_args()
    try:
        result, exit_code = run_profile(config)
        atomic_write_json(config.output, result)
        return exit_code
    except Exception as exc:
        atomic_write_json(
            config.output,
            {
                "profile_mode": config.profile_mode,
                "requested_case_idx": config.case_idx,
                "timed_out": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(f"profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    _set_process_title("EvalEngine")
    raise SystemExit(main())
