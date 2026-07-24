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

import json
import os
import textwrap
import time
from pathlib import Path

try:
    from .msprof_op import remaining_pass_timeout, run_command
except ImportError:  # direct execution: python engine/profile.py
    from msprof_op import remaining_pass_timeout, run_command


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
            spec.loader.exec_module(module)
            return module


        def seed(torch):
            torch.manual_seed(0)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(0)


        def resolve_groups(reference, torch):
            seed(torch)
            if hasattr(reference, "get_input_groups"):
                groups = list(reference.get_input_groups())
                if not groups:
                    raise RuntimeError("get_input_groups() returned no cases")
            elif hasattr(reference, "get_inputs"):
                groups = [reference.get_inputs()]
            else:
                raise RuntimeError(
                    "reference defines neither get_input_groups nor get_inputs"
                )
            normalized = []
            for index, group in enumerate(groups):
                if not isinstance(group, (list, tuple)):
                    raise TypeError(
                        f"case {index} must be a list or tuple of arguments"
                    )
                normalized.append(list(group))
            return normalized


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
            mode = parser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--describe-cases", action="store_true")
            mode.add_argument("--case-idx", type=int)
            args = parser.parse_args()

            sys.path.insert(0, args.engine_dir)
            sys.path.insert(0, args.verify_dir)
            import torch
            from _common_utils import describe_input, enforce_conv_fp32

            reference = load_module(
                f"{args.op_name}_torch",
                os.path.join(args.verify_dir, f"{args.op_name}_torch.py"),
            )
            groups = resolve_groups(reference, torch)
            if args.describe_cases:
                print(json.dumps({
                    "total_cases": len(groups),
                    "cases": [
                        {"case_idx": index, "input_desc": describe_input(group)}
                        for index, group in enumerate(groups)
                    ],
                }))
                return

            index = args.case_idx
            if index is None or index < 0 or index >= len(groups):
                raise IndexError(
                    f"case_idx {index} out of range for {len(groups)} cases"
                )
            enforce_conv_fp32()
            implementation = load_module(
                f"{args.op_name}_{args.triton_impl_name}",
                os.path.join(
                    args.verify_dir,
                    f"{args.op_name}_{args.triton_impl_name}.py",
                ),
            )
            seed(torch)
            model = implementation.ModelNew(
                *reference.get_init_inputs()
            ).to("npu:0")
            inputs = [to_npu(value, torch) for value in groups[index]]
            with torch.no_grad():
                model(*inputs)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            print(json.dumps({
                "case_idx": index,
                "input_desc": describe_input(groups[index]),
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
) -> list[dict]:
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
        "--describe-cases",
    ]
    outcome = run_command(
        command,
        cwd=verify_dir,
        env=env,
        timeout=remaining_pass_timeout(deadline),
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("profile job deadline exceeded")
    if outcome.timed_out:
        raise RuntimeError("case discovery exceeded its per-process timeout")
    if outcome.exit_code != 0:
        raise RuntimeError(f"case discovery failed: {outcome.stderr[-1000:]}")
    payload = _last_json_object(outcome.stdout)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("case discovery returned no cases")
    return cases
