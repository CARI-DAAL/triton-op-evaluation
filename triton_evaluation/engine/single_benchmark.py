#!/usr/bin/env python3
"""Measure one supplied implementation, without correctness or reference timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

try:
    from . import benchmark
except ImportError:  # Direct child-process invocation by BenchmarkEvaluator.
    import benchmark


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("implementation module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def measure_implementation(*, op_name, verify_dir, triton_impl_name, implementation,
                           warmup, repeats, method, case_indices=None, device="npu:0"):
    import torch

    if implementation not in {"torch", "triton"}:
        raise ValueError("a single implementation is required")
    root = Path(verify_dir)
    reference = load_module(f"{op_name}_torch", root / f"{op_name}_torch.py")
    if implementation == "torch":
        model_type = reference.Model
    else:
        candidate = load_module(
            f"{op_name}_{triton_impl_name}", root / f"{op_name}_{triton_impl_name}.py"
        )
        model_type = candidate.ModelNew
    torch.manual_seed(0)
    groups = reference.get_input_groups() if hasattr(reference, "get_input_groups") else [
        reference.get_inputs()
    ]
    if not hasattr(groups, "__len__") or not hasattr(groups, "__getitem__"):
        groups = list(groups)
    selected = list(range(len(groups))) if case_indices is None else list(case_indices)
    if not selected or len(set(selected)) != len(selected) or any(
        type(index) is not int or index < 0 or index >= len(groups) for index in selected
    ):
        raise ValueError("invalid performance case selection")
    measured, unmeasured = [], []
    for index in selected:
        shapes = []
        model = inputs = raw = None
        try:
            raw = groups[index]
            shapes = [
                {"input_index": i, "dimensions": list(value.shape), "dtype": str(value.dtype)}
                for i, value in enumerate(raw) if isinstance(value, torch.Tensor)
            ]
            inputs = benchmark._move_inputs_to_device(raw, device)
            model = model_type(*reference.get_init_inputs()).to(device)
            measurement = benchmark.measure_single(benchmark.MeasureContext(
                model=model, inputs=inputs, warmup=warmup, repeats=repeats,
                profile_name=f"single_{implementation}_{index}", device=device, method=method,
            ))
            _, latency, peak = measurement
            if latency is None or not math.isfinite(latency) or latency <= 0:
                unmeasured.append({"case_idx": index, "input_shapes": shapes,
                                   "reason_code": "NO_STABLE_MEASUREMENT"})
                continue
            case = {"case_idx": index, "input_shapes": shapes, "latency_ms": latency}
            kernel_metrics = getattr(measurement, "kernel_metrics", None)
            if kernel_metrics is not None:
                case["kernel_metrics"] = kernel_metrics
            if peak is not None and math.isfinite(peak) and peak >= 0:
                case["peak_memory_mb"] = peak
            measured.append(case)
        except Exception:
            unmeasured.append({"case_idx": index, "input_shapes": shapes,
                               "reason_code": "UNSUPPORTED_CASE"})
        finally:
            model = inputs = raw = None
            benchmark.cleanup_npu_memory()
    if not measured:
        raise RuntimeError("single-implementation benchmark produced no measurements")
    return {
        "implementation": implementation, "correctness_verified": False,
        "measurement_method": "npu_event_v1" if method == "event" else "profiler_kernel_sum_v1",
        "selected_cases": len(selected), "measured_cases": len(measured),
        "coverage": len(measured) / len(selected), "per_case": measured,
        "unmeasured_cases": unmeasured,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op_name", required=True)
    parser.add_argument("--verify_dir", required=True)
    parser.add_argument("--triton_impl_name", required=True)
    parser.add_argument("--implementation", choices=("torch", "triton"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--measurement-method", choices=("event", "profiler"), required=True)
    parser.add_argument("--case-indices")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    benchmark._ensure_npu_ambient_shared(torch)
    benchmark._enforce_conv_fp32()
    indices = None if args.case_indices is None else [
        int(value) - 1 for value in args.case_indices.split(",")
    ]
    result = measure_implementation(
        op_name=args.op_name, verify_dir=args.verify_dir, triton_impl_name=args.triton_impl_name,
        implementation=args.implementation, warmup=args.warmup, repeats=args.repeats,
        method=args.measurement_method, case_indices=indices,
    )
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
