#!/usr/bin/env python3
"""Timing and kernel-duration traces in one bounded, streaming engine context.

An event timing phase completes before any trace capture starts. Profiler timing
already owns a kernel-duration capture, so its selected cases reuse that exact
capture. This is deliberately not an msprof op/deep metrics implementation.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from . import benchmark, single_benchmark
except ImportError:  # Direct engine subprocess invocation.
    import benchmark
    import single_benchmark


def _clone_inputs(value, memo=None, *, copy_tensors=True):
    """Preserve opaque sentinels and aliases while isolating mutable tensors.

    Lazy providers still retain at most one workload. Eager legacy providers
    keep their original tensors pristine for a later phase, even when .to()
    returns its input because inputs were generated on the device already.
    """
    import torch

    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        if copy_tensors:
            import copy

            # Torch only deep-copies graph leaves. Timing runs under no_grad,
            # so detach non-leaves before copying their storage. A shared memo
            # across the whole input tree preserves distinct views, strides,
            # storage offsets and aliases to already-copied leaf tensors.
            source = value if value.is_leaf else value.detach()
            result = copy.deepcopy(source, memo)
            if not value.is_leaf:
                result.requires_grad_(value.requires_grad)
        else:
            result = value
    elif isinstance(value, list):
        result = []
        memo[identity] = result
        result.extend(_clone_inputs(item, memo, copy_tensors=copy_tensors) for item in value)
    elif isinstance(value, tuple):
        result = tuple(_clone_inputs(item, memo, copy_tensors=copy_tensors) for item in value)
    elif isinstance(value, dict):
        result = {}
        memo[identity] = result
        result.update((key, _clone_inputs(item, memo, copy_tensors=copy_tensors))
                      for key, item in value.items())
    else:
        # In particular, adapter-owned _ABSENT sentinels must retain identity.
        result = value
    memo[identity] = result
    return result


def _rng_state():
    import torch

    cpu = torch.get_rng_state()
    npu = getattr(torch, "npu", None)
    device = npu.get_rng_state() if npu is not None and hasattr(npu, "get_rng_state") else None
    return cpu, device


def _restore_rng(state):
    import torch

    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.npu.set_rng_state(state[1])


class CleanInputGroups(Sequence):
    """Reusable provider with bounded tensor lifetime and per-case RNG replay."""

    def __init__(self, groups):
        self.groups = groups
        self.states = {}
        self.replay = False
        self.fresh = getattr(groups, "_inputs_are_fresh", False) is True

    def __len__(self):
        return len(self.groups)

    def _before_case(self, index):
        if self.fresh:
            # Built-in workload generation already resets RNGs per case and
            # allocates fresh storage. Avoid copying NPU tensors or RNG states.
            return
        if self.replay and index in self.states:
            _restore_rng(self.states[index])
        else:
            self.states[index] = _rng_state()

    def __getitem__(self, index):
        self._before_case(index)
        return _clone_inputs(self.groups[index], copy_tensors=not self.fresh)

    def __iter__(self):
        # Preserve the official adapter's linear full-pass iterator; indexing
        # every case can replay all preceding cases in legacy lazy adapters.
        iterator = iter(self.groups)
        for index in range(len(self)):
            self._before_case(index)
            raw = next(iterator)
            try:
                clean = _clone_inputs(raw, copy_tensors=not self.fresh)
            finally:
                del raw
            yield clean
            del clean


def _prepare(config, implementation, device):
    import torch

    if device is None:
        import torch_npu  # noqa: F401 — register the NPU device before torch.device

        device = torch.device("npu")
        policy = os.environ.get("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
        wait = benchmark._AMBIENT_WAIT_CAP_SECONDS if policy == "npu-strict" else 0.0
        ready = benchmark._ensure_npu_ambient_shared(torch, wait_seconds=wait)
        if policy == "npu-strict" and not ready:
            raise RuntimeError("npu-strict measurement requires an ambient NPU device")
    root = Path(config.verify_dir)
    reference = single_benchmark.load_module(
        f"{config.op_name}_torch", root / f"{config.op_name}_torch.py"
    )
    if implementation == "torch":
        model_type = reference.Model
    else:
        candidate = single_benchmark.load_module(
            f"{config.op_name}_{config.triton_impl_name}",
            root / f"{config.op_name}_{config.triton_impl_name}.py",
        )
        model_type = candidate.ModelNew
    torch.manual_seed(0)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(0)
    groups = (
        benchmark._build_input_groups_shared(reference)
        if hasattr(reference, "get_input_groups") else [reference.get_inputs()]
    )
    if not hasattr(groups, "__len__") or not hasattr(groups, "__getitem__"):
        groups = list(groups)
    if not 1 <= len(groups) <= 128:
        raise ValueError("measurement requires 1..128 input cases")
    return reference, model_type, CleanInputGroups(groups), device


def _error(code, message):
    return {
        "code": code, "phase": "execution", "check": "profile",
        "retryable": False, "message": message,
    }


def _failed_case(index, message):
    return {
        "case_idx": index, "status": "failed", "kernels": [],
        "error": _error("PROFILE_CAPTURE_FAILED", message),
    }


def trace_case(index, operators, max_kernels):
    """Convert the existing capture's microseconds without inventing metrics."""
    if not isinstance(operators, dict) or not operators:
        return _failed_case(index, "Profiler capture contained no kernel durations")
    if any(
        not isinstance(name, str) or not name
        or isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
        for name, value in operators.items()
    ):
        return _failed_case(index, "Profiler capture contained invalid kernel durations")
    total = sum(operators.values())
    if not math.isfinite(total) or total <= 0:
        return _failed_case(index, "Profiler capture contained no positive kernel duration")
    selected = sorted(operators.items(), key=lambda item: (-item[1], item[0]))[:max_kernels]
    partial = len(selected) < len(operators)
    return {
        "case_idx": index, "status": "partial" if partial else "complete",
        "kernels": [
            {"op_name": name, "time_ms": value / 1000, "percentage": 100 * (value / total)}
            for name, value in selected
        ],
        "error": _error(
            "PROFILE_KERNEL_LIMIT",
            f"Returned {len(selected)} of {len(operators)} kernel groups; "
            "percentages use the full captured kernel duration",
        ) if partial else None,
    }


def _capture_case(config, reference, model_type, raw, device, index, implementation):
    import torch

    inputs = model = None
    try:
        if implementation == "comparison":
            inputs = benchmark._prepare_candidate_inputs_shared(reference.Model, raw)
        else:
            inputs = raw
        inputs = benchmark._move_inputs_to_device(inputs, device)
        init_inputs = reference.get_init_inputs()
        if implementation == "comparison":
            # Match _instantiate_bench_models, including its candidate seed.
            torch.manual_seed(0)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed(0)
        model = model_type(*init_inputs).to(device)
        operators, _, _ = benchmark.measure_single(benchmark.MeasureContext(
            model=model, inputs=inputs, warmup=config.warmup, repeats=config.repeats,
            profile_name=f"trace_case{index + 1}", device=device, method="profiler",
        ))
        return operators
    finally:
        raw = inputs = model = None
        benchmark.cleanup_npu_memory()


def run_measurements(config, *, implementation="comparison", trace_indices=None,
                     timing=True, max_kernels=200, performance_ready=None, device=None,
                     performance_progress=None):
    """Run explicit phases; selections use original one-based engine indices.

    The callback receives complete timing before later capture starts. A trace
    failure never discards that result. Profiler-selected cases are never
    captured a second time, including cases where the first capture failed.
    The optional progress callback also receives bounded per-case snapshots
    during timing, before a later case or trace can be interrupted.
    """
    if implementation not in {"comparison", "torch", "triton"}:
        raise ValueError("invalid measurement implementation")
    if type(max_kernels) is not int or not 1 <= max_kernels <= 1000:
        raise ValueError("max_kernels must be within 1..1000")
    reference, model_type, groups, device = _prepare(config, implementation, device)
    operators_by_case = {}
    performance = None
    timing_indices = set()
    if timing:
        timing_indices = {
            index - 1 for index in benchmark._resolve_selected_indices(config.case_indices, len(groups))
        }
        try:
            if implementation == "comparison":
                measured = benchmark.benchmark_implementations(
                    config, input_groups=groups, device=device,
                    model_spec=benchmark.BenchmarkModelSpec(
                        reference.Model, model_type, reference.get_init_inputs,
                    ),
                    checkpoint=performance_progress,
                )
                operators_by_case = {
                    case.case_idx - 1: case.implementation.operators
                    for case in measured.per_shape_results if case.implementation is not None
                }
                performance = benchmark.result_to_dict(measured)
            else:
                performance = single_benchmark.measure_implementation(
                    op_name=config.op_name, verify_dir=config.verify_dir,
                    triton_impl_name=config.triton_impl_name, implementation=implementation,
                    warmup=config.warmup, repeats=config.repeats,
                    method=config.measurement_method, device=device,
                    case_indices=sorted(timing_indices), prepared=(reference, model_type, groups),
                    operators_by_case=operators_by_case,
                    checkpoint=performance_progress,
                )
        except Exception as error:
            # Preserve the orchestrator's stop-on-infrastructure-failure rule.
            # Earlier per-case snapshots remain valid even when the phase
            # cannot finish; no trace capture starts after this failure.
            print(f"Timing phase failed: {type(error).__name__}", file=sys.stderr)
            raise
        if performance is not None and performance_ready is not None:
            performance_ready(performance)

    selected = [index - 1 for index in benchmark._resolve_selected_indices(trace_indices, len(groups))]
    groups.replay = True
    reused_indices = timing_indices if config.measurement_method == "profiler" else set()
    capture_groups = benchmark._SelectedInputGroups(
        groups, [index + 1 for index in selected if index not in reused_indices],
    )
    capture_iterator = iter(capture_groups)
    cases = []
    for index in selected:
        if index in reused_indices:
            operators = operators_by_case.get(index)
            cases.append(trace_case(index, operators, max_kernels))
            continue
        raw = None
        try:
            raw = next(capture_iterator)
            operators = _capture_case(
                config, reference, model_type, raw, device, index, implementation,
            )
            cases.append(trace_case(index, operators, max_kernels))
        except Exception as error:
            cases.append(_failed_case(index, f"Profiler capture failed ({type(error).__name__})"))
        finally:
            # Release before advancing the lazy provider to the next workload.
            raw = None
    result = {
        "mode": "trace", "measurement_method": "profiler_kernel_sum_v1",
        "total_cases": len(groups),
        "counts": {
            "selected": len(cases),
            **{status: sum(case["status"] == status for case in cases)
               for status in ("complete", "partial", "failed")},
        },
        "cases": cases,
    }
    if implementation != "comparison":
        result.update(implementation=implementation, correctness_verified=False)
    return performance, result


def _write_checkpoint(path, result):
    benchmark.write_checkpoint(path, result)


def main():
    parser = benchmark._build_argparser()
    parser.add_argument("--implementation", choices=("comparison", "torch", "triton"),
                        default="comparison")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--trace-case-indices")
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--max-trace-kernels", type=int, default=200)
    args = parser.parse_args()
    if args.reference_only:
        parser.error("reference-only baselines require the isolated benchmark runner")
    if not args.trace_only and not args.output:
        parser.error("joint timing requires --output")
    benchmark._setup_logger()
    benchmark._enforce_conv_fp32()
    if args.implementation == "comparison" and not args.trace_only:
        try:
            benchmark.check_correctness_gate(args)
        except benchmark.VerifyGateError as error:
            return error.exit_code
    config = benchmark._build_config(args, os.path.abspath(args.verify_dir))
    _, trace = run_measurements(
        config, implementation=args.implementation, timing=not args.trace_only,
        trace_indices=benchmark._parse_case_indices(args.trace_case_indices),
        max_kernels=args.max_trace_kernels,
        performance_ready=lambda result: _write_checkpoint(args.output, result),
        performance_progress=lambda result: _write_checkpoint(args.output, result),
    )
    _write_checkpoint(args.trace_output, trace)
    return 0


if __name__ == "__main__":
    benchmark._set_process_title("EvalEngine")
    sys.exit(main())
