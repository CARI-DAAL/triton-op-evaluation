#!/usr/bin/env python3
"""NPU-only profiler feedback loop, never an admission or independent Sample.

Run in a fresh output directory. It exercises the service's actual profiler
MeasureContext (5 warmups / 20 repeats) on a 256x256 matrix product. A pass
requires eight finite positive observations and their median. Raw profiler
artifacts are retained, including failed captures; no timing backend changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def valid_latency(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
        and value > 0
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def probe(output: Path, physical_device: int, samples: int = 8) -> int:
    if samples != 8:
        raise ValueError("The feedback loop requires all eight reference measurements")
    if not 0 <= physical_device < 16:
        raise ValueError("Physical device must be in the sixteen-card fleet")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    isolated_assignment = os.environ.get("TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID")
    if visible != str(physical_device) and not (
        visible == "0" and isolated_assignment == str(physical_device)
    ):
        raise ValueError("Select exactly the requested physical card before Python starts")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_cwd = Path.cwd()
    report: dict[str, Any] = {
        "schema_version": "profiler-latency-probe-v1",
        "is_independent_sample": False,
        "admission_evidence": False,
        "measurement": {"method": "profiler", "warmup": 5, "repeats": 20, "samples": 8},
        "physical_device": physical_device,
        "process_id": os.getpid(),
        "state": "running",
        "samples": [],
        "cleanup_requests_retained": [],
        "started_unix_ns": time.time_ns(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/probe_profiler_latency.py",
                "triton_evaluation/engine/benchmark.py",
                "triton_evaluation/profiler_runtime.py",
            )
        },
    }
    target = output / "report.json"
    write_report(target, report)
    cleanup = None
    try:
        from triton_evaluation.profiler_runtime import describe_runtime

        report["profiler_runtime"] = describe_runtime()
        import torch
        import torch_npu

        from triton_evaluation.engine import benchmark

        versions = {}
        for name in ("torch", "torch-npu", "triton", "triton-ascend"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        report["runtime_versions"] = versions
        torch.npu.set_device(0)
        torch.set_default_device("npu")
        torch.manual_seed(0)
        torch.npu.manual_seed(0)
        report["execution_conditions"] = {
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "logical_device": torch.npu.current_device(),
            "visible_device_count": torch.npu.device_count(),
            "device_name": torch.npu.get_device_name(0),
            "default_device": str(torch.get_default_device()),
            "torch_module_version": torch.__version__,
            "torch_npu_module_version": torch_npu.__version__,
            "reference_device_policy": os.environ.get(
                "TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first"
            ),
        }
        if torch.npu.current_device() != 0 or torch.npu.device_count() != 1:
            raise RuntimeError("The requested physical mask did not select one logical NPU")

        class MatrixProduct(torch.nn.Module):
            def forward(self, value):
                return value @ value

        model = MatrixProduct().eval()
        inputs = [torch.randn(256, 256)]
        with torch.no_grad():
            native_output = model(*inputs)
        torch.npu.synchronize()
        report["native_tensor_proof"] = {
            "input_device": str(inputs[0].device),
            "output_device": str(native_output.device),
            "shape": list(native_output.shape),
            "dtype": str(native_output.dtype),
            "finite": bool(torch.isfinite(native_output).all().item()),
        }
        if (
            str(inputs[0].device) != "npu:0"
            or str(native_output.device) != "npu:0"
            or not report["native_tensor_proof"]["finite"]
        ):
            raise RuntimeError("Minimal matrix product was not observed finite on NPU zero")
        del native_output
        # Failure branches currently remove artifacts even with keep_res=True.
        # Retain only that filesystem effect in this diagnostic process; the
        # actual profiler configuration, capture and latency parser stay intact.
        cleanup = benchmark.cleanup_profile_path

        def retain(path):
            report["cleanup_requests_retained"].append(str(Path(path).resolve()))

        benchmark.cleanup_profile_path = retain
        os.chdir(output)
        for number in range(1, samples + 1):
            started = time.time_ns()
            operators, latency, peak = benchmark.measure_single(
                benchmark.MeasureContext(
                    model=model,
                    inputs=inputs,
                    warmup=5,
                    repeats=20,
                    profile_name=f"sample-{number:02d}",
                    device=torch.device("npu"),
                    method="profiler",
                ),
                clear_l2_cache=False,
                keep_res=True,
            )
            row: dict[str, Any] = {
                "sample": number,
                "latency_ms": float(latency) if valid_latency(latency) else None,
                "raw_latency": repr(latency),
                "operators": operators,
                "peak_memory_mb": peak,
                "elapsed_ns": time.time_ns() - started,
                "passed": valid_latency(latency),
            }
            runtime_traces = list(output.glob(f"sample-{number:02d}_*/profiler-runtime.json"))
            row["profiler_runtime_traces"] = [
                json.loads(path.read_bytes()) for path in runtime_traces
            ]
            # Preserve the actual measurement even if its runtime proof is
            # incomplete. It remains an observed failed diagnostic attempt.
            report["samples"].append(row)
            if report["profiler_runtime"]["enabled"]:
                expected = report["profiler_runtime"]
                traces = row["profiler_runtime_traces"]
                valid_trace = len(traces) == 1 and all(
                    traces[0].get(key) == expected[key]
                    for key in (
                        "manifest_path",
                        "manifest_sha256",
                        "alias_path",
                        "alias_sha256",
                        "library_path",
                        "library_sha256",
                        "consumer_path",
                        "consumer_sha256",
                        "msprof_path",
                        "msprof_sha256",
                    )
                )
                valid_trace = valid_trace and traces[0].get("cann_runtime") == expected.get(
                    "cann_runtime"
                )
                valid_trace = valid_trace and all(
                    traces[0].get(key) == value
                    for key, value in {
                        "enabled": True,
                        "callback_scope_completed": True,
                        "base_ld_restored": True,
                        "main_thread": True,
                        "process_id": os.getpid(),
                        "helper_sha256": report["source_sha256"][
                            "triton_evaluation/profiler_runtime.py"
                        ],
                        "helper_path": str(ROOT / "triton_evaluation/profiler_runtime.py"),
                        "service_package_path": str(ROOT / "triton_evaluation"),
                        "physical_device": str(physical_device),
                        "visible_devices": visible,
                        "reference_device_policy": report["execution_conditions"][
                            "reference_device_policy"
                        ],
                        "effective_ld_prefix": str(Path(expected["alias_path"]).parent),
                        "previous_ld_sha256": expected["base_ld_sha256"],
                    }.items()
                )
                row["runtime_evidence_valid"] = valid_trace
                if not valid_trace:
                    raise RuntimeError(
                        "Configured profiler runtime lacks actual completed-callback evidence"
                    )
            write_report(target, report)
            if not row["passed"]:
                raise RuntimeError(
                    f"sample {number}: reference latency must be finite and positive; "
                    f"got {latency!r}"
                )
        report["median_ms"] = statistics.median(row["latency_ms"] for row in report["samples"])
        report["state"] = "passed"
    except Exception as error:
        report.update(state="failed", error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        if cleanup is not None:
            benchmark.cleanup_profile_path = cleanup
        os.chdir(original_cwd)
        report["finished_unix_ns"] = time.time_ns()
        report["artifacts"] = [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != target
        ]
        write_report(target, report)
    print(
        json.dumps(
            {
                "state": report["state"],
                "report": str(target),
                "observed_samples": len(report["samples"]),
                "median_ms": report.get("median_ms"),
                "error": report.get("error"),
            },
            allow_nan=False,
        )
    )
    return 0 if report["state"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=8)
    parser.add_argument(
        "--sandbox", action="store_true", help="Use the actual worker run_script boundary"
    )
    args = parser.parse_args()
    return (
        isolated_probe(args.output, args.device)
        if args.sandbox
        else probe(args.output, args.device)
    )


def isolated_probe(output: Path, physical_device: int) -> int:
    """Run this exact tracked probe through the production invocation boundary."""
    from triton_evaluation.evaluators.base import run_script
    from triton_evaluation.evaluators.sandbox import (
        CONFIG_ENV,
        DIGEST_ENV,
        OVERLAY_CONFIG_ENV,
        OVERLAY_DIGEST_ENV,
    )

    if any(
        key not in os.environ
        for key in (CONFIG_ENV, DIGEST_ENV, OVERLAY_CONFIG_ENV, OVERLAY_DIGEST_ENV)
    ):
        raise ValueError(
            "Isolated profiler probe requires original anchor and new overlay pin pairs"
        )
    work = output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    target = work / "sandbox-probe.json"
    report = {
        "schema_version": "isolated-profiler-probe-v1",
        "is_independent_sample": False,
        "admission_evidence": False,
        "state": "running",
        "launcher_pid": os.getpid(),
        "physical_device": physical_device,
        "started_unix_ns": time.time_ns(),
        "anchor_manifest_path": os.environ[CONFIG_ENV],
        "anchor_manifest_sha256": os.environ[DIGEST_ENV],
        "overlay_path": os.environ[OVERLAY_CONFIG_ENV],
        "overlay_sha256": os.environ[OVERLAY_DIGEST_ENV],
    }
    write_report(target, report)
    try:
        result = run_script(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--device",
                str(physical_device),
                "--output",
                str(work / "native"),
            ],
            work,
            physical_device,
            budget_seconds=180,
        )
        (work / "child.stdout").write_text(result.stdout)
        (work / "child.stderr").write_text(result.stderr)
        report.update(
            exit_code=result.exit_code, outcome=result.outcome, timed_out=result.timed_out
        )
        native_path = work / "native/report.json"
        if native_path.exists():
            native = json.loads(native_path.read_bytes())
            report["native_report_path"] = str(native_path)
            report["native_report_sha256"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
            report["child_namespace_pid"] = native.get("process_id")
            if (
                result.exit_code == 0
                and result.outcome == "exited"
                and not result.timed_out
                and native.get("state") == "passed"
                and len(native.get("samples", [])) == 8
                and native.get("profiler_runtime", {}).get("enabled") is True
                and native.get("physical_device") == physical_device
                and valid_latency(native.get("median_ms"))
                and all(
                    row.get("passed") is True
                    and row.get("runtime_evidence_valid") is True
                    and valid_latency(row.get("latency_ms"))
                    for row in native["samples"]
                )
                and native.get("native_tensor_proof", {}).get("input_device") == "npu:0"
                and native.get("native_tensor_proof", {}).get("output_device") == "npu:0"
                and native.get("native_tensor_proof", {}).get("finite") is True
            ):
                report["state"] = "passed"
            else:
                report["state"] = "failed"
        else:
            report.update(state="failed", error="Actual sandbox child produced no native report")
    except Exception as error:
        report.update(state="failed", error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        report["finished_unix_ns"] = time.time_ns()
        write_report(target, report)
    print(json.dumps({"state": report["state"], "report": str(target)}))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
