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
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(physical_device):
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
        "state": "running",
        "samples": [],
        "cleanup_requests_retained": [],
        "started_unix_ns": time.time_ns(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/probe_profiler_latency.py",
                "triton_evaluation/engine/benchmark.py",
            )
        },
    }
    target = output / "report.json"
    write_report(target, report)
    cleanup = None
    try:
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
            row = {
                "sample": number,
                "latency_ms": float(latency) if valid_latency(latency) else None,
                "raw_latency": repr(latency),
                "operators": operators,
                "peak_memory_mb": peak,
                "elapsed_ns": time.time_ns() - started,
                "passed": valid_latency(latency),
            }
            report["samples"].append(row)
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
    args = parser.parse_args()
    return probe(args.output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
