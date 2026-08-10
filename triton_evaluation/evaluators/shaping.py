"""Project private engine JSON into the stable evaluation v2 payload."""

import json
import math
from typing import Any


class ShapingError(ValueError):
    """An engine result cannot satisfy the stable v2 contract."""


def _case_failure(raw: dict) -> dict:
    metrics = raw.get("metrics") or {}
    kind = raw.get("kind")
    return {
        "case_idx": raw.get("case_idx"),
        "kind": kind,
        "metrics": {
            "max_abs_diff": metrics.get("max_abs_diff"),
            "mean_abs_err": None,
            "matched_ratio": metrics.get("matched_ratio"),
            "mere": metrics.get("MERE"),
            "small_count": metrics.get("small_count"),
            "normal_count": metrics.get("normal_count"),
            "atol": metrics.get("atol"),
            "rtol": metrics.get("rtol"),
        },
        "sample": raw.get("sample") or [],
        "error": None if kind is not None else (raw.get("error_msg") or raw.get("error_type")),
    }


def shape_correctness(verify_json: dict | None, *, decoy: bool = False) -> dict | None:
    if decoy:
        return {
            "passed": False,
            "compiled": False,
            "decoy": True,
            "passed_cases": 0,
            "total_cases": 0,
            "failures": [],
        }
    if verify_json is None:
        return None
    total = verify_json.get("total_cases", 0)
    passed = verify_json.get("passed_cases", 0)
    failures = verify_json.get("failures", [])
    compiled = not any(failure.get("error_type") == "CompilationError" for failure in failures)
    return {
        "passed": total > 0 and passed == total,
        "compiled": compiled,
        "decoy": False,
        "passed_cases": passed,
        "total_cases": total,
        "failures": [_case_failure(failure) for failure in failures],
    }


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _shape_id(input_desc: list[dict]) -> str:
    shapes = [
        item["shape"]
        for item in input_desc
        if item.get("type") == "tensor" and isinstance(item.get("shape"), list)
    ]
    value = shapes[0] if len(shapes) == 1 else shapes
    return json.dumps(value, separators=(",", ":"))


def shape_performance(perf_json: dict) -> dict:
    overall = perf_json.get("speedup_vs_torch")
    if perf_json.get("failed_cases", 0) != 0 or not _positive_finite(overall):
        raise ShapingError("benchmark result is incomplete")

    per_shape = []
    for raw in perf_json.get("per_shape_results") or []:
        implementation = raw.get("implementation") or {}
        latency = implementation.get("avg_latency_ms")
        speedup = raw.get("speedup_vs_torch")
        if raw.get("status") != "pass" or not _positive_finite(latency):
            raise ShapingError("benchmark contains a failed or invalid shape")
        if not _positive_finite(speedup):
            raise ShapingError("benchmark contains an invalid shape speedup")
        per_shape.append(
            {
                "shape": _shape_id(raw.get("input_desc") or []),
                "latency_ms": float(latency),
                "throughput": None,
                "speedup_vs_torch": float(speedup),
            }
        )
    if not per_shape:
        raise ShapingError("benchmark contains no valid per-shape results")
    return {"speedup_vs_torch": float(overall), "per_shape": per_shape}


def _profile_state(profile_json: dict) -> str:
    selected = profile_json.get("selected_cases", 0)
    completed = profile_json.get("completed_cases", 0)
    partial = profile_json.get("partial_cases", 0)
    failed = profile_json.get("failed_cases", 0)
    timed_out = bool(profile_json.get("timed_out"))
    if not timed_out and selected > 0 and completed >= selected and partial == 0 and failed == 0:
        return "complete"
    return "partial"


def _op_profiles(profile_json: dict) -> list[dict]:
    profiles = []
    for case in profile_json.get("per_case_results") or []:
        timed = [
            (kernel, kernel.get("duration_us"))
            for kernel in case.get("kernels") or []
            if _positive_finite(kernel.get("duration_us"))
        ]
        total_us = sum(duration for _, duration in timed)
        for kernel, duration_us in timed:
            profiles.append(
                {
                    "case_idx": case.get("case_idx"),
                    "op_name": kernel.get("name"),
                    "time_ms": duration_us / 1000.0,
                    "percentage": duration_us / total_us * 100.0 if total_us else None,
                }
            )
    return profiles


def _ratio(ratios: dict, name: str) -> float | None:
    value = ratios.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _scalar_ratio(ratios: dict) -> float | None:
    values = [_ratio(ratios, "SCALAR"), _ratio(ratios, "SCALARLDST")]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _deep_summaries(profile_json: dict) -> list[dict]:
    summaries = []
    for case in profile_json.get("per_case_results") or []:
        bound = case.get("bound") or {}
        ratios = bound.get("ratios") or {}
        per_core = case.get("per_core") or {}
        recommendations = [
            item.get("recommendation")
            for item in case.get("recommendations") or []
            if isinstance(item, dict) and item.get("recommendation")
        ]
        summaries.append(
            {
                "case_idx": case.get("case_idx"),
                "status": case.get("status"),
                "target_kernel": case.get("target_kernel"),
                "bound": bound.get("name"),
                "rule": bound.get("rule"),
                "mte2_ratio": _ratio(ratios, "MTE2"),
                "cube_ratio": _ratio(ratios, "CUBE"),
                "vec_ratio": _ratio(ratios, "PUSHQ"),
                "fixp_ratio": _ratio(ratios, "FIXP"),
                "mte3_ratio": _ratio(ratios, "MTE3"),
                "scalar_ratio": _scalar_ratio(ratios),
                "per_core_spread_ratio": per_core.get("spread_ratio"),
                "per_core_verdict": per_core.get("verdict"),
                "recommendations": recommendations,
                "missing_metrics": list(case.get("missing_metrics") or []),
                "warnings": list(case.get("warnings") or []),
                "error": case.get("error"),
            }
        )
    return summaries


def shape_profile(profile_json: dict) -> dict:
    mode = profile_json.get("profile_mode")
    if mode not in {"op", "deep"}:
        raise ShapingError("profile result has an invalid mode")
    data = {
        "mode": mode,
        "selected_cases": profile_json.get("selected_cases", 0),
        "completed_cases": profile_json.get("completed_cases", 0),
        "partial_cases": profile_json.get("partial_cases", 0),
        "failed_cases": profile_json.get("failed_cases", 0),
        "timed_out": bool(profile_json.get("timed_out")),
        "ops": _op_profiles(profile_json) if mode == "op" else None,
        "summaries": _deep_summaries(profile_json) if mode == "deep" else None,
    }
    return {"profile_state": _profile_state(profile_json), "data": data}


def shape_payload(
    verify_json: dict | None,
    perf_json: dict | None,
    profile_json: dict | None,
    *,
    checks: list[str],
    decoy: bool = False,
) -> dict[str, Any]:
    correctness = (
        shape_correctness(verify_json, decoy=decoy) if "correctness" in checks else None
    )
    correctness_passed = bool(correctness and correctness.get("passed"))
    performance = None
    if "performance" in checks and correctness_passed and perf_json is not None:
        performance = shape_performance(perf_json)
    profile = (
        shape_profile(profile_json)
        if "profile" in checks and profile_json is not None
        else None
    )
    return {"correctness": correctness, "performance": performance, "profile": profile}
