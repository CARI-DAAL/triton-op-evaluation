"""Shaping layer: engine-script JSON -> EvaluationPayload-shaped dict.

No classification lives here (the engine wrote ``kind``/``sample`` into each
failure). This layer only renames/maps fields into the v2 contract shape so the
worker can store one dict and the v2 route can validate it as ``EvaluationResult``.
"""
from typing import Any


def _case_failure(raw: dict) -> dict:
    metrics = raw.get("metrics") or {}
    mere = metrics.get("MERE")  # MERE = mean RELATIVE error (verify.py:_compute_accuracy_state)
    return {
        "case_idx": raw.get("case_idx"),
        "kind": raw.get("kind"),
        "metrics": {
            "max_abs_diff": metrics.get("max_abs_diff"),
            "mean_abs_err": None,  # engine produces no mean ABSOLUTE error (only max_abs_diff + MERE)
            "matched_ratio": metrics.get("matched_ratio"),
            "mere": mere,
            "small_count": metrics.get("small_count"),
            "normal_count": metrics.get("normal_count"),
            "atol": metrics.get("atol"),
            "rtol": metrics.get("rtol"),
        },
        "sample": raw.get("sample") or [],
        "error": raw.get("error_msg") or raw.get("error_type") or "",
    }


def _correctness(verify_json: dict | None, decoy: bool) -> dict | None:
    if decoy:
        return {
            "passed": False, "compiled": False, "decoy": True,
            "passed_cases": 0, "total_cases": 0, "failures": [],
        }
    if verify_json is None:
        return None
    total = verify_json.get("total_cases", 0)
    passed = verify_json.get("passed_cases", 0)
    return {
        "passed": total > 0 and passed == total,
        "compiled": True,
        "decoy": False,
        "passed_cases": passed,
        "total_cases": total,
        "failures": [_case_failure(f) for f in verify_json.get("failures", [])],
    }


def _performance(perf_json: dict | None, correctness_passed: bool) -> dict | None:
    if perf_json is None or not correctness_passed:
        return None
    return {
        "speedup_vs_torch": perf_json.get("speedup_vs_torch"),
        "per_shape": perf_json.get("per_shape_results", []),
    }


def _profile_state(profile_json: dict) -> str:
    selected = profile_json.get("selected_cases", 0)
    completed = profile_json.get("completed_cases", 0)
    failed = profile_json.get("failed_cases", 0)
    if failed == 0 and completed >= selected and selected > 0:
        return "complete"
    return "partial"


def _profile(profile_json: dict | None) -> dict | None:
    if profile_json is None:
        return None
    return {"profile_state": _profile_state(profile_json), "data": dict(profile_json)}


def shape_payload(verify_json: dict | None, perf_json: dict | None,
                  profile_json: dict | None, *, checks: list[str],
                  decoy: bool = False) -> dict[str, Any]:
    correctness = _correctness(verify_json, decoy) if "correctness" in checks else None
    correctness_passed = bool(correctness and correctness.get("passed"))
    performance = (
        _performance(perf_json, correctness_passed) if "performance" in checks else None
    )
    profile = _profile(profile_json) if "profile" in checks else None
    return {"correctness": correctness, "performance": performance, "profile": profile}
