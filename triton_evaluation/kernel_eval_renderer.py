"""Standalone legacy kernel_eval renderer (de-bridged), for service-side use.

Ported verbatim from the bridge's renderer pair:
- /home/e84381970/open-evo-bridge/legacy_contract.py  (observation envelope)
- /home/e84381970/open-evo-bridge/legacy_metrics.py    (diagnostics block)

Upstream sources those files port (do not deviate — the baseline model was
trained against these formats, 2026-09-27 A+ hard-statement):
- kernel_agent/eval/evaluation_observation.py  -> render_evaluation_observation
- kernel_agent/tools/kernel_eval/error_classify.py -> classify_error
- kernel_agent/rewards/triton_eval_client.py   -> verdict semantics (_map_verify_task,
  _map_profile_task, _normalize_benchmark) reimplemented against evaluation-api v2.1
- kernel_agent/eval/session.py                 -> eval_budget accounting ("used/max")

Contract of this module:
- Pure functions only. No bridge imports, no I/O, no network, no clock.
- Input: the evaluation service's terminal resource JSON (schema v2.1:
  {"state": ..., "result": {"correctness": ..., "performance": ..., "profile": ...,
  "compile": ...}}) plus the caller-side session facts (action, kernel_code,
  eval_turn, eval_budget, diagnostics).
- Output: the legacy ``[kernel_eval]`` envelope. ``render`` returns the
  envelope dict; ``render_bytes`` returns its canonical JSON encoding
  (json.dumps, ensure_ascii=False, insertion-ordered) whose ``observation``
  field carries the exact bytes the agent-side kernel_eval tool prints.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# ---------------------------------------------------------------------------
# evaluation_observation.py (verbatim)
# ---------------------------------------------------------------------------


def _truncate(value: object, limit: int = 1500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}\n…[truncated]…"


def _render_verify(result: dict, *, action: str) -> str:
    if result.get("infra_error"):
        return f"[kernel_eval][{action}][infra_error]\nerror={_truncate(result.get('error'))}"
    if not result.get("compiled"):
        label = result.get("error_class") or "compile_error"
        return (
            f"[kernel_eval][{action}][{label}]\n"
            f"error={_truncate(result.get('error') or '(no traceback)')}"
        )
    if result.get("decoy"):
        return (
            f"[kernel_eval][{action}][decoy]\n"
            f"error={_truncate(result.get('error') or 'custom kernel not detected')}"
        )
    if not result.get("correct"):
        label = result.get("error_class") or "incorrect"
        return (
            f"[kernel_eval][{action}][{label}] "
            f"cases={int(result.get('passed_cases') or 0)}/"
            f"{int(result.get('total_cases') or 0)}\n"
            f"error={_truncate(result.get('error') or 'outputs differ from reference')}"
        )
    cases = f"cases={int(result.get('passed_cases') or 0)}/{int(result.get('total_cases') or 0)}"
    if action == "verify":
        return f"[kernel_eval][verify][correct] {cases}"
    if not result.get("benchmark_available"):
        return f"[kernel_eval][benchmark][unavailable] verify_{cases}"
    benchmark_json = json.dumps(result.get("benchmark"), ensure_ascii=False, sort_keys=True)
    return (
        f"[kernel_eval][benchmark][measured] verify_{cases} "
        f"speedup_geomean={float(result.get('speedup') or 0.0):.3f}x\n"
        f"benchmark={_truncate(benchmark_json)}"
    )


def _render_profile(result: dict) -> str:
    if result.get("infra_error"):
        return f"[kernel_eval][profile][infra_error]\nerror={_truncate(result.get('error'))}"
    state = str(result.get("profile_state") or "complete")
    status = str(result.get("task_status") or "SUCCEEDED")
    lines = [f"[kernel_eval][profile][{state}] task_status={status}"]
    snapshot = result.get("result")
    if snapshot is not None:
        lines.append(
            "result=" + _truncate(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        )
    if result.get("error") is not None:
        lines.append(f"error={_truncate(result.get('error'))}")
    return "\n".join(lines)


def render_evaluation_observation(result: dict) -> str:
    """Render a complete accepted/blocked session envelope for the model."""
    lines = [
        "[kernel_eval]",
        f"action={result['action']}",
        f"artifact_sha256={result['artifact_sha256']}",
        f"eval_budget={result['eval_budget']}",
        f"request_status={result['request_status']}",
    ]
    if result["request_status"] == "budget_exhausted":
        lines.append(
            "feedback=interactive evaluation budget exhausted; no evaluator request started"
        )
    elif result["request_status"] == "infra_error":
        lines.append(f"error={_truncate(result.get('error'))}")
    elif result["action"] == "profile":
        lines.append(_render_profile(result))
    else:
        lines.append(_render_verify(result, action=result["action"]))
    diagnostics = result.get("diagnostics")
    if diagnostics:
        lines.append(diagnostics)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# error_classify.py (verbatim)
# ---------------------------------------------------------------------------

_JIT_MARKERS = (
    "min_dot_size",  # tl.dot dim < 16 assert (triton frontend)
    "compilationerror",  # triton.compiler.errors.CompilationError
    "ptxaserror",
    "outofresources",  # triton.runtime.errors.OutOfResources
    "kernel compilation failed",  # evaluator compiled=False error_message
    "triton/compiler/",  # traceback frames inside the Triton compiler
    "triton/language/",  # frontend semantic asserts (dtype/shape in tl ops)
)


def classify_error(traceback_text: str | None) -> str:
    """Classify a grader error/traceback text into one taxonomy class.

    Pure text heuristic — order matters: JIT markers first (a JIT failure is
    wrapped in "Kernel execution failed: ..."), then the validation-specific
    messages (NaN/Inf, shape, numeric), then generic runtime, else "other".
    Empty/unknown text → "other" (callers must not change their verdict on it).
    """
    t = (traceback_text or "").strip()
    if not t:
        return "other"
    low = t.lower()
    if any(m in low for m in _JIT_MARKERS) or (
        "ptxas" in low and ("error" in low or "fatal" in low)
    ):
        return "jit_compile_error"
    if (
        "nan position mismatch" in low
        or "inf position mismatch" in low
        or "inf sign mismatch" in low
    ):
        return "nan_inf"
    if "shape mismatch" in low:
        return "shape_mismatch"
    if (
        "output mismatch" in low
        or "produced incorrect results" in low
        or "allclose" in low
        or "max_abs_err" in low
        or "max_difference" in low
    ):
        return "numerical_mismatch"
    if (
        "kernel execution failed" in low
        or "cuda error" in low
        or "illegal memory access" in low
        or "device-side assert" in low
        or "traceback (most recent call last)" in low
    ):
        return "runtime_error"
    return "other"


# ---------------------------------------------------------------------------
# v2.1 resource -> legacy flat verdict (semantics of triton_eval_client)
# ---------------------------------------------------------------------------

# v2.1 failure kinds -> legacy error_type names that classify_error maps onto
# the same class the old service's messages produced.
_KIND_TYPE = {
    "shape_mismatch": "Shape mismatch",
    "value_mismatch": "AccuracyError",
    "non_finite_mismatch": "NonFiniteMismatch",
}

_DOMAIN_TYPE = {
    "SUBJECT_COMPILE_ERROR": "CompilationError",
    "SUBJECT_RUNTIME_ERROR": "RuntimeError",
    "SUBJECT_EXECUTION_TIMEOUT": "TimeoutError",
}


def _blank_verdict(*, error: str | None = None, infra_error: bool = False) -> dict:
    """The stable public verdict shape (triton_eval_client._blank_verdict)."""
    return {
        "correct": False,
        "correctness_raw": None,
        "decoy": False,
        "speedup": 0.0,
        "compiled": False,
        "error": error,
        "error_class": None,
        "max_abs_err": None,
        "mean_abs_err": None,
        "pass_rate": None,
        "infra_error": infra_error,
        "total_cases": 0,
        "passed_cases": 0,
        "failures": [],
        "benchmark_available": False,
        "benchmark": None,
    }


def _case_message(failure: dict) -> str:
    """Render one v2.1 CaseFailure with the grader-message markers the legacy
    classifier (and the model) expect."""
    kind = failure.get("kind")
    metrics = failure.get("metrics") or {}
    sample = failure.get("sample") or []
    if kind == "shape_mismatch":
        parts = ["Output shape differs from the reference shape"]
        if metrics:
            parts.append(f"metrics={json.dumps(metrics, ensure_ascii=False)}")
        return "; ".join(parts)
    if kind == "non_finite_mismatch":
        parts = ["NaN position mismatch between candidate and reference"]
        if sample:
            parts.append(f"sample={json.dumps(sample, ensure_ascii=False)}")
        return "; ".join(parts)
    parts = ["Kernel produced incorrect results"]
    detail = {
        key: metrics[key]
        for key in ("max_abs_diff", "mere", "matched_ratio", "atol", "rtol")
        if metrics.get(key) is not None
    }
    if detail:
        parts.append(json.dumps(detail, ensure_ascii=False))
    if sample:
        parts.append(f"sample={json.dumps(sample, ensure_ascii=False)}")
    return "; ".join(parts)


def _fold_failures(failures: list[dict]) -> str:
    """Legacy _fold_failures: one "[case N] Type: msg" line per failure."""
    return "\n".join(
        f"[case {failure.get('case_idx')}] "
        f"{_KIND_TYPE.get(failure.get('kind'), failure.get('kind'))}: "
        f"{_case_message(failure)}"
        for failure in failures
    )


def _numeric_from_failures(failures: list[dict]) -> dict:
    for failure in failures:
        metrics = failure.get("metrics") or {}
        if metrics.get("max_abs_diff") is not None:
            return {
                "max_abs_err": metrics.get("max_abs_diff"),
                "mean_abs_err": metrics.get("mere"),
                "pass_rate": metrics.get("matched_ratio"),
            }
    return {"max_abs_err": None, "mean_abs_err": None, "pass_rate": None}


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _legacy_benchmark(perf_check: dict | None) -> tuple[bool, dict | None, float]:
    """v2.1 PerformanceResult -> legacy BenchmarkResult shape.

    Legacy benchmark dicts exposed: speedup_vs_torch, total/passed_cases,
    per_shape_results (case_idx, input_desc, status, framework/
    implementation latencies, speedup_vs_torch) — see verifier-ref
    triton-op-verifier/scripts/benchmark.py BenchmarkResult.
    """
    if perf_check is None or perf_check.get("status") != "succeeded":
        return False, None, 0.0
    data = perf_check.get("data") or {}
    speedup = _finite_positive(data.get("overall_speedup_vs_reference"))

    def input_desc(shapes: list[dict]) -> list[dict]:
        return [
            {
                "input_index": shape.get("input_index"),
                "dimensions": shape.get("dimensions"),
                "dtype": shape.get("dtype"),
            }
            for shape in shapes or []
        ]

    per_shape: list[dict] = []
    for case in data.get("per_case") or []:
        per_shape.append(
            {
                "case_idx": case.get("case_idx"),
                "input_desc": input_desc(case.get("input_shapes")),
                "status": "pass",
                "framework": {"avg_latency_ms": case.get("reference_latency_ms")},
                "implementation": {"avg_latency_ms": case.get("implementation_latency_ms")},
                "speedup_vs_torch": case.get("speedup_vs_reference"),
            }
        )
    for case in data.get("unmeasured_cases") or []:
        per_shape.append(
            {
                "case_idx": case.get("case_idx"),
                "input_desc": input_desc(case.get("input_shapes")),
                "status": "fail",
                "framework": None,
                "implementation": None,
                "speedup_vs_torch": None,
                "error_type": case.get("reason_code"),
                "error_msg": case.get("reason_code"),
            }
        )
    benchmark = {
        "speedup_vs_torch": speedup,
        "total_cases": data.get("selected_cases"),
        "passed_cases": data.get("measured_cases"),
        "coverage": data.get("coverage"),
        "per_shape_results": per_shape,
    }
    return speedup is not None, benchmark, speedup or 0.0


def verify_verdict(result: dict, action: str) -> dict:
    """Map an evaluation-api v2.1 result onto the legacy verify verdict.

    Mirrors triton_eval_client._map_verify_task: decoy first, compiled from
    compile outcome, correct from case counts, error folded per case, class
    from the legacy text classifier.
    """
    check = result.get("correctness") or {}
    status = check.get("status")
    verdict = _blank_verdict()

    if status == "succeeded":
        data = check.get("data") or {}
        outcome = data.get("outcome")
        total = int(data.get("total_cases") or 0)
        passed = int(data.get("passed_cases") or 0)
        failures = [item for item in (data.get("failures") or []) if isinstance(item, dict)]
        verdict["total_cases"] = total
        verdict["passed_cases"] = passed
        verdict["failures"] = failures
        verdict["correctness_raw"] = data

        if data.get("decoy"):
            verdict["decoy"] = True
            verdict["compiled"] = True
            verdict["error"] = "decoy: torch passthrough detected"
            return verdict

        lines = [_fold_failures(failures)]
        domain = data.get("domain_failure")
        if domain:
            lines.append(
                f"[case {domain.get('case_idx')}] "
                f"{_DOMAIN_TYPE.get(domain.get('code'), domain.get('code'))}: "
                f"{domain.get('message')}"
            )
        error = "\n".join(part for part in lines if part)

        compiled = outcome != "compile_error"
        verdict["compiled"] = compiled
        if not compiled:
            compile_check = result.get("compile") or {}
            excerpt = None
            if compile_check.get("status") == "succeeded":
                excerpt = (compile_check.get("data") or {}).get("error_excerpt")
            verdict["error"] = error or excerpt or "kernel compilation failed"
            verdict["error_class"] = classify_error(verdict["error"])
            return verdict

        if total > 0 and passed == total:
            verdict["correct"] = True
            if action == "benchmark":
                available, benchmark, speedup = _legacy_benchmark(result.get("performance"))
                verdict["benchmark_available"] = available
                verdict["benchmark"] = benchmark
                verdict["speedup"] = speedup
            return verdict

        verdict["error"] = error or "verify failed"
        if outcome == "execution_timeout":
            # Legacy TIMEOUT tasks classified as runtime_error.
            verdict["error_class"] = "runtime_error"
            if not verdict["error"] or verdict["error"] == "verify failed":
                verdict["error"] = "verify timeout"
        else:
            verdict["error_class"] = classify_error(verdict["error"]) if verdict["error"] else None
        verdict.update(_numeric_from_failures(failures))
        return verdict

    if status == "timeout":
        error_info = check.get("error") or {}
        verdict["compiled"] = True
        verdict["error"] = error_info.get("message") or "verify timeout"
        verdict["error_class"] = "runtime_error"
        return verdict

    error_info = check.get("error") or {}
    if status == "skipped":
        reason = check.get("reason") or {}
        verdict["error"] = f"verify skipped: {reason.get('code')}: {reason.get('message')}"
    elif error_info:
        verdict["error"] = f"{error_info.get('code')}: {error_info.get('message')}"
    else:
        verdict["error"] = f"verify {status}" if status else "verify result missing"
    verdict["infra_error"] = True
    return verdict


def profile_verdict(check: dict | None) -> dict:
    """Map a v2.1 profile check onto the legacy profile result dict
    (triton_eval_client._map_profile_task semantics)."""
    if check is None:
        return {
            "task_status": "REJECTED",
            "profile_state": "infra_error",
            "result": None,
            "error": "profile result missing",
            "infra_error": True,
        }
    status = check.get("status")
    if status == "succeeded":
        data = check.get("data") or {}
        counts = data.get("counts") or {}
        selected = int(counts.get("selected") or 0)
        failed = int(counts.get("failed") or 0)
        partial = int(counts.get("partial") or 0)
        result = {
            "selected_cases": selected,
            "failed_cases": failed,
            "partial_cases": partial,
            "remaining_case_indices": [],
            "timed_out": False,
            "cases": data.get("cases") or [],
        }
        if selected > 0 and failed == selected:
            state = "failed"
        elif partial or failed:
            state = "partial"
        else:
            state = "complete"
        return {
            "task_status": "SUCCEEDED",
            "profile_state": state,
            "result": result,
            "error": None,
            "infra_error": False,
        }
    if status == "timeout":
        error_info = check.get("error") or {}
        return {
            "task_status": "TIMEOUT",
            "profile_state": "timeout",
            "result": None,
            "error": error_info.get("message") or "profile timeout",
            "infra_error": False,
        }
    error_info = check.get("error") or {}
    if status == "failed":
        return {
            "task_status": "FAILED",
            "profile_state": "failed",
            "result": None,
            "error": (
                f"{error_info.get('code')}: {error_info.get('message')}"
                if error_info
                else "profile failed"
            ),
            "infra_error": False,
        }
    reason = check.get("reason") or {}
    return {
        "task_status": "REJECTED",
        "profile_state": "infra_error",
        "result": None,
        "error": (
            f"{reason.get('code')}: {reason.get('message')}" if reason else "profile rejected"
        ),
        "infra_error": True,
    }


def envelope(
    verdict: dict,
    *,
    action: str,
    kernel_code: str,
    eval_turn: int | None,
    eval_budget: str,
    request_status: str,
    diagnostics: str | None = None,
) -> dict:
    """Attach the session envelope fields (session._store_interactive /
    session_server._normalize_result) and render the observation."""
    result = {
        **verdict,
        "action": action,
        "artifact_sha256": hashlib.sha256(kernel_code.encode("utf-8")).hexdigest(),
        "eval_turn": eval_turn,
        "eval_budget": eval_budget,
        "request_status": request_status,
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    result["observation"] = render_evaluation_observation(result)
    return result


# ---------------------------------------------------------------------------
# legacy_metrics.py (verbatim) — diagnostics block for correct verify/benchmark
# ---------------------------------------------------------------------------

_VEC_MIN = 0.10
_SCALAR_MAX = 0.30
_MTE2_MAX = 0.50
_CUBE_MIN = 0.30


def kernel_metrics_from_profile(profile_check: dict[str, Any] | None) -> dict[str, Any]:
    """Project a v2.1 profile check onto the legacy kernel_metrics schema."""
    if not profile_check or profile_check.get("status") != "succeeded":
        return {}
    data = profile_check.get("data") or {}
    # The v2.1 shaped profile document names the case list ``cases`` (the
    # engine's private output calls it ``per_case_results``; ``op_profile_cases``
    # was an earlier draft name and matches nothing the contract emits).
    cases = data.get("cases") or []
    # The dominant case mirrors the legacy aggregator: the first complete case.
    case = next((c for c in cases if c.get("status") == "complete"), None)
    if case is None:
        return {}
    metrics: dict[str, Any] = {}
    for kernel in case.get("kernels") or []:
        ratios = kernel.get("metrics") or {}
        metrics[kernel.get("op_name") or "unknown"] = {
            "time_ms": kernel.get("time_ms"),
            "percentage": kernel.get("percentage"),
            "aiv_vec_ratio": ratios.get("aiv_vec_ratio"),
            "aiv_mte2_ratio": ratios.get("aiv_mte2_ratio"),
            "aiv_mte3_ratio": ratios.get("aiv_mte3_ratio"),
            "aic_cube_ratio": ratios.get("aic_cube_ratio"),
            "aic_mte2_ratio": ratios.get("aic_mte2_ratio"),
            "gm_to_ub_usage": ratios.get("gm_to_ub_usage"),
            "ub_to_gm_usage": ratios.get("ub_to_gm_usage"),
            "l2_hit": ratios.get("aiv_l2_hit", ratios.get("aic_l2_hit")),
        }
    return metrics


def _time_us(entry: dict) -> float:
    value = entry.get("time_ms")
    try:
        return float(value) * 1000.0 if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def compute_bottleneck_hint(kernel_metrics: dict[str, Any]) -> list[str]:
    """The four KB1 rules over the dominant (longest) kernel."""
    if not kernel_metrics:
        return ["no_bottleneck_data"]
    try:
        dominant = max(kernel_metrics, key=lambda k: _time_us(kernel_metrics[k]))
    except ValueError:
        return ["no_bottleneck_data"]
    m = kernel_metrics.get(dominant) or {}
    if not m:
        return ["no_bottleneck_data"]

    hints: list[str] = []
    vec = m.get("aiv_vec_ratio")
    mte2 = m.get("aiv_mte2_ratio")
    cube = m.get("aic_cube_ratio")

    if vec is not None and vec < _VEC_MIN:
        hints.append("vector_underutilized")
    # Derived scalar share: what is neither vector nor transfer on the AIV.
    if vec is not None and mte2 is not None:
        scalar_share = max(0.0, 1.0 - vec - mte2)
        if scalar_share > _SCALAR_MAX and vec < _SCALAR_MAX:
            hints.append("scalar_degradation_possible")
    if mte2 is not None and mte2 > _MTE2_MAX:
        hints.append("memory_transfer_bound")
    if cube is not None and cube < _CUBE_MIN:
        hints.append("cube_stall")
    if not hints:
        hints.append("no_obvious_bottleneck")
    return hints


def render_diagnostics(kernel_metrics: dict[str, Any]) -> str:
    """Legacy-style diagnostic lines for the observation body."""
    if not kernel_metrics:
        return ""
    dominant = max(kernel_metrics, key=lambda k: _time_us(kernel_metrics[k]))
    entry = kernel_metrics[dominant]
    lines = [f"kernel_metrics: dominant={dominant} time_ms={entry.get('time_ms')}"]
    for name in (
        "aiv_vec_ratio",
        "aiv_mte2_ratio",
        "aic_cube_ratio",
        "gm_to_ub_usage",
        "ub_to_gm_usage",
        "l2_hit",
    ):
        value = entry.get(name)
        if value is not None:
            lines.append(f"  {name}={float(value):.3f}")
    hints = compute_bottleneck_hint(kernel_metrics)
    lines.append(f"bottleneck_hint: {', '.join(hints)}")
    return "\n".join(lines)


def compose_diagnostics(profile_check: dict | None) -> str | None:
    """Diagnostics string (or None) from an internal single-case profile check.

    Service-side replacement for the bridge's auto-diagnostics composition:
    feed the profile check result of the internal case-0 op-profile evaluation.
    """
    return render_diagnostics(kernel_metrics_from_profile(profile_check)) or None


# ---------------------------------------------------------------------------
# service-facing entry points (the de-bridged call surface)
# ---------------------------------------------------------------------------


def render_envelope(
    resource: dict,
    *,
    action: str,
    kernel_code: str,
    eval_turn: int | None,
    eval_budget: str,
    diagnostics: str | None = None,
) -> dict:
    """Render a terminal v2.1 resource into the legacy envelope dict.

    Mirrors bridge.Adapter._evaluate's terminal step: verdict by action,
    request_status derived from the verdict, diagnostics appended only when
    provided (caller composes them for correct verify/benchmark).
    """
    result = resource.get("result") or {}
    if resource.get("state") == "failed":
        message = "unknown failure"
        for check in ("correctness", "performance", "profile", "compile"):
            error = (result.get(check) or {}).get("error")
            if error:
                message = f"{error.get('code', 'UNKNOWN')}: {error.get('message', '')}"
                break
        return render_infra_error(
            action=action,
            kernel_code=kernel_code,
            eval_turn=eval_turn,
            eval_budget=eval_budget,
            message=f"evaluation {resource.get('evaluation_id')} failed: {message}",
        )
    if action == "profile":
        verdict = profile_verdict(result.get("profile"))
    else:
        verdict = verify_verdict(result, action)
    return envelope(
        verdict,
        action=action,
        kernel_code=kernel_code,
        eval_turn=eval_turn,
        eval_budget=eval_budget,
        request_status="infra_error" if verdict.get("infra_error") else "completed",
        diagnostics=diagnostics,
    )


def render_infra_error(
    *,
    action: str,
    kernel_code: str,
    eval_turn: int | None,
    eval_budget: str,
    message: str,
) -> dict:
    """Enveloped transport/admission failure (bridge.Adapter._infra)."""
    return envelope(
        {"infra_error": True, "error": message},
        action=action,
        kernel_code=kernel_code,
        eval_turn=eval_turn,
        eval_budget=eval_budget,
        request_status="infra_error",
    )


def render_budget_exhausted(
    *,
    action: str,
    kernel_code: str,
    eval_budget: str,
) -> dict:
    """Enveloped per-trial budget block (bridge.Adapter.handle_action)."""
    return envelope(
        {},
        action=action,
        kernel_code=kernel_code,
        eval_turn=None,
        eval_budget=eval_budget,
        request_status="budget_exhausted",
    )


def envelope_bytes(envelope_dict: dict) -> bytes:
    """Canonical byte form of the envelope (deterministic serialization)."""
    return json.dumps(envelope_dict, ensure_ascii=False).encode("utf-8")


def render(
    resource: dict,
    *,
    action: str,
    kernel_code: str,
    eval_turn: int | None,
    eval_budget: str,
    diagnostics: str | None = None,
) -> bytes:
    """Terminal resource -> legacy envelope bytes (canonical JSON)."""
    return envelope_bytes(
        render_envelope(
            resource,
            action=action,
            kernel_code=kernel_code,
            eval_turn=eval_turn,
            eval_budget=eval_budget,
            diagnostics=diagnostics,
        )
    )
