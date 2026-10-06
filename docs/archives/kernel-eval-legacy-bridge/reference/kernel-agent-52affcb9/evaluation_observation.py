"""One model-facing formatter for direct and HTTP session results."""
from __future__ import annotations

import json


def _truncate(value: object, limit: int = 1500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}\n…[truncated]…"


def _render_verify(result: dict, *, action: str) -> str:
    if result.get("infra_error"):
        return (
            f"[kernel_eval][{action}][infra_error]\n"
            f"error={_truncate(result.get('error'))}"
        )
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
    cases = (
        f"cases={int(result.get('passed_cases') or 0)}/"
        f"{int(result.get('total_cases') or 0)}"
    )
    if action == "verify":
        return f"[kernel_eval][verify][correct] {cases}"
    if not result.get("benchmark_available"):
        return f"[kernel_eval][benchmark][unavailable] verify_{cases}"
    return (
        f"[kernel_eval][benchmark][measured] verify_{cases} "
        f"speedup_geomean={float(result.get('speedup') or 0.0):.3f}x\n"
        f"benchmark={_truncate(json.dumps(result.get('benchmark'), ensure_ascii=False, sort_keys=True))}"
    )


def _render_profile(result: dict) -> str:
    if result.get("infra_error"):
        return (
            "[kernel_eval][profile][infra_error]\n"
            f"error={_truncate(result.get('error'))}"
        )
    state = str(result.get("profile_state") or "complete")
    status = str(result.get("task_status") or "SUCCEEDED")
    lines = [f"[kernel_eval][profile][{state}] task_status={status}"]
    snapshot = result.get("result")
    if snapshot is not None:
        lines.append(
            "result="
            + _truncate(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
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
            "feedback=interactive evaluation budget exhausted; "
            "no evaluator request started"
        )
    elif result["request_status"] == "infra_error":
        lines.append(f"error={_truncate(result.get('error'))}")
    elif result["action"] == "profile":
        lines.append(_render_profile(result))
    else:
        lines.append(_render_verify(result, action=result["action"]))
    return "\n".join(lines)
