"""Legacy kernel_metrics and bottleneck_hint synthesis from v2.1 op profiles.

Ports the legacy run_verify diagnostics (_compute_bottleneck_hint in
uni_agent/tools/run_verify/run_verify, KB1 profiling-guide rules) onto the
v2.1 OpProfileCase contract (OpKernel.time_ms / percentage / metrics with
aiv_vec_ratio, aiv_mte2_ratio, aic_cube_ratio, ...). The composed dict
matches the legacy summary schema so the observation is agent-imperceptible
regardless of whether a sandbox-local NPU or the service served the call.
"""

from __future__ import annotations

from typing import Any

# Legacy KB1 rules (profiling-guide.md 155-160):
#   1. aiv_vec_ratio  < 10% -> Vector underutilized
#   2. aiv_scalar_ratio > 30% -> scalar degradation (possible; pipelining may hide it)
#   3. aiv_mte2_ratio  > 50% -> memory-transfer bound
#   4. aic_cube_ratio  < 30% -> Cube stall
# v2.1 has no aiv_scalar_ratio column; it is derived as the unaccounted
# non-transfer, non-vector share and only reported when both are low enough
# that scalar work plausibly dominates (the rule was advisory anyway).

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


def _time_us(entry: dict[str, Any]) -> float:
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
