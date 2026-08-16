"""Prior-major (v1) wire shapes and the v2 -> v1 projection.

Everything legacy lives here and in ``legacy_routes.py``: the core
``contract.py``, storage records, and worker stay v2-only. The adapters call
the same ``admit()`` and ``EvaluationStore.create_or_replay()`` APIs as
``/evaluations``; they never import service implementation from another
repository.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contract import EvaluationResource


class _LegacyModel(BaseModel):
    """Legacy bodies reject unknown fields; v2-only fields cannot be smuggled."""

    model_config = ConfigDict(extra="forbid")


class LegacyVerifyRequest(_LegacyModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    non_compute: bool = False
    timeout: int = Field(default=900, ge=10, le=3600)
    measure_speed: bool = False
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)


class LegacyProfileRequest(_LegacyModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    non_compute: bool = False
    timeout: int = Field(default=900, ge=10, le=3600)
    mode: Literal["op", "deep"] = "op"
    case_idx: int | None = Field(default=None, ge=0)


class LegacyTaskView(BaseModel):
    task_id: str
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT"]
    error: str | None = None
    result: dict | None = None


_LEGACY_STATE = {
    "queued": "QUEUED",
    "running": "RUNNING",
    "completed": "SUCCEEDED",
    "failed": "FAILED",
    "timed_out": "TIMEOUT",
}
_REJECTED = "result projection unavailable"


def verify_to_request(body: LegacyVerifyRequest) -> dict:
    """Translate a legacy verify body into a v2 trial request payload."""
    payload = {
        "schema_version": "2.0",
        "purpose": "trial",
        "op_name": body.op_name,
        "reference_code": body.reference_code,
        "kernel_code": body.kernel_code,
        "triton_impl_name": body.triton_impl_name,
        "checks": ["correctness", "performance"]
        if body.measure_speed
        else ["correctness"],
        "non_compute": body.non_compute,
        "deadline_seconds": body.timeout,
    }
    if body.measure_speed:
        payload["performance"] = {"warmup": body.warmup, "repeats": body.repeats}
    return payload


def profile_to_request(body: LegacyProfileRequest) -> dict:
    payload = {
        "schema_version": "2.0",
        "purpose": "trial",
        "op_name": body.op_name,
        "reference_code": body.reference_code,
        "kernel_code": body.kernel_code,
        "triton_impl_name": body.triton_impl_name,
        "checks": ["profile"],
        "non_compute": body.non_compute,
        "deadline_seconds": body.timeout,
        "profile": {"mode": body.mode},
    }
    if body.case_idx is not None:
        payload["profile"]["case_indices"] = [body.case_idx]
    return payload


def project_task(resource: EvaluationResource) -> LegacyTaskView:
    """Project a v2 resource into the legacy task view.

    Unrepresentable values fail closed to a redacted legacy failure; source
    code and private storage never appear.
    """
    status = _LEGACY_STATE[resource.state]
    if resource.state in ("queued", "running") or resource.result is None:
        return LegacyTaskView(task_id=resource.evaluation_id, status=status)
    try:
        result = _project_result(resource)
    except Exception:
        return LegacyTaskView(
            task_id=resource.evaluation_id, status="FAILED", error=_REJECTED
        )
    if resource.state == "failed":
        return LegacyTaskView(
            task_id=resource.evaluation_id, status="FAILED",
            error=_first_error(resource), result=result,
        )
    if resource.state == "timed_out":
        return LegacyTaskView(
            task_id=resource.evaluation_id, status="TIMEOUT",
            error=_first_error(resource), result=result,
        )
    return LegacyTaskView(task_id=resource.evaluation_id, status=status,
                          result=result)


def _first_error(resource) -> str | None:
    result = resource.result
    for outcome in (result.correctness, result.performance, result.profile):
        if outcome is not None and getattr(outcome, "error", None) is not None:
            return outcome.error.message
    return None


def _project_result(resource) -> dict:
    result = resource.result
    projected: dict = {}

    correctness = result.correctness
    if correctness is not None:
        if correctness.status != "succeeded" or correctness.data is None:
            raise ValueError("unrepresentable correctness outcome")
        data = correctness.data
        projected["correctness"] = {
            "passed": data.passed,
            "compiled": data.outcome != "compile_error",
            "decoy": data.decoy,
            "passed_cases": data.passed_cases,
            "total_cases": data.total_cases,
            "failures": [
                {
                    "case_idx": failure.case_idx,
                    "kind": failure.kind,
                    "metrics": failure.metrics.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "sample": [
                        element.model_dump(mode="json", exclude_none=True)
                        for element in failure.sample
                    ],
                }
                for failure in data.failures
            ],
        }

    performance = result.performance
    if performance is not None:
        if performance.status != "succeeded" or performance.data is None:
            raise ValueError("unrepresentable performance outcome")
        data = performance.data
        projected["performance"] = {
            "speedup_vs_torch": data.overall_speedup_vs_reference,
            "per_shape": [
                {
                    "shape": "x".join(
                        str(dim)
                        for shape in case.input_shapes
                        for dim in shape.dimensions
                    )
                    or str(case.case_idx),
                    "latency_ms": case.implementation_latency_ms,
                    "speedup_vs_torch": case.speedup_vs_reference,
                }
                for case in data.per_case
            ],
        }

    profile = result.profile
    if profile is not None:
        if profile.status != "succeeded" or profile.data is None:
            raise ValueError("unrepresentable profile outcome")
        data = profile.data
        counts = data.counts
        projected["profile"] = {
            "profile_state": (
                "complete" if counts.partial == 0 and counts.failed == 0 else "partial"
            ),
            "data": {
                "mode": data.mode,
                "selected_cases": counts.selected,
                "completed_cases": counts.complete,
                "partial_cases": counts.partial,
                "failed_cases": counts.failed,
                "timed_out": False,
                "ops": [
                    {
                        "case_idx": case.case_idx,
                        "op_name": kernel.op_name,
                        "time_ms": kernel.time_ms,
                        "percentage": kernel.percentage,
                    }
                    for case in getattr(data, "cases", [])
                    for kernel in case.kernels
                ]
                if data.mode == "op"
                else None,
                "summaries": [
                    {
                        "case_idx": case.case_idx,
                        "status": case.status,
                        "target_kernel": case.target_kernel,
                        "bound": case.bound,
                        "mte2_ratio": case.ratios.mte2,
                        "cube_ratio": case.ratios.cube,
                        "vec_ratio": case.ratios.vec,
                        "fixp_ratio": case.ratios.fixp,
                        "mte3_ratio": case.ratios.mte3,
                        "scalar_ratio": case.ratios.scalar,
                        "per_core_spread_ratio": case.per_core_spread_ratio,
                        "recommendations": case.recommendations,
                        "missing_metrics": case.missing_metrics,
                        "warnings": case.warnings,
                        "error": case.error.message if case.error else None,
                    }
                    for case in getattr(data, "cases", [])
                ]
                if data.mode == "deep"
                else None,
            },
        }

    return projected
