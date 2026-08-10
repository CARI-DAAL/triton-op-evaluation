import asyncio
import re
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import ValidationError

from ..broker import Broker
from ..models import Job, JobSpec, JobType
from ..worker import compute_session_lease_ttl
from .contract import EvaluationPayload, EvaluationRequest, EvaluationResult
from .dependencies import get_broker

router = APIRouter()

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
TERMINAL = {"SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"}


def _monotonic() -> float:
    return time.monotonic()


def _validate_session_id(value: str | None) -> str:
    if value is None or SESSION_ID_RE.fullmatch(value) is None:
        raise HTTPException(status_code=400, detail="invalid or missing X-Session-ID")
    return value


def _to_evaluation_result(record: dict) -> EvaluationResult:
    try:
        payload = EvaluationPayload.model_validate(record.get("result") or {})
        return EvaluationResult(
            task_id=record["task_id"],
            status=record["status"],
            error=record.get("error"),
            result=payload,
        )
    except (KeyError, TypeError, ValidationError):
        return EvaluationResult(
            task_id=record.get("task_id") or "unknown",
            status="FAILED",
            error="invalid worker result",
            result=EvaluationPayload(),
        )


async def _wait_for_result(
    broker: Broker,
    task_id: str,
    deadline: float,
    poll_interval: float,
) -> EvaluationResult:
    while True:
        remaining = deadline - _monotonic()
        if remaining <= 0:
            return EvaluationResult(
                task_id=task_id,
                status="TIMEOUT",
                error="evaluation request timed out",
                result=EvaluationPayload(),
            )
        record = await asyncio.to_thread(broker.get_task, task_id)
        remaining = deadline - _monotonic()
        if remaining <= 0:
            return EvaluationResult(
                task_id=task_id,
                status="TIMEOUT",
                error="evaluation request timed out",
                result=EvaluationPayload(),
            )
        if record is not None and record.get("status") in TERMINAL:
            return _to_evaluation_result(record)
        await asyncio.sleep(min(poll_interval, remaining))


@router.post("/evaluations", response_model=EvaluationResult)
async def submit_evaluation(
    request: EvaluationRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    broker: Broker = Depends(get_broker),
):
    deadline = _monotonic() + request.timeout
    session_id = _validate_session_id(x_session_id)
    if "performance" in request.checks and "correctness" not in request.checks:
        raise HTTPException(status_code=400, detail="performance requires correctness")

    task_id = uuid.uuid4().hex
    spec = JobSpec(
        op_name=request.op_name,
        reference_code=request.reference_code,
        kernel_code=request.kernel_code,
        triton_impl_name=request.triton_impl_name,
        non_compute=request.non_compute,
        timeout=request.timeout,
        checks=list(request.checks),
        warmup=request.performance.warmup if request.performance else 5,
        repeats=request.performance.repeats if request.performance else 50,
        profile_mode=request.profile.mode if request.profile else "op",
        case_idx=request.profile.case_idx if request.profile else None,
        profile_launch_count=request.profile.launch_count if request.profile else None,
        session_id=session_id,
    )
    lease_ttl = compute_session_lease_ttl(spec, broker.s)
    acquired = await asyncio.to_thread(
        broker.acquire_session,
        session_id,
        task_id,
        lease_ttl,
    )
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="session already has an in-flight evaluation",
            headers={"Retry-After": "1"},
        )
    try:
        await asyncio.to_thread(
            broker.create_task,
            Job(task_id, JobType.EVALUATION.value, spec),
        )
    except Exception:
        await asyncio.to_thread(broker.release_session, session_id, task_id)
        raise

    return await _wait_for_result(
        broker,
        task_id,
        deadline,
        broker.s.api_poll_interval_seconds,
    )


@router.get("/healthz")
def healthz(broker: Broker = Depends(get_broker)):
    try:
        broker.ping()
        return {"redis": "ok"}
    except Exception:
        return {"redis": "down"}
