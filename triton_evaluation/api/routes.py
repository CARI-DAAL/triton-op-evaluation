import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..broker import Broker
from ..models import (
    BenchmarkRequest,
    Job,
    JobSpec,
    JobType,
    SubmitResponse,
    TaskResponse,
    VerifyRequest,
)
from .dependencies import get_broker

router = APIRouter()


@router.post("/verify", response_model=SubmitResponse)
def submit_verify(req: VerifyRequest, broker: Broker = Depends(get_broker)):
    task_id = uuid.uuid4().hex
    spec = JobSpec(
        op_name=req.op_name,
        reference_code=req.reference_code,
        kernel_code=req.kernel_code,
        triton_impl_name=req.triton_impl_name,
        non_compute=req.non_compute,
        timeout=req.timeout,
    )
    broker.create_task(Job(task_id, JobType.VERIFY.value, spec))
    return SubmitResponse(task_id=task_id)


@router.post("/benchmark", response_model=SubmitResponse)
def submit_benchmark(req: BenchmarkRequest, broker: Broker = Depends(get_broker)):
    task_id = uuid.uuid4().hex
    spec = JobSpec(
        op_name=req.op_name,
        reference_code=req.reference_code,
        kernel_code=req.kernel_code,
        triton_impl_name=req.triton_impl_name,
        warmup=req.warmup,
        repeats=req.repeats,
        timeout=req.timeout,
        verify_task_id=req.verify_task_id,
        verify_not_required=req.verify_not_required,
    )
    broker.create_task(Job(task_id, JobType.BENCHMARK.value, spec))
    return SubmitResponse(task_id=task_id)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, broker: Broker = Depends(get_broker)):
    rec = broker.get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskResponse(**rec)


@router.get("/healthz")
def healthz(broker: Broker = Depends(get_broker)):
    try:
        broker.ping()
        return {"redis": "ok"}
    except Exception:
        return {"redis": "down"}
