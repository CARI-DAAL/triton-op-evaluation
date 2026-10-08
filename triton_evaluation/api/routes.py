"""Durable POST/GET evaluation endpoints.

POST admits (or replays) one durable evaluation and may wait briefly for a
terminal state; GET recovers a visible resource by ID. Once a resource
exists, execution errors are persisted outcomes — never HTTP 5xx. Transport
wait, headers, and session correlation never enter admission.
"""
import asyncio
import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import Response

from ..evaluators.operator import to_evaluation_request
from ..service.admission import ForbiddenError
from ..service.broker import StorageError
from ..service.instance import EvaluationInstance, InstanceNotReadyError, QueueUnavailableError
from ..service.security import Caller
from . import errors
from .auth import require_observer, require_read_caller, require_submit_caller
from .contract import EvaluationResource
from .dependencies import AppContext, get_context, get_instance
from .errors import error_responses
from .operator import OperatorRequest
from .schemas import IdempotencyKey, Liveness, Readiness, SessionId

router = APIRouter()

TERMINAL_STATES = ("completed", "failed", "timed_out")


async def wait_until_terminal(
    instance, caller, evaluation_id, *, budget_seconds, poll_seconds, is_disconnected
) -> EvaluationResource | None:
    """Poll the durable resource until terminal, budget end, or disconnect.

    Waiting never terminalizes and never cancels; the caller simply receives
    the current nonterminal resource when its budget ends.
    """
    deadline = time.monotonic() + budget_seconds
    resource = await asyncio.to_thread(instance.get, evaluation_id, caller)
    while resource is not None and resource.state not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or await _maybe_await(is_disconnected):
            break
        await asyncio.sleep(min(poll_seconds, max(remaining, 0)))
        resource = await asyncio.to_thread(instance.get, evaluation_id, caller)
    return resource


async def _maybe_await(value):
    import inspect

    result = value()
    if inspect.isawaitable(result):
        return await result
    return result


SUBMIT_RESPONSES = {
    **error_responses(400, 401, 403, 409, 413, 422, 429, 500, 503),
    202: {
        "model": EvaluationResource,
        "description": "Accepted or replayed evaluation still running.",
        "headers": {"Location": {
            "description": "Resource recovery URL.", "schema": {"type": "string"},
        }},
    },
}


@router.post("/evaluations", response_model=EvaluationResource, responses=SUBMIT_RESPONSES)
@router.post("/verify", response_model=EvaluationResource, responses=SUBMIT_RESPONSES)
@router.post("/benchmark", response_model=EvaluationResource, responses=SUBMIT_RESPONSES)
@router.post("/profile", response_model=EvaluationResource, responses=SUBMIT_RESPONSES)
async def submit_evaluation(
    request: Request,
    response: Response,
    evaluation_request: OperatorRequest,
    caller: Annotated[Caller, Depends(require_submit_caller)],
    instance: Annotated[EvaluationInstance, Depends(get_instance)],
    session_id: Annotated[SessionId | None, Header(alias="X-Session-ID")] = None,
    idempotency_key: Annotated[IdempotencyKey | None, Header(alias="Idempotency-Key")] = None,
    wait_seconds: Annotated[int, Query(
        ge=0, le=60,
        description="Wait for a terminal result; expiry returns 202 without cancelling work.",
    )] = 0,
) -> EvaluationResource:
    checks = {
        "verify": ["correctness"],
        "benchmark": (
            ["correctness", "performance"]
            if evaluation_request.torch_code is not None
            and evaluation_request.triton_code is not None
            else ["performance"]
        ),
        "profile": ["profile"],
    }.get(request.scope["route"].path.rsplit("/", 1)[-1])
    try:
        internal_request = to_evaluation_request(evaluation_request, checks=checks)
    except ValueError:
        raise errors.invalid_request_schema("request failed schema validation") from None
    session_id = session_id or ("default" if idempotency_key else uuid4().hex)
    idempotency_key = idempotency_key or uuid4().hex
    try:
        outcome = await asyncio.to_thread(
            instance.submit, internal_request, caller,
            session_id=session_id, idempotency_key=idempotency_key,
        )
    except ForbiddenError:
        raise errors.forbidden() from None
    except InstanceNotReadyError:
        raise errors.not_ready() from None
    except QueueUnavailableError:
        raise errors.queue_unavailable() from None
    except StorageError:
        raise errors.internal() from None

    if outcome.decision == "conflict":
        raise errors.idempotency_conflict()
    if outcome.decision == "rate_limited":
        raise errors.rate_limited()
    if outcome.decision == "limit_exceeded":
        raise errors.limit_exceeded("caller inflight limit exceeded")
    resource = outcome.resource
    assert resource is not None

    if resource.state in TERMINAL_STATES:
        # A replayed terminal resource answers 200 immediately.
        return resource

    if wait_seconds > 0:
        final = await wait_until_terminal(
            instance,
            caller,
            resource.evaluation_id,
            budget_seconds=wait_seconds,
            poll_seconds=instance.settings.api_poll_interval_seconds,
            is_disconnected=request.is_disconnected,
        )
        if final is not None and final.state in TERMINAL_STATES:
            return final

    response.status_code = 202
    response.headers["Location"] = request.url_for(
        "get_evaluation", evaluation_id=resource.evaluation_id
    ).path
    return resource


@router.get(
    "/evaluations/{evaluation_id}", response_model=EvaluationResource,
    responses={
        **error_responses(401, 403, 404, 413, 422, 500),
        200: {"headers": {"Retry-After": {
            "description": "Polling delay for a pending resource, in seconds.",
            "schema": {"type": "string", "const": "1"},
        }}},
    },
)
async def get_evaluation(
    evaluation_id: str,
    response: Response,
    caller: Annotated[Caller, Depends(require_read_caller)],
    instance: Annotated[EvaluationInstance, Depends(get_instance)],
) -> EvaluationResource:
    resource = await asyncio.to_thread(
        instance.get, evaluation_id, caller
    )
    if resource is None:
        # Unknown, invisible, and invalidly shaped IDs are indistinguishable.
        raise errors.not_found()
    if resource.state not in TERMINAL_STATES:
        response.headers.update(errors.RETRY_AFTER_ONE)
    return resource


@router.get("/livez", response_model=Liveness, responses=error_responses(413, 500))
async def livez():
    """Constant process liveness; touches no store or worker state."""
    return {"status": "ok"}


@router.get("/health", response_model=Readiness, responses=error_responses(413, 500, 503))
@router.get("/readyz", response_model=Readiness, responses=error_responses(413, 500, 503))
async def readyz(instance: Annotated[EvaluationInstance, Depends(get_instance)]):
    ready = await asyncio.to_thread(instance.is_ready)
    if not ready:
        # No component detail: readiness failures are uniform.
        raise errors.not_ready()
    return {"ready": True}


@router.get(
    "/internal/metrics", response_class=Response, dependencies=[Depends(require_observer)],
    responses={
        **error_responses(401, 403, 404, 413, 500),
        200: {"content": {"application/openmetrics-text": {"schema": {"type": "string"}}}},
    },
)
def internal_metrics(context: Annotated[AppContext, Depends(get_context)]):
    metrics = context.instance.metrics
    if metrics is None:
        raise errors.not_found()
    return Response(
        content=metrics.render_openmetrics(),
        media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
    )
