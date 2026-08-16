"""Time-boxed legacy adapter routes (compatibility does not weaken v2 security).

``POST /verify`` and ``POST /profile`` translate prior-major bodies into one
trial evaluation through the same admission and durable-store APIs as
``POST /evaluations``; ``GET /tasks/{task_id}`` projects the persisted
``evaluation_id`` back into the legacy task view. Common Authorization,
session, and idempotency headers stay mandatory. The routes disappear when
the window is disabled or past its published sunset instant.
"""
import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..admission import admit
from ..models import CreateResourceCommand
from . import errors
from .auth import require_read_caller, require_submit_caller
from .dependencies import AppContext, get_context
from .legacy_contract import (
    LegacyProfileRequest,
    LegacyTaskView,
    LegacyVerifyRequest,
    profile_to_request,
    project_task,
    verify_to_request,
)
from .routes import _algorithms, _bump, _validate_idempotency_key, _validate_session_id

router = APIRouter()


def _bump_legacy(context: AppContext, route: str) -> None:
    _bump(context, "legacy_route_requests_total", route=route)


@router.post("/verify")
async def legacy_verify(
    request: Request,
    caller=Depends(require_submit_caller),
    context=Depends(get_context),
):
    _bump_legacy(context, "verify")
    if caller.caller_class != "trial":
        raise errors.forbidden("legacy routes accept trial callers only")
    session_id = _validate_session_id(request.headers.get("X-Session-ID"))
    idempotency_key = _validate_idempotency_key(request.headers.get("Idempotency-Key"))
    body = _parse_legacy(LegacyVerifyRequest, await _json_body(request))
    payload = verify_to_request(body)
    return await _admit_legacy(context, caller, session_id, idempotency_key, payload)


@router.post("/profile")
async def legacy_profile(
    request: Request,
    caller=Depends(require_submit_caller),
    context=Depends(get_context),
):
    _bump_legacy(context, "profile")
    if caller.caller_class != "trial":
        raise errors.forbidden("legacy routes accept trial callers only")
    session_id = _validate_session_id(request.headers.get("X-Session-ID"))
    idempotency_key = _validate_idempotency_key(request.headers.get("Idempotency-Key"))
    body = _parse_legacy(LegacyProfileRequest, await _json_body(request))
    payload = profile_to_request(body)
    return await _admit_legacy(context, caller, session_id, idempotency_key, payload)


@router.get("/tasks/{task_id}")
async def legacy_task(
    task_id: str,
    caller=Depends(require_read_caller),
    context=Depends(get_context),
):
    _bump_legacy(context, "tasks")
    resource = await asyncio.to_thread(context.store.get_visible, task_id, caller)
    if resource is None:
        raise errors.not_found()
    view: LegacyTaskView = project_task(resource)
    return JSONResponse(content=view.model_dump(mode="json", exclude_none=False))


def _parse_legacy(model, payload: dict):
    """Unrepresentable legacy values are rejected, never clamped or reinterpreted."""
    from pydantic import ValidationError

    try:
        return model.model_validate(payload)
    except ValidationError:
        raise errors.invalid_request(
            "legacy values are not representable in the current contract"
        ) from None


async def _json_body(request: Request) -> dict:
    import json

    raw = await request.body()
    try:
        payload = json.loads(raw)
    except ValueError:
        raise errors.invalid_request("request body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise errors.invalid_request("request body must be a JSON object")
    return payload


async def _admit_legacy(context, caller, session_id, idempotency_key, payload):
    from ..api.contract import EvaluationRequest

    try:
        evaluation_request = EvaluationRequest.model_validate(payload)
    except Exception:
        raise errors.invalid_request(
            "legacy values are not representable in the current contract"
        ) from None
    admission = admit(evaluation_request, caller, context.catalog)
    command = CreateResourceCommand(
        evaluation_id=_new_evaluation_id(),
        caller_id=caller.caller_id,
        caller_class=caller.caller_class,
        traffic_class="trial",
        session_id=session_id,
        idempotency_key=idempotency_key,
        admission=admission,
        evaluator_build=context.settings.evaluator_build,
        algorithms=_algorithms(context),
    )
    if (
        await asyncio.to_thread(
            context.store.lookup_idempotent, caller.caller_id, idempotency_key
        )
        is None
    ):
        from .routes import _require_ready

        _require_ready(context, caller)
    outcome = await asyncio.to_thread(context.store.create_or_replay, command)
    if outcome.decision == "conflict":
        raise errors.idempotency_conflict()
    if outcome.decision == "rate_limited":
        raise errors.rate_limited()
    if outcome.decision == "limit_exceeded":
        raise errors.limit_exceeded("caller inflight limit exceeded")
    resource = outcome.resource
    assert resource is not None
    # Legacy POST returns immediately after durable admission.
    return JSONResponse(
        content={"task_id": resource.evaluation_id},
        status_code=202,
        headers={"Location": f"/tasks/{resource.evaluation_id}"},
    )


def _new_evaluation_id() -> str:
    from ..models import new_evaluation_id

    return new_evaluation_id()
