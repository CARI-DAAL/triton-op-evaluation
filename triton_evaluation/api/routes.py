"""Durable POST/GET evaluation endpoints.

POST admits (or replays) one durable evaluation and may wait briefly for a
terminal state; GET recovers a visible resource by ID. Once a resource
exists, execution errors are persisted outcomes — never HTTP 5xx. Transport
wait, headers, and session correlation never enter admission.
"""
import asyncio
import json
import re
import time

from fastapi import APIRouter, Depends, Query, Request
from pydantic import ValidationError

from ..admission import ForbiddenError
from ..broker import StorageError
from ..catalog import ManifestBindingError
from ..instance import InstanceNotReadyError, QueueUnavailableError
from . import errors
from .auth import require_read_caller, require_submit_caller
from .contract import EvaluationRequest, EvaluationResource, dump_contract
from .dependencies import get_context, get_instance

router = APIRouter()

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
IDEMPOTENCY_KEY_MAX = 128
_SOURCE_LIMIT_RE = re.compile(r"must be at most \d+ UTF-8 bytes")
TERMINAL_STATES = ("completed", "failed", "timed_out")


def _reject_nonstandard_number(token: str):
    raise ValueError(f"nonstandard JSON constant: {token}")


def _validate_session_id(value: str | None) -> str:
    if value is None or SESSION_ID_RE.fullmatch(value) is None:
        raise errors.invalid_request("invalid or missing X-Session-ID")
    return value


def _validate_idempotency_key(value: str | None) -> str:
    if value is None or not 1 <= len(value) <= IDEMPOTENCY_KEY_MAX:
        raise errors.invalid_request("invalid or missing Idempotency-Key")
    if any(not 0x21 <= ord(char) <= 0x7E for char in value):
        raise errors.invalid_request("invalid or missing Idempotency-Key")
    return value


async def _decode_and_validate(request: Request) -> EvaluationRequest:
    content_type = request.headers.get("Content-Type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        raise errors.invalid_request("content type must be application/json")
    raw = await request.body()
    try:
        payload = json.loads(raw, parse_constant=_reject_nonstandard_number)
    except (ValueError, UnicodeDecodeError):
        raise errors.invalid_request("request body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise errors.invalid_request("request body must be a JSON object")
    try:
        return EvaluationRequest.model_validate(payload)
    except ValidationError as exc:
        text = str(exc)
        if _SOURCE_LIMIT_RE.search(text):
            raise errors.limit_exceeded("source exceeds the size limit") from None
        if all(error["type"] == "value_error" for error in exc.errors()):
            raise errors.invalid_request("request violates a contract combination") \
                from None
        raise errors.invalid_request_schema("request failed schema validation") \
            from None


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


@router.post("/evaluations")
async def submit_evaluation(
    request: Request,
    wait_seconds: int = Query(default=0),
    caller=Depends(require_submit_caller),
    instance=Depends(get_instance),
):
    if not 0 <= wait_seconds <= 60:
        raise errors.invalid_request("wait_seconds must be between 0 and 60")
    session_id = _validate_session_id(request.headers.get("X-Session-ID"))
    idempotency_key = _validate_idempotency_key(request.headers.get("Idempotency-Key"))

    evaluation_request = await _decode_and_validate(request)

    try:
        outcome = await asyncio.to_thread(
            instance.submit, evaluation_request, caller,
            session_id=session_id, idempotency_key=idempotency_key,
        )
    except ForbiddenError:
        raise errors.forbidden() from None
    except ManifestBindingError:
        raise errors.manifest_binding_failed() from None
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
        return _resource_response(resource)

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
            return _resource_response(final)

    return _resource_response(resource, accepted=True)


def _resource_response(resource: EvaluationResource, accepted: bool = False):
    from fastapi.responses import JSONResponse

    headers = {}
    if accepted:
        headers["Location"] = f"/evaluations/{resource.evaluation_id}"
    return JSONResponse(
        content=json.loads(dump_contract(resource)),
        status_code=202 if accepted else 200,
        headers=headers,
    )


@router.get("/evaluations/{evaluation_id}")
async def get_evaluation(
    evaluation_id: str,
    caller=Depends(require_read_caller),
    instance=Depends(get_instance),
):
    resource = await asyncio.to_thread(
        instance.get, evaluation_id, caller
    )
    if resource is None:
        # Unknown, invisible, and invalidly shaped IDs are indistinguishable.
        raise errors.not_found()
    headers = (
        dict(errors.RETRY_AFTER_ONE) if resource.state not in TERMINAL_STATES else None
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content=json.loads(dump_contract(resource)),
        status_code=200,
        headers=headers,
    )


@router.get("/livez")
async def livez():
    """Constant process liveness; touches no store, catalog, or worker state."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(instance=Depends(get_instance)):
    ready = await asyncio.to_thread(instance.is_ready)
    if not ready:
        # No component detail: readiness failures are uniform.
        raise errors.not_ready()
    return {"ready": True}


@router.get("/internal/metrics")
async def internal_metrics(request: Request, context=Depends(get_context)):
    caller = _require_observer(request, context)
    if caller is None:
        raise errors.forbidden("metrics require observer scope")
    metrics = context.instance.metrics
    if metrics is None:
        raise errors.not_found()
    from fastapi.responses import Response

    return Response(
        content=metrics.render_openmetrics(),
        media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
    )


def _require_observer(request: Request, context):
    from .auth import authenticate

    caller = authenticate(request, context.registry)
    if "observer:read" not in caller.scopes:
        return None
    return caller
