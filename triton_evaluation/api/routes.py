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

from ..admission import ForbiddenError, admit
from ..broker import StorageError
from ..catalog import ManifestBindingError
from ..models import CreateResourceCommand
from ..security import Caller
from . import errors
from .auth import require_read_caller, require_submit_caller
from .contract import EvaluationRequest, EvaluationResource
from .dependencies import AppContext, get_context

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


def _algorithms(context: AppContext):
    from ..models import AlgorithmVersions

    settings = context.settings
    return AlgorithmVersions(
        correctness=settings.correctness_algorithm_version,
        decoy=settings.decoy_algorithm_version,
        benchmark=settings.benchmark_algorithm_version,
        profiler=settings.profiler_algorithm_version,
    )


def _require_ready(context: AppContext, caller: Caller) -> None:
    if not context.artifacts_verified:
        raise errors.not_ready("published contract artifacts are not verified")
    if not context.store.probe_storage():
        raise errors.queue_unavailable()
    if not context.store.has_fresh_compatible_worker(
        type(
            "Compatibility",
            (),
            {"evaluator_build": context.settings.evaluator_build,
             "algorithms": _algorithms(context)},
        )()
    ):
        raise errors.not_ready()


def _build_command(context, caller, session_id, idempotency_key, admission, purpose):
    from ..models import new_evaluation_id

    return CreateResourceCommand(
        evaluation_id=new_evaluation_id(),
        caller_id=caller.caller_id,
        caller_class=caller.caller_class,
        traffic_class="trial" if purpose == "trial" else "trusted",
        session_id=session_id,
        idempotency_key=idempotency_key,
        admission=admission,
        evaluator_build=context.settings.evaluator_build,
        algorithms=_algorithms(context),
    )


async def wait_until_terminal(
    store, caller, evaluation_id, *, budget_seconds, poll_seconds, is_disconnected
) -> EvaluationResource | None:
    """Poll the durable resource until terminal, budget end, or disconnect.

    Waiting never terminalizes and never cancels; the caller simply receives
    the current nonterminal resource when its budget ends.
    """
    deadline = time.monotonic() + budget_seconds
    resource = await asyncio.to_thread(store.get_visible, evaluation_id, caller)
    while resource is not None and resource.state not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or await _maybe_await(is_disconnected):
            break
        await asyncio.sleep(min(poll_seconds, max(remaining, 0)))
        resource = await asyncio.to_thread(store.get_visible, evaluation_id, caller)
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
    context=Depends(get_context),
):
    if not 0 <= wait_seconds <= 60:
        raise errors.invalid_request("wait_seconds must be between 0 and 60")
    session_id = _validate_session_id(request.headers.get("X-Session-ID"))
    idempotency_key = _validate_idempotency_key(request.headers.get("Idempotency-Key"))

    evaluation_request = await _decode_and_validate(request)

    try:
        admission = admit(evaluation_request, caller, context.catalog)
    except ForbiddenError:
        raise errors.forbidden("caller lacks the required scope") from None
    except ManifestBindingError:
        raise errors.manifest_binding_failed() from None

    command = _build_command(
        context, caller, session_id, idempotency_key, admission,
        evaluation_request.purpose,
    )

    # An idempotent replay stays recoverable through transient unreadiness:
    # the record is consulted before readiness applies to a new admission.
    replayed = await asyncio.to_thread(
        context.store.lookup_idempotent, caller.caller_id, idempotency_key
    )
    if replayed is None:
        _require_ready(context, caller)

    try:
        outcome = await asyncio.to_thread(context.store.create_or_replay, command)
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
            context.store,
            caller,
            resource.evaluation_id,
            budget_seconds=wait_seconds,
            poll_seconds=context.settings.api_poll_interval_seconds,
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
        content=resource.model_dump(mode="json", exclude_none=False),
        status_code=202 if accepted else 200,
        headers=headers,
    )


@router.get("/evaluations/{evaluation_id}")
async def get_evaluation(
    evaluation_id: str,
    caller=Depends(require_read_caller),
    context=Depends(get_context),
):
    resource = await asyncio.to_thread(
        context.store.get_visible, evaluation_id, caller
    )
    if resource is None:
        # Unknown, invisible, and invalidly shaped IDs are indistinguishable.
        raise errors.not_found()
    headers = (
        dict(errors.RETRY_AFTER_ONE) if resource.state not in TERMINAL_STATES else None
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content=resource.model_dump(mode="json", exclude_none=False),
        status_code=200,
        headers=headers,
    )
