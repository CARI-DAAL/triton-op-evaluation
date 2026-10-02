"""Native kernel_eval tool protocol over the owning durable evaluation instance.

This module has no adapter, bridge process, upstream HTTP or alternate catalog
dependency. The single sandbox token authenticates a fixed trial caller;
instance.submit/get retain readiness, admission quotas and visibility checks.
"""

import asyncio
import hashlib
import logging
import secrets
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import kernel_eval_renderer as renderer
from ..admission import ForbiddenError
from ..broker import StorageError
from ..catalog import ManifestBindingError
from ..instance import InstanceNotReadyError, QueueUnavailableError
from ..kernel_eval_budget import EvalBudgetRegistry
from ..security import Caller
from . import errors
from .contract import EvaluationRequest, ProfileParams
from .dependencies import get_context
from .routes import SESSION_ID_RE, TERMINAL_STATES, _validate_idempotency_key

logger = logging.getLogger(__name__)
router = APIRouter()
TOOL_CALLER = Caller("kernel-eval", "trial", frozenset({"trial:submit", "trial:read"}))


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _enabled(settings) -> bool:
    return settings.kernel_eval_enabled and bool(settings.kernel_eval_token.get_secret_value())


def _session_id(request: Request, body: dict) -> str | None:
    body_session = body.get("session")
    header_session = request.headers.get("X-Session-ID")
    if body_session is not None and header_session is not None and body_session != header_session:
        return None
    value = body_session if body_session is not None else header_session
    if value is None:
        # Compatibility with the original per-container-IP contract. New
        # callers supply a trial ID to separate trials sharing one host/IP.
        remote = request.client.host if request.client is not None else "unknown"
        return "client-" + hashlib.sha256(remote.encode()).hexdigest()[:32]
    if not isinstance(value, str) or SESSION_ID_RE.fullmatch(value) is None:
        return None
    return value


def _profile_options(body: dict) -> dict | None:
    mode = body.get("profile_mode", "op")
    case = body.get("case_idx")
    launches = body.get("profile_launch_count")
    if (
        mode not in ("op", "deep")
        or (case is not None and (type(case) is not int or case < 0))
        or (launches is not None and (type(launches) is not int or not 1 <= launches <= 100))
        or (mode == "deep" and launches is not None)
    ):
        return None
    profile = {"mode": mode}
    if case is not None:
        profile["case_indices"] = [case]
    # The legacy contract validates launch_count but never forwards it.
    return profile


@router.post("/verify")
@router.post("/benchmark")
@router.post("/profile")
async def handle_action(request: Request):
    context = get_context(request)
    instance = context.instance
    settings = instance.settings
    action = request.url.path.rsplit("/", 1)[-1]
    if not _enabled(settings):
        raise errors.not_found()
    presented = request.headers.get("Authorization", "")
    expected = "Bearer " + settings.kernel_eval_token.get_secret_value()
    if not secrets.compare_digest(presented.encode(), expected.encode()):
        return _error("unauthorized", 401)
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _error("invalid json", 400)
    if not isinstance(body, dict):
        return _error("invalid json", 400)
    allowed = {"task_id", "kernel_code", "session"}
    if action == "profile":
        allowed.update({"profile_mode", "case_idx", "profile_launch_count"})
    if any("reference" in str(key).lower() for key in body) or set(body) - allowed:
        return _error("reference fields are forbidden", 400)
    task_id, kernel_code = body.get("task_id"), body.get("kernel_code")
    if not isinstance(task_id, str) or not isinstance(kernel_code, str):
        return _error("task_id and kernel_code are required", 400)
    try:
        task = instance.catalog.resolve_official_task(
            task_id, manifest_id=settings.kernel_eval_manifest_id
        )
    except ManifestBindingError:
        return _error("task_id mismatch", 403)
    profile = _profile_options(body) if action == "profile" else None
    if action == "profile" and profile is None:
        return _error("invalid profile options", 400)
    session_id = _session_id(request, body)
    if session_id is None:
        return _error("invalid session", 400)
    try:
        evaluation_request = EvaluationRequest.model_validate(
            {
                "schema_version": "2.1",
                "purpose": "trial",
                "op_name": task.op_name,
                "reference_code": task.reference_code,
                "kernel_code": kernel_code,
                "triton_impl_name": "triton_ascend_impl",
                "checks": (
                    ["profile"]
                    if profile
                    else ["correctness", "performance"]
                    if action == "benchmark"
                    else ["correctness"]
                ),
                "deadline_seconds": min(900, int(settings.kernel_eval_max_wait_seconds)),
                "performance": {
                    "warmup": settings.kernel_eval_performance_warmup,
                    "repeats": settings.kernel_eval_performance_repeats,
                    "method": "profiler",
                }
                if action == "benchmark"
                else None,
                "profile": profile,
            }
        )
    except ValidationError:
        return _error("invalid evaluation request", 400)
    payload = evaluation_request.model_dump_json(exclude_none=False).encode()
    digest = hashlib.sha256(payload).hexdigest()
    operation_id = request.headers.get("Idempotency-Key")
    if operation_id is None:
        # Legacy callers consume a turn on each HTTP tool call, even when
        # identical source reuses a durable evaluation. Modern clients may
        # supply a stable key so a transport retry reuses its reserved turn.
        operation_id = uuid.uuid4().hex
        evaluation_identity = hashlib.sha256(payload + action.encode()).hexdigest()
    else:
        try:
            _validate_idempotency_key(operation_id)
        except errors.ApiError:
            return _error("invalid Idempotency-Key", 400)
        evaluation_identity = operation_id
    # The owning store is the only state backend. Both keyspaces share its
    # connection and namespace; no extra Redis or bridge config is required.
    budget = EvalBudgetRegistry(
        instance.store._r,
        prefix=settings.redis_prefix,
        maximum=settings.kernel_eval_max_turns,
        retention_seconds=settings.retention_seconds,
    )
    try:
        reservation = await asyncio.to_thread(
            budget.reserve, TOOL_CALLER.caller_id, session_id, operation_id, digest
        )
    except StorageError:
        return JSONResponse(
            renderer.render_infra_error(
                action=action,
                kernel_code=kernel_code,
                eval_turn=None,
                eval_budget="unknown",
                message=f"{action} request failed: storage unavailable",
            )
        )
    eval_budget = (
        f"{reservation.used}/{settings.kernel_eval_max_turns}"
        if settings.kernel_eval_max_turns >= 0
        else "unlimited"
    )
    if reservation.decision == "budget_exhausted":
        return JSONResponse(
            renderer.render_budget_exhausted(
                action=action,
                kernel_code=kernel_code,
                eval_budget=eval_budget,
            )
        )
    if reservation.decision == "conflict":
        return _error("idempotency conflict", 409)
    idempotency = hashlib.sha256(
        f"kernel-eval\0{session_id}\0{evaluation_identity}".encode()
    ).hexdigest()
    deadline = time.monotonic() + settings.kernel_eval_max_wait_seconds
    try:
        resource = await _evaluate(instance, evaluation_request, session_id, idempotency, deadline)
    except ToolEvaluationError as error:
        return JSONResponse(
            renderer.render_infra_error(
                action=action,
                kernel_code=kernel_code,
                eval_turn=reservation.turn,
                eval_budget=eval_budget,
                message=f"{action} request failed: {error}",
            )
        )
    diagnostics = None
    verdict = renderer.verify_verdict(resource.get("result") or {}, action)
    if (
        action in ("verify", "benchmark")
        and verdict.get("correct")
        and settings.kernel_eval_auto_diagnostics
    ):
        diagnostic_request = evaluation_request.model_copy(
            update={
                "checks": ["profile"],
                "performance": None,
                "profile": ProfileParams(mode="op", case_indices=[0]),
            }
        )
        try:
            diagnostic = await _evaluate(
                instance,
                diagnostic_request,
                session_id,
                hashlib.sha256((idempotency + ":diagnostics:case0").encode()).hexdigest(),
                deadline,
            )
            diagnostics = renderer.compose_diagnostics(
                (diagnostic.get("result") or {}).get("profile")
            )
        except ToolEvaluationError as error:
            logger.debug("optional kernel metrics unavailable (%s)", error)
    return JSONResponse(
        renderer.render_envelope(
            resource,
            action=action,
            kernel_code=kernel_code,
            eval_turn=reservation.turn,
            eval_budget=eval_budget,
            diagnostics=diagnostics,
        )
    )


class ToolEvaluationError(Exception):
    """A bounded, redacted failure rendered inside the tool envelope."""


def _admission_backoff_seconds(attempt: int, remaining: float) -> float:
    return min(15.0, 2.0 * (attempt + 1), remaining)


async def _evaluate(instance, evaluation_request, session_id, idempotency, deadline) -> dict:
    for attempt in range(8):
        if time.monotonic() >= deadline:
            raise ToolEvaluationError("evaluation admission did not succeed within the wait budget")
        try:
            outcome = await asyncio.to_thread(
                instance.submit,
                evaluation_request,
                TOOL_CALLER,
                session_id=session_id,
                idempotency_key=idempotency,
            )
        except (InstanceNotReadyError, QueueUnavailableError, StorageError):
            raise ToolEvaluationError("evaluation service unavailable") from None
        except (ForbiddenError, ManifestBindingError):
            raise ToolEvaluationError("evaluation admission rejected") from None
        if outcome.decision in ("rate_limited", "limit_exceeded"):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(_admission_backoff_seconds(attempt, remaining))
            continue
        if outcome.decision == "conflict":
            raise ToolEvaluationError("evaluation idempotency conflict")
        resource = outcome.resource
        if resource is None:
            raise ToolEvaluationError("evaluation admission returned no resource")
        break
    else:
        raise ToolEvaluationError("evaluation admission kept being rate limited")
    while resource.state not in TERMINAL_STATES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolEvaluationError(
                f"evaluation {resource.evaluation_id} did not finish within the wait budget; "
                "it may complete later"
            )
        await asyncio.sleep(min(instance.settings.api_poll_interval_seconds, remaining))
        try:
            recovered = await asyncio.to_thread(instance.get, resource.evaluation_id, TOOL_CALLER)
        except StorageError:
            raise ToolEvaluationError(f"polling failed for {resource.evaluation_id}") from None
        if recovered is None:
            raise ToolEvaluationError(f"evaluation {resource.evaluation_id} is not visible")
        resource = recovered
    return resource.model_dump(mode="json", exclude_none=False)


@router.get("/health")
async def health(request: Request):
    instance = get_context(request).instance
    if not _enabled(instance.settings):
        return _error("not found", 404)
    return {"ready": await asyncio.to_thread(instance.is_ready)}
