"""HTTP adapter for kernel evaluation: authenticate, decode and map responses.

The injected KernelEvaluation module owns tool policy and durable recovery.
"""

import asyncio
import hashlib
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..kernel_evaluation import ToolInputError
from . import errors
from .dependencies import get_context
from .routes import SESSION_ID_RE

router = APIRouter()


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
    try:
        envelope = await context.kernel_eval.evaluate(
            action, body, session_id=_session_id(request, body),
            operation_id=request.headers.get("Idempotency-Key"),
        )
    except ToolInputError as error:
        status = {"invalid": 400, "forbidden": 403, "conflict": 409}[error.kind]
        return _error(str(error), status)
    return JSONResponse(envelope)


@router.get("/health")
async def health(request: Request):
    instance = get_context(request).instance
    if not _enabled(instance.settings):
        return _error("not found", 404)
    return {"ready": await asyncio.to_thread(instance.is_ready)}
