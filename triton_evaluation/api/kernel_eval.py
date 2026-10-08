"""Typed HTTP tool endpoints over the durable KernelEvaluation module."""

import asyncio
import hashlib
import secrets
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from ..kernel_evaluation import ToolInputError
from . import errors
from .auth import tool_bearer
from .dependencies import AppContext, get_context
from .errors import error_responses
from .schemas import IdempotencyKey, Readiness, SessionId
from .tool_schemas import (
    BenchmarkResponse,
    ProfileResponse,
    ToolError,
    ToolProfileRequest,
    ToolRequest,
    VerifyResponse,
)

router = APIRouter()

_TOOL_ERRORS = {
    **error_responses(400, 403, 409, model=ToolError),
    **error_responses(401, 404, 413, 422, 500),
}


def _enabled(settings) -> bool:
    return settings.kernel_eval_enabled and bool(settings.kernel_eval_token.get_secret_value())


def require_tool_context(
    context: Annotated[AppContext, Depends(get_context)],
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(tool_bearer)],
) -> AppContext:
    settings = context.instance.settings
    if not _enabled(settings):
        raise errors.not_found()
    expected = settings.kernel_eval_token.get_secret_value()
    if credential is None or not secrets.compare_digest(
        credential.credentials.encode(), expected.encode(),
    ):
        raise errors.unauthenticated()
    return context


ToolContext = Annotated[AppContext, Depends(require_tool_context)]
SessionHeader = Annotated[SessionId | None, Header(alias="X-Session-ID")]
OperationHeader = Annotated[IdempotencyKey | None, Header(alias="Idempotency-Key")]


def _session_id(request: Request, body: ToolRequest, header_session: str | None) -> str | None:
    if body.session is not None and header_session is not None and body.session != header_session:
        return None
    value = body.session if body.session is not None else header_session
    if value is None:
        # A missing explicit trial retains the original per-client-IP budget.
        remote = request.client.host if request.client is not None else "unknown"
        return "client-" + hashlib.sha256(remote.encode()).hexdigest()[:32]
    return value


async def _evaluate(
    action: Literal["verify", "benchmark", "profile"],
    request: Request,
    body: ToolRequest,
    context: AppContext,
    session_id: str | None,
    operation_id: str | None,
) -> dict | JSONResponse:
    try:
        return await context.kernel_eval.evaluate(
            action, body.model_dump(exclude_unset=True),
            session_id=_session_id(request, body, session_id), operation_id=operation_id,
        )
    except ToolInputError as error:
        status = {"invalid": 400, "forbidden": 403, "conflict": 409}[error.kind]
        return JSONResponse({"error": str(error)}, status_code=status)


@router.post(
    "/verify", response_model=VerifyResponse, response_model_exclude_unset=True,
    responses=_TOOL_ERRORS,
)
async def verify(
    request: Request, body: ToolRequest, context: ToolContext,
    session_id: SessionHeader = None, operation_id: OperationHeader = None,
):
    return await _evaluate("verify", request, body, context, session_id, operation_id)


@router.post(
    "/benchmark", response_model=BenchmarkResponse, response_model_exclude_unset=True,
    responses=_TOOL_ERRORS,
)
async def benchmark(
    request: Request, body: ToolRequest, context: ToolContext,
    session_id: SessionHeader = None, operation_id: OperationHeader = None,
):
    return await _evaluate("benchmark", request, body, context, session_id, operation_id)


@router.post(
    "/profile", response_model=ProfileResponse, response_model_exclude_unset=True,
    responses=_TOOL_ERRORS,
)
async def profile(
    request: Request, body: ToolProfileRequest, context: ToolContext,
    session_id: SessionHeader = None, operation_id: OperationHeader = None,
):
    return await _evaluate("profile", request, body, context, session_id, operation_id)


@router.get(
    "/health", response_model=Readiness, responses=error_responses(404, 413, 500),
)
async def health(context: Annotated[AppContext, Depends(get_context)]):
    instance = context.instance
    if not _enabled(instance.settings):
        raise errors.not_found()
    return {"ready": await asyncio.to_thread(instance.is_ready)}
