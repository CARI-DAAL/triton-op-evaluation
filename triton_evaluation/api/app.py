"""Durable evaluation API application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..service.metrics import increment
from . import errors
from .contract import ErrorEnvelope
from .dependencies import AppContext
from .middleware import RequestIdBodyLimitMiddleware, new_request_id
from .routes import router

logger = logging.getLogger(__name__)

_MAINTENANCE_BATCH = 100


def _envelope_response(request: Request, error: errors.ApiError) -> JSONResponse:
    _count_error_response(request, error)
    request_id = getattr(request.state, "request_id", None) or new_request_id()
    envelope = ErrorEnvelope(
        schema_version="2.1",
        request_id=request_id,
        error=error.info,
    )
    return JSONResponse(
        status_code=error.http_status,
        content=envelope.model_dump(mode="json", exclude_none=False),
        headers=dict(error.headers),
    )


def _count_error_response(request: Request, error: errors.ApiError) -> None:
    """Best-effort bounded error counter; never affects the response."""
    metrics = getattr(request.app.state, "context", None)
    metrics = getattr(getattr(metrics, "instance", None), "metrics", None)
    increment(metrics, "api_error_responses_total", code=error.info.code)
    if error.info.phase == "auth":
        increment(metrics, "auth_failures_total", code=error.info.code)


def create_app(context: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runtime readiness depends on durable storage and compatible workers;
        # schemas and golden examples are verified by tests, never at startup.
        context.instance.store.probe_storage()  # fail fast when the store rejects writes

        stop = asyncio.Event()
        app.state.maintenance_stop = stop
        task = asyncio.create_task(_maintenance_loop(context, stop))
        try:
            yield
        finally:
            stop.set()
            await task

    app = FastAPI(title="triton-evaluation", lifespan=lifespan)
    app.state.context = context
    app.add_middleware(RequestIdBodyLimitMiddleware)
    app.include_router(router)
    _install_exception_handlers(app)
    return app


async def _maintenance_loop(context: AppContext, stop: asyncio.Event) -> None:
    """Bounded watchdog/reaper/cleanup so queued deadlines terminalize even
    with no workers running. Every mutation is fenced/CAS, so duplicate
    maintenance (API plus workers) is safe."""
    interval = context.instance.settings.maintenance_interval_seconds
    while not stop.is_set():
        try:
            await asyncio.to_thread(_run_maintenance, context)
        except Exception as error:  # noqa: BLE001 — a daemon loop never dies
            logger.warning("api maintenance error (%s); continuing", type(error).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _run_maintenance(context: AppContext) -> None:
    store = context.instance.store
    store.expire_due_resources(limit=_MAINTENANCE_BATCH)
    store.recover_expired_leases(limit=_MAINTENANCE_BATCH)
    store.purge_expired_terminal_indexes(limit=_MAINTENANCE_BATCH)


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(errors.ApiError)
    async def handle_api_error(request: Request, exc: errors.ApiError):
        return await asyncio.to_thread(_envelope_response, request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        return await asyncio.to_thread(
            _envelope_response, request,
            errors.invalid_request_schema("request failed schema validation"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 400:
            error = errors.invalid_request("request body could not be parsed")
        elif exc.status_code == 404:
            error = errors.not_found()
        elif exc.status_code == 405:
            error = errors.not_found()
        elif exc.status_code == 401:
            error = errors.unauthenticated()
        else:
            error = errors.internal()
        return await asyncio.to_thread(_envelope_response, request, error)

    @app.exception_handler(Exception)
    async def handle_uncaught(request: Request, exc: Exception):
        logger.warning(
            "unhandled error (%s); responding with a redacted envelope",
            type(exc).__name__,
        )
        return await asyncio.to_thread(_envelope_response, request, errors.internal())
