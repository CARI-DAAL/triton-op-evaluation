"""Durable evaluation API application factory."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
        schema_version="2.0",
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
    metrics = getattr(metrics, "metrics", None) if metrics is not None else None
    if metrics is None:
        return
    try:
        metrics.increment("api_error_responses_total", {"code": error.info.code})
        if error.info.phase == "auth":
            metrics.increment("auth_failures_total", {"code": error.info.code})
    except Exception as error:  # noqa: BLE001 — never affect the response
        logger.warning("metrics write failed (%s)", type(error).__name__)


def create_app(context: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail closed before serving: verify published artifact bytes and
        # load security/catalog data. A development credential is never
        # constructed here — only the configured secret file is loaded.
        from ..api.contract_artifacts import CONTRACT_ROOT, verify_artifacts

        if verify_artifacts(CONTRACT_ROOT):
            raise RuntimeError("published contract artifacts failed verification")
        context.artifacts_verified = True
        context.store.probe_storage()  # fail fast when the store rejects writes

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
    interval = context.settings.maintenance_interval_seconds
    while not stop.is_set():
        try:
            await asyncio.to_thread(_run_maintenance, context)
        except Exception as error:  # noqa: BLE001 — a daemon loop never dies
            logger.warning(
                "api maintenance error (%s); continuing", type(error).__name__
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _run_maintenance(context: AppContext) -> None:
    store = context.store
    store.expire_due_resources(limit=_MAINTENANCE_BATCH)
    store.recover_expired_leases(limit=_MAINTENANCE_BATCH)
    store.purge_expired_terminal_indexes(limit=_MAINTENANCE_BATCH)


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(errors.ApiError)
    async def handle_api_error(request: Request, exc: errors.ApiError):
        return _envelope_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        return _envelope_response(request, errors.invalid_request_schema(
            "request failed schema validation"
        ))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            error = errors.not_found()
        elif exc.status_code == 405:
            error = errors.not_found()
        elif exc.status_code == 401:
            error = errors.unauthenticated()
        else:
            error = errors.internal()
        return _envelope_response(request, error)

    @app.exception_handler(Exception)
    async def handle_uncaught(request: Request, exc: Exception):
        logger.warning(
            "unhandled error (%s); responding with a redacted envelope",
            type(exc).__name__,
        )
        return _envelope_response(request, errors.internal())
