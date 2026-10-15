"""Durable evaluation API entrypoint."""

import asyncio
from pathlib import Path

import redis
import uvicorn
from redis.backoff import NoBackoff
from redis.retry import Retry

from triton_evaluation.api import create_app
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.config import Settings
from triton_evaluation.service.broker import EvaluationStore
from triton_evaluation.service.instance import EvaluationInstance
from triton_evaluation.service.metrics import RedisMetrics
from triton_evaluation.service.security import CapabilityRegistry
from triton_evaluation.versioning import startup_identity


def build_app() -> "FastAPI":  # noqa: F821 — FastAPI imported via api
    settings = Settings.from_env()
    # uvicorn imports this module's `app`, so this is the API's formal startup
    # path: verify the release before any listener exists.
    release = startup_identity(settings)
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    metrics_client = redis.Redis.from_url(
        settings.redis_url, decode_responses=False,
        socket_connect_timeout=0.25, socket_timeout=0.25,
        retry=Retry(NoBackoff(), 0), retry_on_timeout=False, retry_on_error=[],
    )
    metrics = RedisMetrics(metrics_client, settings.redis_prefix)
    store = EvaluationStore(client, settings, metrics=metrics)
    registry = CapabilityRegistry.from_secret_file(Path(settings.capability_secret_path))
    instance = EvaluationInstance(settings=settings, store=store, metrics=metrics)
    context = AppContext(
        instance=instance,
        registry=registry,
        release=release,
    )
    return create_app(context)


app = build_app()


def main() -> None:
    settings = app.state.context.instance.settings
    if settings.api_tls_port is not None:
        from triton_evaluation.api.listeners import serve_app

        asyncio.run(serve_app(app, settings))
    else:
        uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
