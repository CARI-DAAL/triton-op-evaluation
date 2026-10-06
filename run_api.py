"""Durable evaluation API entrypoint."""

import asyncio
from pathlib import Path

import redis
import uvicorn

from triton_evaluation.api import create_app
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.broker import EvaluationStore
from triton_evaluation.catalog import ReleaseCatalog
from triton_evaluation.config import Settings
from triton_evaluation.instance import EvaluationInstance
from triton_evaluation.security import CapabilityRegistry


def build_app() -> "FastAPI":  # noqa: F821 — FastAPI imported via api
    settings = Settings.from_env()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    store = EvaluationStore(client, settings)
    registry = CapabilityRegistry.from_secret_file(Path(settings.capability_secret_path))
    catalog = ReleaseCatalog.load(Path(settings.catalog_root), Path(settings.catalog_manifest_path))
    context = AppContext(
        instance=EvaluationInstance(settings=settings, store=store, catalog=catalog),
        registry=registry,
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
