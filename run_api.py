"""Durable evaluation API entrypoint."""
import os
from pathlib import Path

import redis
import uvicorn

from triton_evaluation.api import create_app
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.broker import EvaluationStore
from triton_evaluation.catalog import ReleaseCatalog
from triton_evaluation.config import Settings
from triton_evaluation.security import CapabilityRegistry


def build_app() -> "FastAPI":  # noqa: F821 — FastAPI imported via api
    settings = Settings.from_env()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    store = EvaluationStore(client, settings)
    registry = CapabilityRegistry.from_secret_file(
        Path(settings.capability_secret_path)
    )
    catalog = ReleaseCatalog.load(
        Path(settings.catalog_root), Path(settings.catalog_manifest_path)
    )
    context = AppContext(
        settings=settings, store=store, registry=registry, catalog=catalog
    )
    return create_app(context)


app = build_app()


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("TRITON_EVAL_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRITON_EVAL_API_PORT", "11907")),
    )


if __name__ == "__main__":
    main()
