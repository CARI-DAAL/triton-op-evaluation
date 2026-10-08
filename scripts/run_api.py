"""Durable evaluation API entrypoint."""

import asyncio
from pathlib import Path

import redis
import uvicorn

from triton_evaluation.api import create_app
from triton_evaluation.api.dependencies import AppContext
from triton_evaluation.config import Settings
from triton_evaluation.kernel_eval_budget import EvalBudgetRegistry
from triton_evaluation.kernel_evaluation import KernelEvaluation
from triton_evaluation.service.broker import EvaluationStore
from triton_evaluation.service.catalog import ReleaseCatalog
from triton_evaluation.service.instance import EvaluationInstance
from triton_evaluation.service.security import CapabilityRegistry


def build_app() -> "FastAPI":  # noqa: F821 — FastAPI imported via api
    settings = Settings.from_env()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    store = EvaluationStore(client, settings)
    registry = CapabilityRegistry.from_secret_file(Path(settings.capability_secret_path))
    catalog = ReleaseCatalog.load(Path(settings.catalog_root), Path(settings.catalog_manifest_path))
    instance = EvaluationInstance(settings=settings, store=store, catalog=catalog)
    budget = EvalBudgetRegistry(
        client, prefix=settings.redis_prefix, maximum=settings.kernel_eval_max_turns,
        retention_seconds=settings.retention_seconds,
    )
    context = AppContext(
        instance=instance,
        registry=registry,
        kernel_eval=KernelEvaluation(instance, budget),
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
