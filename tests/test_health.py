"""/livez and /readyz semantics: liveness touches nothing; readiness is
dependency-aware and reveals no component detail."""
import pytest
from test_api import build_app, seed_worker

from triton_evaluation.api.contract import ErrorEnvelope

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


async def test_livez_is_constant_and_touches_nothing(app, monkeypatch):
    """Liveness must not read the store, catalog, or worker state."""
    from triton_evaluation.broker import EvaluationStore

    def _forbidden(*args, **kwargs):
        raise AssertionError("livez must not touch the store")

    monkeypatch.setattr(EvaluationStore, "probe_storage", _forbidden)
    monkeypatch.setattr(EvaluationStore, "has_fresh_compatible_worker", _forbidden)
    async with app.http as client:
        response = await client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_requires_fresh_compatible_worker(app):
    async with app.http as client:
        response = await client.get("/readyz")
    assert response.status_code == 503
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "NOT_READY"
    assert envelope.error.retryable is True
    assert response.headers.get("Retry-After") == "1"


async def test_readyz_succeeds_with_dependencies(app):
    seed_worker(app)
    async with app.http as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


async def test_readyz_failure_body_reveals_no_internals(app):
    async with app.http as client:
        response = await client.get("/readyz")
    text = response.text.lower()
    for leak in ("worker", "heartbeat", "queue", "redis", "artifact", "catalog",
                 "build", "card", "count", "6379", "127.0.0.1"):
        assert leak not in text


async def test_readyz_fails_closed_when_artifacts_unverified(
    app, release_catalog_files, tmp_path
):
    seed_worker(app)
    app.app.state.context.artifacts_verified = False
    async with app.http as client:
        response = await client.get("/readyz")
    assert response.status_code == 503
    assert ErrorEnvelope.model_validate_json(response.content).error.code == (
        "NOT_READY"
    )
