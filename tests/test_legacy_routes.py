"""Time-boxed legacy adapters over the same durable admission path."""
import json

import pytest
from test_api import (
    GRADE_TOKEN,
    TRIAL_TOKEN,
    make_test_settings,
    seed_worker,
    trial_headers,
)

from triton_evaluation.api.contract import EvaluationResource

pytestmark = pytest.mark.anyio

SUNSET = "2026-10-15T00:00:00Z"
VERIFY_BODY = {
    "op_name": "addmm",
    "reference_code": "REF",
    "kernel_code": "IMPL",
    "measure_speed": False,
}
PROFILE_BODY = {
    "op_name": "addmm",
    "reference_code": "REF",
    "kernel_code": "IMPL",
    "mode": "deep",
    "case_idx": 1,
}


def legacy_settings(tmp_path):
    settings = make_test_settings(tmp_path)
    return settings.model_copy(
        update={"legacy_routes_enabled": True, "legacy_routes_sunset_at": SUNSET}
    )


@pytest.fixture
def app(tmp_path, release_catalog_files):
    from test_api import build_app as _build

    return _build(tmp_path, release_catalog_files, settings=legacy_settings(tmp_path))


def legacy_headers(key="legacy-key-1", session="legacy-session"):
    return {
        "Authorization": f"Bearer {TRIAL_TOKEN}",
        "X-Session-ID": session,
        "Idempotency-Key": key,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Adapter translation
# ---------------------------------------------------------------------------


async def test_verify_translates_measure_speed_into_checks(app):
    seed_worker(app)
    async with app.http as client:
        plain = await client.post("/verify", json=VERIFY_BODY,
                                  headers=legacy_headers(key="k1"))
        speedy = await client.post(
            "/verify", json=VERIFY_BODY | {"measure_speed": True},
            headers=legacy_headers(key="k2"),
        )
    assert plain.status_code == 202
    assert speedy.status_code == 202
    plain_task = plain.json()["task_id"]
    stored_plain = app.store.get_visible(
        plain_task, _trial_caller()
    )
    assert stored_plain is not None
    # The durable resource carries the translated v2 request.
    from triton_evaluation.api.contract import EvaluationRequest

    raw = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{plain_task}", b"canonical_request"
    )
    request = EvaluationRequest.model_validate_json(raw)
    assert request.checks == ["correctness"]
    assert request.purpose == "trial"

    raw_speedy = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{speedy.json()['task_id']}",
        b"canonical_request",
    )
    request_speedy = EvaluationRequest.model_validate_json(raw_speedy)
    assert request_speedy.checks == ["correctness", "performance"]
    assert request_speedy.performance is not None


def _trial_caller():
    from conftest import make_caller

    return make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))


async def test_profile_translates_mode_and_case_selection(app):
    seed_worker(app)
    async with app.http as client:
        response = await client.post("/profile", json=PROFILE_BODY,
                                     headers=legacy_headers())
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    raw = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{task_id}", b"canonical_request"
    )
    from triton_evaluation.api.contract import EvaluationRequest

    request = EvaluationRequest.model_validate_json(raw)
    assert request.checks == ["profile"]
    assert request.profile.mode == "deep"
    assert request.profile.case_indices == [1]


async def test_tasks_maps_states_to_legacy_names(app):
    seed_worker(app)
    async with app.http as client:
        created = await client.post("/verify", json=VERIFY_BODY,
                                    headers=legacy_headers())
        task_id = created.json()["task_id"]
        view = await client.get(f"/tasks/{task_id}", headers=legacy_headers())
    assert view.status_code == 200
    payload = view.json()
    assert payload["task_id"] == task_id
    assert payload["status"] == "QUEUED"
    assert payload["result"] is None


async def test_terminal_task_projects_old_field_names(app):
    seed_worker(app)
    async with app.http as client:
        created = await client.post("/verify", json=VERIFY_BODY,
                                    headers=legacy_headers())
        task_id = created.json()["task_id"]
        _complete_as_v2(app, task_id)
        view = await client.get(f"/tasks/{task_id}", headers=legacy_headers())
    payload = view.json()
    assert payload["status"] == "SUCCEEDED"
    correctness = payload["result"]["correctness"]
    assert correctness["passed"] is True
    assert correctness["compiled"] is True
    assert "reference_code" not in view.text
    assert "canonical_request" not in view.text


def _complete_as_v2(app, task_id):
    from conftest import make_completed_result
    from test_api import algorithms

    from triton_evaluation.models import WorkerIdentity, new_worker_id

    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    assert lease is not None
    assert app.store.commit_terminal(
        lease, "completed", make_completed_result(app.settings)
    )


async def test_failed_projection_is_redacted_legacy_failure(app):
    seed_worker(app)
    async with app.http as client:
        created = await client.post("/verify", json=VERIFY_BODY,
                                    headers=legacy_headers())
        task_id = created.json()["task_id"]
        # Terminalize with a result the projection cannot represent.
        from test_api import algorithms

        from triton_evaluation.models import WorkerIdentity, new_worker_id

        lease = app.store.claim_next(
            WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                           algorithms(app.settings))
        )
        result = _protocol_failed_result(app)
        app.store.commit_terminal(lease, "failed", result)
        view = await client.get(f"/tasks/{task_id}", headers=legacy_headers())
    payload = view.json()
    assert payload["status"] == "FAILED"
    assert payload["error"]
    assert "error" not in json.dumps(payload.get("result") or {})


def _protocol_failed_result(app):
    from triton_evaluation.models import AlgorithmVersions
    from triton_evaluation.worker import _protocol_failure_result

    lease = type(
        "L", (),
        {"evaluator_build": app.settings.evaluator_build,
         "algorithms": AlgorithmVersions(
             correctness=app.settings.correctness_algorithm_version,
             decoy=app.settings.decoy_algorithm_version,
             benchmark=app.settings.benchmark_algorithm_version,
             profiler=app.settings.profiler_algorithm_version,
         )},
    )()
    return _protocol_failure_result(
        requested_checks=["correctness"], persisted_outcomes={},
        code="WORKER_LOST", phase="execution", retryable=True,
        message="worker lease expired before a subject outcome was committed",
        lease=lease,
    )


# ---------------------------------------------------------------------------
# Security parity
# ---------------------------------------------------------------------------


async def test_legacy_requires_trial_capability_and_headers(app):
    async with app.http as client:
        anonymous = await client.post("/verify", json=VERIFY_BODY)
        no_key = await client.post(
            "/verify", json=VERIFY_BODY,
            headers={"Authorization": f"Bearer {TRIAL_TOKEN}",
                     "X-Session-ID": "s"},
        )
        graded = await client.post(
            "/verify", json=VERIFY_BODY,
            headers=legacy_headers() | {"Authorization": f"Bearer {GRADE_TOKEN}"},
        )
    assert anonymous.status_code == 401
    assert no_key.status_code == 400
    assert graded.status_code == 403  # grade capability cannot use legacy surface


async def test_legacy_cannot_submit_grade(app):
    seed_worker(app)
    async with app.http as client:
        response = await client.post(
            "/verify", json=VERIFY_BODY | {"purpose": "grade"},
            headers=legacy_headers(),
        )
    assert response.status_code in (400, 422)


async def test_legacy_shares_the_durable_store_and_fifo(app):
    seed_worker(app)
    async with app.http as client:
        legacy = await client.post("/verify", json=VERIFY_BODY,
                                   headers=legacy_headers())
        task_id = legacy.json()["task_id"]
        # The same resource is visible through the v2 endpoint.
        v2_view = await client.get(f"/evaluations/{task_id}",
                                   headers=trial_headers())
    assert v2_view.status_code == 200
    assert EvaluationResource.model_validate_json(
        json.dumps(v2_view.json())
    ).state == "queued"


async def test_legacy_routes_do_not_import_service_internals():
    import inspect

    import triton_evaluation.api.legacy_routes as legacy_routes

    source = inspect.getsource(legacy_routes)
    assert "from ..broker import" not in source
    assert "import evaluate" not in source
    assert "Broker" not in source


async def test_legacy_usage_metric_emitted(app):
    seed_worker(app)
    async with app.http as client:
        await client.post("/verify", json=VERIFY_BODY, headers=legacy_headers())
    rendered = app.app.state.context.metrics.render_openmetrics().decode()
    assert "legacy_route_requests_total" in rendered
    assert 'route="verify"' in rendered


async def test_invisible_task_ids_are_404(app):
    async with app.http as client:
        response = await client.get(
            f"/tasks/ev_{'0' * 32}", headers=legacy_headers()
        )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------


async def test_legacy_timeout_above_v2_maximum_is_rejected(app):
    seed_worker(app)
    async with app.http as client:
        response = await client.post(
            "/verify", json=VERIFY_BODY | {"timeout": 3601},
            headers=legacy_headers(),
        )
    assert response.status_code in (400, 422)
    assert "3601" not in response.text  # unrepresentable, never echoed


async def test_legacy_idempotent_replay(app):
    seed_worker(app)
    async with app.http as client:
        first = await client.post("/verify", json=VERIFY_BODY,
                                  headers=legacy_headers())
        replay = await client.post("/verify", json=VERIFY_BODY,
                                   headers=legacy_headers())
    assert replay.status_code == 202
    assert replay.json()["task_id"] == first.json()["task_id"]


async def test_performance_dependency_is_translated_not_invented(app):
    seed_worker(app)
    async with app.http as client:
        response = await client.post(
            "/verify", json=VERIFY_BODY | {"measure_speed": True, "warmup": 2,
                                           "repeats": 10},
            headers=legacy_headers(key="perf-key"),
        )
    raw = app.store._r.hget(
        f"{app.settings.redis_prefix}:resource:{response.json()['task_id']}",
        b"canonical_request",
    )
    from triton_evaluation.api.contract import EvaluationRequest

    request = EvaluationRequest.model_validate_json(raw)
    assert request.performance.warmup == 2
    assert request.performance.repeats == 10


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


async def test_legacy_routes_absent_when_disabled(tmp_path, release_catalog_files):
    from test_api import build_app as _build

    app = _build(tmp_path, release_catalog_files)
    async with app.http as client:
        verify = await client.post("/verify", json=VERIFY_BODY,
                                   headers=legacy_headers())
        tasks = await client.get(f"/tasks/ev_{'0' * 32}",
                                 headers=legacy_headers())
    assert verify.status_code == 404
    assert tasks.status_code == 404


async def test_enabling_without_exact_sunset_fails(tmp_path):
    from triton_evaluation.config import Settings

    with pytest.raises(ValueError):
        Settings(
            engine_dir=str(tmp_path),
            legacy_routes_enabled=True,
            legacy_routes_sunset_at="2027-01-01T00:00:00Z",
        )


async def test_enabling_after_sunset_fails(tmp_path, monkeypatch):
    import datetime as dt

    import triton_evaluation.config as config_module
    from triton_evaluation.config import Settings

    monkeypatch.setattr(
        config_module, "_utcnow",
        lambda: dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc),
    )
    with pytest.raises(ValueError):
        Settings(
            engine_dir=str(tmp_path),
            legacy_routes_enabled=True,
            legacy_routes_sunset_at=SUNSET,
        )
