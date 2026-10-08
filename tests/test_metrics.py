"""Bounded Redis metrics: allowlist enforcement, cardinality, instrumentation."""
import pytest
from test_api import build_app, post, seed_worker, trial_headers

from triton_evaluation.service.metrics import RedisMetrics

pytestmark = pytest.mark.anyio

OBSERVER_HEADERS = {"Authorization": "Bearer observer-token-1"}


@pytest.fixture
def app(tmp_path, release_catalog_files):
    built = build_app(tmp_path, release_catalog_files)
    built.app.state.context.instance.metrics = RedisMetrics(
        built.store._r, built.settings.redis_prefix
    )
    return built


# ---------------------------------------------------------------------------
# Registry allowlist
# ---------------------------------------------------------------------------


def test_unknown_metric_names_are_rejected(app):
    metrics = app.app.state.context.instance.metrics
    with pytest.raises(ValueError):
        metrics.increment("arbitrary_metric_total", {"caller_class": "trial"})


@pytest.mark.parametrize("name, labels", [
    ("unknown_gauge", {}),
    ("card_health_state", {"card": "unbounded-card-id"}),
    ("card_health_state", {"worker_id": "private-worker"}),
    ("worker_attempts_total", {}),
])
def test_gauge_writes_preserve_metric_kind_and_label_allowlists(app, name, labels):
    with pytest.raises(ValueError):
        app.app.state.context.instance.metrics.set_gauge(name, labels, 1)


def test_unknown_label_names_are_rejected(app):
    metrics = app.app.state.context.instance.metrics
    with pytest.raises(ValueError):
        metrics.increment("evaluation_admissions_total", {"caller_id": "trial-caller"})


def test_unknown_label_values_are_rejected(app):
    metrics = app.app.state.context.instance.metrics
    with pytest.raises(ValueError):
        metrics.increment("evaluation_admissions_total", {"caller_class": "vip"})


def test_caller_values_aggregate_to_bounded_classes(app):
    metrics = app.app.state.context.instance.metrics
    metrics.increment("evaluation_admissions_total", {"decision": "created",
                                                      "caller_class": "trial"})
    rendered = metrics.render_openmetrics().decode()
    assert 'evaluation_admissions_total{caller_class="trial",decision="created"} 1' \
        in rendered


def test_leakage_attempts_are_rejected(app):
    metrics = app.app.state.context.instance.metrics
    for bad in (
        {"caller_class": "trial", "evaluation_id": "ev_1"},
        {"caller_class": "trial-caller"},
        {"check": "correctness", "session": "session-1"},
        {"check": "/abs/path"},
    ):
        with pytest.raises(ValueError):
            metrics.increment("evaluation_admissions_total", bad)


def test_no_graderecord_metrics_in_this_service(app):
    """GradeRecord scored/unscored metrics belong to Eval Hub, not here."""
    from triton_evaluation.service.metrics import ALLOWED_METRICS

    assert not any("grade_record" in name for name in ALLOWED_METRICS)
    assert "correctness_domain_outcomes_total" in ALLOWED_METRICS
    assert "grade_eligibility_total" in ALLOWED_METRICS


# ---------------------------------------------------------------------------
# Histograms and rendering
# ---------------------------------------------------------------------------


def test_histogram_uses_fixed_buckets(app):
    metrics = app.app.state.context.instance.metrics
    metrics.observe("resource_terminal_duration_seconds", {"state": "completed"}, 0.05)
    metrics.observe("resource_terminal_duration_seconds", {"state": "completed"}, 2.0)
    rendered = metrics.render_openmetrics().decode()
    assert "# TYPE resource_terminal_duration_seconds histogram" in rendered
    # Cumulative buckets: 0.05 lands in le="0.1" and above; 2.0 only in le="5"+.
    assert 'le="0.1"} 1' in rendered
    assert 'le="5"} 2' in rendered
    assert '_count{state="completed"} 2' in rendered


def test_rendering_is_deterministic_and_ends_with_eof(app):
    metrics = app.app.state.context.instance.metrics
    metrics.increment("evaluation_admissions_total", {"decision": "created",
                                                      "caller_class": "trial"})
    first = metrics.render_openmetrics()
    second = metrics.render_openmetrics()
    assert first == second
    assert first.endswith(b"# EOF\n")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def test_internal_metrics_requires_observer_scope(app):
    async with app.http as client:
        denied = await client.get("/internal/metrics", headers=trial_headers())
        anonymous = await client.get("/internal/metrics")
    assert denied.status_code == 403
    assert anonymous.status_code == 401


async def test_internal_metrics_renders_openmetrics(app):
    seed_worker(app)
    async with app.http as client:
        await post(client)
        response = await client.get("/internal/metrics", headers=OBSERVER_HEADERS)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == (
        "application/openmetrics-text; version=1.0.0; charset=utf-8"
    )
    body = response.text
    assert body.endswith("# EOF\n")
    assert "evaluation_admissions_total" in body


async def test_metrics_do_not_render_ids_or_queue_depth(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        body = (await client.get("/internal/metrics", headers=OBSERVER_HEADERS)).text
    evaluation_id = created.json()["evaluation_id"]
    assert evaluation_id not in body
    assert "session-1" not in body
    assert "queue_depth" not in body
    assert "trial-caller" not in body  # caller IDs aggregate to classes


async def test_metrics_failure_never_changes_resource_outcomes(app, monkeypatch):
    seed_worker(app)

    def broken_increment(self, name, labels, amount=1):
        raise RuntimeError("metrics backend down")

    monkeypatch.setattr(RedisMetrics, "increment", broken_increment)
    async with app.http as client:
        response = await post(client)
    assert response.status_code == 202  # admission unaffected
    assert app.store.get_visible(
        response.json()["evaluation_id"], _trial_caller()
    ).state == "queued"


def _trial_caller():
    from conftest import make_caller

    return make_caller("trial-caller", "trial", ("trial:submit", "trial:read"))


# ---------------------------------------------------------------------------
# Instrumentation dimensions
# ---------------------------------------------------------------------------


async def test_post_dimensions_are_instrumented(app):
    seed_worker(app)
    async with app.http as client:
        await post(client)
        await post(client)  # replay
        body = (await client.get("/internal/metrics", headers=OBSERVER_HEADERS)).text
    assert 'decision="created"' in body
    assert 'decision="replayed"' in body


async def test_auth_and_error_dimensions_are_instrumented(app):
    async with app.http as client:
        headers = trial_headers()
        del headers["Authorization"]
        await post(client, headers=headers)
        await post(client)  # not ready -> 503
        body = (await client.get("/internal/metrics", headers=OBSERVER_HEADERS)).text
    assert "api_error_responses_total" in body
    assert 'code="UNAUTHENTICATED"' in body
    assert 'code="NOT_READY"' in body


async def test_terminal_dimensions_are_instrumented(app, make_command):
    from conftest import make_caller, make_completed_result
    from test_api import algorithms

    from triton_evaluation.service.models import WorkerIdentity, new_worker_id

    app.store.create_or_replay(
        make_command(caller=make_caller("trial-caller", "trial",
                                        ("trial:submit", "trial:read")))
    )
    lease = app.store.claim_next(
        WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                       algorithms(app.settings))
    )
    app.store.commit_terminal(lease, "completed", make_completed_result(app.settings))
    body = app.app.state.context.instance.metrics.render_openmetrics().decode()
    assert "resource_terminal_transitions_total" in body
    assert 'state="completed"' in body
    assert "resource_terminal_duration_seconds" in body


def test_worker_dimensions_are_instrumented(app):
    from test_api import algorithms

    from triton_evaluation.service.models import WorkerIdentity, new_worker_id

    identity = WorkerIdentity(new_worker_id(), app.settings.evaluator_build,
                              algorithms(app.settings))
    app.store.heartbeat(identity)
    app.store.note_fenced_write()
    body = app.app.state.context.instance.metrics.render_openmetrics().decode()
    assert "worker_heartbeats_total" in body
    assert "fenced_write_rejections_total" in body


def test_shaping_and_catalog_dimensions_exist(app):
    from triton_evaluation.service.metrics import ALLOWED_METRICS

    assert "shaping_failures_total" in ALLOWED_METRICS
    assert "catalog_binding_outcomes_total" in ALLOWED_METRICS
    assert "worker_compatibility_total" in ALLOWED_METRICS
    assert "check_outcomes_total" in ALLOWED_METRICS
    assert "check_skips_total" in ALLOWED_METRICS
    assert "correctness_domain_outcomes_total" in ALLOWED_METRICS
    assert "decoy_detections_total" in ALLOWED_METRICS
    assert "performance_unmeasured_total" in ALLOWED_METRICS
    assert "lease_renewals_total" in ALLOWED_METRICS
    assert "reaped_leases_total" in ALLOWED_METRICS
    assert "terminal_commit_cas_total" in ALLOWED_METRICS
    assert "resource_deadline_expiries_total" in ALLOWED_METRICS
