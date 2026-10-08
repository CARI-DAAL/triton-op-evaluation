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


def test_shaping_and_execution_dimensions_exist(app):
    from triton_evaluation.service.metrics import ALLOWED_METRICS

    assert "shaping_failures_total" in ALLOWED_METRICS
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


@pytest.mark.parametrize("operation, arguments", [
    ("increment", ("worker_attempts_total",)),
    ("observe", ("performance_coverage", 0.5)),
    ("set_gauge", ("card_health_state", 1)),
])
def test_best_effort_emission_accepts_optional_or_failing_adapters(operation, arguments):
    from triton_evaluation.service import metrics

    class FailingMetrics:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise RuntimeError("unavailable")
            return fail

    emit = getattr(metrics, operation)
    emit(None, *arguments)
    emit(FailingMetrics(), *arguments)


def test_best_effort_emission_preserves_labels_and_values():
    from triton_evaluation.service.metrics import increment, observe, set_gauge

    calls = []

    class RecordingMetrics:
        def increment(self, *args):
            calls.append(("increment", args))

        def observe(self, *args):
            calls.append(("observe", args))

        def set_gauge(self, *args):
            calls.append(("set_gauge", args))

    adapter = RecordingMetrics()
    increment(adapter, "api_error_responses_total", code="INTERNAL")
    observe(adapter, "performance_coverage", 0.5)
    set_gauge(adapter, "card_health_state", 1, card="0")
    assert calls == [
        ("increment", ("api_error_responses_total", {"code": "INTERNAL"})),
        ("observe", ("performance_coverage", {}, 0.5)),
        ("set_gauge", ("card_health_state", {"card": "0"}, 1)),
    ]


async def test_metrics_read_failure_preserves_internal_error_envelope(app):
    import httpx

    class BrokenRedis:
        def hgetall(self, key):
            raise RuntimeError("private backend details")

        def hincrby(self, *args):
            raise RuntimeError("unavailable")

    app.app.state.context.instance.metrics = RedisMetrics(BrokenRedis(), "test")
    transport = httpx.ASGITransport(app=app.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/internal/metrics", headers=OBSERVER_HEADERS)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL"
    assert "private backend details" not in response.text


@pytest.mark.parametrize("path, headers, operation, expected_status", [
    ("/internal/metrics", OBSERVER_HEADERS, "read", 200),
    ("/internal/metrics", {}, "increment", 401),
    ("/unknown", {}, "increment", 404),
])
async def test_liveness_progresses_while_metrics_redis_is_blocked(
    app, path, headers, operation, expected_status
):
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    writes = []

    class BlockingRedis:
        def block(self):
            entered.set()
            release.wait()

        def hgetall(self, key):
            if operation == "read":
                self.block()
            return {}

        def hincrby(self, key, field, amount):
            writes.append((key, field, amount))
            if operation == "increment":
                self.block()

    app.app.state.context.instance.metrics = RedisMetrics(BlockingRedis(), "test")
    # A watchdog only prevents a broken implementation deadlocking the suite.
    # Correctness is event ordering: /livez must finish before Redis is released.
    watchdog = threading.Timer(10, release.set)
    watchdog.start()
    pending = None
    try:
        async with app.http as client:
            pending = asyncio.create_task(client.get(path, headers=headers))
            assert await asyncio.to_thread(entered.wait, 10)
            liveness = await client.get("/livez")
            assert liveness.status_code == 200
            assert not release.is_set()
            assert not pending.done()
            release.set()
            response = await pending
            assert response.status_code == expected_status
            if expected_status == 401:
                assert [field for _, field, _ in writes] == [
                    "code=UNAUTHENTICATED", "code=UNAUTHENTICATED",
                ]
                assert [key.rsplit(":", 1)[-1] for key, _, _ in writes] == [
                    "api_error_responses_total", "auth_failures_total",
                ]
    finally:
        release.set()
        watchdog.cancel()
        if pending is not None:
            await pending
