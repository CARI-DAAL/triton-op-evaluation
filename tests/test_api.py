import fakeredis
from fastapi.testclient import TestClient

from triton_evaluation.api import create_app
from triton_evaluation.broker import Broker


def _client(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    broker = Broker(client, settings)
    return TestClient(create_app(broker)), broker


def test_post_verify_enqueues(settings):
    client, broker = _client(settings)
    r = client.post("/verify", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    rec = broker.get_task(task_id)
    assert rec["status"] == "QUEUED"
    assert rec["type"] == "verify"
    job = broker.dequeue()
    assert job.task_id == task_id
    assert job.spec.measure_speed is False


def test_post_verify_measure_speed_sets_spec(settings):
    client, broker = _client(settings)
    r = client.post(
        "/verify",
        json={
            "op_name": "s",
            "reference_code": "R",
            "kernel_code": "K",
            "measure_speed": True,
            "warmup": 2,
            "repeats": 20,
        },
    )
    assert r.status_code == 200
    job = broker.dequeue()
    assert job.type == "verify"
    assert job.spec.measure_speed is True
    assert job.spec.warmup == 2
    assert job.spec.repeats == 20


def test_post_profile_defaults_to_all_cases(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={"op_name": "softmax", "reference_code": "R", "kernel_code": "K"},
    )
    assert response.status_code == 200
    job = broker.dequeue()
    assert job.type == "profile"
    assert job.spec.profile_mode == "op"
    assert job.spec.case_idx is None
    assert job.spec.timeout == 3600


def test_post_profile_accepts_one_deep_case(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={
            "op_name": "softmax",
            "reference_code": "R",
            "kernel_code": "K",
            "profile_mode": "deep",
            "case_idx": 3,
            "timeout": 7200,
        },
    )
    assert response.status_code == 200
    job = broker.dequeue()
    assert job.spec.profile_mode == "deep"
    assert job.spec.case_idx == 3
    assert job.spec.timeout == 7200


def test_post_profile_rejects_deep_launch_count(settings):
    client, broker = _client(settings)
    response = client.post(
        "/profile",
        json={
            "op_name": "softmax",
            "reference_code": "R",
            "kernel_code": "K",
            "profile_mode": "deep",
            "profile_launch_count": 2,
        },
    )
    assert response.status_code == 422
    assert broker.dequeue() is None


def test_benchmark_route_is_gone(settings):
    client, _ = _client(settings)
    r = client.post("/benchmark", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"})
    assert r.status_code == 404


def test_get_task_unknown_is_404(settings):
    client, _ = _client(settings)
    assert client.get("/tasks/nope").status_code == 404


def test_get_task_returns_record(settings):
    client, broker = _client(settings)
    task_id = client.post(
        "/verify", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"}
    ).json()["task_id"]
    r = client.get(f"/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "QUEUED"


def test_healthz(settings):
    client, _ = _client(settings)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["redis"] == "ok"


class _RecordingBroker:
    """Minimal broker double to observe lifespan startup/shutdown hooks."""

    def __init__(self):
        self.pinged = 0
        self.closed = 0

    def ping(self):
        self.pinged += 1

    def close(self):
        self.closed += 1


def test_lifespan_pings_on_startup_and_closes_on_shutdown():
    b = _RecordingBroker()
    app = create_app(b)
    assert b.pinged == 0 and b.closed == 0
    with TestClient(app):  # entering the context runs the lifespan startup
        assert b.pinged == 1  # fail-fast Redis ping on startup
    assert b.closed == 1  # pool released on shutdown
