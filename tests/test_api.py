import fakeredis
from fastapi.testclient import TestClient

from triton_evaluation.broker import Broker
from triton_evaluation.api import create_app


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
    assert broker.dequeue().task_id == task_id


def test_post_benchmark_without_gate_is_422(settings):
    client, _ = _client(settings)
    r = client.post("/benchmark", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"})
    assert r.status_code == 422


def test_post_benchmark_with_verify_task_id_ok(settings):
    client, broker = _client(settings)
    r = client.post("/benchmark", json={
        "op_name": "s", "reference_code": "R", "kernel_code": "K", "verify_task_id": "v1",
    })
    assert r.status_code == 200
    job = broker.dequeue()
    assert job.type == "benchmark"
    assert job.spec.verify_task_id == "v1"


def test_get_task_unknown_is_404(settings):
    client, _ = _client(settings)
    assert client.get("/tasks/nope").status_code == 404


def test_get_task_returns_record(settings):
    client, broker = _client(settings)
    task_id = client.post("/verify", json={"op_name": "s", "reference_code": "R", "kernel_code": "K"}).json()["task_id"]
    r = client.get(f"/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "QUEUED"


def test_healthz(settings):
    client, _ = _client(settings)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["redis"] == "ok"
