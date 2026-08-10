import math

import fakeredis
from fastapi.testclient import TestClient

from triton_evaluation.api import create_app
from triton_evaluation.broker import Broker


def _request_body(**overrides):
    body = {
        "op_name": "addmm",
        "reference_code": "R",
        "kernel_code": "K",
        "checks": ["correctness"],
        "timeout": 10,
    }
    body.update(overrides)
    return body


def _client(settings, broker_class=Broker):
    redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
    broker = broker_class(redis_client, settings)
    return TestClient(create_app(broker)), broker


class _CompletingBroker(Broker):
    def get_task(self, task_id):
        record = super().get_task(task_id)
        if record is not None and record["status"] == "QUEUED":
            self.store_result(
                task_id,
                "SUCCEEDED",
                result={
                    "correctness": {
                        "passed": True,
                        "compiled": True,
                        "decoy": False,
                        "passed_cases": 1,
                        "total_cases": 1,
                        "failures": [],
                    },
                    "performance": None,
                    "profile": None,
                },
            )
            return super().get_task(task_id)
        return record


class _MalformedCompletingBroker(Broker):
    def get_task(self, task_id):
        record = super().get_task(task_id)
        if record is not None and record["status"] == "QUEUED":
            self.store_result(
                task_id,
                "SUCCEEDED",
                result={"correctness": {"passed": True}},
            )
            return super().get_task(task_id)
        return record


class _NonFiniteCompletingBroker(Broker):
    def get_task(self, task_id):
        record = super().get_task(task_id)
        if record is not None and record["status"] == "QUEUED":
            self.store_result(
                task_id,
                "SUCCEEDED",
                result={
                    "correctness": {
                        "passed": False,
                        "compiled": True,
                        "decoy": False,
                        "passed_cases": 0,
                        "total_cases": 1,
                        "failures": [
                            {
                                "case_idx": 1,
                                "kind": "scattered",
                                "sample": [
                                    {
                                        "idx": [0],
                                        "candidate": math.nan,
                                        "reference": 0.0,
                                        "diff": math.nan,
                                    }
                                ],
                                "error": None,
                            }
                        ],
                    },
                    "performance": None,
                    "profile": None,
                },
            )
            return super().get_task(task_id)
        return record


def test_post_evaluations_waits_and_returns_terminal_result(settings):
    client, broker = _client(settings, _CompletingBroker)

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "container-17"},
        json=_request_body(
            checks=["correctness", "performance"],
            performance={"warmup": 2, "repeats": 20},
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"task_id", "status", "error", "result"}
    assert body["status"] == "SUCCEEDED"
    assert body["result"]["correctness"]["passed"] is True
    assert body["result"]["performance"] is None
    job = broker.dequeue()
    assert job.type == "evaluation"
    assert job.spec.checks == ["correctness", "performance"]
    assert job.spec.warmup == 2
    assert job.spec.repeats == 20
    assert job.spec.session_id == "container-17"


def test_malformed_worker_result_becomes_contract_shaped_failure(settings):
    client, _ = _client(settings, _MalformedCompletingBroker)

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "container-17"},
        json=_request_body(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"] == "invalid worker result"
    assert response.json()["result"] == {
        "correctness": None,
        "performance": None,
        "profile": None,
    }


def test_non_finite_worker_float_becomes_contract_shaped_failure(settings):
    client, _ = _client(settings, _NonFiniteCompletingBroker)

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "container-17"},
        json=_request_body(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"] == "invalid worker result"


def test_missing_or_invalid_session_header_is_400(settings):
    client, broker = _client(settings)

    missing = client.post("/evaluations", json=_request_body())
    invalid = client.post(
        "/evaluations",
        headers={"X-Session-ID": "contains a space"},
        json=_request_body(),
    )

    assert missing.status_code == 400
    assert invalid.status_code == 400
    assert broker.dequeue() is None


def test_performance_without_correctness_is_400(settings):
    client, broker = _client(settings)

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "session-a"},
        json=_request_body(checks=["performance"]),
    )

    assert response.status_code == 400
    assert broker.dequeue() is None


def test_existing_session_lease_is_429(settings):
    client, broker = _client(settings)
    assert broker.acquire_session("session-a", "existing-task", 60) is True

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "session-a"},
        json=_request_body(),
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert broker.dequeue() is None


def test_request_timeout_returns_terminal_payload_without_cancelling_job(
    settings,
    monkeypatch,
):
    import triton_evaluation.api.routes as routes

    ticks = iter([100.0, 111.0])
    monkeypatch.setattr(routes, "_monotonic", lambda: next(ticks))
    client, broker = _client(settings)

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "session-a"},
        json=_request_body(timeout=10),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "TIMEOUT"
    assert body["error"] == "evaluation request timed out"
    assert body["result"] == {"correctness": None, "performance": None, "profile": None}
    assert broker.get_task(body["task_id"])["status"] == "QUEUED"
    assert broker.acquire_session("session-a", "duplicate-task", 60) is False


def test_removed_v1_and_polling_routes_are_404(settings):
    client, _ = _client(settings)

    assert client.post("/verify", json={}).status_code == 404
    assert client.post("/profile", json={}).status_code == 404
    assert client.get("/tasks/t1").status_code == 404
    assert client.get("/evaluations/t1").status_code == 404


def test_healthz(settings):
    client, _ = _client(settings)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"redis": "ok"}


def test_end_to_end_sync_combination_contract(settings):
    from threading import Thread

    from triton_evaluation.evaluators.base import EvaluatorResult
    from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
    from triton_evaluation.models import TaskStatus
    from triton_evaluation.worker import Worker

    class _FailVerify:
        def run(self, spec, verify_dir, device_id):
            return EvaluatorResult(
                TaskStatus.SUCCEEDED.value,
                result={
                    "op_name": spec.op_name,
                    "total_cases": 2,
                    "passed_cases": 1,
                    "failed_cases": 1,
                    "failures": [
                        {
                            "case_idx": 2,
                            "error_type": "AccuracyError",
                            "error_msg": "classified traceback",
                            "kind": "constant_offset",
                            "sample": [],
                        }
                    ],
                },
            )

    class _MustNotBenchmark:
        def run(self, spec, verify_dir, device_id):
            raise AssertionError("performance must not run for a wrong kernel")

    class _CompleteProfile:
        def run(self, spec, verify_dir, device_id):
            return EvaluatorResult(
                TaskStatus.SUCCEEDED.value,
                result={
                    "profile_mode": "op",
                    "selected_cases": 1,
                    "completed_cases": 1,
                    "partial_cases": 0,
                    "failed_cases": 0,
                    "timed_out": False,
                    "per_case_results": [
                        {
                            "case_idx": 0,
                            "status": "complete",
                            "kernels": [
                                {"name": "softmax_kernel", "duration_us": 400.0}
                            ],
                        }
                    ],
                },
            )

    class _HarnessEvaluation(EvaluationEvaluator):
        def _verify_evaluator(self):
            return _FailVerify()

        def _benchmark_evaluator(self):
            return _MustNotBenchmark()

        def _profile_evaluator(self):
            return _CompleteProfile()

    client, broker = _client(settings)
    worker = Worker(
        broker,
        {"evaluation": _HarnessEvaluation("unused")},
        device_id=0,
        settings=settings,
    )
    worker_thread = Thread(target=worker.run_once)
    worker_thread.start()

    response = client.post(
        "/evaluations",
        headers={"X-Session-ID": "e2e-session"},
        json=_request_body(checks=["correctness", "performance", "profile"]),
    )
    worker_thread.join(timeout=5)

    assert worker_thread.is_alive() is False
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["result"]["correctness"]["passed"] is False
    assert body["result"]["correctness"]["failures"][0]["kind"] == "constant_offset"
    assert body["result"]["correctness"]["failures"][0]["error"] is None
    assert body["result"]["performance"] is None
    assert body["result"]["profile"]["profile_state"] == "complete"
    assert broker.acquire_session("e2e-session", "next-task", 60) is True
