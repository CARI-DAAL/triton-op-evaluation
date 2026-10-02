"""Native routes exercise real catalog binding and durable resource ownership.

Only kernel execution is replaced: a test worker claims and fences terminal
outcomes in the real store. Authentication, request translation, budgets,
admission/replay, polling, quotas and visibility run through production code.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_api import algorithms, build_app, make_test_settings, seed_worker, trial_headers

from triton_evaluation.api.contract import EvaluationResult
from triton_evaluation.api.kernel_eval import TOOL_CALLER, ToolEvaluationError, _evaluate
from triton_evaluation.models import WorkerIdentity, new_worker_id
from triton_evaluation.official_task import load_task

TOKEN = "native-tool-token"
TASK_ID = "official_l3_Demo"
FIXTURES = Path(__file__).parent / "fixtures" / "kernel_eval"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_native_app(tmp_path):
    task_dir = tmp_path / "official" / "Demo"
    task_dir.mkdir(parents=True)
    (task_dir / "definition.json").write_text(
        json.dumps(
            {
                "name": "Demo",
                "inputs": {"x": {"dtype": "float32"}},
                "reference": "def run(x):\n    return x + 1\n",
            }
        )
    )
    (task_dir / "workload.jsonl").write_text(
        json.dumps({"inputs": {"x": {"type": "random", "dtype": "float32", "shape": [2]}}}) + "\n"
    )
    task = load_task(task_dir, problem_id=TASK_ID)
    root = tmp_path / "catalog"
    refs = root / "references"
    refs.mkdir(parents=True)
    relative = f"references/{TASK_ID}.py"
    (root / relative).write_text(task["reference_source"])
    provenance = {
        "schema_version": "1.0",
        "manifest_id": "official-release",
        "problem_id": TASK_ID,
        "reference_digest": task["reference_digest"],
        "official_digest": task["official_digest"],
        "op_name": "Demo",
        "adapter_version": task["adapter_version"],
        "cases": 1,
    }
    (root / (relative + ".PROVENANCE.json")).write_text(json.dumps(provenance))
    manifest = root / "catalog.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": [
                    {
                        "manifest_id": "official-release",
                        "problem_id": TASK_ID,
                        "reference_digest": task["reference_digest"],
                        "reference_path": relative,
                    }
                ],
            }
        )
    )
    files = SimpleNamespace(root=root, manifest_path=manifest)
    settings = make_test_settings(tmp_path).model_copy(
        update={
            "kernel_eval_enabled": True,
            "kernel_eval_auto_diagnostics": False,
        }
    )
    # Revalidate rather than bypassing SecretStr coercion with model_copy.
    settings = type(settings).model_validate({**settings.model_dump(), "kernel_eval_token": TOKEN})
    app = build_app(tmp_path, files, settings=settings)
    app.task = task
    return app


@pytest.fixture
def native_app(tmp_path):
    return build_native_app(tmp_path)


def payload(**overrides):
    body = {"task_id": TASK_ID, "kernel_code": "KERNEL", "session": "trial-1"}
    body.update(overrides)
    return body


def headers(key=None):
    result = {"Authorization": f"Bearer {TOKEN}"}
    if key is not None:
        result["Idempotency-Key"] = key
    return result


def outcome(name, settings):
    fixture = json.loads((FIXTURES / f"{name}.input.json").read_bytes())
    result = EvaluationResult.model_validate(fixture["resource"]["result"])
    # The terminal worker result belongs to this deployment/official trial.
    from conftest import make_provenance

    return result.model_copy(update={"provenance": make_provenance(settings)})


async def complete_one(app, request_task, name="verify_correct"):
    worker = WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
    for _ in range(500):
        lease = await asyncio.to_thread(app.store.claim_next, worker)
        if lease is not None:
            result = outcome(name, app.settings)
            assert await asyncio.to_thread(app.store.commit_terminal, lease, "completed", result)
            return lease
        if request_task.done():
            response = await request_task
            pytest.fail(f"request finished before durable work was admitted: {response.text}")
        await asyncio.sleep(0.005)
    request_task.cancel()
    pytest.fail("test worker could not claim a request")


@pytest.mark.parametrize("route", ["verify", "benchmark", "profile"])
async def test_single_token_and_legacy_refusals(native_app, route):
    app = native_app
    async with app.http as client:
        unauthorized = await client.post(f"/{route}", content="not json")
        assert (unauthorized.status_code, unauthorized.json()) == (401, {"error": "unauthorized"})
        v21_token = await client.post(f"/{route}", json=payload(), headers=trial_headers())
        assert v21_token.status_code == 401
        malformed = await client.post(f"/{route}", content="not json", headers=headers())
        assert (malformed.status_code, malformed.json()) == (400, {"error": "invalid json"})
        reference = await client.post(
            f"/{route}", json=payload(reference_code="ATTACK"), headers=headers()
        )
        assert (reference.status_code, reference.json()) == (
            400,
            {"error": "reference fields are forbidden"},
        )
        missing = await client.post(f"/{route}", json={"task_id": TASK_ID}, headers=headers())
        assert missing.json() == {"error": "task_id and kernel_code are required"}
        mismatch = await client.post(f"/{route}", json=payload(task_id="other"), headers=headers())
        assert (mismatch.status_code, mismatch.json()) == (403, {"error": "task_id mismatch"})
    assert not app.store._r.keys(f"{app.settings.redis_prefix}:kernel-eval:budget:*")


@pytest.mark.parametrize("changes", [{"kernel_eval_enabled": False}, {"kernel_eval_token": ""}])
async def test_disabled_or_empty_token_fails_closed(native_app, changes):
    from pydantic import SecretStr

    for name, value in changes.items():
        setattr(native_app.settings, name, SecretStr(value) if name.endswith("token") else value)
    async with native_app.http as client:
        response = await client.post("/verify", json=payload(), headers=headers())
        health = await client.get("/health")
    assert response.status_code == health.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize(
    "options",
    [
        {"profile_mode": "invalid"},
        {"case_idx": True},
        {"case_idx": -1},
        {"profile_launch_count": 0},
        {"profile_launch_count": True},
        {"profile_mode": "deep", "profile_launch_count": 2},
    ],
)
async def test_profile_option_rejection_does_not_reserve_turn(native_app, options):
    async with native_app.http as client:
        response = await client.post("/profile", json=payload(**options), headers=headers())
    assert (response.status_code, response.json()) == (400, {"error": "invalid profile options"})
    assert not native_app.store._r.keys(f"{native_app.settings.redis_prefix}:kernel-eval:budget:*")


@pytest.mark.parametrize(
    "route,name,checks",
    [
        ("verify", "verify_correct", ["correctness"]),
        ("benchmark", "benchmark_measured", ["correctness", "performance"]),
        ("profile", "profile_complete", ["profile"]),
    ],
)
async def test_native_catalog_admission_polling_and_owned_replay(native_app, route, name, checks):
    app = native_app
    seed_worker(app)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post(f"/{route}", json=payload(), headers=headers("call-1"))
        )
        lease = await complete_one(app, pending, name)
        first = await pending
        replay = await client.post(f"/{route}", json=payload(), headers=headers("call-1"))
        invisible = await client.get(f"/evaluations/{lease.evaluation_id}", headers=trial_headers())
        direct_token = await client.get(f"/evaluations/{lease.evaluation_id}", headers=headers())
    canonical = json.loads(lease.canonical_request)
    assert canonical["reference_code"] == app.task["reference_source"]
    assert canonical["op_name"] == "Demo"
    assert canonical["purpose"] == "trial"
    assert canonical["checks"] == checks
    if route == "benchmark":
        assert canonical["performance"] == {"warmup": 5, "repeats": 20, "method": "profiler"}
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["eval_budget"] == "1/18"
    assert first.json()["request_status"] == "completed"
    assert invisible.status_code == 404
    assert direct_token.status_code == 401  # the tool token is confined to its facade
    assert app.store.get_visible(lease.evaluation_id, TOOL_CALLER).state == "completed"
    assert (
        app.store.claim_next(
            WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
        )
        is None
    )


async def test_catalog_reference_mutation_is_rejected(native_app):
    reference = Path(native_app.settings.catalog_root) / f"references/{TASK_ID}.py"
    reference.write_text("MUTATED")
    async with native_app.http as client:
        response = await client.post("/verify", json=payload(), headers=headers())
    assert (response.status_code, response.json()) == (403, {"error": "task_id mismatch"})


async def test_replay_conflict_and_budget_share_across_actions(native_app):
    app = native_app
    app.settings.kernel_eval_max_turns = 1
    seed_worker(app)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("call-1"))
        )
        await complete_one(app, pending)
        await pending
        replay = await client.post("/verify", json=payload(), headers=headers("call-1"))
        conflict = await client.post(
            "/verify", json=payload(kernel_code="CHANGED"), headers=headers("call-1")
        )
        blocked = await client.post("/profile", json=payload(), headers=headers("call-2"))
        second_trial = asyncio.create_task(
            client.post("/verify", json=payload(session="trial-2"), headers=headers("call-1"))
        )
        await complete_one(app, second_trial)
        second = await second_trial
    assert replay.json()["eval_budget"] == "1/1"
    assert (conflict.status_code, conflict.json()) == (409, {"error": "idempotency conflict"})
    assert blocked.json()["request_status"] == "budget_exhausted"
    assert blocked.json()["eval_turn"] is None
    assert second.json()["eval_budget"] == "1/1"


async def test_no_session_uses_legacy_remote_and_explicit_trial_is_separate(native_app):
    app = native_app
    app.settings.kernel_eval_max_turns = 1
    seed_worker(app)
    body = payload()
    body.pop("session")
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=body, headers=headers("legacy-1"))
        )
        await complete_one(app, pending)
        await pending
        blocked = await client.post("/verify", json=body, headers=headers("legacy-2"))
        explicit = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("legacy-2"))
        )
        await complete_one(app, explicit)
        independent = await explicit
    assert blocked.json()["request_status"] == "budget_exhausted"
    assert independent.json()["request_status"] == "completed"


async def test_legacy_repeated_tool_calls_charge_turns_while_evaluation_replays(native_app):
    app = native_app
    app.settings.kernel_eval_max_turns = 2
    seed_worker(app)
    async with app.http as client:
        pending = asyncio.create_task(client.post("/verify", json=payload(), headers=headers()))
        lease = await complete_one(app, pending)
        first = await pending
        second = await client.post("/verify", json=payload(), headers=headers())
        blocked = await client.post("/verify", json=payload(), headers=headers())
    assert first.json()["eval_budget"] == "1/2"
    assert second.json()["eval_budget"] == "2/2"
    assert second.json()["correct"] is True
    assert blocked.json()["request_status"] == "budget_exhausted"
    assert app.store.get_visible(lease.evaluation_id, TOOL_CALLER).state == "completed"
    worker = WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
    assert app.store.claim_next(worker) is None


async def test_readiness_failure_consumes_one_durable_turn(native_app):
    async with native_app.http as client:
        response = await client.post("/verify", json=payload(), headers=headers("retry-me"))
        replay = await client.post("/verify", json=payload(), headers=headers("retry-me"))
    assert response.json()["request_status"] == "infra_error"
    assert response.json()["eval_budget"] == replay.json()["eval_budget"] == "1/18"
    assert "service unavailable" in response.json()["error"]


async def test_case0_diagnostics_do_not_charge_a_second_turn(native_app):
    app = native_app
    app.settings.kernel_eval_auto_diagnostics = True
    app.settings.kernel_eval_max_turns = 1
    seed_worker(app)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("call-1"))
        )
        await complete_one(app, pending)
        diagnostic_lease = await complete_one(app, pending, "profile_complete")
        response = await pending
        replay = await client.post("/verify", json=payload(), headers=headers("call-1"))
    canonical = json.loads(diagnostic_lease.canonical_request)
    assert canonical["checks"] == ["profile"]
    assert canonical["profile"]["mode"] == "op"
    assert canonical["profile"]["case_indices"] == [0]
    assert response.json()["eval_budget"] == "1/1"
    assert response.json() == replay.json()
    assert response.json()["diagnostics"].startswith("kernel_metrics:")


async def test_poll_timeout_leaves_work_recoverable(native_app):
    import time

    from triton_evaluation.api.contract import EvaluationRequest

    app = native_app
    seed_worker(app)
    request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "Demo",
            "reference_code": app.task["reference_source"],
            "kernel_code": "KERNEL",
            "checks": ["correctness"],
        }
    )
    with pytest.raises(ToolEvaluationError, match="may complete later"):
        await _evaluate(
            app.app.state.context.instance,
            request,
            "trial-1",
            "timeout-key",
            time.monotonic() + 0.02,
        )
    worker = WorkerIdentity(new_worker_id(), app.settings.evaluator_build, algorithms(app.settings))
    lease = app.store.claim_next(worker)
    assert lease is not None
    assert app.store.commit_terminal(lease, "completed", outcome("verify_correct", app.settings))
    recovered = await _evaluate(
        app.app.state.context.instance, request, "trial-1", "timeout-key", time.monotonic() + 1
    )
    assert recovered["evaluation_id"] == lease.evaluation_id
    assert recovered["state"] == "completed"


@pytest.mark.parametrize("limited_by", ["rate_limited", "limit_exceeded"])
async def test_actual_durable_quota_contention_retries_one_idempotent_admission(
    native_app, monkeypatch, limited_by
):
    from triton_evaluation.api import kernel_eval
    from triton_evaluation.api.contract import EvaluationRequest

    app = native_app
    if limited_by == "rate_limited":
        app.settings.trial_rate_limit_per_minute = 1
    else:
        app.settings.trial_max_inflight = 1
    seed_worker(app)
    instance = app.app.state.context.instance
    blocking_request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": "Demo",
            "reference_code": app.task["reference_source"],
            "kernel_code": "BLOCKER",
            "checks": ["correctness"],
        }
    )
    blocker = instance.submit(
        blocking_request, TOOL_CALLER, session_id="another-trial", idempotency_key="blocker"
    )
    assert blocker.decision == "created"
    # Keep real admission/quota transactions; shorten only the transport
    # retry delay, and observe which real store decision triggers the retry.
    monkeypatch.setattr(kernel_eval, "_admission_backoff_seconds", lambda _a, _r: 0.01)
    attempts = []
    real_submit = instance.submit

    def observe_submit(*args, **kwargs):
        result = real_submit(*args, **kwargs)
        attempts.append((result.decision, kwargs["idempotency_key"]))
        return result

    monkeypatch.setattr(instance, "submit", observe_submit)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("contention"))
        )
        for _ in range(500):
            if attempts:
                break
            await asyncio.sleep(0.001)
        assert attempts[0][0] == limited_by
        worker = WorkerIdentity(
            new_worker_id(), app.settings.evaluator_build, algorithms(app.settings)
        )
        blocker_lease = app.store.claim_next(worker)
        assert blocker_lease.evaluation_id == blocker.resource.evaluation_id
        assert app.store.commit_terminal(
            blocker_lease, "completed", outcome("verify_correct", app.settings)
        )
        if limited_by == "rate_limited":
            app.clock.advance_ms(60_000)
            seed_worker(app)
        lease = await complete_one(app, pending)
        response = await pending
        replay = await client.post("/verify", json=payload(), headers=headers("contention"))
    assert response.json() == replay.json()
    assert response.json()["eval_budget"] == "1/18"
    assert response.json()["request_status"] == "completed"
    assert len({key for _decision, key in attempts}) == 1
    assert sum(decision == "created" for decision, _key in attempts) == 1
    assert instance.get(lease.evaluation_id, TOOL_CALLER).state == "completed"


async def test_optional_diagnostics_failure_preserves_correctness(native_app, monkeypatch):
    from triton_evaluation.instance import InstanceNotReadyError

    app = native_app
    app.settings.kernel_eval_auto_diagnostics = True
    seed_worker(app)
    instance = app.app.state.context.instance
    real_submit = instance.submit

    def fail_diagnostics(request, *args, **kwargs):
        if request.checks == ["profile"]:
            raise InstanceNotReadyError()
        return real_submit(request, *args, **kwargs)

    monkeypatch.setattr(instance, "submit", fail_diagnostics)
    async with app.http as client:
        pending = asyncio.create_task(
            client.post("/verify", json=payload(), headers=headers("diagnostics-fail"))
        )
        await complete_one(app, pending)
        response = await pending
    assert response.json()["correct"] is True
    assert response.json()["request_status"] == "completed"
    assert response.json()["eval_budget"] == "1/18"
    assert "diagnostics" not in response.json()
