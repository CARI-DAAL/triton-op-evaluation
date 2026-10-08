"""Caller intent through its public interface with real admission and fenced completion.

No HTTP application or internal method patches: deterministic waits drive a real
store and a worker adapter that supplies already-tested engine result fixtures.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import fakeredis
import pytest
from conftest import FakeClock
from native_eval_helpers import TASK_ID, outcome, task_catalog, worker_identity

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.config import Settings
from triton_evaluation.kernel_eval_budget import EvalBudgetRegistry
from triton_evaluation.kernel_evaluation import TOOL_CALLER, KernelEvaluation, ToolInputError
from triton_evaluation.service.broker import EvaluationStore
from triton_evaluation.service.catalog import ReleaseCatalog
from triton_evaluation.service.instance import EvaluationInstance

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def payload(**overrides):
    return {"task_id": TASK_ID, "kernel_code": "KERNEL", **overrides}


class WaitClock:
    def __init__(self):
        self.now = 0.0
        self.delays = []
        self.on_wait = None

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay
        if self.on_wait is not None:
            self.on_wait()


def build_tool(tmp_path, **overrides):
    files, task = task_catalog(tmp_path)
    settings = Settings(
        engine_dir=str(tmp_path), cann_fingerprint="test-cann", card_canary_on_startup=False,
        catalog_root=str(files.root), catalog_manifest_path=str(files.manifest_path),
        kernel_eval_auto_diagnostics=False, **overrides,
    )
    client = fakeredis.FakeStrictRedis()
    clock = FakeClock()
    store = EvaluationStore(client, settings, clock=clock)
    instance = EvaluationInstance(
        settings=settings, store=store,
        catalog=ReleaseCatalog.load(files.root, files.manifest_path),
    )
    budget = EvalBudgetRegistry(
        client, prefix=settings.redis_prefix, maximum=settings.kernel_eval_max_turns,
        retention_seconds=settings.retention_seconds,
    )
    wait = WaitClock()
    tool = KernelEvaluation(instance, budget, clock=wait, sleep=wait.sleep)
    harness = SimpleNamespace(
        tool=tool, instance=instance, budget=budget, settings=settings,
        client=client, store=store, clock=clock, wait=wait, task=task, leases=[],
        worker=worker_identity(settings),
    )
    return harness


def ready(harness):
    harness.store.heartbeat(harness.worker)


def complete_next(harness):
    lease = harness.store.claim_next(harness.worker)
    if lease is None:
        return None
    checks = json.loads(lease.canonical_request)["checks"]
    name = (
        "profile_complete" if checks == ["profile"] else
        "benchmark_measured" if "performance" in checks else "verify_correct"
    )
    assert harness.store.commit_terminal(lease, "completed", outcome(name, harness.settings))
    harness.leases.append(lease)
    return lease


def executing(harness):
    ready(harness)
    harness.wait.on_wait = lambda: complete_next(harness)


async def evaluate(harness, action="verify", operation="call-1", session="trial-1", **changes):
    return await harness.tool.evaluate(
        action, payload(**changes), session_id=session, operation_id=operation,
    )


async def test_replay_conflict_budget_and_sessions_survive_module_reconstruction(tmp_path):
    h = build_tool(tmp_path, kernel_eval_max_turns=1)
    executing(h)
    first = await evaluate(h)
    # A fresh module/adapter over the same Redis still owns the turn and identity.
    h.tool = KernelEvaluation(h.instance, EvalBudgetRegistry(
        h.client, prefix=h.settings.redis_prefix, maximum=1,
        retention_seconds=h.settings.retention_seconds,
    ), clock=h.wait, sleep=h.wait.sleep)
    assert await evaluate(h) == first
    with pytest.raises(ToolInputError) as error:
        await evaluate(h, kernel_code="CHANGED")
    assert error.value.kind == "conflict"
    blocked = await evaluate(h, action="profile", operation="call-2")
    assert blocked["request_status"] == "budget_exhausted"
    assert blocked["eval_turn"] is None
    assert (await evaluate(h, session="trial-2"))["eval_budget"] == "1/1"
    assert len(h.leases) == 2
    assert len({lease.evaluation_id for lease in h.leases}) == 2


async def test_legacy_calls_charge_each_turn_but_replay_same_evaluation(tmp_path):
    h = build_tool(tmp_path, kernel_eval_max_turns=2)
    executing(h)
    first = await evaluate(h, operation=None)
    second = await evaluate(h, operation=None)
    assert first["eval_budget"] == "1/2"
    assert second["eval_budget"] == "2/2"
    assert second["correct"] is True
    assert (await evaluate(h, operation=None))["request_status"] == "budget_exhausted"
    assert len(h.leases) == 1
    assert h.store.claim_next(h.worker) is None


async def test_readiness_failure_consumes_one_durable_turn(tmp_path):
    h = build_tool(tmp_path)
    first = await evaluate(h)
    replay = await evaluate(h)
    assert first["request_status"] == "infra_error"
    assert first["eval_budget"] == replay["eval_budget"] == "1/18"
    assert "service unavailable" in first["error"]


async def test_case0_diagnostics_share_turn_and_replay(tmp_path):
    h = build_tool(tmp_path, kernel_eval_max_turns=1)
    h.settings.kernel_eval_auto_diagnostics = True
    executing(h)
    first = await evaluate(h)
    assert first["eval_budget"] == "1/1"
    assert first["diagnostics"].startswith("kernel_metrics:")
    assert await evaluate(h) == first
    assert len(h.leases) == 2
    diagnostic = json.loads(h.leases[1].canonical_request)
    assert diagnostic["checks"] == ["profile"]
    assert diagnostic["profile"]["mode"] == "op"
    assert diagnostic["profile"]["case_indices"] == [0]


async def test_poll_timeout_keeps_accepted_work_and_same_operation_recovers(tmp_path):
    h = build_tool(tmp_path, kernel_eval_max_wait_seconds=10, api_poll_interval_seconds=4)
    ready(h)
    first = await evaluate(h)
    assert first["request_status"] == "infra_error"
    assert "may complete later" in first["error"]
    lease = complete_next(h)
    assert lease is not None
    recovered = await evaluate(h)
    assert recovered["request_status"] == "completed"
    assert recovered["eval_budget"] == first["eval_budget"] == "1/18"
    assert h.instance.get(lease.evaluation_id, TOOL_CALLER).state == "completed"
    assert h.store.claim_next(h.worker) is None


async def test_diagnostics_share_wait_deadline_and_timeout_preserves_correctness(tmp_path):
    h = build_tool(tmp_path, kernel_eval_max_wait_seconds=10, api_poll_interval_seconds=4)
    h.settings.kernel_eval_auto_diagnostics = True
    ready(h)

    def only_complete_primary():
        if not h.leases:
            complete_next(h)

    h.wait.on_wait = only_complete_primary
    result = await evaluate(h)
    assert result["correct"] is True
    assert result["request_status"] == "completed"
    assert "diagnostics" not in result
    assert result["eval_budget"] == "1/18"
    assert h.wait.now == pytest.approx(10.0)
    assert h.wait.delays == pytest.approx([4, 4, 2])
    # Optional timeout cannot cancel the already-admitted diagnostic either.
    assert complete_next(h) is not None
    assert len(h.leases) == 2


@pytest.mark.parametrize("limited_by", ["rate_limited", "limit_exceeded"])
async def test_real_quota_contention_retries_one_admission(tmp_path, limited_by):
    options = {"trial_rate_limit_per_minute": 1} if limited_by == "rate_limited" else {
        "trial_max_inflight": 1,
    }
    h = build_tool(tmp_path, **options)
    ready(h)
    request = EvaluationRequest.model_validate({
        "schema_version": "2.1", "op_name": "Demo",
        "reference_code": h.task["reference_source"], "kernel_code": "BLOCKER",
        "checks": ["correctness"],
    })
    blocker = h.instance.submit(
        request, TOOL_CALLER, session_id="blocker-session", idempotency_key="blocker",
    )
    assert blocker.decision == "created"

    def release_contention():
        lease = complete_next(h)
        assert lease is not None
        if lease.evaluation_id == blocker.resource.evaluation_id and limited_by == "rate_limited":
            h.clock.advance_ms(60_000)
            ready(h)

    h.wait.on_wait = release_contention
    result = await evaluate(h)
    assert result["request_status"] == "completed"
    assert result["eval_budget"] == "1/18"
    assert await evaluate(h) == result
    assert h.wait.delays[0] == 2.0  # The first wait is admission backoff, not polling.
    assert len(h.leases) == 2
    assert h.store.claim_next(h.worker) is None


@pytest.mark.parametrize("options", [
    {"profile_mode": "invalid"}, {"case_idx": True}, {"case_idx": -1},
    {"profile_launch_count": 0}, {"profile_launch_count": True},
    {"profile_mode": "deep", "profile_launch_count": 2},
])
async def test_invalid_profile_never_reserves_a_turn(tmp_path, options):
    h = build_tool(tmp_path)
    with pytest.raises(ToolInputError, match="invalid profile options") as error:
        await evaluate(h, action="profile", **options)
    assert error.value.kind == "invalid"
    assert h.client.dbsize() == 0


@pytest.mark.parametrize("changes", [
    {"action": "unknown"}, {"operation": "bad key"}, {"session": "bad session"},
    {"reference_code": "ATTACK"}, {"kernel_code": None},
])
async def test_invalid_input_never_reserves_a_turn(tmp_path, changes):
    h = build_tool(tmp_path)
    with pytest.raises(ToolInputError) as error:
        await evaluate(h, **changes)
    assert error.value.kind == "invalid"
    assert h.client.dbsize() == 0


async def test_catalog_mutation_rejected_before_reserving_turn(tmp_path):
    h = build_tool(tmp_path)
    reference = Path(h.settings.catalog_root) / h.task["reference_path"]
    reference.write_text("MUTATED")
    with pytest.raises(ToolInputError) as error:
        await evaluate(h)
    assert error.value.kind == "forbidden"
    assert h.client.dbsize() == 0


async def test_optional_diagnostics_readiness_failure_preserves_primary_result(tmp_path):
    h = build_tool(tmp_path)
    h.settings.kernel_eval_auto_diagnostics = True
    ready(h)

    def finish_then_lose_capacity():
        assert complete_next(h) is not None
        h.clock.advance_ms(h.settings.lease_seconds * 1000 + 1)

    h.wait.on_wait = finish_then_lose_capacity
    result = await evaluate(h)
    assert result["correct"] is True
    assert result["request_status"] == "completed"
    assert result["eval_budget"] == "1/18"
    assert "diagnostics" not in result
    assert len(h.leases) == 1
    assert h.store.claim_next(h.worker) is None
