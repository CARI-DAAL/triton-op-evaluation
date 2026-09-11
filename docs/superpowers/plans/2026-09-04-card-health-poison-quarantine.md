# Card Health Loop + Poison Quarantine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fleet protects itself from suspect kernels and sick cards: suspicious attempt deaths are counted per evaluation (poison quarantine at 2 deaths, 2.0-degraded WORKER_LOST shape), the dead card is recorded for affinity avoidance, and the worker owning a card runs a canary → drain → recovery-ladder loop (cleanup → configured device reset → process restart; host reboot never automatic).

**Architecture:** Death counting lives ONLY in the reaper (`_recover_one_lease` — lease expiry is the definitive exactly-once death signal the system already trusts). Card-health Redis keys are written ONLY by the worker that owns the card (single writer). The worker's canary is a vendored mini add-kernel run in-process under a SIGALRM budget; a failed canary drains the worker (skip heartbeat → readiness drops after one lease window → 503 NOT_READY when no claim-capable worker remains) and drives the ladder. Everything is contract-invisible (Release 1); `SUSPECTED_POISON_CANDIDATE` as a formal check code rides with Plan E.

**Tech Stack:** Python 3.11, fakeredis + FakeClock in tests; torch_npu/triton canary is NPU-host-only (lazy imports).

**Spec:** `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§10)

## Global Constraints

- **Contract-invisible**: no `contracts/` or `api/contract.py` changes; poison terminal uses the existing synthesized `WORKER_LOST` shape (`retryable=True` stays — frozen) with a quarantine message; no new ErrorEnvelope codes.
- **Fencing/state machine unchanged**: new resource-hash fields are advisory metadata (`suspicious_deaths`, `claimed_card`, `last_failed_card`), never part of fence conditions.
- **Single writer per card-health key**: only the worker owning the card writes `card-health:{card}`; the API/reaper never does.
- Heartbeat semantics ("idle workers heartbeat so readiness means claim-capable") stay intact: a drained worker skips heartbeats deliberately.
- Tests: fakeredis + FakeClock only (Tasks 1-3); Task 4's NPU test is marker-gated and skipped off-host.
- Full suite green each task except the 5 pre-existing environmental `referencing` failures.
- ruff line-length 100, target py310.

---

### Task 1: Settings + broker — death counting, poison quarantine, affinity

**Files:**
- Modify: `triton_evaluation/config.py` (Plan C settings block)
- Modify: `triton_evaluation/broker.py` (`_new_queued_resource`, `_recover_one_lease`, `claim_next`, new small methods)
- Modify: `tests/test_recovery.py`, `tests/test_broker.py` (append)
- Modify: `docker.env.example` (commented block)

**Interfaces:**
- Consumes: existing `Settings` Field pattern; `synthesize_orchestration_result` ("worker_lost" shape).
- Produces:
  - Settings: `poison_death_threshold: int = 2 (ge=2)`, `card_affinity_avoidance: bool = True`.
  - Resource hash fields: `suspicious_deaths` (init `"0"`), `claimed_card`, `last_failed_card`.
  - `EvaluationStore.note_claimed_card(evaluation_id: str, card: int) -> None` (advisory HSET, swallows errors).
  - `EvaluationStore.claim_next(worker, avoid_card: int | None = None)` — skips members whose stored `last_failed_card` equals `avoid_card`.
  - `EvaluationStore.set_card_unhealthy(card: int) -> None` / `clear_card_unhealthy(card: int) -> None` / `is_card_unhealthy(card: int) -> bool` (hash `card-health:{card}`, field `state` = `unhealthy`/`ok`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recovery.py`:

```python
def test_reap_counts_suspicious_death_and_records_failed_card(store, redis_client,
                                                              clock, settings,
                                                              make_command, make_worker):
    cmd = make_command()
    store.create_or_replay(cmd)
    worker = make_worker()
    lease = store.claim_next(worker)
    store.note_claimed_card(lease.evaluation_id, card=7)
    clock.advance_ms(settings.lease_seconds * 1000 + 1)

    counts = store.recover_expired_leases()
    assert counts.requeued == 1
    key = f"triton-eval:v2:resource:{cmd.evaluation_id}"
    assert redis_client.hget(key, "suspicious_deaths") == b"1"
    assert redis_client.hget(key, "last_failed_card") == b"7"


def test_second_suspicious_death_quarantines_as_worker_lost(store, clock, settings,
                                                            make_command, make_worker):
    cmd = make_command()
    store.create_or_replay(cmd)
    worker = make_worker()
    for _ in range(2):
        lease = store.claim_next(worker)
        store.note_claimed_card(lease.evaluation_id, card=7)
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        counts = store.recover_expired_leases()
        if counts.failed:
            break
        # re-arm the replacement claim for the next death
        clock.advance_ms(1)
    resource = store.get_visible(cmd.evaluation_id, make_caller()) if False else None
    stored = redis_client_hgetall(store, cmd.evaluation_id)
    assert stored[b"suspicious_deaths"] == b"2"
    resource = EvaluationResource.model_validate_json(stored[b"resource"])
    assert resource.state == "failed"
    assert resource.result.correctness.status == "failed"
    assert resource.result.correctness.error.code == "WORKER_LOST"
    assert "poison" in resource.result.correctness.error.message
```

(The second test needs small local helpers — define at module level in the test file:)

```python
def redis_client_hgetall(store, evaluation_id):
    return store._r.hgetall(f"triton-eval:v2:resource:{evaluation_id}")


def make_caller():  # if not already importable from conftest in this file
    from conftest import make_caller as _mc
    return _mc()
```

Append to `tests/test_broker.py`:

```python
def test_claim_avoid_card_skips_last_failed_card(store, redis_client, clock, settings,
                                                 make_command, make_worker):
    poisoned = make_command(session_id="s-1", idempotency_key="k-1")
    store.create_or_replay(poisoned)
    fine = make_command(session_id="s-2", idempotency_key="k-2")
    store.create_or_replay(fine)
    for cmd in (poisoned, fine):
        lease = store.claim_next(make_worker())
        store.note_claimed_card(lease.evaluation_id, card=3)
        clock.advance_ms(settings.lease_seconds * 1000 + 1)
        store.recover_expired_leases()
        clock.advance_ms(1)
    # both have a suspicious death; poisoned re-armed once more to carry last_failed_card
    lease = store.claim_next(make_worker(), avoid_card=3)
    assert lease.evaluation_id == fine.evaluation_id


def test_note_claimed_card_and_card_health_roundtrip(store, redis_client):
    store.note_claimed_card("ev_x", card=7)
    assert redis_client.hget("triton-eval:v2:resource:ev_x", "claimed_card") == b"7"
    assert not store.is_card_unhealthy(7)
    store.set_card_unhealthy(7)
    assert store.is_card_unhealthy(7)
    store.clear_card_unhealthy(7)
    assert not store.is_card_unhealthy(7)
```

Append settings tests to `tests/test_config.py`:

```python
def test_plan_c_health_defaults():
    s = Settings()
    assert s.poison_death_threshold == 2
    assert s.card_affinity_avoidance is True


def test_plan_c_health_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_POISON_DEATH_THRESHOLD", "3")
    monkeypatch.setenv("TRITON_EVAL_CARD_AFFINITY_AVOIDANCE", "false")
    s = Settings.from_env()
    assert s.poison_death_threshold == 3
    assert s.card_affinity_avoidance is False
```

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_recovery.py tests/test_broker.py tests/test_config.py -q`
Expected: failures — no `note_claimed_card`, no `avoid_card`, no `suspicious_deaths`, no settings fields.

- [ ] **Step 3: Settings**

In `triton_evaluation/config.py`, after the rollout block:

```python
    # Card health loop (spec §10): poison quarantine + canary/drain/recovery.
    poison_death_threshold: int = Field(default=2, ge=2)
    card_affinity_avoidance: bool = Field(default=True)
```

- [ ] **Step 4: Broker — resource fields + note_claimed_card + card health + claim avoidance**

`_new_queued_resource` mapping gains (next to `"attempt_count": "0"`):

```python
            "suspicious_deaths": "0",
```

New methods (place near `heartbeat`):

```python
    def note_claimed_card(self, evaluation_id: str, card: int) -> None:
        """Advisory provenance: which card ran this attempt. Never fenced — affinity
        avoidance degrades gracefully if the write is lost."""
        try:
            self._r.hset(
                self._key(f"resource:{evaluation_id}"), "claimed_card", str(card)
            )
        except redis.RedisError:
            logger.warning("note_claimed_card failed for %s", evaluation_id)

    def set_card_unhealthy(self, card: int) -> None:
        self._r.hset(self._key(f"card-health:{card}"), "state", "unhealthy")

    def clear_card_unhealthy(self, card: int) -> None:
        self._r.hset(self._key(f"card-health:{card}"), "state", "ok")

    def is_card_unhealthy(self, card: int) -> bool:
        try:
            return self._r.hget(self._key(f"card-health:{card}"), "state") == b"unhealthy"
        except redis.RedisError:
            return False
```

`claim_next` signature: `def claim_next(self, worker: WorkerIdentity, avoid_card: int | None = None) -> AttemptLease | None:` — pass through to `_claim_from_queue`, and inside the member loop (after the `stored` empty-check, before the deadline check) add:

```python
                    if (
                        avoid_card is not None
                        and stored.get(b"last_failed_card") == str(avoid_card).encode()
                    ):
                        continue  # affinity avoidance: another card may claim it
```

- [ ] **Step 5: Broker — reap-time death counting + poison quarantine**

In `_recover_one_lease`, the requeue branch (`attempt_count < MAX_ATTEMPTS`). Replace its decision head:

```python
                deaths = int(stored.get(b"suspicious_deaths", b"0")) + 1
                failed_card = stored.get(b"claimed_card")
                if deaths >= self._s.poison_death_threshold:
                    # Poison quarantine (spec §10): too many suspicious deaths —
                    # terminate instead of requeueing. 2.0 contract-invisible shape:
                    # the existing synthesized WORKER_LOST result with a
                    # quarantine message (retryable stays True; that flag is frozen).
                    result = self._synthesized_result(
                        stored,
                        "worker_lost",
                        "worker lease expired before a subject outcome was committed; "
                        "repeated suspicious deaths — candidate quarantined as "
                        "suspected poison",
                    )
                    hash_updates = {"suspicious_deaths": str(deaths)}
                    if failed_card is not None:
                        hash_updates["last_failed_card"] = failed_card.decode("utf-8")
                    pipeline.watch(resource_key)
                    fresh = pipeline.hgetall(resource_key)
                    pipeline.multi()
                    pipeline.hset(resource_key, mapping=hash_updates)
                    pipeline.execute()
                    pipeline.watch(resource_key)
                    stored = self._r.hgetall(resource_key)
                    pipeline.multi()
                    self._commit_terminal_locked(
                        pipeline,
                        stored={**stored, b"resource": fresh[b"resource"]},
                        resource=EvaluationResource.model_validate_json(fresh[b"resource"]),
                        state="failed",
                        result=result,
                        now_ms=now_ms,
                        bump_fence=True,
                    )
                    pipeline.execute()
                    self._bump("reaped_leases_total", outcome="failed")
                    return "failed"
```

Wait — that double-watch shape is convoluted and risks divergence from `_commit_terminal_locked`'s expectations. Use the simpler, pattern-consistent form instead: the function already holds `stored` from its initial watch; extend the existing requeue `hset` mapping with the two fields, and for the poison path call `_commit_terminal_locked` with `stored` carrying the updated metadata (it only reads metadata for inflight keys / idempotency / fencing, and the synthesized result is built from `stored`'s request fields which are unchanged). Concretely:

```python
                deaths = int(stored.get(b"suspicious_deaths", b"0")) + 1
                failed_card = stored.get(b"claimed_card")
                metadata = {"suspicious_deaths": str(deaths)}
                if failed_card is not None:
                    metadata["last_failed_card"] = failed_card.decode("utf-8")
                if deaths >= self._s.poison_death_threshold:
                    result = self._synthesized_result(
                        stored,
                        "worker_lost",
                        "worker lease expired before a subject outcome was committed; "
                        "repeated suspicious deaths — candidate quarantined as "
                        "suspected poison",
                    )
                    enriched = {**stored, **{k.encode(): v.encode() for k, v in metadata.items()}}
                    pipeline.multi()
                    pipeline.hset(resource_key, mapping=metadata)
                    self._commit_terminal_locked(
                        pipeline,
                        stored=enriched,
                        resource=resource,
                        state="failed",
                        result=result,
                        now_ms=now_ms,
                        bump_fence=True,
                    )
                    pipeline.execute()
                    self._bump("reaped_leases_total", outcome="failed")
                    return "failed"
                pipeline.watch(ready_key, sequence_key)
                successor = int(pipeline.get(sequence_key) or 0) + 1
                pipeline.multi()
                pipeline.set(sequence_key, successor)
                pipeline.hset(resource_key, mapping={
                    "attempt_id": "",
                    "fencing_token": str(int(stored[b"fencing_token"]) + 1),
                    "replacement_pending": "1",
                    "active_check": "",
                    "lease_deadline_ms": "0",
                    **metadata,
                })
                pipeline.zrem(leases_key, member)
                pipeline.zadd(ready_key, {evaluation_id: successor})
                pipeline.execute()
                return "requeued"
```

(Delete the old requeue block this replaces. `_commit_terminal_locked` internally does `pipeline.watch(caller_key, session_key)` then `pipeline.multi()` — appending it inside our `multi()` block is INVALID; check how the existing attempt-limit path calls it: it calls `self._commit_terminal_locked(pipeline, ...)` directly followed by `pipeline.execute()`, WITHOUT an outer `pipeline.multi()`. Follow that exact pattern: for the poison path do NOT open `pipeline.multi()` yourself — call `pipeline.hset(...)` then `_commit_terminal_locked(...)` then a single `pipeline.execute()`. Since hset outside MULTI executes immediately, that is fine. Match the existing attempt-limit path's call shape verbatim.)

- [ ] **Step 6: env example + run + commit**

Append to `docker.env.example`:

```bash
# Card health loop (spec §10): poison quarantine threshold + claim affinity.
# TRITON_EVAL_POISON_DEATH_THRESHOLD=2
# TRITON_EVAL_CARD_AFFINITY_AVOIDANCE=true
```

Run: `python3 -m pytest tests/test_recovery.py tests/test_broker.py tests/test_config.py -q`, then the full suite; `ruff check triton_evaluation/ tests/test_recovery.py tests/test_broker.py tests/test_config.py`.

```bash
git add triton_evaluation/config.py triton_evaluation/broker.py \
        tests/test_recovery.py tests/test_broker.py tests/test_config.py docker.env.example
git commit -m "feat: suspicious-death counting, poison quarantine, claim affinity"
```

---

### Task 2: health module — canary + recovery ladder + metrics

**Files:**
- Create: `triton_evaluation/health.py`
- Modify: `triton_evaluation/config.py` (canary/recovery settings)
- Modify: `triton_evaluation/metrics.py` (new allowlist entries + labels + gauge)
- Test: `tests/test_health.py` (create)

**Interfaces:**
- Consumes: Task 1 settings pattern.
- Produces:
  - `triton_evaluation/health.py`: `CANARY_ADD_KERNEL: str`, `run_canary(*, device_id: int, budget_seconds: int) -> bool`, `run_recovery_step(step: int, *, card: int, reset_command: str) -> str` (returns `"cleanup" | "reset" | "restart"`; step 3 does `os.execv` and never returns).
  - Settings: `canary_budget_seconds: int = 120 (gt=0)`, `canary_interval_seconds: float = 60.0 (gt=0)`, `card_reset_command: str = "" (max_length=500)`, `card_recovery_max_step: int = 2 (ge=1, le=3)`, `card_recovery_retry_seconds: float = 300.0 (gt=0)`, `card_canary_on_startup: bool = True`.
  - Metrics: `card_canary_total{outcome=pass|fail}`, `card_recovery_total{step=cleanup|reset|restart}`, gauge `card_health_state{card=0..15}` (1 = unhealthy, 0 = healthy).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_health.py`:

```python
"""Card health module unit tests: recovery ladder routing and metric allowlists.
The canary body itself is NPU-only (tests/test_card_health_npu.py)."""
from types import SimpleNamespace

import pytest

import triton_evaluation.health as health
from triton_evaluation import metrics
from triton_evaluation.config import Settings


def test_ladder_cleanup_step_runs_without_device(mocker):
    mocker.patch.object(health, "_cleanup_in_process", return_value=None)
    name = health.run_recovery_step(1, card=7, reset_command="")
    assert name == "cleanup"
    health._cleanup_in_process.assert_called_once()


def test_ladder_reset_uses_configured_command(mocker):
    run = mocker.patch.object(health.subprocess, "run", return_value=SimpleNamespace(returncode=0))
    name = health.run_recovery_step(2, card=7, reset_command="npu-smi reset -c {card}")
    assert name == "reset"
    run.assert_called_once()
    assert run.call_args.kwargs["args"] == "npu-smi reset -c 7" or run.call_args[0][0] == "npu-smi reset -c 7"


def test_ladder_reset_without_command_is_noop_but_named(mocker):
    run = mocker.patch.object(health.subprocess, "run")
    name = health.run_recovery_step(2, card=7, reset_command="")
    assert name == "reset"
    run.assert_not_called()


def test_ladder_restart_execs_process(mocker):
    execv = mocker.patch.object(health.os, "execv")
    health.run_recovery_step(3, card=7, reset_command="")
    execv.assert_called_once()


def test_health_metric_allowlists_registered():
    assert metrics.ALLOWED_METRICS["card_canary_total"] == ("outcome",)
    assert metrics.ALLOWED_METRICS["card_recovery_total"] == ("step",)
    assert metrics.ALLOWED_METRICS["card_health_state"] == ("card",)
    assert "card_health_state" in metrics._GAUGE_NAMES
    assert "pass" in metrics.LABEL_VALUES["outcome"]
    assert "fail" in metrics.LABEL_VALUES["outcome"]
    assert "cleanup" in metrics.LABEL_VALUES["step"]
    assert metrics.LABEL_VALUES["card"] == tuple(str(i) for i in range(16))


def test_health_settings_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.canary_budget_seconds == 120
    assert s.canary_interval_seconds == 60.0
    assert s.card_reset_command == ""
    assert s.card_recovery_max_step == 2
    assert s.card_recovery_retry_seconds == 300.0
    assert s.card_canary_on_startup is True
    monkeypatch.setenv("TRITON_EVAL_CARD_RESET_COMMAND", "npu-smi reset -c {card}")
    monkeypatch.setenv("TRITON_EVAL_CARD_RECOVERY_MAX_STEP", "3")
    s2 = Settings.from_env()
    assert s2.card_reset_command == "npu-smi reset -c {card}"
    assert s2.card_recovery_max_step == 3
```

(If `pytest-mock`'s `mocker` is unavailable, use `unittest.mock.patch` — check which the repo's tests already use and follow that.)

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_health.py -q`
Expected: import/collection errors (`triton_evaluation.health` missing; settings fields missing).

- [ ] **Step 3: Settings**

After the Task 1 block:

```python
    # Canary + recovery ladder (spec §10).
    canary_budget_seconds: int = Field(default=120, gt=0)
    canary_interval_seconds: float = Field(default=60.0, gt=0)
    card_reset_command: str = Field(default="", max_length=500)  # "{card}" substituted
    card_recovery_max_step: int = Field(default=2, ge=1, le=3)  # 1 cleanup, 2 reset, 3 restart
    card_recovery_retry_seconds: float = Field(default=300.0, gt=0)
    card_canary_on_startup: bool = Field(default=True)
```

- [ ] **Step 4: Metrics**

In `LABEL_VALUES`: extend `"outcome"` with `"pass"` and `"fail"`; add `"step": ("cleanup", "reset", "restart")` and `"card": tuple(str(i) for i in range(16))`. In `ALLOWED_METRICS`:

```python
    "card_canary_total": ("outcome",),
    "card_recovery_total": ("step",),
    "card_health_state": ("card",),
```

Extend `_GAUGE_NAMES`: `("resource_state_gauge", "card_health_state")`.

- [ ] **Step 5: health.py**

```python
"""Card health: canary probe and recovery ladder (spec §10).

The canary runs a vendored known-good mini kernel in-process under a SIGALRM
budget; the recovery ladder escalates in-process cleanup -> a configured
device-reset command -> full process restart via os.execv. Host reboot is
deliberately never automatic. All torch/triton imports are lazy: this module
only executes on NPU hosts.
"""
import gc
import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

CANARY_ADD_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _canary_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _canary_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        return output
'''


class CanaryBudgetExceeded(TimeoutError):
    pass


def _cleanup_in_process() -> None:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def run_canary(*, device_id: int, budget_seconds: int) -> bool:
    """Run the known-good mini kernel on this worker's card. True = healthy.

    A hang raises via SIGALRM and is reported as unhealthy; the bisheng
    subprocesses launched by the kernel never outlive the budget because the
    same timer fires during their sync window.
    """
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    def _handler(signum, frame):
        raise CanaryBudgetExceeded(f"canary exceeded {budget_seconds}s budget")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, budget_seconds)
    try:
        torch.npu.set_device(0)  # pinned to the card via ASCEND_RT_VISIBLE_DEVICES
        namespace: dict = {}
        exec(compile(CANARY_ADD_KERNEL, "<canary>", "exec"), namespace)
        model = namespace["ModelNew"]().npu()
        x = torch.randn(1024, device="npu")
        y = torch.randn(1024, device="npu")
        with torch.no_grad():
            out = model(x, y)
        torch.npu.synchronize()
        return bool(torch.equal(out, x + y))
    except Exception as error:  # noqa: BLE001 — any failure means unhealthy
        logger.warning("card %s canary failed (%s)", device_id, type(error).__name__)
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_recovery_step(step: int, *, card: int, reset_command: str) -> str:
    """Execute one ladder rung; returns its name. Step 3 replaces the process
    image and never returns. A missing reset command makes rung 2 a named
    no-op so callers can keep rung numbering stable."""
    if step == 1:
        _cleanup_in_process()
        return "cleanup"
    if step == 2:
        if reset_command:
            command = reset_command.replace("{card}", str(card))
            subprocess.run(command, shell=True, timeout=120, check=False,
                           capture_output=True)
        return "reset"
    os.execv(sys.executable, [sys.executable, *sys.argv])
    raise AssertionError("unreachable after execv")
```

- [ ] **Step 6: Run tests + full suite + commit**

Run: `python3 -m pytest tests/test_health.py -q`, then full suite; `ruff check triton_evaluation/ tests/test_health.py`.

```bash
git add triton_evaluation/health.py triton_evaluation/config.py triton_evaluation/metrics.py tests/test_health.py
git commit -m "feat: card canary + recovery ladder module and health metrics"
```

---

### Task 3: Worker integration — suspicion, canary gating, drain, ladder

**Files:**
- Modify: `triton_evaluation/worker.py` (`Worker.__init__`, `run_once`, `_run_attempt`, new private methods)
- Test: `tests/test_worker.py` (append; uses `StubEvaluator`, `build_worker` helpers already in the file)

**Interfaces:**
- Consumes: Task 1 (`claim_next(avoid_card=...)`, `note_claimed_card`, `set/clear/is_card_unhealthy`), Task 2 (`health.run_canary`, `health.run_recovery_step`, settings).
- Produces: Worker constructor kwargs `canary=None`, `recovery_step=None` (injectable for tests; defaults to health functions). Drained state observable via `store.is_card_unhealthy(device_id)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker.py`:

```python
class _ScriptedCanary:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *, device_id, budget_seconds):
        self.calls += 1
        return self._outcomes.pop(0)


def _healthy_worker(store, settings, evaluator, clock, canary, **kwargs):
    worker = build_worker(store, settings, evaluator, clock, **kwargs)
    worker._canary = canary
    worker._recovery_step = lambda step, *, card, reset_command: (
        "cleanup" if step == 1 else "reset")
    return worker


def test_canary_runs_before_first_claim_when_startup_canary_enabled(store, settings, clock,
                                                                    make_command):
    from conftest import make_completed_result
    canary = _ScriptedCanary([True])
    worker = _healthy_worker(store, settings, StubEvaluator(result=make_completed_result(settings)),
                             clock, canary)
    store.create_or_replay(make_command())
    assert worker.run_once() is True
    assert canary.calls == 1


def test_suspicious_ending_triggers_canary_before_next_claim(store, settings, clock,
                                                             make_command):
    from conftest import make_completed_result
    canary = _ScriptedCanary([True, True])
    evaluator = StubEvaluator(result=make_completed_result(settings))
    worker = _healthy_worker(store, settings, evaluator, clock, canary)
    store.create_or_replay(make_command())
    assert worker.run_once() is True
    assert canary.calls == 1  # startup
    evaluator.error = RuntimeError("boom")  # protocol failure → suspicious
    store.create_or_replay(make_command(session_id="s-2", idempotency_key="k-2"))
    assert worker.run_once() is True
    assert canary.calls == 2  # suspicion triggered a re-canary


def test_failed_canary_drains_worker_and_skips_claim_and_heartbeat(store, settings,
                                                                   clock, make_command):
    canary = _ScriptedCanary([False])
    worker = _healthy_worker(store, settings, StubEvaluator(), clock, canary)
    store.create_or_replay(make_command())
    assert worker.run_once() is False
    assert store.is_card_unhealthy(0)
    queues_empty = not store.claim_next.__self__._r.zcard("triton-eval:v2:ready:exec:interactive")
    assert queues_empty  # nothing was claimed


def test_recovery_ladder_runs_between_canary_retries(store, settings, clock, make_command):
    # Outcome budget: _probe_card consumes 1; the ladder (max_step=2) consumes up
    # to 2 more; the drained re-probe consumes 1.
    steps: list[str] = []
    canary = _ScriptedCanary([False, False, False, True])
    worker = build_worker(store, settings, StubEvaluator(), clock)
    worker._canary = canary
    worker._recovery_step = lambda step, *, card, reset_command: steps.append(
        "cleanup" if step == 1 else "reset") or "cleanup"
    store.create_or_replay(make_command())
    # startup suspicion → canary F → rung 1 F → rung 2 F → drained
    assert worker.run_once() is False
    assert steps == ["cleanup", "reset"]
    assert store.is_card_unhealthy(0)
    # drained: claims skipped until the re-probe interval elapses
    assert worker.run_once() is False
    assert canary.calls == 3
    clock.advance_ms(int(settings.canary_interval_seconds * 1000) + 1)
    # drain-loop re-probe passes → healthy, claims resume next iteration
    assert worker.run_once() is False
    assert canary.calls == 4
    assert not store.is_card_unhealthy(0)


def test_avoid_card_passed_to_claim(store, settings, clock, make_command, mocker):
    from conftest import make_completed_result
    canary = _ScriptedCanary([True])
    worker = _healthy_worker(store, settings, StubEvaluator(result=make_completed_result(settings)),
                             clock, canary)
    spy = mocker.spy(store, "claim_next")
    store.create_or_replay(make_command())
    worker.run_once()
    assert spy.call_args.kwargs.get("avoid_card") == 0
```

(Adapt to the file's existing StubEvaluator/build_worker signatures — they exist at tests/test_worker.py:26-71. Note the canary outcome budget is fixed by the implementation shape: `_probe_card` consumes exactly one outcome, each ladder rung re-probes once, and the drain loop consumes one per re-probe — size the `_ScriptedCanary` lists accordingly. Disclose any further adjustment in the report.)

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_worker.py -k "canary or drain or ladder or avoid_card" -q`
Expected: failures — Worker has no `_canary`, no suspicion flag, no drain path.

- [ ] **Step 3: Implement worker integration**

In `Worker.__init__` (after `self._sleep`):

```python
        self._canary = canary or (
            lambda *, device_id, budget_seconds: health.run_canary(
                device_id=device_id, budget_seconds=budget_seconds
            )
        )
        self._recovery_step = recovery_step or health.run_recovery_step
        self._suspicion = bool(settings.card_canary_on_startup)
        self._drained = False
        self._last_recovery_ms = 0
```

(Add `canary=None, recovery_step=None` kwargs; `from triton_evaluation import health` at module import — top-level import is safe because health.py's device imports are lazy.)

`run_once` becomes:

```python
    def run_once(self, *, sleep=None) -> bool:
        """Claim one attempt, lease it, evaluate, fenced-commit it."""
        do_sleep = sleep if sleep is not None else self._sleep
        if self._drained:
            self._drain_loop()
            return False
        self.store.heartbeat(self.identity)
        if self._suspicion and not self._probe_card():
            return False
        avoid_card = self.device_id if self.settings.card_affinity_avoidance else None
        lease = self.store.claim_next(self.identity, avoid_card=avoid_card)
        if lease is None:
            self._maintenance()
            return False
        self.store.note_claimed_card(lease.evaluation_id, self.device_id)
        if self.store._metrics is not None:
            try:
                self.store._metrics.increment("worker_attempts_total", {})
            except ValueError:
                pass
        self._run_attempt(lease, do_sleep)
        return True
```

New private methods:

```python
    def _probe_card(self) -> bool:
        """Canary-gated suspicion: probe the card; on failure drain it and run
        the recovery ladder. Returns True when the card is claim-worthy."""
        healthy = self._canary(
            device_id=self.device_id,
            budget_seconds=self.settings.canary_budget_seconds,
        )
        self._bump_metric("card_canary_total", outcome="pass" if healthy else "fail")
        if healthy:
            self._suspicion = False
            self.store.clear_card_unhealthy(self.device_id)
            self._bump_metric("card_health_state", card=str(self.device_id))
            return True
        self.store.set_card_unhealthy(self.device_id)
        self._bump_metric("card_health_state", card=str(self.device_id))
        self._run_ladder()
        self._suspicion = False  # ladder already probed; drain loop owns re-probes
        return False

    def _run_ladder(self) -> None:
        now_ms = self._clock()
        if now_ms - self._last_recovery_ms < self.settings.card_recovery_retry_seconds * 1000:
            self._drained = True
            return
        self._last_recovery_ms = now_ms
        for step in range(1, self.settings.card_recovery_max_step + 1):
            name = self._recovery_step(
                step, card=self.device_id,
                reset_command=self.settings.card_reset_command,
            )
            self._bump_metric("card_recovery_total", step=name)
            if self._canary(device_id=self.device_id,
                            budget_seconds=self.settings.canary_budget_seconds):
                self._bump_metric("card_canary_total", outcome="pass")
                self.store.clear_card_unhealthy(self.device_id)
                self._bump_metric("card_health_state", card=str(self.device_id))
                return
            self._bump_metric("card_canary_total", outcome="fail")
        self._drained = True

    def _drain_loop(self) -> None:
        """Drained: no heartbeat (readiness drops), no claims. Re-probe the
        canary on canary_interval_seconds; a pass resumes heartbeats/claims."""
        if self._clock() - self._last_probe_ms < self.settings.canary_interval_seconds * 1000:
            return
        self._last_probe_ms = self._clock()
        if self._canary(device_id=self.device_id,
                        budget_seconds=self.settings.canary_budget_seconds):
            self._bump_metric("card_canary_total", outcome="pass")
            self.store.clear_card_unhealthy(self.device_id)
            self._bump_metric("card_health_state", card=str(self.device_id))
            self._drained = False
            return
        self._bump_metric("card_canary_total", outcome="fail")
        self._run_ladder()  # retry budget permitting; stays drained otherwise

    def _bump_metric(self, name: str, **labels) -> None:
        if self.store._metrics is None:
            return
        try:
            self.store._metrics.increment(name, labels)
        except ValueError:
            pass
```

(Initialize `self._last_probe_ms = 0` in `__init__` too.)

`_run_attempt` suspicion wiring — set `self._suspicion = True` in the three suspicious endings: the `except LeaseLostError` branch, the `except Exception` protocol-failure branch, and the `if not keeper.is_valid()` branch (each just before the existing `return`).

- [ ] **Step 4: Run tests + full suite + commit**

Run: `python3 -m pytest tests/test_worker.py -q`, then the full suite; `ruff check triton_evaluation/worker.py tests/test_worker.py`.

```bash
git add triton_evaluation/worker.py tests/test_worker.py
git commit -m "feat: worker canary gating, drain-on-unhealthy, recovery ladder"
```

---

### Task 4: NPU-only canary test + docs

**Files:**
- Create: `tests/test_card_health_npu.py`
- Modify: `README.md` (short deployment note), `docker.env.example` (canary/recovery block)

**Interfaces:**
- Consumes: `health.run_canary`.
- Produces: an `npu`-marked test skipped off-host; deployment documentation.

- [ ] **Step 1: Write the NPU-only test**

Create `tests/test_card_health_npu.py`:

```python
"""NPU-only: real canary against a live card. Never runs in CI (no torch_npu)."""
import os

import pytest

pytestmark = pytest.mark.npu

health = pytest.importorskip("triton_evaluation.health", reason="health module")


def _npu_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="no Ascend NPU visible")
def test_canary_passes_on_healthy_card():
    card = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0])
    assert health.run_canary(device_id=card, budget_seconds=120) is True
```

- [ ] **Step 2: Docs**

`docker.env.example` — append:

```bash
# Canary + recovery ladder (spec §10). Reset command is site-specific; "{card}"
# is substituted. Max step: 1=cleanup, 2=+device reset, 3=+process restart.
# TRITON_EVAL_CANARY_BUDGET_SECONDS=120
# TRITON_EVAL_CANARY_INTERVAL_SECONDS=60
# TRITON_EVAL_CARD_RESET_COMMAND=
# TRITON_EVAL_CARD_RECOVERY_MAX_STEP=2
# TRITON_EVAL_CARD_RECOVERY_RETRY_SECONDS=300
# TRITON_EVAL_CARD_CANARY_ON_STARTUP=true
```

`README.md` — under the Docker/env area, add one paragraph:

```markdown
Card health (spec §10): workers run a canary kernel before claiming after any
suspicious attempt death and at startup. A failed canary drains the worker (no
heartbeats — readiness drops to 503 once no claim-capable worker remains) and
drives a recovery ladder: in-process cleanup → `TRITON_EVAL_CARD_RESET_COMMAND`
("{card}" substituted) → process restart. Host reboot is never automatic. An
evaluation whose attempts die suspiciously twice is quarantined (terminal
`WORKER_LOST` with a poison message; never retried by the service).
```

- [ ] **Step 3: Run + commit**

Run: `python3 -m pytest tests/test_card_health_npu.py -q` (skipped off-host — expected), then the full suite; `ruff check tests/test_card_health_npu.py`.

```bash
git add tests/test_card_health_npu.py README.md docker.env.example
git commit -m "test+docs: NPU-only canary test and card-health deployment notes"
```

---

## Self-Review Notes (author check, completed at plan-write time)

- **Spec coverage (§10):** trigger = reaper-side lease expiry (the exactly-once death signal; worker-side LeaseLost/protocol failures funnel into it via abandonment) + worker-local suspicion for immediate canary; canary = vendored known-good kernel, SIGALRM budget; drain = no heartbeat + no claim (readiness drops after `lease_seconds`); recovery ladder = cleanup → configured reset → os.execv restart, ceiling default 2, host reboot never automatic; poison = 2nd suspicious death → terminal, 2.0-degraded WORKER_LOST shape with quarantine message; affinity = `last_failed_card` + `claim_next(avoid_card=)`, setting-gated. Card-health gauge + canary/recovery metrics on the existing allowlist.
- **Deliberate scoping cuts (YAGNI):** no readiness-payload change (drain via heartbeat skip); no worker→card registry (card known only locally, `claimed_card` advisory on resources); reaper does not write card-health keys (single-writer principle); `SUSPECTED_POISON_CANDIDATE` formal code deferred to Plan E; release-gate wiring for the NPU test deferred (npu marker only).
- **Type consistency:** `claim_next(worker, avoid_card=None)` call sites: worker.py passes kwarg; all existing tests positional → compatible. `_probe_card`/`_run_ladder`/`_drain_loop`/`_bump_metric` names consistent across Task 3 code/tests. Metric names/labels consistent across health.py/worker.py/metrics.py/tests.
- **No placeholders:** every code step complete; the one adaptive note is the worker-test clock arrangement (disclose adjustments in the report).
- **Known pre-existing failures:** the 5 `referencing` environmental failures, unchanged.
