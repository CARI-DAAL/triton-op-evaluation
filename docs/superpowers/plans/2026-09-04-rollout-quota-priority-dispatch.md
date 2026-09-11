# Rollout Traffic Class + Strict-Priority Dispatch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let RL rollout traffic submit a whole step durably (new "rollout" quota bucket: 600/min, in-flight 256) while strict-priority dual dispatch queues keep interactive traffic (trial/grade) from being head-of-line blocked by bulk floods — with zero API-contract changes.

**Architecture:** The quota class attaches server-side (`Settings.rollout_caller_ids`, resolved in `_build_command`); the broker gains a rollout branch in `_quota_limits`, persists `traffic_class` on the resource hash (terminal inflight release + metrics read it), and the single `ready` zset splits into `ready:exec:interactive` + `ready:exec:bulk` scanned strictly in that order, with the legacy `ready` zset draining last for durable-volume compatibility.

**Tech Stack:** Python 3.11, pydantic-settings, redis (fakeredis in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§7 scheduling & quotas; §6 Release-1 contract-invisibility requirement)

## Global Constraints

- **Contract-invisible**: no changes under `contracts/`, no changes to `triton_evaluation/api/contract.py`, no new endpoints, no new ErrorEnvelope codes (reuse `rate_limited` 429 / `limit_exceeded` 413).
- **State machine and fencing semantics unchanged**; dual queues alter only ready-key routing and claim scan order.
- **Durable-volume compatibility**: resources created before this deploy lack a `traffic_class` hash field — derive from purpose (`trial`→trial, else trusted) as fallback; the old `ready` zset must still drain (claim scans it last).
- **Quota class attach is server-side only**: `Settings.rollout_caller_ids` — no capability-secret file schema change.
- Grade purpose never maps to rollout; legacy routes keep hardcoding `traffic_class="trial"`.
- Tests use fakeredis + the injected `FakeClock` only; the full suite must stay green each task (5 pre-existing `test_portable_schema_probe.py` failures from the missing `referencing` dep on this box are environmental and not regressions).
- ruff line-length 100, target py310.

---

### Task 1: Rollout quota bucket

**Files:**
- Modify: `triton_evaluation/config.py:75-80` (quota settings block)
- Modify: `triton_evaluation/models.py:17-19` (TrafficClass literal)
- Modify: `triton_evaluation/broker.py:150-153` (`_quota_limits`)
- Modify: `triton_evaluation/metrics.py:29` (caller_class label values)
- Modify: `tests/conftest.py:115-131` (`make_command` gains `traffic_class` override)
- Modify: `tests/test_config.py` (settings tests)
- Modify: `tests/test_idempotency.py` (rollout quota tests)
- Modify: `docker.env.example` (commented rollout block)

**Interfaces:**
- Consumes: existing `Settings` quota fields; `_store_with` helper and quota-test patterns in `tests/test_idempotency.py`.
- Produces:
  - `Settings.rollout_rate_limit_per_minute: int` (default 600), `Settings.rollout_max_inflight: int` (default 256), `Settings.rollout_caller_ids: list[str]` (default `[]`). Env: `TRITON_EVAL_ROLLOUT_RATE_LIMIT_PER_MINUTE` / `..._MAX_INFLIGHT` / `..._CALLER_IDS` (JSON list).
  - `TrafficClass` extended with `"rollout"` (used by Task 2-4).
  - `make_command(..., traffic_class: str | None = None)` override (used by Task 2/3 tests).

- [ ] **Step 1: Write the failing settings tests**

Append to `tests/test_config.py` (matching the file's existing style):

```python
def test_rollout_quota_defaults():
    s = Settings()
    assert s.rollout_rate_limit_per_minute == 600
    assert s.rollout_max_inflight == 256
    assert s.rollout_caller_ids == []


def test_rollout_settings_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_ROLLOUT_CALLER_IDS", '["rl-1", "rl-2"]')
    monkeypatch.setenv("TRITON_EVAL_ROLLOUT_MAX_INFLIGHT", "512")
    s = Settings.from_env()
    assert s.rollout_caller_ids == ["rl-1", "rl-2"]
    assert s.rollout_max_inflight == 512
```

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError`/`AssertionError` on the new fields (fields don't exist yet).

- [ ] **Step 3: Add the settings fields**

In `triton_evaluation/config.py`, after the `session_max_inflight` line (the block reads `# --- admission quotas; trial and trusted buckets stay independent ---`):

```python
    # Rollout bucket: server-declared RL traffic (routes._build_command maps
    # declared callers here). Contract-invisible; grade never lands here.
    rollout_rate_limit_per_minute: int = Field(default=600, gt=0)
    rollout_max_inflight: int = Field(default=256, gt=0)
    rollout_caller_ids: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Extend TrafficClass and _quota_limits**

`triton_evaluation/models.py:17-19` becomes:

```python
# Quota buckets: trial and trusted derive from purpose; rollout is a
# server-declared class for RL traffic (settings.rollout_caller_ids). An
# observer granted grade:submit consumes trusted capacity, never an observer
# bucket.
TrafficClass = Literal["trial", "trusted", "rollout"]
```

`triton_evaluation/broker.py` `_quota_limits` (lines 150-153) becomes:

```python
    def _quota_limits(self, traffic_class: str) -> tuple[int, int]:
        if traffic_class == "trial":
            return self._s.trial_rate_limit_per_minute, self._s.trial_max_inflight
        if traffic_class == "rollout":
            return self._s.rollout_rate_limit_per_minute, self._s.rollout_max_inflight
        return self._s.trusted_rate_limit_per_minute, self._s.trusted_max_inflight
```

- [ ] **Step 5: Extend the metrics label allowlist**

`triton_evaluation/metrics.py:29`:

```python
    "caller_class": ("trial", "trusted", "observer", "rollout"),
```

(The label stays a superset: auth-class bumps still validate; only the terminal-transitions metric passes `"rollout"` after Task 2 persists the class.)

- [ ] **Step 6: conftest make_command override**

In `tests/conftest.py` `_make` (lines 115-131): add the parameter and replace the traffic_class expression:

```python
    def _make(
        *,
        caller=None,
        request=None,
        session_id="session-1",
        idempotency_key="key-1",
        evaluation_id=None,
        traffic_class=None,
    ):
        ...
        return CreateResourceCommand(
            evaluation_id=evaluation_id or new_evaluation_id(),
            caller_id=caller.caller_id,
            caller_class=caller.caller_class,
            traffic_class=traffic_class
            or ("trial" if request.purpose == "trial" else "trusted"),
            ...
        )
```

- [ ] **Step 7: Write the rollout quota tests**

Append to `tests/test_idempotency.py` (same file that holds `test_trial_and_trusted_quotas_are_independent`; reuse its `_store_with` helper and `make_caller` import; build raw keys as that file already does — if it imports a key helper use that, else `from triton_evaluation.broker import _digest_text`):

```python
def test_rollout_inflight_limit_is_independent(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, rollout_max_inflight=1)
    rollout_caller = make_caller(caller_id="rl-runner-1")

    first = make_command(caller=rollout_caller, session_id="s-rl-1",
                         idempotency_key="k-rl-1", traffic_class="rollout")
    assert store.create_or_replay(first).decision == "created"

    blocked = make_command(caller=rollout_caller, session_id="s-rl-2",
                           idempotency_key="k-rl-2", traffic_class="rollout")
    assert store.create_or_replay(blocked).decision == "limit_exceeded"

    trial = make_command()  # separate bucket, unaffected
    assert store.create_or_replay(trial).decision == "created"

    rollout_key = "triton-eval:v2:inflight:caller:rollout:" + _digest_text("rl-runner-1")
    assert int(redis_client.get(rollout_key)) == 1
    trial_key = "triton-eval:v2:inflight:caller:trial:" + _digest_text("caller-1")
    assert int(redis_client.get(trial_key)) == 1


def test_rollout_rate_limit_uses_rollout_bucket(redis_client, clock, make_command):
    store = _store_with(redis_client, clock, rollout_rate_limit_per_minute=1)
    caller = make_caller(caller_id="rl-runner-1")
    first = make_command(caller=caller, session_id="s1", idempotency_key="k1",
                         traffic_class="rollout")
    assert store.create_or_replay(first).decision == "created"
    second = make_command(caller=caller, session_id="s2", idempotency_key="k2",
                          traffic_class="rollout")
    assert store.create_or_replay(second).decision == "rate_limited"
```

- [ ] **Step 8: Run the tests — verify they pass**

Run: `python3 -m pytest tests/test_config.py tests/test_idempotency.py -q`
Expected: all pass.

- [ ] **Step 9: env example + full suite + commit**

Append to `docker.env.example`:

```bash
# Rollout traffic class: server-declared RL callers (JSON list of caller_id).
# TRITON_EVAL_ROLLOUT_CALLER_IDS='["rl-runner-1"]'
# TRITON_EVAL_ROLLOUT_RATE_LIMIT_PER_MINUTE=600
# TRITON_EVAL_ROLLOUT_MAX_INFLIGHT=256
```

Run: `python3 -m pytest -q` (full suite; the 5 environmental `referencing` failures excepted) and `ruff check triton_evaluation/ tests/test_config.py tests/test_idempotency.py tests/conftest.py`.

```bash
git add triton_evaluation/config.py triton_evaluation/models.py triton_evaluation/broker.py \
        triton_evaluation/metrics.py tests/conftest.py tests/test_config.py \
        tests/test_idempotency.py docker.env.example
git commit -m "feat: rollout quota bucket (settings, TrafficClass, _quota_limits)"
```

---

### Task 2: Persist traffic_class; terminal release via stored class

**Files:**
- Modify: `triton_evaluation/broker.py:279-308` (`_new_queued_resource` mapping)
- Modify: `triton_evaluation/broker.py:695-701` (`_inflight_keys`)
- Modify: `triton_evaluation/broker.py:615-621` (terminal-transitions metric)
- Test: `tests/test_broker.py` (append)

**Interfaces:**
- Consumes: Task 1's `TrafficClass` extension and metrics label superset.
- Produces: `EvaluationStore._stored_traffic_class(stored: dict) -> str` (static helper) — Task 3's recovery requeue and any future reader use it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_broker.py` (uses the `store`, `redis_client`, `make_command`, `make_worker`, `make_completed_result`, `settings` fixtures; import `_digest_text` the way this file already builds raw keys, or add `from triton_evaluation.broker import _digest_text`):

```python
def test_terminal_commit_releases_rollout_inflight(
    store, redis_client, make_command, make_worker, make_completed_result, settings,
):
    cmd = make_command(caller=make_caller(caller_id="rl-1"), session_id="s1",
                       idempotency_key="k1", traffic_class="rollout")
    store.create_or_replay(cmd)
    rollout_key = f"triton-eval:v2:inflight:caller:rollout:{_digest_text('rl-1')}"
    assert int(redis_client.get(rollout_key)) == 1

    lease = store.claim_next(make_worker())
    assert lease is not None
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    assert int(redis_client.get(rollout_key)) == 0


def test_terminal_release_derives_class_when_field_missing(
    store, redis_client, make_command, make_worker, make_completed_result, settings,
):
    """Pre-deploy resources lack traffic_class; purpose-derived fallback releases
    the right bucket (durable Redis volume outlives the deploy)."""
    cmd = make_command()  # trial caller-1
    store.create_or_replay(cmd)
    redis_client.hdel(f"triton-eval:v2:resource:{cmd.evaluation_id}", "traffic_class")

    lease = store.claim_next(make_worker())
    assert store.commit_terminal(lease, "completed", make_completed_result(settings))
    trial_key = f"triton-eval:v2:inflight:caller:trial:{_digest_text('caller-1')}"
    assert int(redis_client.get(trial_key)) == 0
```

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_broker.py -k "rollout_inflight or derives_class" -q`
Expected: FAIL — the rollout counter is written under `inflight:caller:trial:...` (fallback derivation maps non-trial-purpose? no: purpose is trial → trial bucket) so the first test's rollout-key assert fails; also `traffic_class` not persisted.

- [ ] **Step 3: Persist traffic_class at creation**

In `_new_queued_resource` mapping (`triton_evaluation/broker.py:279-308`), add after `"purpose"`:

```python
            "traffic_class": command.traffic_class,
```

- [ ] **Step 4: Add the stored-class helper and use it**

In `EvaluationStore`, next to `_inflight_keys`:

```python
    @staticmethod
    def _stored_traffic_class(stored: dict) -> str:
        """Class persisted since the rollout-class deploy; older resources on the
        durable volume derive it from purpose."""
        raw = stored.get(b"traffic_class")
        if raw is not None:
            return raw.decode("utf-8")
        purpose = stored[b"purpose"].decode("utf-8")
        return "trial" if purpose == "trial" else "trusted"
```

`_inflight_keys` (lines 695-701) becomes:

```python
    def _inflight_keys(self, stored: dict) -> tuple[str, str]:
        owner = stored[b"owner"].decode("utf-8")
        traffic_class = self._stored_traffic_class(stored)
        caller_key = self._key(f"inflight:caller:{traffic_class}:{_digest_text(owner)}")
        session_key = self._key(
            f"inflight:session:{stored[b'session_digest'].decode('utf-8')}"
        )
        return caller_key, session_key
```

(Keep the existing exact key construction for the session key — match what's there today.)

The metric bump in `commit_terminal` (lines 615-621) becomes:

```python
                self._bump(
                    "resource_terminal_transitions_total",
                    state=state,
                    caller_class=self._stored_traffic_class(stored),
                )
```

- [ ] **Step 5: Run tests + full suite + commit**

Run: `python3 -m pytest tests/test_broker.py -q` then `python3 -m pytest -q`; `ruff check triton_evaluation/broker.py tests/test_broker.py`.

```bash
git add triton_evaluation/broker.py tests/test_broker.py
git commit -m "feat: persist traffic_class; terminal inflight release reads stored class"
```

---

### Task 3: Strict-priority dual dispatch queues

**Files:**
- Modify: `triton_evaluation/broker.py:11-37` (keyspace docstring)
- Modify: `triton_evaluation/broker.py:218` (create zadd)
- Modify: `triton_evaluation/broker.py:368-436` (`claim_next` refactor)
- Modify: `triton_evaluation/broker.py:685` (terminal zrem)
- Modify: `triton_evaluation/broker.py:724-770` (`_recover_one_lease` requeue)
- Test: `tests/test_broker.py` (append); update any existing raw-`ready`-key assertions found by the Step 1 grep

**Interfaces:**
- Consumes: Task 2's `_stored_traffic_class`.
- Produces: `EvaluationStore._ready_key(traffic_class: str) -> str`; keys `ready:exec:interactive` / `ready:exec:bulk` (legacy `ready` drains last). External behavior: `claim_next` returns interactive-class leases before bulk-class ones regardless of enqueue order.

- [ ] **Step 1: Find existing raw-`ready`-key assertions**

Run: `grep -rn '"ready"' tests/ triton_evaluation/ | grep -v 'ready:exec' | grep -v running-leases`
For each hit that zadd/zrange/zrem-scores the old `ready` key in tests, update it to `ready:exec:interactive` (all existing tests create trial/trusted traffic). List what you changed in the report.

- [ ] **Step 2: Write the failing dispatch tests**

Append to `tests/test_broker.py`:

```python
def _queue_members(redis_client):
    return {
        "interactive": set(
            redis_client.zrange("triton-eval:v2:ready:exec:interactive", 0, -1)
        ),
        "bulk": set(redis_client.zrange("triton-eval:v2:ready:exec:bulk", 0, -1)),
        "legacy": set(redis_client.zrange("triton-eval:v2:ready", 0, -1)),
    }


def _rollout_command(make_command, n):
    return make_command(caller=make_caller(caller_id="rl-1"), session_id=f"s-{n}",
                        idempotency_key=f"k-{n}", traffic_class="rollout")


def test_rollout_routes_to_bulk_queue(store, redis_client, make_command):
    cmd = _rollout_command(make_command, 1)
    store.create_or_replay(cmd)
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["bulk"]
    assert not queues["interactive"] and not queues["legacy"]


def test_trial_routes_to_interactive_queue(store, redis_client, make_command):
    cmd = make_command()
    store.create_or_replay(cmd)
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["interactive"]
    assert not queues["bulk"]


def test_interactive_claim_precedes_older_bulk(store, redis_client, make_command,
                                               make_worker):
    older_bulk = _rollout_command(make_command, 1)
    store.create_or_replay(older_bulk)
    newer_interactive = make_command(session_id="s-2", idempotency_key="k-2")
    store.create_or_replay(newer_interactive)

    worker = make_worker()
    first = store.claim_next(worker)
    assert first.evaluation_id == newer_interactive.evaluation_id
    second = store.claim_next(worker)
    assert second.evaluation_id == older_bulk.evaluation_id


def test_bulk_drains_in_fifo_order_when_interactive_empty(store, make_command,
                                                          make_worker):
    cmds = [_rollout_command(make_command, n) for n in (1, 2)]
    for cmd in cmds:
        store.create_or_replay(cmd)
    worker = make_worker()
    assert store.claim_next(worker).evaluation_id == cmds[0].evaluation_id
    assert store.claim_next(worker).evaluation_id == cmds[1].evaluation_id
    assert store.claim_next(worker) is None


def test_recovery_requeues_rollout_to_bulk(store, redis_client, clock, settings,
                                           make_command, make_worker):
    cmd = _rollout_command(make_command, 1)
    store.create_or_replay(cmd)
    store.claim_next(make_worker())
    clock.advance_ms(settings.lease_seconds * 1000 + 1)
    counts = store.recover_expired_leases()
    assert counts.requeued == 1
    queues = _queue_members(redis_client)
    assert cmd.evaluation_id.encode() in queues["bulk"]
    assert not queues["interactive"]


def test_legacy_ready_zset_still_drains(store, redis_client, make_command,
                                        make_worker):
    """Pre-deploy members of the old ready zset remain claimable (durable volume)."""
    cmd = make_command()
    store.create_or_replay(cmd)
    member = cmd.evaluation_id.encode()
    redis_client.zrem("triton-eval:v2:ready:exec:interactive", member)
    redis_client.zadd("triton-eval:v2:ready", {member: 1})
    lease = store.claim_next(make_worker())
    assert lease is not None and lease.evaluation_id == cmd.evaluation_id
```

- [ ] **Step 3: Run them — verify they fail**

Run: `python3 -m pytest tests/test_broker.py -k "queue or bulk or interactive or legacy" -q`
Expected: FAIL — everything still lands in / scans the old `ready` zset.

- [ ] **Step 4: Keyspace docstring**

In the broker module docstring (`triton_evaluation/broker.py:16-17`), replace the `ready` entry:

```
    ready:exec:interactive                       zset, score = queue sequence
    ready:exec:bulk                              zset, score = queue sequence (rollout class)
    ready                                        legacy zset, drain-only (pre-dual-queue deploys)
```

- [ ] **Step 5: Routing helper + create + terminal cleanup + recovery**

Add next to `_quota_limits`:

```python
    def _ready_key(self, traffic_class: str) -> str:
        """Strict-priority dispatch: rollout drains from bulk, everything else
        from interactive; workers always scan interactive first."""
        suffix = "bulk" if traffic_class == "rollout" else "interactive"
        return self._key(f"ready:exec:{suffix}")
```

`create_or_replay` line 218: `pipeline.zadd(self._key("ready"), ...)` →
`pipeline.zadd(self._ready_key(command.traffic_class), {command.evaluation_id: successor})`.

`_commit_terminal_locked` (line 685): replace the single `pipeline.zrem(self._key("ready"), evaluation_id)` with:

```python
        pipeline.zrem(self._key("ready:exec:interactive"), evaluation_id)
        pipeline.zrem(self._key("ready:exec:bulk"), evaluation_id)
        pipeline.zrem(self._key("ready"), evaluation_id)  # legacy drain-only queue
```

`_recover_one_lease`: the local `ready_key = self._key("ready")` (line 728) is used for the requeue zadd at line 768. `stored` is loaded at line 734 before the `pipeline.watch(ready_key, sequence_key)` at line 753, so move the derivation down and class-route it:

```python
        # after `stored` is validated as a live running lease, before the watch:
        ready_key = self._ready_key(self._stored_traffic_class(stored))
```

(Delete the old top-of-function `ready_key` local; keep `leases_key`/`sequence_key` where they are.)

- [ ] **Step 6: claim_next refactor — extract per-queue scan**

Replace `claim_next` (`triton_evaluation/broker.py:368-436`) with a queue-ordered driver plus the existing body as `_claim_from_queue`. Preserve the existing docstring's fencing description and add the priority sentence:

```python
    def claim_next(self, worker: WorkerIdentity) -> AttemptLease | None:
        """Atomically claim the head of the dispatch queues for ``worker``.

        Strict priority: the interactive queue (trial/trusted) scans first, the
        bulk queue (rollout class) only when interactive is empty, and the legacy
        ``ready`` zset (pre-dual-queue deploys) drains last. Interactive callers
        are quota-capped, so priority cannot starve bulk traffic; bulk floods
        cannot head-of-line block interactive latency. A queued claim removes the
        ready member, flips the public resource to running with a bumped revision
        and the first ``started_at``, installs attempt/lease fields, and moves the
        state indexes. A replacement-pending running resource keeps its public
        state, revision, and started time and consumes the higher token the reaper
        reserved. Past-deadline members never start; duplicate members for claimed
        resources are removed.
        """
        leases_key = self._key("running-leases")
        for ready_key in (
            self._key("ready:exec:interactive"),
            self._key("ready:exec:bulk"),
            self._key("ready"),  # legacy, drain-only
        ):
            lease = self._claim_from_queue(worker, ready_key, leases_key)
            if lease is not None:
                return lease
        return None

    def _claim_from_queue(
        self, worker: WorkerIdentity, ready_key: str, leases_key: str
    ) -> AttemptLease | None:
        """Single-queue claim scan; fencing and replacement semantics unchanged."""
        while True:
            now_ms = self._clock()
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(ready_key)
                members = pipeline.zrange(ready_key, 0, _CLAIM_SCAN_BATCH - 1)
                if not members:
                    return None
                stale: list[bytes] = []
                claim = None
                for member in members:
                    evaluation_id = member.decode("utf-8")
                    resource_key = self._key(f"resource:{evaluation_id}")
                    pipeline.watch(resource_key)
                    stored = pipeline.hgetall(resource_key)
                    if not stored:
                        stale.append(member)
                        continue
                    resource = EvaluationResource.model_validate_json(stored[b"resource"])
                    if now_ms >= int(stored[b"deadline_ms"]):
                        continue  # the deadline watchdog owns this resource
                    if resource.state == "queued":
                        claim = (member, stored, resource, False)
                        break
                    if resource.state == "running" and (
                        stored.get(b"replacement_pending") == b"1"
                    ):
                        claim = (member, stored, resource, True)
                        break
                    stale.append(member)  # duplicate delivery, already claimed
                if claim is None:
                    if not stale:
                        return None
                    pipeline.multi()
                    for member in stale:
                        pipeline.zrem(ready_key, member)
                    pipeline.execute()
                    continue
                member, stored, resource, replacement = claim
                pipeline.multi()
                lease = self._install_claim(
                    pipeline,
                    member=member,
                    stored=stored,
                    resource=resource,
                    replacement=replacement,
                    now_ms=now_ms,
                    ready_key=ready_key,
                    leases_key=leases_key,
                )
                pipeline.execute()
                return lease
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()
```

- [ ] **Step 7: Run tests + full suite + commit**

Run: `python3 -m pytest tests/test_broker.py -q`, then the full suite; `ruff check triton_evaluation/broker.py tests/test_broker.py`.

```bash
git add triton_evaluation/broker.py tests/test_broker.py
git commit -m "feat: strict-priority dual dispatch queues (interactive before bulk)"
```

---

### Task 4: Admission wiring — declared rollout callers

**Files:**
- Modify: `triton_evaluation/api/routes.py:129-142` (`_build_command`)
- Test: `tests/test_rollout_class.py` (create)

**Interfaces:**
- Consumes: Tasks 1-3 (a rollout command lands in bulk, consumes the rollout bucket).
- Produces: nothing further; this closes the server-side loop from HTTP admission to queue routing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rollout_class.py`:

```python
"""Rollout traffic-class wiring: declared callers map to the rollout bucket."""
from types import SimpleNamespace

from triton_evaluation.admission import admit
from triton_evaluation.api.routes import _build_command
from triton_evaluation.catalog import ReleaseCatalog  # match conftest's import path

from .conftest import make_caller, make_grade_caller, make_grade_request, make_request


def test_build_command_maps_declared_rollout_callers(settings, release_catalog):
    rollout_settings = settings.model_copy(update={"rollout_caller_ids": ["rl-1"]})
    context = SimpleNamespace(settings=rollout_settings)

    rollout_caller = make_caller(caller_id="rl-1")
    admission = admit(make_request(), rollout_caller, ReleaseCatalog({}))
    cmd = _build_command(context, rollout_caller, "s1", "k1", admission, "trial")
    assert cmd.traffic_class == "rollout"

    plain_caller = make_caller(caller_id="someone-else")
    admission2 = admit(make_request(), plain_caller, ReleaseCatalog({}))
    cmd2 = _build_command(context, plain_caller, "s2", "k2", admission2, "trial")
    assert cmd2.traffic_class == "trial"


def test_build_command_never_maps_grade_to_rollout(settings, release_catalog):
    rollout_settings = settings.model_copy(update={"rollout_caller_ids": ["grader-1"]})
    context = SimpleNamespace(settings=rollout_settings)
    grade_caller = make_grade_caller()
    admission = admit(make_grade_request(), grade_caller, release_catalog)
    cmd = _build_command(context, grade_caller, "s1", "k1", admission, "grade")
    assert cmd.traffic_class == "trusted"
```

(If `make_caller` etc. are importable from `tests.conftest` as plain functions — they are module-level helpers — prefer importing from `conftest` without the relative import; match how other test files import them. `release_catalog` and `settings` are fixtures.)

- [ ] **Step 2: Run it — verify it fails**

Run: `python3 -m pytest tests/test_rollout_class.py -q`
Expected: FAIL — `traffic_class` is `"trial"` for the declared rollout caller.

- [ ] **Step 3: Wire _build_command**

`triton_evaluation/api/routes.py:129-142` becomes:

```python
def _build_command(context, caller, session_id, idempotency_key, admission, purpose):
    from ..models import new_evaluation_id

    is_rollout = (
        purpose == "trial" and caller.caller_id in context.settings.rollout_caller_ids
    )
    return CreateResourceCommand(
        evaluation_id=new_evaluation_id(),
        caller_id=caller.caller_id,
        caller_class=caller.caller_class,
        traffic_class=(
            "rollout"
            if is_rollout
            else ("trial" if purpose == "trial" else "trusted")
        ),
        session_id=session_id,
        idempotency_key=idempotency_key,
        admission=admission,
        evaluator_build=context.settings.evaluator_build,
        algorithms=_algorithms(context),
    )
```

- [ ] **Step 4: Run tests + full suite + lint + commit**

Run: `python3 -m pytest tests/test_rollout_class.py tests/test_api_v2.py -q`, then the full suite; `ruff check triton_evaluation/api/routes.py tests/test_rollout_class.py`.

```bash
git add triton_evaluation/api/routes.py tests/test_rollout_class.py
git commit -m "feat: declared rollout callers map to the rollout traffic class"
```

---

## Self-Review Notes (author check, completed at plan-write time)

- **Spec coverage:** §7 quota table (rollout 600/256/300s — the 300s default deadline is a caller-supplied request field today, not a server default, so nothing to implement), strict-priority queues with the no-starvation invariant, server-side class declaration, lane-by-quota-class routing. §6 Release-1 contract-invisibility holds: no contract/ files touched, no new error codes.
- **Durable-volume edge cases covered:** resources lacking `traffic_class` (fallback derivation, Task 2 test) and legacy `ready` members (drain-last scan + test) — both verified against the Redis AOF persistence story in the README.
- **Type consistency:** `_stored_traffic_class` / `_ready_key` names used identically across tasks; `make_command(traffic_class=...)` override consumed by Tasks 2-3 tests; metric label superset added before Task 2 first passes `"rollout"` to it.
- **No placeholders:** every code step is complete; the only adaptive notes are import-path guidance where the repo has an existing idiom (raw-key construction in test_idempotency.py, conftest import style in the new test file).
- **Known pre-existing failures:** 5 `test_portable_schema_probe.py` failures (missing `referencing` dep on the dev box) — baseline-identical, not regressions; verify by comparing failure count before/after each task.
