"""Durable idempotent resource admission store (design sections 3.2 and 6).

``EvaluationStore`` replaces the old list-based ``Broker``: resource creation
is one optimistic ``WATCH``/``MULTI``/``EXEC`` transaction that atomically
writes the idempotency record, the resource hash, the FIFO ``ready`` entry,
the deadline and state indexes, and the quota counters. Connections are
binary (``decode_responses=False``) so canonical request bytes are stored and
compared exactly, never decoded and re-encoded. No Lua scripts and no
in-memory fallback.

Fixed keyspace under ``settings.redis_prefix`` (default ``triton-eval:v2``)::

    resource:{evaluation_id}                     hash (canonical request, public
                                                 resource JSON, owner, quotas metadata)
    idempotency:{sha256(caller_id NUL key)}      hash: evaluation_id, request_digest
    ready:exec:interactive                       zset, score = queue sequence
    ready:exec:bulk                              zset, score = queue sequence (rollout class)
    ready                                        legacy zset, drain-only (pre-dual-queue deploys)
    queue-sequence                               integer
    deadlines                                    zset, score = deadline epoch ms
    running-leases                               zset, score = lease expiry epoch ms
    worker-heartbeats                            hash: worker ID -> compatibility JSON
    worker-heartbeat-expiry                      zset, score = heartbeat expiry ms
    metrics:fenced-write-rejected                integer
    state:{queued|running|completed|failed|timed_out}  zsets of evaluation IDs;
                                                 queued/running scored by deadline,
                                                 terminal by retention-expiry instant
    inflight:caller:{traffic_class}:{sha256(caller_id)}        integer
    inflight:session:{sha256(caller_id NUL session_id)}        integer
    rate:{traffic_class}:{sha256(caller_id)}:{minute}          integer, 120s expiry

Only the rate-window key expires; nonterminal records never carry a TTL, and
terminal resource/idempotency records share the retention TTL. Persisted
timestamps come from the store clock — Redis ``TIME`` in production, an
injected deterministic clock in tests. The state machine accepts only
``queued -> running``, ``queued -> failed|timed_out``, and
``running -> completed|failed|timed_out``; every attempt write is fenced by
running state, exact attempt ID, exact fencing token, expected public
revision, and ``now < deadline_at``.
"""
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import redis

from triton_evaluation.api.contract import (
    Check,
    CheckOutcome,
    EvaluationResource,
    EvaluationResult,
)
from triton_evaluation.api.contract_artifacts import canonical_json_bytes, sha256_hex
from triton_evaluation.config import Settings
from triton_evaluation.models import (
    AlgorithmVersions,
    AttemptLease,
    CreateResourceCommand,
    CreateResourceResult,
    RecoveryCounts,
    TerminalState,
    WorkerCompatibility,
    WorkerIdentity,
    new_attempt_id,
)
from triton_evaluation.security import Caller
from triton_evaluation.terminal_results import (
    persisted_check_outcomes,
    synthesize_orchestration_result,
)

_RATE_WINDOW_EXPIRY_SECONDS = 120
_MINUTE_MS = 60_000
# Retry-safe workload: a crashed attempt may be replaced twice before the
# reaper commits WORKER_LOST.
MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)
_CLAIM_SCAN_BATCH = 10
_TERMINAL_STATES = ("completed", "failed", "timed_out")


def _no_op_failpoint(evaluation_id: str) -> None:
    """Production failpoint: an immutable no-op.

    Tests inject callables (``broker._create_pre_exec = ...``) around the
    transaction boundaries; they can never be configured from requests or
    the environment.
    """


_create_pre_exec = _no_op_failpoint
_create_post_exec = _no_op_failpoint
_commit_pre_exec = _no_op_failpoint
_commit_post_exec = _no_op_failpoint


class StorageError(Exception):
    """Durable storage is internally inconsistent; never a caller-facing detail."""

    code = "INTERNAL"
    phase = "storage"
    retryable = False

    def __init__(self) -> None:
        super().__init__("unexpected internal error")


def _redis_time_ms(client: redis.Redis) -> int:
    """Production store clock: epoch milliseconds from Redis ``TIME``."""
    seconds, micros = client.time()
    return int(seconds) * 1000 + int(micros) // 1000


def _digest_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def _scoped_digest(*parts: str) -> str:
    return sha256_hex("\0".join(parts).encode("utf-8"))


class EvaluationStore:
    def __init__(
        self,
        client: redis.Redis,
        settings: Settings,
        clock: Callable[[], int] | None = None,
        metrics=None,
    ) -> None:
        if settings.retention_seconds < settings.max_recovery_age_seconds:
            raise ValueError("retention_seconds must cover max_recovery_age_seconds")
        self._r = client
        self._s = settings
        self._clock = clock or (lambda: _redis_time_ms(self._r))
        self._metrics = metrics

    def _bump(self, name: str, **labels) -> None:
        """Best-effort bounded counter; never changes a transition result."""
        if self._metrics is None:
            return
        try:
            self._metrics.increment(name, labels)
        except Exception as error:  # noqa: BLE001 — never change a transition
            logger.warning("metrics write failed (%s)", type(error).__name__)

    def _key(self, suffix: str) -> str:
        return f"{self._s.redis_prefix}:{suffix}"

    def _quota_limits(self, traffic_class: str) -> tuple[int, int]:
        if traffic_class == "trial":
            return self._s.trial_rate_limit_per_minute, self._s.trial_max_inflight
        if traffic_class == "rollout":
            return self._s.rollout_rate_limit_per_minute, self._s.rollout_max_inflight
        return self._s.trusted_rate_limit_per_minute, self._s.trusted_max_inflight

    def _ready_key(self, traffic_class: str) -> str:
        """Strict-priority dispatch: rollout drains from bulk, everything else
        from interactive; workers always scan interactive first."""
        suffix = "bulk" if traffic_class == "rollout" else "interactive"
        return self._key(f"ready:exec:{suffix}")

    def create_or_replay(self, command: CreateResourceCommand) -> CreateResourceResult:
        """Atomically write idempotency, resource, FIFO entry, deadline, and quotas.

        A retrying optimistic transaction: watch the idempotency, quota, and
        queue-sequence keys; replay or conflict short-circuit before any write;
        a new record stores the watched sequence's successor and all admission
        writes in one ``MULTI/EXEC``. Only ``redis.WatchError`` is retried.
        """
        caller_digest = _digest_text(command.caller_id)
        session_digest = _scoped_digest(command.caller_id, command.session_id)
        idempotency_key = self._key(
            f"idempotency:{_scoped_digest(command.caller_id, command.idempotency_key)}"
        )
        inflight_caller_key = self._key(
            f"inflight:caller:{command.traffic_class}:{caller_digest}"
        )
        inflight_session_key = self._key(f"inflight:session:{session_digest}")
        sequence_key = self._key("queue-sequence")
        rate_limit, inflight_limit = self._quota_limits(command.traffic_class)

        while True:
            now_ms = self._clock()
            rate_key = self._key(
                f"rate:{command.traffic_class}:{caller_digest}:{now_ms // _MINUTE_MS}"
            )
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(
                    idempotency_key,
                    rate_key,
                    inflight_caller_key,
                    inflight_session_key,
                    sequence_key,
                )
                record = pipeline.hgetall(idempotency_key)
                if record:
                    return self._replay(record, command, pipeline)

                if int(pipeline.get(rate_key) or 0) >= rate_limit:
                    return CreateResourceResult("rate_limited", None)
                if (
                    int(pipeline.get(inflight_caller_key) or 0) >= inflight_limit
                    or int(pipeline.get(inflight_session_key) or 0)
                    >= self._s.session_max_inflight
                ):
                    return CreateResourceResult("limit_exceeded", None)

                successor = int(pipeline.get(sequence_key) or 0) + 1
                resource, mapping, deadline_ms = self._new_queued_resource(
                    command, idempotency_key, session_digest, now_ms
                )
                pipeline.multi()
                pipeline.set(sequence_key, successor)
                pipeline.hset(
                    self._key(f"resource:{command.evaluation_id}"), mapping=mapping
                )
                pipeline.hset(
                    idempotency_key,
                    mapping={
                        "evaluation_id": command.evaluation_id,
                        "request_digest": command.admission.request_digest,
                    },
                )
                pipeline.zadd(
                    self._ready_key(command.traffic_class),
                    {command.evaluation_id: successor},
                )
                pipeline.zadd(self._key("deadlines"), {command.evaluation_id: deadline_ms})
                pipeline.zadd(
                    self._key("state:queued"), {command.evaluation_id: deadline_ms}
                )
                pipeline.incr(inflight_caller_key)
                pipeline.incr(inflight_session_key)
                pipeline.incr(rate_key)
                pipeline.expire(rate_key, _RATE_WINDOW_EXPIRY_SECONDS)
                _create_pre_exec(command.evaluation_id)
                pipeline.execute()
                _create_post_exec(command.evaluation_id)
                return CreateResourceResult("created", resource)
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()

    def _replay(self, record: dict, command: CreateResourceCommand, pipeline):
        """Compare a watched idempotency record and its resource, byte-exactly."""
        evaluation_id = record[b"evaluation_id"].decode("utf-8")
        resource_key = self._key(f"resource:{evaluation_id}")
        pipeline.watch(resource_key)
        stored = pipeline.hgetall(resource_key)
        if not stored:
            # A mapped-but-missing resource is a storage failure, never
            # permission to reuse the idempotency key.
            raise StorageError()
        if (
            record[b"request_digest"].decode("utf-8") != command.admission.request_digest
            or stored[b"canonical_request"] != command.admission.canonical_request
        ):
            return CreateResourceResult("conflict", None)
        resource = EvaluationResource.model_validate_json(stored[b"resource"])
        return CreateResourceResult("replayed", resource)

    def _new_queued_resource(
        self,
        command: CreateResourceCommand,
        idempotency_key: str,
        session_digest: str,
        accepted_ms: int,
    ) -> tuple[EvaluationResource, dict, int]:
        payload = json.loads(command.admission.canonical_request)
        deadline_seconds = int(payload["deadline_seconds"])
        deadline_ms = accepted_ms + deadline_seconds * 1000
        accepted_at = datetime.fromtimestamp(accepted_ms / 1000, tz=timezone.utc)
        resource = EvaluationResource(
            schema_version="2.0",
            evaluation_id=command.evaluation_id,
            revision=1,
            state="queued",
            created_at=accepted_at,
            updated_at=accepted_at,
            started_at=None,
            finished_at=None,
            deadline_at=accepted_at + timedelta(seconds=deadline_seconds),
            request_digest=command.admission.request_digest,
            reference_digest=command.admission.reference_digest,
            result=None,
        )
        mapping = {
            "owner": command.caller_id,
            "purpose": payload["purpose"],
            "traffic_class": command.traffic_class,
            "session_digest": session_digest,
            "idempotency_key": idempotency_key,
            "canonical_request": command.admission.canonical_request,
            "request_digest": command.admission.request_digest,
            "reference_digest": command.admission.reference_digest,
            "requested_checks": canonical_json_bytes(
                list(command.admission.requested_checks)
            ),
            "resource": canonical_json_bytes(
                resource.model_dump(mode="json", exclude_none=False)
            ),
            "evaluator_build": command.evaluator_build,
            "correctness_algorithm": command.algorithms.correctness,
            "decoy_algorithm": command.algorithms.decoy,
            "benchmark_algorithm": command.algorithms.benchmark,
            "profiler_algorithm": command.algorithms.profiler,
            "attempt_count": "0",
            "suspicious_deaths": "0",
            "attempt_id": "",
            "fencing_token": "0",
            "active_check": "",
            "check_outcomes": "{}",
            "lease_deadline_ms": "0",
            "replacement_pending": "0",
            "created_at_ms": str(accepted_ms),
            "updated_at_ms": str(accepted_ms),
            "deadline_ms": str(deadline_ms),
        }
        return resource, mapping, deadline_ms

    def get_visible(self, evaluation_id: str, caller: Caller) -> EvaluationResource | None:
        """Return the resource iff the caller may read it; hide everything else.

        Owners need the read scope matching the resource purpose
        (``trial:read``/``grade:read``); non-owners need ``observer:read``.
        Unknown and invisible records return the same ``None``. Sessions never
        authorize reads, and observer reads never change ownership.
        """
        stored = self._r.hgetall(self._key(f"resource:{evaluation_id}"))
        if not stored:
            return None
        owner = stored[b"owner"].decode("utf-8")
        purpose = stored[b"purpose"].decode("utf-8")
        if caller.caller_id == owner:
            required = "trial:read" if purpose == "trial" else "grade:read"
            if required not in caller.scopes:
                return None
        elif "observer:read" not in caller.scopes:
            return None
        return EvaluationResource.model_validate_json(stored[b"resource"])

    def lookup_idempotent(
        self, caller_id: str, idempotency_key: str
    ) -> EvaluationResource | None:
        """Return the resource an existing idempotency record maps to, if any.

        Read-only pre-check used by admission so an idempotent replay stays
        recoverable through transient unreadiness; readiness applies only to
        genuinely new admissions.
        """
        record_key = self._key(
            f"idempotency:{_scoped_digest(caller_id, idempotency_key)}"
        )
        record = self._r.hgetall(record_key)
        if not record:
            return None
        resource_key = self._key(f"resource:{record[b'evaluation_id'].decode('utf-8')}")
        stored = self._r.hget(resource_key, b"resource")
        if stored is None:
            # A mapped-but-missing resource is a storage failure, never a
            # silent pass-through to new admission.
            raise StorageError("idempotency record maps to a missing resource")
        return EvaluationResource.model_validate_json(stored)

    def probe_storage(self) -> bool:
        """One ephemeral write/delete proving the store accepts writes."""
        probe_key = self._key(f"readiness-probe:{uuid.uuid4().hex}")
        pipeline = self._r.pipeline()
        pipeline.set(probe_key, b"1", nx=True, px=5000)
        pipeline.delete(probe_key)
        results = pipeline.execute()
        return bool(results and results[0])

    # ------------------------------------------------------------------
    # Claims and leases (design section 6)
    # ------------------------------------------------------------------

    def claim_next(
        self, worker: WorkerIdentity, avoid_card: int | None = None
    ) -> AttemptLease | None:
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
        resources are removed. ``avoid_card`` skips members whose stored
        ``last_failed_card`` matches (affinity avoidance): the member stays in the
        queue for a worker on another card.
        """
        leases_key = self._key("running-leases")
        for ready_key in (
            self._key("ready:exec:interactive"),
            self._key("ready:exec:bulk"),
            self._key("ready"),  # legacy, drain-only
        ):
            lease = self._claim_from_queue(worker, ready_key, leases_key, avoid_card)
            if lease is not None:
                return lease
        return None

    def _claim_from_queue(
        self,
        worker: WorkerIdentity,
        ready_key: str,
        leases_key: str,
        avoid_card: int | None = None,
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
                    if (
                        avoid_card is not None
                        and stored.get(b"last_failed_card") == str(avoid_card).encode()
                    ):
                        continue  # affinity avoidance: another card may claim it
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

    def _install_claim(
        self,
        pipeline,
        *,
        member: bytes,
        stored: dict,
        resource: EvaluationResource,
        replacement: bool,
        now_ms: int,
        ready_key: str,
        leases_key: str,
    ) -> AttemptLease:
        evaluation_id = resource.evaluation_id
        resource_key = self._key(f"resource:{evaluation_id}")
        deadline_ms = int(stored[b"deadline_ms"])
        lease_expiry_ms = now_ms + self._s.lease_seconds * 1000
        attempt_id = new_attempt_id()
        attempt_count = int(stored[b"attempt_count"]) + 1
        if replacement:
            token = int(stored[b"fencing_token"])  # reserved by the reaper
            revision = resource.revision
        else:
            token = int(stored[b"fencing_token"]) + 1
            revision = resource.revision + 1
            started = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
            updated = EvaluationResource.model_validate(
                {
                    **resource.model_dump(exclude_none=False),
                    "revision": revision,
                    "state": "running",
                    "started_at": started,
                    "updated_at": started,
                }
            )
            pipeline.hset(
                resource_key,
                mapping={
                    "resource": canonical_json_bytes(
                        updated.model_dump(mode="json", exclude_none=False)
                    ),
                    "updated_at_ms": str(now_ms),
                },
            )
            pipeline.zrem(self._key("state:queued"), evaluation_id)
            pipeline.zadd(self._key("state:running"), {evaluation_id: deadline_ms})
        pipeline.hset(
            resource_key,
            mapping={
                "attempt_id": attempt_id,
                "fencing_token": str(token),
                "attempt_count": str(attempt_count),
                "lease_deadline_ms": str(lease_expiry_ms),
                "replacement_pending": "0",
                "active_check": "",
                "check_outcomes": "{}",
            },
        )
        pipeline.zrem(ready_key, member)
        pipeline.zadd(leases_key, {evaluation_id: lease_expiry_ms})
        return AttemptLease(
            evaluation_id=evaluation_id,
            attempt_id=attempt_id,
            fencing_token=token,
            expected_revision=revision,
            deadline_at_ms=deadline_ms,
            canonical_request=stored[b"canonical_request"],
            evaluator_build=stored[b"evaluator_build"].decode("utf-8"),
            algorithms=AlgorithmVersions(
                correctness=stored[b"correctness_algorithm"].decode("utf-8"),
                decoy=stored[b"decoy_algorithm"].decode("utf-8"),
                benchmark=stored[b"benchmark_algorithm"].decode("utf-8"),
                profiler=stored[b"profiler_algorithm"].decode("utf-8"),
            ),
        )

    def _fence_conditions_met(self, stored: dict, lease: AttemptLease, now_ms: int) -> bool:
        resource = EvaluationResource.model_validate_json(stored[b"resource"])
        return (
            resource.state == "running"
            and resource.revision == lease.expected_revision
            and stored.get(b"attempt_id", b"").decode("utf-8") == lease.attempt_id
            and int(stored[b"fencing_token"]) == lease.fencing_token
            and now_ms < int(stored[b"deadline_ms"])
        )

    def _fenced_write(self, lease: AttemptLease, mutate) -> bool:
        """Run ``mutate(pipeline, stored, now_ms)`` iff every fence holds."""
        resource_key = self._key(f"resource:{lease.evaluation_id}")
        while True:
            now_ms = self._clock()
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(resource_key)
                stored = pipeline.hgetall(resource_key)
                if not stored or not self._fence_conditions_met(stored, lease, now_ms):
                    self.note_fenced_write()
                    self._bump("terminal_commit_cas_total", outcome="lost")
                    return False
                pipeline.multi()
                mutate(pipeline, stored, now_ms)
                _commit_pre_exec(lease.evaluation_id)
                pipeline.execute()
                _commit_post_exec(lease.evaluation_id)
                return True
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()

    def renew_lease(self, lease: AttemptLease) -> bool:
        """Extend only the internal lease expiry; the public revision stays."""
        resource_key = self._key(f"resource:{lease.evaluation_id}")
        leases_key = self._key("running-leases")

        def mutate(pipeline, stored, now_ms):
            expiry_ms = now_ms + self._s.lease_seconds * 1000
            pipeline.hset(resource_key, "lease_deadline_ms", str(expiry_ms))
            pipeline.zadd(leases_key, {lease.evaluation_id: expiry_ms})

        return self._fenced_write(lease, mutate)

    def set_active_check(self, lease: AttemptLease, check: Check) -> bool:
        resource_key = self._key(f"resource:{lease.evaluation_id}")

        def mutate(pipeline, stored, now_ms):
            pipeline.hset(resource_key, "active_check", check)

        return self._fenced_write(lease, mutate)

    def save_check_outcome(self, lease: AttemptLease, check: Check, outcome: CheckOutcome) -> bool:
        """Persist a partial outcome privately; it never appears nonterminal."""
        resource_key = self._key(f"resource:{lease.evaluation_id}")

        def mutate(pipeline, stored, now_ms):
            outcomes = json.loads(stored[b"check_outcomes"])
            outcomes[check] = outcome.model_dump(mode="json", exclude_none=False)
            pipeline.hset(resource_key, "check_outcomes", canonical_json_bytes(outcomes))

        return self._fenced_write(lease, mutate)

    # ------------------------------------------------------------------
    # Terminal commit
    # ------------------------------------------------------------------

    def commit_terminal(
        self, lease: AttemptLease, state: TerminalState, result: EvaluationResult
    ) -> bool:
        """Single-winner fenced terminal transition; stale writes get ``False``."""
        resource_key = self._key(f"resource:{lease.evaluation_id}")
        while True:
            now_ms = self._clock()
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(resource_key)
                stored = pipeline.hgetall(resource_key)
                if not stored or not self._fence_conditions_met(stored, lease, now_ms):
                    self.note_fenced_write()
                    self._bump("terminal_commit_cas_total", outcome="lost")
                    return False
                resource = EvaluationResource.model_validate_json(stored[b"resource"])
                requested = json.loads(stored[b"requested_checks"])
                validated = EvaluationResult.model_validate(
                    result.model_dump(mode="json", exclude_none=False),
                    context={"requested_checks": requested},
                )
                _commit_pre_exec(lease.evaluation_id)
                self._commit_terminal_locked(
                    pipeline,
                    stored=stored,
                    resource=resource,
                    state=state,
                    result=validated,
                    now_ms=now_ms,
                    bump_fence=False,
                )
                _commit_post_exec(lease.evaluation_id)
                self._bump("terminal_commit_cas_total", outcome="won")
                self._bump(
                    "resource_terminal_transitions_total",
                    state=state,
                    caller_class=self._stored_traffic_class(stored),
                )
                started_ms = stored.get(b"created_at_ms")
                if started_ms is not None and self._metrics is not None:
                    try:
                        self._metrics.observe(
                            "resource_terminal_duration_seconds",
                            {"state": state},
                            max(0.0, (now_ms - int(started_ms)) / 1000),
                        )
                    except ValueError:
                        pass
                return True
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()

    def _commit_terminal_locked(
        self,
        pipeline,
        *,
        stored: dict,
        resource: EvaluationResource,
        state: TerminalState,
        result: EvaluationResult,
        now_ms: int,
        bump_fence: bool,
    ) -> None:
        """Issue one MULTI/EXEC terminal transition; caller holds the watches."""
        evaluation_id = resource.evaluation_id
        resource_key = self._key(f"resource:{evaluation_id}")
        caller_key, session_key = self._inflight_keys(stored)
        pipeline.watch(caller_key, session_key)
        caller_inflight = int(pipeline.get(caller_key) or 0)
        session_inflight = int(pipeline.get(session_key) or 0)

        finished = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        updated = EvaluationResource.model_validate(
            {
                **resource.model_dump(exclude_none=False),
                "revision": resource.revision + 1,
                "state": state,
                "updated_at": finished,
                "finished_at": finished,
                "result": result.model_dump(mode="json", exclude_none=False),
            }
        )
        expiry_ms = now_ms + self._s.retention_seconds * 1000
        hash_updates = {
            "resource": canonical_json_bytes(
                updated.model_dump(mode="json", exclude_none=False)
            ),
            "updated_at_ms": str(now_ms),
            "attempt_id": "",
            "replacement_pending": "0",
        }
        if bump_fence:
            hash_updates["fencing_token"] = str(int(stored[b"fencing_token"]) + 1)
        pipeline.multi()
        pipeline.hset(resource_key, mapping=hash_updates)
        pipeline.expire(resource_key, self._s.retention_seconds)
        pipeline.expire(
            stored[b"idempotency_key"].decode("utf-8"), self._s.retention_seconds
        )
        pipeline.zrem(self._key("ready:exec:interactive"), evaluation_id)
        pipeline.zrem(self._key("ready:exec:bulk"), evaluation_id)
        pipeline.zrem(self._key("ready"), evaluation_id)  # legacy drain-only queue
        pipeline.zrem(self._key("deadlines"), evaluation_id)
        pipeline.zrem(self._key("running-leases"), evaluation_id)
        pipeline.zrem(self._key("state:queued"), evaluation_id)
        pipeline.zrem(self._key("state:running"), evaluation_id)
        pipeline.zadd(self._key(f"state:{state}"), {evaluation_id: expiry_ms})
        pipeline.set(caller_key, max(0, caller_inflight - 1))
        pipeline.set(session_key, max(0, session_inflight - 1))
        pipeline.execute()

    @staticmethod
    def _stored_traffic_class(stored: dict) -> str:
        """Class persisted since the rollout-class deploy; older resources on the
        durable volume derive it from purpose."""
        raw = stored.get(b"traffic_class")
        if raw is not None:
            return raw.decode("utf-8")
        purpose = stored[b"purpose"].decode("utf-8")
        return "trial" if purpose == "trial" else "trusted"

    def _inflight_keys(self, stored: dict) -> tuple[str, str]:
        owner = stored[b"owner"].decode("utf-8")
        traffic_class = self._stored_traffic_class(stored)
        caller_key = self._key(f"inflight:caller:{traffic_class}:{_digest_text(owner)}")
        session_digest = stored[b"session_digest"].decode("utf-8")
        return caller_key, self._key(f"inflight:session:{session_digest}")

    # ------------------------------------------------------------------
    # Recovery: lease reaper, deadline watchdog, terminal-index cleanup
    # ------------------------------------------------------------------

    def recover_expired_leases(self, limit: int = 100) -> RecoveryCounts:
        """Requeue replaceable expired leases; commit WORKER_LOST at the limit."""
        leases_key = self._key("running-leases")
        now_ms = self._clock()
        members = self._r.zrangebyscore(leases_key, "-inf", now_ms, start=0, num=limit)
        requeued = 0
        failed = 0
        for member in members:
            outcome = self._recover_one_lease(member, now_ms)
            if outcome == "requeued":
                requeued += 1
                self._bump("reaped_leases_total", outcome="requeued")
            elif outcome == "failed":
                failed += 1
                self._bump("reaped_leases_total", outcome="failed")
        return RecoveryCounts(requeued=requeued, failed=failed)

    def _recover_one_lease(self, member: bytes, now_ms: int) -> str | None:
        evaluation_id = member.decode("utf-8")
        resource_key = self._key(f"resource:{evaluation_id}")
        leases_key = self._key("running-leases")
        sequence_key = self._key("queue-sequence")
        while True:
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(resource_key, leases_key)
                stored = pipeline.hgetall(resource_key)
                if not stored:
                    pipeline.multi()
                    pipeline.zrem(leases_key, member)
                    pipeline.execute()
                    return None
                resource = EvaluationResource.model_validate_json(stored[b"resource"])
                lease_deadline_ms = int(stored[b"lease_deadline_ms"])
                installed = pipeline.zscore(leases_key, member)
                if (
                    resource.state != "running"
                    or installed is None
                    or int(installed) != lease_deadline_ms
                    or lease_deadline_ms > now_ms
                ):
                    return None  # committed, renewed, or already replaced
                if now_ms >= int(stored[b"deadline_ms"]):
                    return None  # the deadline watchdog owns this resource
                # Suspicious deaths are counted only when card provenance exists:
                # without it there is no card to attribute the death to, and
                # card-unaware traffic keeps the exact Plan B retry semantics.
                failed_card = stored.get(b"claimed_card")
                if failed_card is not None:
                    deaths = int(stored.get(b"suspicious_deaths", b"0")) + 1
                    metadata = {
                        "suspicious_deaths": str(deaths),
                        "last_failed_card": failed_card.decode("utf-8"),
                    }
                else:
                    deaths = 0
                    metadata = {}
                if deaths >= self._s.poison_death_threshold:
                    # Poison quarantine (spec §10): too many suspicious deaths —
                    # terminate instead of requeueing. 2.0 contract-invisible shape:
                    # the existing synthesized WORKER_LOST result with a
                    # quarantine message (retryable stays True; that flag is frozen).
                    # Mirrors the attempt-limit path: `_commit_terminal_locked`
                    # issues its own MULTI/EXEC, so the advisory metadata write
                    # must not touch the watched hash before EXEC (a self-write
                    # dirtying the watch would abort the transaction).
                    result = self._synthesized_result(
                        stored,
                        "worker_lost",
                        "worker lease expired before a subject outcome was committed; "
                        "repeated suspicious deaths — candidate quarantined as "
                        "suspected poison",
                    )
                    self._commit_terminal_locked(
                        pipeline,
                        stored=stored,
                        resource=resource,
                        state="failed",
                        result=result,
                        now_ms=now_ms,
                        bump_fence=True,
                    )
                    self._r.hset(resource_key, mapping=metadata)
                    return "failed"
                if int(stored[b"attempt_count"]) < MAX_ATTEMPTS:
                    ready_key = self._ready_key(self._stored_traffic_class(stored))
                    pipeline.watch(ready_key, sequence_key)
                    successor = int(pipeline.get(sequence_key) or 0) + 1
                    pipeline.multi()
                    pipeline.set(sequence_key, successor)
                    pipeline.hset(
                        resource_key,
                        mapping={
                            "attempt_id": "",
                            "fencing_token": str(int(stored[b"fencing_token"]) + 1),
                            "replacement_pending": "1",
                            "active_check": "",
                            "lease_deadline_ms": "0",
                            **metadata,
                        },
                    )
                    pipeline.zrem(leases_key, member)
                    pipeline.zadd(ready_key, {evaluation_id: successor})
                    pipeline.execute()
                    return "requeued"
                result = self._synthesized_result(
                    stored,
                    "worker_lost",
                    "worker lease expired before a subject outcome was committed",
                )
                self._commit_terminal_locked(
                    pipeline,
                    stored=stored,
                    resource=resource,
                    state="failed",
                    result=result,
                    now_ms=now_ms,
                    bump_fence=True,
                )
                return "failed"
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()

    def expire_due_resources(self, limit: int = 100) -> int:
        """Deadline watchdog: CAS due queued/running resources to timed_out."""
        deadlines_key = self._key("deadlines")
        now_ms = self._clock()
        members = self._r.zrangebyscore(deadlines_key, "-inf", now_ms, start=0, num=limit)
        expired = 0
        for member in members:
            if self._expire_one_resource(member, now_ms):
                expired += 1
        return expired

    def _expire_one_resource(self, member: bytes, now_ms: int) -> bool:
        evaluation_id = member.decode("utf-8")
        resource_key = self._key(f"resource:{evaluation_id}")
        deadlines_key = self._key("deadlines")
        while True:
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(resource_key, deadlines_key)
                stored = pipeline.hgetall(resource_key)
                if not stored:
                    pipeline.multi()
                    pipeline.zrem(deadlines_key, member)
                    pipeline.execute()
                    return False
                resource = EvaluationResource.model_validate_json(stored[b"resource"])
                if resource.state in _TERMINAL_STATES:
                    pipeline.multi()
                    pipeline.zrem(deadlines_key, member)
                    pipeline.execute()
                    return False
                message = (
                    "resource deadline expired before any attempt started"
                    if int(stored[b"attempt_count"]) == 0
                    else "resource deadline expired"
                )
                result = self._synthesized_result(stored, "deadline", message)
                self._commit_terminal_locked(
                    pipeline,
                    stored=stored,
                    resource=resource,
                    state="timed_out",
                    result=result,
                    now_ms=now_ms,
                    bump_fence=True,
                )
                return True
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()

    def purge_expired_terminal_indexes(self, limit: int = 100) -> int:
        """Remove terminal index members only when the resource hash expired.

        Idempotent under concurrent API/worker maintenance loops: a member
        whose resource hash still exists is never removed, and ``ZREM`` of an
        already-removed member is a no-op.
        """
        now_ms = self._clock()
        removed = 0
        for state in _TERMINAL_STATES:
            if removed >= limit:
                break
            index_key = self._key(f"state:{state}")
            members = self._r.zrangebyscore(
                index_key, "-inf", now_ms, start=0, num=limit - removed
            )
            for member in members:
                evaluation_id = member.decode("utf-8")
                if not self._r.exists(self._key(f"resource:{evaluation_id}")):
                    removed += self._r.zrem(index_key, member)
        return removed

    def _synthesized_result(
        self, stored: dict, failure: str, message: str
    ) -> EvaluationResult:
        payload = json.loads(stored[b"canonical_request"])
        grade_context = payload.get("grade_context") or {}
        return synthesize_orchestration_result(
            requested_checks=json.loads(stored[b"requested_checks"]),
            persisted_outcomes=persisted_check_outcomes(stored[b"check_outcomes"]),
            failure=failure,
            failure_message=message,
            evaluator_build=stored[b"evaluator_build"].decode("utf-8"),
            algorithms=AlgorithmVersions(
                correctness=stored[b"correctness_algorithm"].decode("utf-8"),
                decoy=stored[b"decoy_algorithm"].decode("utf-8"),
                benchmark=stored[b"benchmark_algorithm"].decode("utf-8"),
                profiler=stored[b"profiler_algorithm"].decode("utf-8"),
            ),
            manifest_id=grade_context.get("manifest_id"),
            problem_id=grade_context.get("problem_id"),
        )

    # ------------------------------------------------------------------
    # Worker heartbeats and fencing metric
    # ------------------------------------------------------------------

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

    def heartbeat(self, worker: WorkerIdentity) -> None:
        """Record a worker heartbeat; fresh means within one lease window."""
        now_ms = self._clock()
        payload = canonical_json_bytes(
            {
                "evaluator_build": worker.evaluator_build,
                "correctness_algorithm": worker.algorithms.correctness,
                "decoy_algorithm": worker.algorithms.decoy,
                "benchmark_algorithm": worker.algorithms.benchmark,
                "profiler_algorithm": worker.algorithms.profiler,
            }
        )
        self._bump("worker_heartbeats_total")
        pipeline = self._r.pipeline()
        pipeline.hset(self._key("worker-heartbeats"), worker.worker_id, payload)
        pipeline.zadd(
            self._key("worker-heartbeat-expiry"),
            {worker.worker_id: now_ms + self._s.lease_seconds * 1000},
        )
        pipeline.execute()

    def has_fresh_compatible_worker(self, compatibility: WorkerCompatibility) -> bool:
        """True iff a non-pruned heartbeat matches build and all algorithms."""
        heartbeats_key = self._key("worker-heartbeats")
        expiry_key = self._key("worker-heartbeat-expiry")
        now_ms = self._clock()
        stale = self._r.zrangebyscore(expiry_key, "-inf", now_ms)
        if stale:
            pipeline = self._r.pipeline()
            pipeline.hdel(heartbeats_key, *stale)
            pipeline.zrem(expiry_key, *stale)
            pipeline.execute()
        expected = {
            "evaluator_build": compatibility.evaluator_build,
            "correctness_algorithm": compatibility.algorithms.correctness,
            "decoy_algorithm": compatibility.algorithms.decoy,
            "benchmark_algorithm": compatibility.algorithms.benchmark,
            "profiler_algorithm": compatibility.algorithms.profiler,
        }
        for payload in self._r.hgetall(heartbeats_key).values():
            try:
                record = json.loads(payload)
            except ValueError:
                continue
            if record == expected:
                return True
        return False

    def note_fenced_write(self) -> None:
        """Count a late, duplicate, or lease-losing attempt write."""
        self._r.incr(self._key("metrics:fenced-write-rejected"))
        self._bump("fenced_write_rejections_total")
