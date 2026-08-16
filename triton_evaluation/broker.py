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
    ready                                        zset, score = queue sequence
    queue-sequence                               integer
    deadlines                                    zset, score = deadline epoch ms
    state:{queued|running|completed|failed|timed_out}  zsets of evaluation IDs;
                                                 queued/running scored by deadline,
                                                 terminal by retention-expiry instant
    inflight:caller:{traffic_class}:{sha256(caller_id)}        integer
    inflight:session:{sha256(caller_id NUL session_id)}        integer
    rate:{traffic_class}:{sha256(caller_id)}:{minute}          integer, 120s expiry

Only the rate-window key expires; nonterminal records never carry a TTL.
Persisted timestamps come from the store clock — Redis ``TIME`` in
production, an injected deterministic clock in tests.
"""
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import redis

from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.api.contract_artifacts import canonical_json_bytes, sha256_hex
from triton_evaluation.config import Settings
from triton_evaluation.models import CreateResourceCommand, CreateResourceResult
from triton_evaluation.security import Caller

_RATE_WINDOW_EXPIRY_SECONDS = 120
_MINUTE_MS = 60_000


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
    ) -> None:
        if settings.retention_seconds < settings.max_recovery_age_seconds:
            raise ValueError("retention_seconds must cover max_recovery_age_seconds")
        self._r = client
        self._s = settings
        self._clock = clock or (lambda: _redis_time_ms(self._r))

    def _key(self, suffix: str) -> str:
        return f"{self._s.redis_prefix}:{suffix}"

    def _quota_limits(self, traffic_class: str) -> tuple[int, int]:
        if traffic_class == "trial":
            return self._s.trial_rate_limit_per_minute, self._s.trial_max_inflight
        return self._s.trusted_rate_limit_per_minute, self._s.trusted_max_inflight

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
                pipeline.zadd(self._key("ready"), {command.evaluation_id: successor})
                pipeline.zadd(self._key("deadlines"), {command.evaluation_id: deadline_ms})
                pipeline.zadd(
                    self._key("state:queued"), {command.evaluation_id: deadline_ms}
                )
                pipeline.incr(inflight_caller_key)
                pipeline.incr(inflight_session_key)
                pipeline.incr(rate_key)
                pipeline.expire(rate_key, _RATE_WINDOW_EXPIRY_SECONDS)
                pipeline.execute()
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
            "attempt_id": "",
            "fencing_token": "0",
            "active_check": "",
            "check_outcomes": "{}",
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
