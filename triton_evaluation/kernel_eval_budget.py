"""Durable per-trial tool budget, independent of evaluation admission quotas.

One session identifies one trial. A replay reserves the original turn again;
distinct tool operations share the 18-turn cap. Redis WATCH/MULTI keeps this
true across API processes, concurrent calls and restarts. Admission failures
consume their reserved turn, matching the original tool contract. There is
no process-local fallback when durable storage is unavailable.
"""

import hashlib
import json
from dataclasses import dataclass

import redis

from .service.broker import StorageError


@dataclass(frozen=True)
class BudgetReservation:
    turn: int | None
    used: int
    decision: str


class EvalBudgetRegistry:
    def __init__(self, client, *, prefix: str, maximum: int, retention_seconds: int):
        self.client = client
        self.prefix = prefix
        self.maximum = maximum
        self.retention_seconds = retention_seconds

    def reserve(
        self, caller_id: str, session_id: str, operation_id: str, request_digest: str
    ) -> BudgetReservation:
        identity = hashlib.sha256(f"{caller_id}\0{session_id}".encode()).hexdigest()
        key = f"{self.prefix}:kernel-eval:budget:{identity}"
        field = "operation:" + hashlib.sha256(operation_id.encode()).hexdigest()
        try:
            while True:
                with self.client.pipeline() as pipe:
                    try:
                        pipe.watch(key)
                        used = int(pipe.hget(key, "used") or 0)
                        previous = pipe.hget(key, field)
                        if previous is not None:
                            operation = json.loads(previous)
                            if operation["request_digest"] != request_digest:
                                return BudgetReservation(None, used, "conflict")
                            turn = int(operation["turn"])
                            return BudgetReservation(turn, turn, "replayed")
                        if self.maximum >= 0 and used >= self.maximum:
                            return BudgetReservation(None, used, "budget_exhausted")
                        turn = used + 1
                        pipe.multi()
                        pipe.hset(
                            key,
                            mapping={
                                "used": turn,
                                field: json.dumps({"request_digest": request_digest, "turn": turn}),
                            },
                        )
                        pipe.expire(key, self.retention_seconds)
                        pipe.execute()
                        return BudgetReservation(turn, turn, "reserved")
                    except redis.WatchError:
                        continue
        except redis.RedisError:
            raise StorageError() from None
