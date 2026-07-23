import json
import time

import redis

from .config import Settings
from .models import Job, TaskStatus


def _parse_task(raw: dict) -> dict:
    def _f(v):
        return float(v) if v not in (None, "") else None

    return {
        "task_id": raw.get("task_id"),
        "type": raw.get("type"),
        "status": raw.get("status"),
        "result": json.loads(raw["result"]) if raw.get("result") else None,
        "error": raw.get("error") or None,
        "device_id": int(raw["device_id"]) if raw.get("device_id") not in (None, "") else None,
        "created_at": _f(raw.get("created_at")),
        "started_at": _f(raw.get("started_at")),
        "finished_at": _f(raw.get("finished_at")),
        "deadline": _f(raw.get("deadline")),
    }


class Broker:
    def __init__(self, client: "redis.Redis", settings: Settings):
        self.r = client
        self.s = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "Broker":
        return cls(redis.from_url(settings.redis_url, decode_responses=True), settings)

    def _task_key(self, task_id: str) -> str:
        return f"{self.s.task_key_prefix}{task_id}"

    def ping(self) -> None:
        self.r.ping()

    def close(self) -> None:
        self.r.close()

    def create_task(self, job: Job) -> None:
        key = self._task_key(job.task_id)
        self.r.hset(
            key,
            mapping={
                "task_id": job.task_id,
                "type": job.type,
                "status": TaskStatus.QUEUED.value,
                "created_at": repr(time.time()),
            },
        )
        self.r.expire(key, self.s.task_ttl_seconds)
        self.r.lpush(self.s.queue_key, job.to_json())

    def dequeue(self) -> Job | None:
        try:
            item = self.r.brpop(self.s.queue_key, timeout=self.s.dequeue_block_seconds)
        except redis.exceptions.TimeoutError:
            # A blocking-read timeout is just "no task" — same event as BRPOP-nil.
            # Honouring the Job | None contract here decouples socket_timeout from
            # dequeue_block_seconds and keeps an empty queue from crashing the worker.
            return None
        if item is None:
            return None
        _, payload = item
        return Job.from_json(payload)

    def mark_running(self, task_id: str, device_id: int, deadline: float) -> None:
        key = self._task_key(task_id)
        self.r.hset(
            key,
            mapping={
                "status": TaskStatus.RUNNING.value,
                "started_at": repr(time.time()),
                "device_id": str(device_id),
                "deadline": repr(deadline),
            },
        )
        self.r.expire(key, self.s.task_ttl_seconds)

    def store_result(
        self, task_id: str, status: str, result: dict | None = None, error: str | None = None
    ) -> None:
        key = self._task_key(task_id)
        mapping = {"status": status, "finished_at": repr(time.time())}
        if result is not None:
            mapping["result"] = json.dumps(result)
        if error is not None:
            mapping["error"] = error
        self.r.hset(key, mapping=mapping)
        self.r.expire(key, self.s.task_ttl_seconds)

    def get_task(self, task_id: str) -> dict | None:
        key = self._task_key(task_id)
        if not self.r.exists(key):
            return None
        return _parse_task(self.r.hgetall(key))

    def iter_running_tasks(self) -> list:
        out = []
        for key in self.r.scan_iter(match=f"{self.s.task_key_prefix}*"):
            raw = self.r.hgetall(key)
            if raw.get("status") == TaskStatus.RUNNING.value:
                out.append(_parse_task(raw))
        return out
