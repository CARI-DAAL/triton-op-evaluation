import logging
import shutil
import tempfile
import time

import redis

from .broker import Broker
from .config import BENCHMARK_TIMEOUT_SECONDS, Settings
from .models import TaskStatus

logger = logging.getLogger(__name__)


def compute_max_runtime(spec) -> float:
    """Wall-clock budget for a job, accounting for v2 checks.

    correctness/profile each may take up to ``spec.timeout`` (their subprocess
    timeout is spec.timeout + 60); performance adds the benchmark hard cap."""
    budget = spec.timeout
    checks = spec.checks
    if checks:
        if "performance" in checks:
            budget += BENCHMARK_TIMEOUT_SECONDS
        if "profile" in checks:
            budget += spec.timeout
    elif getattr(spec, "measure_speed", False):
        budget += BENCHMARK_TIMEOUT_SECONDS
    return budget


class Worker:
    def __init__(self, broker: Broker, registry: dict, device_id: int, settings: Settings):
        self.broker = broker
        self.registry = registry
        self.device_id = device_id
        self.settings = settings

    def run_once(self) -> bool:
        job = self.broker.dequeue()
        if job is None:
            return False
        max_runtime = compute_max_runtime(job.spec)
        deadline = time.time() + max_runtime + self.settings.reaper_grace_seconds
        self.broker.mark_running(job.task_id, self.device_id, deadline)
        try:
            evaluator = self.registry[job.type]
            verify_dir = tempfile.mkdtemp(prefix=f"eval_{job.task_id}_")
            try:
                res = evaluator.run(job.spec, verify_dir, self.device_id)
            finally:
                shutil.rmtree(verify_dir, ignore_errors=True)
            self.broker.store_result(job.task_id, res.status, res.result, res.error)
        except Exception as e:  # dispatch/setup failure — never crash the worker
            self.broker.store_result(job.task_id, TaskStatus.FAILED.value, error=repr(e))
        return True

    def serve_forever(self) -> None:
        # A daemon's top loop must never exit on a recoverable error. dequeue()
        # owns "no task"; this boundary owns transient infra faults, with bounded
        # exponential backoff so a Redis outage does not become a busy loop.
        errs = 0
        while True:
            try:
                processed = self.run_once()
                errs = 0
                if not processed:
                    reap_stale_tasks(self.broker, self.settings)
            except redis.exceptions.ConnectionError:
                errs += 1
                logger.warning("redis connection error; backing off (attempt %d)", errs)
                time.sleep(min(2**errs, 30))
            except Exception:
                logger.exception("worker loop error; continuing")
                time.sleep(1)


def reap_stale_tasks(broker: Broker, settings: Settings) -> int:
    now = time.time()
    reaped = 0
    for rec in broker.iter_running_tasks():
        deadline = rec.get("deadline")
        if deadline is not None and now > deadline:
            broker.store_result(rec["task_id"], TaskStatus.FAILED.value, error="worker_lost")
            reaped += 1
    return reaped
