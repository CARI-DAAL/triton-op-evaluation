import shutil
import tempfile
import time

from .broker import Broker
from .config import Settings
from .models import JobType, TaskStatus


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
        deadline = time.time() + job.spec.timeout + self.settings.reaper_grace_seconds
        self.broker.mark_running(job.task_id, self.device_id, deadline)
        try:
            evaluator = self.registry[job.type]
            spec = job.spec
            if (
                job.type == JobType.BENCHMARK.value
                and spec.verify_task_id
                and not spec.verify_not_required
            ):
                prior = self.broker.get_task(spec.verify_task_id)
                if prior is None or prior.get("status") != TaskStatus.SUCCEEDED.value:
                    # The referenced verify hasn't finished (still queued/running) or errored.
                    # That is a race, not a gate failure — report it distinctly, not as REJECTED.
                    state = "not found" if prior is None else prior.get("status")
                    self.broker.store_result(
                        job.task_id,
                        TaskStatus.FAILED.value,
                        error=(
                            f"benchmark gate: verify task {spec.verify_task_id} is not "
                            f"SUCCEEDED (state={state}); submit benchmark only after verify completes"
                        ),
                    )
                    return True
                spec.verify_result_seed = prior.get("result")
            verify_dir = tempfile.mkdtemp(prefix=f"eval_{job.task_id}_")
            try:
                res = evaluator.run(spec, verify_dir, self.device_id)
            finally:
                shutil.rmtree(verify_dir, ignore_errors=True)
            self.broker.store_result(job.task_id, res.status, res.result, res.error)
        except Exception as e:  # dispatch/setup failure — never crash the worker
            self.broker.store_result(job.task_id, TaskStatus.FAILED.value, error=repr(e))
        return True

    def serve_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                reap_stale_tasks(self.broker, self.settings)


def reap_stale_tasks(broker: Broker, settings: Settings) -> int:
    now = time.time()
    reaped = 0
    for rec in broker.iter_running_tasks():
        deadline = rec.get("deadline")
        if deadline is not None and now > deadline:
            broker.store_result(rec["task_id"], TaskStatus.FAILED.value, error="worker_lost")
            reaped += 1
    return reaped
