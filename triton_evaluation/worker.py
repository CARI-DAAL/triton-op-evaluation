"""Durable worker loop: fenced claims, continuous leases, discarded fenced work.

One ``Worker.run_once()`` claims a single attempt from the durable store,
maintains its lease through a ``LeaseKeeper`` daemon thread while the
evaluator runs, shapes a complete result, and fenced-commits it. A worker
that loses its lease commits nothing — the reaper and a replacement attempt
own recovery. Logs carry IDs and error codes only: never source, credentials,
session values, absolute work paths, or tracebacks.
"""
import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import redis

from . import health
from .api.contract import Check, CheckOutcome, EvaluationRequest, EvaluationResult
from .api.contract_artifacts import canonical_json_bytes, sha256_hex
from .broker import EvaluationStore
from .config import Settings
from .evaluators.base import LeaseLostError, PinnedProvenance
from .evaluators.evaluation import EvaluationEvaluator
from .execution_environment import build_execution_environment
from .models import (
    AlgorithmVersions,
    AttemptLease,
    WorkerIdentity,
    new_worker_id,
)
from .terminal_results import synthesize_orchestration_result

logger = logging.getLogger(__name__)

_BACKOFF_CAP_SECONDS = 30
_MAINTENANCE_BATCH = 100


def _algorithms(settings: Settings) -> AlgorithmVersions:
    return AlgorithmVersions(
        correctness=settings.correctness_algorithm_version,
        decoy=settings.decoy_algorithm_version,
        benchmark=settings.benchmark_algorithm_version,
        profiler=settings.profiler_algorithm_version,
    )


class LeaseKeeper:
    """One daemon thread per active attempt: heartbeat plus fenced renewal.

    Any false renewal or Redis exception permanently clears validity — a
    later successful renewal cannot revive it. Joining is bounded to two
    heartbeat intervals.
    """

    def __init__(
        self,
        store: EvaluationStore,
        lease: AttemptLease,
        settings: Settings,
        *,
        identity: WorkerIdentity | None = None,
        interval: float | None = None,
        sleep=None,
    ):
        self._store = store
        self._lease = lease
        self._settings = settings
        self._identity = identity
        self._interval = interval if interval is not None else (
            settings.heartbeat_interval_seconds
        )
        self._sleep = sleep or time.sleep
        self._stop = threading.Event()
        self._invalid = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LeaseKeeper":
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="lease-keeper"
        )
        self._thread.start()
        return self

    def is_valid(self) -> bool:
        return not self._invalid.is_set()

    def stop(self) -> None:
        self._stop.set()
        # A stopped keeper is no longer maintaining the lease; validity ends
        # with the attempt rather than lingering until the lease expires.
        self._invalid.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._interval)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _bump(self, name: str, **labels) -> None:
        bump = getattr(self._store, "_bump", None)
        if bump is not None:
            bump(name, **labels)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if self._identity is not None:
                    self._store.heartbeat(self._identity)
                if not self._store.renew_lease(self._lease):
                    self._bump("lease_renewals_total", outcome="rejected")
                    self._invalid.set()
                    return
                self._bump("lease_renewals_total", outcome="renewed")
            except redis.RedisError:
                self._bump("lease_renewals_total", outcome="rejected")
                self._invalid.set()
                return


class FencedProgress:
    """ProgressSink funneling check progress through fenced store writes."""

    def __init__(self, store: EvaluationStore, lease: AttemptLease):
        self._store = store
        self._lease = lease
        self.saved: dict[str, dict] = {}

    def begin(self, check: Check) -> bool:
        return self._store.set_active_check(self._lease, check)

    def save(self, check: Check, outcome: CheckOutcome) -> bool:
        if not self._store.save_check_outcome(self._lease, check, outcome):
            return False
        self.saved[check] = outcome.model_dump(mode="json", exclude_none=False)
        return True


class _CorruptedRequestError(RuntimeError):
    """The stored canonical request bytes no longer round-trip canonically."""


def _protocol_failure_result(
    *,
    requested_checks,
    persisted_outcomes: dict[str, dict],
    code: str,
    phase: str,
    retryable: bool,
    message: str,
    lease: AttemptLease,
) -> EvaluationResult:
    """A redacted failed result when no trusted subject outcome exists."""
    requested = [
        check for check in ("correctness", "performance", "profile")
        if check in requested_checks
    ]
    outcomes: dict[str, dict] = {}
    target = next((check for check in requested if check not in persisted_outcomes), None)
    for check in requested:
        if check in persisted_outcomes:
            outcomes[check] = persisted_outcomes[check]
        elif check == target:
            outcomes[check] = {
                "status": "failed",
                "data": None,
                "error": {
                    "code": code,
                    "phase": phase,
                    "check": check,
                    "retryable": retryable,
                    "message": message,
                },
                "reason": None,
            }
        else:
            outcomes[check] = {
                "status": "skipped",
                "data": None,
                "error": None,
                "reason": {
                    "code": "PRIOR_CHECK_FAILED",
                    "depends_on": "correctness" if check == "performance" else None,
                    "message": "a prior requested check did not complete",
                },
            }
    return EvaluationResult.model_validate(
        {
            "correctness": outcomes.get("correctness"),
            "performance": outcomes.get("performance"),
            "profile": outcomes.get("profile"),
            "provenance": {
                "schema_version": "2.0",
                "evaluator_build": lease.evaluator_build,
                "correctness_algorithm": lease.algorithms.correctness,
                "decoy_algorithm": lease.algorithms.decoy,
                "benchmark_algorithm": lease.algorithms.benchmark,
                "profiler_algorithm": lease.algorithms.profiler,
                "grade_eligible": False,
                "manifest_id": None,
                "problem_id": None,
                "execution_environment": None,
            },
        },
        context={"requested_checks": requested},
    )


class Worker:
    """One claim-capable worker bound to a single NPU device."""

    def __init__(
        self,
        store: EvaluationStore,
        evaluator: EvaluationEvaluator,
        *,
        device_id: int,
        settings: Settings,
        identity: WorkerIdentity | None = None,
        clock: Callable[[], int] | None = None,
        sleep=None,
        canary=None,
        recovery_step=None,
    ):
        self.store = store
        self.evaluator = evaluator
        self.device_id = device_id
        self.settings = settings
        self.identity = identity or WorkerIdentity(
            new_worker_id(), settings.evaluator_build, _algorithms(settings)
        )
        self._clock = clock or (lambda: int(time.time() * 1000))
        self._sleep = sleep or time.sleep
        self._health = health.CardHealth(
            store, settings, device_id=device_id, clock=clock,
            canary=canary, recovery_step=recovery_step, metrics=store._metrics,
        )

    def run_once(self, *, sleep=None) -> bool:
        """Claim one attempt, lease it, evaluate, fenced-commit it."""
        do_sleep = sleep if sleep is not None else self._sleep
        if not self._health.prepare_claim():
            return False
        # Only claim-capable workers heartbeat, including when the queue is empty.
        self.store.heartbeat(self.identity)
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
        # Suspicious endings require a canary immediately after the attempt.
        # Healthy attempts pass through without probing.
        self._health.prepare_claim()
        return True

    def serve_forever(self, stop_event=None, *, sleep=None) -> None:
        """Serve until stopped; transient infra faults back off, never crash."""
        stop = stop_event if stop_event is not None else threading.Event()
        do_sleep = sleep if sleep is not None else self._sleep
        errors = 0
        while not stop.is_set():
            try:
                processed = self.run_once(sleep=do_sleep)
                errors = 0
                if not processed:
                    # Never busy-spin an empty queue.
                    stop.wait(self.settings.queue_poll_interval_seconds)
            except redis.exceptions.ConnectionError:
                errors += 1
                self._backoff(do_sleep, errors)
            except Exception as error:  # noqa: BLE001 — a daemon loop never dies
                errors += 1
                logger.warning(
                    "worker loop error (%s) on evaluation attempt; continuing",
                    type(error).__name__,
                )
                self._backoff(do_sleep, errors)

    @staticmethod
    def _backoff(do_sleep, errors: int) -> None:
        do_sleep(min(2**errors, _BACKOFF_CAP_SECONDS))

    def _maintenance(self) -> None:
        # Fenced/CAS-safe by construction; several workers may run these.
        self.store.expire_due_resources(limit=_MAINTENANCE_BATCH)
        self.store.recover_expired_leases(limit=_MAINTENANCE_BATCH)
        self.store.purge_expired_terminal_indexes(limit=_MAINTENANCE_BATCH)

    def _run_attempt(self, lease: AttemptLease, do_sleep) -> None:
        work_dir: str | None = None
        try:
            try:
                request = self._parse_and_verify(lease)
            except _CorruptedRequestError:
                self._commit_protocol_failure(
                    lease, requested_checks=["correctness"], persisted={}, code="INTERNAL",
                    phase="storage", message="stored request failed canonical verification",
                )
                return

            deadline_at = datetime.fromtimestamp(
                lease.deadline_at_ms / 1000, tz=timezone.utc
            )
            if self._clock() >= lease.deadline_at_ms:
                # Late delivery: expire rather than start past the deadline.
                self._commit_deadline(lease, request.checks, {})
                return

            digest = sha256_hex(
                canonical_json_bytes(
                    request.model_dump(mode="json", exclude_none=False)
                )
            )
            work_dir = tempfile.mkdtemp(prefix=f"eval_{digest[:24]}_")
            progress = FencedProgress(self.store, lease)
            with LeaseKeeper(
                self.store, lease, self.settings,
                identity=self.identity, sleep=do_sleep,
            ) as keeper:
                try:
                    result = self.evaluator.run(
                        request,
                        work_dir,
                        self.device_id,
                        deadline_at,
                        keeper.is_valid,
                        build_execution_environment(self.settings),
                        PinnedProvenance.for_request(
                            request,
                            evaluator_build=lease.evaluator_build,
                            algorithms=lease.algorithms,
                        ),
                        progress,
                    )
                except LeaseLostError:
                    # Fenced work is discarded; the reaper owns recovery.
                    self.store.note_fenced_write()
                    self._health.mark_suspicious()
                    return
                except Exception as error:  # noqa: BLE001 — redacted, never a crash
                    logger.warning(
                        "evaluator protocol failure (%s) on attempt %s",
                        type(error).__name__, lease.attempt_id,
                    )
                    self._commit_protocol_failure(
                        lease,
                        requested_checks=request.checks,
                        persisted=progress.saved,
                        code="EVALUATOR_EXECUTION_FAILED",
                        phase="execution",
                        message="evaluator failed before a subject outcome",
                    )
                    self._health.mark_suspicious()
                    return
                if not keeper.is_valid():
                    self.store.note_fenced_write()
                    self._health.mark_suspicious()
                    return

            if self._clock() >= lease.deadline_at_ms:
                self._commit_deadline(lease, request.checks, progress.saved)
                return
            validated = EvaluationResult.model_validate(
                result.model_dump(mode="json", exclude_none=False),
                context={"requested_checks": request.checks},
            )
            # False means another winner already terminalized the resource.
            self.store.commit_terminal(lease, "completed", validated)
        finally:
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _parse_and_verify(lease: AttemptLease) -> EvaluationRequest:
        canonical = lease.canonical_request
        request = EvaluationRequest.model_validate_json(canonical)
        recomputed = canonical_json_bytes(
            request.model_dump(mode="json", exclude_none=False)
        )
        if recomputed != canonical:
            raise _CorruptedRequestError
        return request

    def _commit_deadline(
        self, lease: AttemptLease, requested_checks, persisted: dict
    ) -> None:
        result = synthesize_orchestration_result(
            requested_checks=requested_checks,
            persisted_outcomes=persisted,
            failure="deadline",
            failure_message="resource deadline expired before a subject outcome",
            evaluator_build=lease.evaluator_build,
            algorithms=lease.algorithms,
            manifest_id=None,
            problem_id=None,
        )
        # False means the watchdog or a faster worker terminalized it first.
        self.store.commit_terminal(lease, "timed_out", result)

    def _commit_protocol_failure(
        self,
        lease: AttemptLease,
        *,
        requested_checks,
        persisted: dict,
        code: str,
        phase: str,
        message: str,
    ) -> None:
        result = _protocol_failure_result(
            requested_checks=requested_checks,
            persisted_outcomes=persisted,
            code=code,
            phase=phase,
            retryable=code != "INTERNAL",
            message=message,
            lease=lease,
        )
        self.store.commit_terminal(lease, "failed", result)
