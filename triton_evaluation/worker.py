"""Durable worker loop: fenced claims, continuous leases, discarded fenced work.

One ``Worker.run_once()`` claims a single attempt from the durable store,
maintains its lease through a ``LeaseKeeper`` daemon thread while the
evaluator runs, shapes a complete result, and fenced-commits it. A worker
that loses its lease commits nothing — the reaper and a replacement attempt
own recovery. Logs carry IDs and error codes only: never source, credentials,
session values, absolute work paths, or tracebacks.
"""
import json
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


def _default_canary(*, device_id: int, budget_seconds: int) -> bool:
    """Canary infra failures (e.g. a lazy device import) read as unhealthy —
    a canary that cannot run is never evidence the card is claim-worthy."""
    try:
        return health.run_canary(device_id=device_id, budget_seconds=budget_seconds)
    except Exception as error:  # noqa: BLE001 — probe failure, not a crash
        logger.warning(
            "card %s canary failed (%s)", device_id, type(error).__name__
        )
        return False


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
        check for check in ("compile", "correctness", "performance", "profile")
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
            "compile": outcomes.get("compile"),
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
        self._canary = canary or _default_canary
        self._recovery_step = recovery_step or health.run_recovery_step
        self._suspicion = bool(settings.card_canary_on_startup)
        self._drained = False
        self._last_recovery_ms = 0
        self._last_probe_ms = 0

    def run_once(self, *, sleep=None) -> bool:
        """Claim one attempt, lease it, evaluate, fenced-commit it."""
        do_sleep = sleep if sleep is not None else self._sleep
        if self._drained:
            self._drain_loop()
            return False
        # Idle workers heartbeat too, so readiness means claim-capable workers.
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
        if self._suspicion:
            # A suspicious attempt ending re-arms suspicion; re-probe the card
            # now so a failure drains before the next claim.
            self._probe_card()
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

    def _probe_card(self) -> bool:
        """Canary-gated suspicion: probe the card; on failure drain it and run
        the recovery ladder. Returns True when the card is claim-worthy."""
        self._last_probe_ms = self._clock()
        healthy = self._canary(
            device_id=self.device_id,
            budget_seconds=self.settings.canary_budget_seconds,
        )
        self._bump_metric("card_canary_total", outcome="pass" if healthy else "fail")
        if healthy:
            self._suspicion = False
            self.store.clear_card_unhealthy(self.device_id)
            self._bump_metric("card_health_state", card=str(self.device_id),
                              amount=-1)
            return True
        self.store.set_card_unhealthy(self.device_id)
        self._bump_metric("card_health_state", card=str(self.device_id))
        self._run_ladder()
        self._suspicion = False  # ladder already probed; drain loop owns re-probes
        return False

    def _run_ladder(self) -> None:
        now_ms = self._clock()
        if now_ms - self._last_recovery_ms < (
            self.settings.card_recovery_retry_seconds * 1000
        ):
            self._drained = True
            return
        self._last_recovery_ms = now_ms
        for step in range(1, self.settings.card_recovery_max_step + 1):
            try:
                name = self._recovery_step(
                    step, card=self.device_id,
                    reset_command=self.settings.card_reset_command,
                )
            except Exception as error:  # noqa: BLE001 — a failed rung escalates
                logger.warning(
                    "card %s recovery step %s failed (%s)",
                    self.device_id, step, type(error).__name__,
                )
                name = None
            if name is not None:
                self._bump_metric("card_recovery_total", step=name)
            if self._canary(device_id=self.device_id,
                            budget_seconds=self.settings.canary_budget_seconds):
                self._bump_metric("card_canary_total", outcome="pass")
                self.store.clear_card_unhealthy(self.device_id)
                self._bump_metric("card_health_state", card=str(self.device_id),
                                  amount=-1)
                return
            self._bump_metric("card_canary_total", outcome="fail")
        self._drained = True

    def _drain_loop(self) -> None:
        """Drained: no heartbeat (readiness drops), no claims. Re-probe the
        canary on canary_interval_seconds; a pass resumes heartbeats/claims."""
        if self._clock() - self._last_probe_ms < (
            self.settings.canary_interval_seconds * 1000
        ):
            return
        self._last_probe_ms = self._clock()
        if self._canary(device_id=self.device_id,
                        budget_seconds=self.settings.canary_budget_seconds):
            self._bump_metric("card_canary_total", outcome="pass")
            self.store.clear_card_unhealthy(self.device_id)
            self._bump_metric("card_health_state", card=str(self.device_id),
                              amount=-1)
            self._drained = False
            return
        self._bump_metric("card_canary_total", outcome="fail")
        self._run_ladder()  # retry budget permitting; stays drained otherwise

    def _bump_metric(self, name: str, amount: int = 1, **labels) -> None:
        if self.store._metrics is None:
            return
        try:
            self.store._metrics.increment(name, labels, amount=amount)
        except ValueError:
            pass

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
                    self._suspicion = True
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
                    self._suspicion = True
                    return
                if not keeper.is_valid():
                    self.store.note_fenced_write()
                    self._suspicion = True
                    return

            if self._clock() >= lease.deadline_at_ms:
                self._commit_deadline(lease, request.checks, progress.saved)
                return
            validated = EvaluationResult.model_validate(
                result.model_dump(mode="json", exclude_none=False),
                context={"requested_checks": request.checks},
            )
            # False means another winner already terminalized the resource.
            committed = self.store.commit_terminal(lease, "completed", validated)
            if committed and self.settings.verdict_cache_enabled:
                self._record_compile_verdict(lease, validated)
        finally:
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _record_compile_verdict(self, lease, validated) -> None:
        """Engine-authoritative compile verdicts feed the admission cache."""
        from triton_evaluation import verdict_cache

        correctness = validated.correctness
        if correctness is None or correctness.status != "succeeded":
            return
        data = correctness.data
        if data is None or getattr(data, "outcome", None) != "compile_error":
            return
        try:
            payload = json.loads(lease.canonical_request)
            digest = verdict_cache.digest_for(
                payload["kernel_code"],
                payload.get("triton_impl_name", "triton_ascend_impl"),
                self.settings.cann_fingerprint,
            )
            verdict_cache.put(
                self.store, digest, "compile_error",
                self.settings.verdict_cache_ttl_seconds,
            )
        except (KeyError, ValueError):
            logger.warning("compile verdict feedback skipped (unparseable request)")

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
            # The version follows the owning resource; a 2.1 request whose
            # checks include compile keeps the compile carrier (the 2.0 wire
            # shape is unchanged — the synthesis drops it for 2.0 itself).
            version=json.loads(lease.canonical_request).get("schema_version", "2.0"),
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
