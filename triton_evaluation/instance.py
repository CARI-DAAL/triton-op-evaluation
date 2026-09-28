"""An evaluation instance owns admission and recovery independently of HTTP.

Submission resolves authority and request identity before reserving durable
work. Candidate code belongs exclusively to a leased worker attempt; neither
new admissions, replays nor rejected requests execute it here.
"""
import logging

from .admission import admit
from .api.contract import EvaluationRequest, EvaluationResource
from .broker import EvaluationStore
from .catalog import ManifestBindingError, ReleaseCatalog
from .config import Settings
from .models import (
    AlgorithmVersions,
    CreateResourceCommand,
    CreateResourceResult,
    ExecutionClass,
    WorkerCompatibility,
    new_evaluation_id,
)
from .security import Caller

logger = logging.getLogger(__name__)


class InstanceNotReadyError(Exception):
    """The requested execution class cannot currently accept new work."""


class QueueUnavailableError(Exception):
    """Durable storage cannot currently accept a new resource."""


class EvaluationInstance:
    """Authorize, admit and recover evaluations within one owning instance."""

    def __init__(
        self, *, settings: Settings, store: EvaluationStore,
        catalog: ReleaseCatalog, metrics=None, artifacts_verified: bool = False,
    ) -> None:
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self.metrics = metrics
        self.artifacts_verified = artifacts_verified
        self._algorithms = AlgorithmVersions(
            correctness=settings.effective_correctness_algorithm_version,
            decoy=settings.decoy_algorithm_version,
            benchmark=settings.benchmark_algorithm_version,
            profiler=settings.profiler_algorithm_version,
        )

    def submit(
        self, request: EvaluationRequest, caller: Caller, *,
        session_id: str, idempotency_key: str,
    ) -> CreateResourceResult:
        try:
            admission = admit(request, caller, self.catalog)
        except ManifestBindingError:
            self._bump("catalog_binding_outcomes_total", outcome="rejected_binding")
            raise
        if request.purpose == "grade":
            self._bump("catalog_binding_outcomes_total", outcome="bound")
        is_rollout = request.purpose == "rollout" or (
            request.purpose == "trial"
            and caller.caller_id in self.settings.rollout_caller_ids
        )
        command = CreateResourceCommand(
            evaluation_id=new_evaluation_id(),
            caller_id=caller.caller_id,
            caller_class=caller.caller_class,
            traffic_class=(
                "rollout" if is_rollout
                else "trial" if request.purpose == "trial" else "trusted"
            ),
            session_id=session_id,
            idempotency_key=idempotency_key,
            admission=admission,
            evaluator_build=self.settings.evaluator_build,
            algorithms=self._algorithms,
        )
        # Existing work remains recoverable while workers or publication
        # readiness are temporarily unavailable. The atomic operation below
        # owns the final byte-exact replay/conflict decision.
        replay = self.store.lookup_idempotent(caller.caller_id, idempotency_key)
        if replay is None:
            self._require_ready(command.execution_class)
        outcome = self.store.create_or_replay(command)
        self._bump(
            "evaluation_admissions_total", decision=outcome.decision,
            caller_class=caller.caller_class,
        )
        if outcome.decision == "replayed":
            self._bump("idempotency_replays_total", caller_class=caller.caller_class)
        return outcome

    def get(self, evaluation_id: str, caller: Caller) -> EvaluationResource | None:
        return self.store.get_visible(evaluation_id, caller)

    def is_ready(self) -> bool:
        """Whole-instance readiness requires capacity for both execution classes."""
        try:
            self._require_ready("cpu")
            self._require_ready("npu")
        except (InstanceNotReadyError, QueueUnavailableError):
            return False
        return True

    def _require_ready(self, execution_class: ExecutionClass) -> None:
        if not self.artifacts_verified:
            self._bump("worker_compatibility_total", outcome="incompatible")
            raise InstanceNotReadyError()
        if not self.store.probe_storage():
            raise QueueUnavailableError()
        compatible = self.store.has_fresh_compatible_worker(WorkerCompatibility(
            evaluator_build=self.settings.evaluator_build,
            algorithms=self._algorithms,
            execution_class=execution_class,
        ))
        self._bump(
            "worker_compatibility_total",
            outcome="compatible" if compatible else "incompatible",
        )
        if not compatible:
            raise InstanceNotReadyError()

    def _bump(self, name: str, **labels) -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.increment(name, labels)
        except Exception as error:  # noqa: BLE001 — metrics never affect admission
            logger.warning("metrics write failed (%s)", type(error).__name__)
