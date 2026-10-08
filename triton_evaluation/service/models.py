"""Immutable storage records for durable v2.1 evaluations.

Commands, attempts and worker identities belong to the leased state machine;
public request and response models belong to ``api.contract``.
"""
import uuid
from dataclasses import dataclass
from typing import Literal

from triton_evaluation.api.contract import Check, EvaluationRequest, EvaluationResource, Purpose
from triton_evaluation.service.admission import CanonicalAdmission
from triton_evaluation.service.security import CallerClass

# Quota buckets: trial and trusted derive from purpose; rollout is a
# server-declared class for RL traffic (settings.rollout_caller_ids). An
# observer granted grade:submit consumes trusted capacity, never an observer
# bucket.
TrafficClass = Literal["trial", "trusted", "rollout"]
ExecutionClass = Literal["cpu", "npu"]
CreateDecision = Literal["created", "replayed", "conflict", "rate_limited", "limit_exceeded"]
TerminalState = Literal["completed", "failed", "timed_out"]


def new_evaluation_id() -> str:
    """Generate a resource ID: ``ev_`` plus 32 lowercase UUID4 hex characters."""
    return "ev_" + uuid.uuid4().hex


def new_attempt_id() -> str:
    """Generate an attempt ID: ``at_`` plus 32 lowercase UUID4 hex characters."""
    return "at_" + uuid.uuid4().hex


def new_worker_id() -> str:
    """Generate a process-random worker ID; it never encodes host/card identity."""
    return "wk_" + uuid.uuid4().hex


@dataclass(frozen=True)
class AlgorithmVersions:
    """Provenance algorithm versions pinned at admission."""

    correctness: str
    decoy: str
    benchmark: str
    profiler: str


@dataclass(frozen=True)
class PinnedProvenance:
    """Admission facts shared by execution and every recovery outcome."""

    evaluator_build: str
    algorithms: AlgorithmVersions
    purpose: Purpose
    manifest_id: str | None
    problem_id: str | None

    @classmethod
    def for_request(
        cls, request: EvaluationRequest, *, evaluator_build: str,
        algorithms: AlgorithmVersions,
    ) -> "PinnedProvenance":
        context = request.grade_context
        return cls(
            evaluator_build=evaluator_build,
            algorithms=algorithms,
            purpose=request.purpose,
            manifest_id=context.manifest_id if context is not None else None,
            problem_id=context.problem_id if context is not None else None,
        )


@dataclass(frozen=True)
class CreateResourceCommand:
    evaluation_id: str
    caller_id: str
    caller_class: CallerClass
    traffic_class: TrafficClass
    session_id: str
    idempotency_key: str
    admission: CanonicalAdmission
    evaluator_build: str
    algorithms: AlgorithmVersions

    @property
    def execution_class(self) -> ExecutionClass:
        """Compile-only work has no physical-card requirement."""
        return "cpu" if tuple(self.admission.requested_checks) == ("compile",) else "npu"


@dataclass(frozen=True)
class CreateResourceResult:
    decision: CreateDecision
    resource: EvaluationResource | None


@dataclass(frozen=True)
class WorkerCompatibility:
    """Exact evaluator build and algorithm versions a worker must match."""

    evaluator_build: str
    algorithms: AlgorithmVersions
    execution_class: ExecutionClass = "npu"


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    evaluator_build: str
    algorithms: AlgorithmVersions
    execution_class: ExecutionClass = "npu"


@dataclass(frozen=True)
class AttemptLease:
    """Fenced claim on one running resource; every write rechecks all of it."""

    evaluation_id: str
    attempt_id: str
    fencing_token: int
    expected_revision: int
    deadline_at_ms: int
    canonical_request: bytes
    request_digest: str
    requested_checks: tuple[Check, ...]
    pinned: PinnedProvenance


@dataclass(frozen=True)
class RecoveryCounts:
    requeued: int
    failed: int
