"""Immutable durable-core storage records for evaluation API v2.

The old list-based queue models (``Job``/``JobSpec``/``TaskStatus``) are
replaced by the immutable command/result records below plus the attempt and
worker records used by the leased state machine. The legacy synchronous HTTP
request/response models are intentionally absent — the compatibility task
re-creates them in ``legacy_contract.py``.
"""
import uuid
from dataclasses import dataclass
from typing import Literal

from triton_evaluation.admission import CanonicalAdmission
from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.security import CallerClass

# Quota buckets: trial and trusted derive from purpose; rollout is a
# server-declared class for RL traffic (settings.rollout_caller_ids). An
# observer granted grade:submit consumes trusted capacity, never an observer
# bucket.
TrafficClass = Literal["trial", "trusted", "rollout"]
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


@dataclass(frozen=True)
class CreateResourceResult:
    decision: CreateDecision
    resource: EvaluationResource | None


@dataclass(frozen=True)
class WorkerCompatibility:
    """Exact evaluator build and algorithm versions a worker must match."""

    evaluator_build: str
    algorithms: AlgorithmVersions


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    evaluator_build: str
    algorithms: AlgorithmVersions


@dataclass(frozen=True)
class AttemptLease:
    """Fenced claim on one running resource; every write rechecks all of it."""

    evaluation_id: str
    attempt_id: str
    fencing_token: int
    expected_revision: int
    deadline_at_ms: int
    canonical_request: bytes
    evaluator_build: str
    algorithms: AlgorithmVersions


@dataclass(frozen=True)
class RecoveryCounts:
    requeued: int
    failed: int
