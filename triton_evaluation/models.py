"""Immutable durable-core storage records for evaluation API v2.

The old list-based queue models (``Job``/``JobSpec``/``TaskStatus``) are
replaced by the immutable command/result records below; attempt records
arrive with the state-machine task. The legacy synchronous HTTP
request/response models are intentionally absent — the compatibility task
re-creates them in ``legacy_contract.py``.
"""
import uuid
from dataclasses import dataclass
from typing import Literal

from triton_evaluation.admission import CanonicalAdmission
from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.security import CallerClass

# Quota buckets exist only for trial and trusted traffic; an observer granted
# grade:submit consumes trusted capacity, never an observer bucket.
TrafficClass = Literal["trial", "trusted"]
CreateDecision = Literal["created", "replayed", "conflict", "rate_limited", "limit_exceeded"]


def new_evaluation_id() -> str:
    """Generate a resource ID: ``ev_`` plus 32 lowercase UUID4 hex characters."""
    return "ev_" + uuid.uuid4().hex


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
