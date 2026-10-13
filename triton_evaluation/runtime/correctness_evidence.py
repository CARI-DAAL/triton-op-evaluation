"""Explicit reuse of a completed correctness observation for the same subject.

Authority comes from the durable owner-scoped lookup, never from caller-supplied
proof. The worker repeats freshness/environment checks just before publishing the
reused check and writes a private, source-bound handoff for benchmark's gate.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from triton_evaluation.api.contract import (
    CorrectnessResult,
    Digest,
    EvaluationRequest,
    EvaluationResource,
    ExecutionEnvironment,
    Provenance,
    StrictModel,
)
from triton_evaluation.runtime.execution_environment import is_complete_execution_environment
from triton_evaluation.serialization import sha256_hex
from triton_evaluation.service.models import PinnedProvenance

HANDOFF_FILENAME = "correctness_evidence.json"
_MAX_HANDOFF_BYTES = 2 * 1024 * 1024


class CorrectnessEvidenceError(ValueError):
    """Nonretryable request failure; never authorizes an implicit fresh run."""

    code = "CORRECTNESS_EVIDENCE_INVALID"
    phase = "request"
    retryable = False

    def __init__(self, message="correctness evidence is unavailable or incompatible"):
        super().__init__(message)


class CorrectnessSubject(StrictModel):
    op_name: str
    reference_sha256: Digest
    candidate_sha256: Digest
    triton_impl_name: str
    implementation: Literal["comparison"]
    non_compute: bool

    @classmethod
    def from_request(cls, request: EvaluationRequest):
        return cls(
            op_name=request.op_name,
            reference_sha256=sha256_hex(request.reference_code.encode("utf-8")),
            candidate_sha256=sha256_hex(request.kernel_code.encode("utf-8")),
            triton_impl_name=request.triton_impl_name,
            implementation=request.implementation,
            non_compute=request.non_compute,
        )


class CorrectnessEvidence(StrictModel):
    schema_version: Literal["correctness-evidence-v1"] = "correctness-evidence-v1"
    evaluation_id: str = Field(pattern=r"^ev_[0-9a-f]{32}$")
    subject: CorrectnessSubject
    correctness: CorrectnessResult
    provenance: Provenance
    # Earliest possible observation time: conservatively use the source attempt
    # start, so a long subsequent profile cannot extend correctness freshness.
    observed_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def complete_original_observation(self):
        data = self.correctness
        if (data.outcome != "correct" or not data.passed
                or data.passed_cases != data.total_cases or data.failures
                or data.reused_from != self.evaluation_id):
            raise ValueError("handoff requires complete passing correctness evidence")
        if data.coverage is not None and (
            data.coverage.stop_reason != "complete"
            or data.coverage.evaluated_case_indices != list(range(data.total_cases))
        ):
            raise ValueError("handoff requires complete whole-task coverage")
        if (not is_complete_execution_environment(self.provenance.execution_environment)
                or self.expires_at <= self.observed_at):
            raise ValueError("handoff requires complete environment and finite freshness window")
        return self


def resolve_source(
    request: EvaluationRequest,
    source_request: EvaluationRequest,
    resource: EvaluationResource,
    *,
    pinned: PinnedProvenance,
    now: datetime,
    max_age_seconds: int,
    execution_environment: ExecutionEnvironment | None = None,
    check_environment: bool = False,
) -> CorrectnessEvidence:
    """Validate an already owner-authorized immutable source observation."""
    if (request.correctness_from != resource.evaluation_id or request.purpose == "grade"
            or resource.state != "completed" or resource.result is None
            or resource.started_at is None or resource.finished_at is None):
        raise CorrectnessEvidenceError()
    correctness = resource.result.correctness
    if (correctness is None or correctness.status != "succeeded"
            or correctness.data.outcome != "correct"
            or "correctness" not in source_request.checks
            or source_request.correctness_from is not None
            or correctness.data.reused_from is not None):
        raise CorrectnessEvidenceError("correctness evidence must be an original full-task pass")
    if CorrectnessSubject.from_request(request) != CorrectnessSubject.from_request(source_request):
        raise CorrectnessEvidenceError("correctness evidence does not match the submitted subject")
    provenance = resource.result.provenance
    if (provenance.evaluator_build != pinned.evaluator_build
            or provenance.correctness_algorithm != pinned.algorithms.correctness
            or provenance.decoy_algorithm != pinned.algorithms.decoy):
        raise CorrectnessEvidenceError("correctness evidence uses a different evaluator protocol")
    if not is_complete_execution_environment(provenance.execution_environment) or (
        check_environment and (
            not is_complete_execution_environment(execution_environment)
            or provenance.execution_environment != execution_environment
        )
    ):
        raise CorrectnessEvidenceError("correctness evidence uses a different worker environment")
    expires_at = resource.started_at + timedelta(seconds=max_age_seconds)
    if not resource.started_at <= resource.finished_at <= now < expires_at:
        raise CorrectnessEvidenceError("correctness evidence is outside its freshness window")
    try:
        return CorrectnessEvidence(
            evaluation_id=resource.evaluation_id,
            subject=CorrectnessSubject.from_request(request),
            correctness=CorrectnessResult.model_validate({
                **correctness.data.model_dump(mode="json"),
                "reused_from": resource.evaluation_id,
            }),
            provenance=provenance,
            observed_at=resource.started_at,
            expires_at=expires_at,
        )
    except ValueError as error:
        raise CorrectnessEvidenceError() from error


def write_handoff(directory: Path, evidence: CorrectnessEvidence) -> Path:
    target = Path(directory) / HANDOFF_FILENAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(evidence.model_dump_json(), encoding="utf-8")
    os.replace(temporary, target)
    return target


def validate_handoff(
    path: str | Path, *, reference_path: str | Path, candidate_path: str | Path,
    triton_impl_name: str, non_compute: bool = False,
) -> int:
    """Engine gate: validate explicit proof and current materialized source bytes.

    Provenance/owner authorization has already been checked by the worker.
    This gate binds that handoff to the actual files the engine is about to run.
    """
    try:
        with Path(path).open("rb") as handle:
            payload = handle.read(_MAX_HANDOFF_BYTES + 1)
        if len(payload) > _MAX_HANDOFF_BYTES:
            raise ValueError("oversized evidence")
        evidence = CorrectnessEvidence.model_validate_json(payload)
        now = datetime.now(timezone.utc)
        subject = evidence.subject
        if not evidence.observed_at <= now < evidence.expires_at:
            raise ValueError("expired evidence")
        if (subject.reference_sha256 != sha256_hex(Path(reference_path).read_bytes())
                or subject.candidate_sha256 != sha256_hex(Path(candidate_path).read_bytes())
                or subject.triton_impl_name != triton_impl_name
                or subject.non_compute != non_compute):
            raise ValueError("source identity mismatch")
        return evidence.correctness.total_cases
    except (OSError, ValueError) as error:
        raise CorrectnessEvidenceError("correctness evidence handoff is invalid") from error
