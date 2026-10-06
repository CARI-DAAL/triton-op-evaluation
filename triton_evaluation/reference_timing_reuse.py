"""Opt-in reuse of historical reference timings validated by sampled evidence.

A narrowly declared, owner-pinned policy artifact authorizes the engine to reuse
the OLD reference cache's per-case timing evidence (the eight real
warmup5/repeats20 profiler samples and their median recorded in sealed
manifests) under one explicitly approved source-runtime -> target-runtime
transition. Reuse is TIMING ONLY: correctness evidence, quality gates and
admission are untouched, and the old deployment, index and manifests are never
modified or rekeyed.

Fail-closed rules:
- Artifact bytes must match the pinned SHA256 (tampering is an infrastructure
  error, never a silent miss).
- The declared source index must match its pinned SHA256 and its deployment
  digest; the declared timing contract must equal the source deployment
  measurement.
- The target runtime identity (patched runtime file hashes and the service
  source hash) must match the running engine; any mismatch denies reuse so the
  engine follows its original execution path (fresh measurement).
- Lookup requires an exact CaseBinding match: same reference source, official
  digest, case, seed-zero inputs and reference policy. Timing evidence is
  never borrowed across devices unless the policy declares cross_card_scope
  (v4): then the UNIQUE sealed entry for the workload — whichever declared
  card audited it — may serve the CURRENT card, and every such evaluation is
  validated by a fresh eight-sample sentinel measured on the current card;
  drift beyond the unchanged 3% keeps the existing per-card monotone
  remeasurement marker for the current card.
"""

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .reference_cache import (
    DEVICE_ENV,
    CacheIndex,
    CaseManifest,
    ReferenceCache,
    ReferenceCacheError,
    ReferenceCacheMiss,
    current_npu_model,
    sha256,
)

POLICY_ENV = "TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY"
POLICY_SHA_ENV = "TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256"
POLICY_ENV_KEYS = frozenset({POLICY_ENV, POLICY_SHA_ENV})
POLICY_SCHEMA = "validated-timing-reuse-policy-v2"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# Evidence from the sampled patched-runtime comparison (GRADE8-RESULT-20261005).
SAMPLED_PROBLEM = "official_l3_BatchMatmul"
SAMPLED_CASES_ONE_BASED = [1, 6, 11, 18, 33, 43]


class TimingReuseError(RuntimeError):
    """Pinned reuse evidence is inconsistent; reuse must fail closed."""


class _TimingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: Literal["profiler"]
    warmup: int = Field(ge=0)
    repeats: int = Field(ge=1)
    samples: Literal[8]


class TimingReusePolicy(BaseModel):
    """Owner-declared artifact; every identity below is verified at load."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[POLICY_SCHEMA] = POLICY_SCHEMA
    # Honest scope: the sampled evidence supports this owner-adopted policy; it
    # is NOT a remeasurement or individual validation of every source case.
    scope_statement: str = Field(min_length=1)
    source_deployment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_cache_dir: str = Field(min_length=1)
    source_index_path: str = Field(min_length=1)
    source_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Dedicated writable state directory for the historical sentinel's
    # monotone drift invalidation markers (same semantics as the live cache).
    source_state_dir: str = Field(min_length=1)
    # Cards on which this policy may serve sealed references. Without
    # cross_card_scope (v2/v3 semantics) these are the only cards whose exact
    # per-card bindings are served. With cross_card_scope (v4) this is the
    # fleet scope of the current card; the sealed entry itself may come from
    # any card declared by the source deployment.
    source_devices: list[str] = Field(min_length=1)
    # v4 scope: sealed entries audited on ANY declared card may serve the
    # current card. The equivalence premise is verified per evaluation by the
    # fresh eight-sample sentinel on the CURRENT card against the served
    # entry; >3% drift keeps the existing per-current-card monotone
    # remeasurement. Absent (v2/v3 artifacts): exact per-card binding only.
    cross_card_scope: bool = False
    # NPU model binding (2026-10-06 ruling: reference evidence binds to the
    # card MODEL, never the card index). When declared, the live device must
    # report exactly this model or reuse misses (fresh measurement); the
    # sealed runtime/toolchain fingerprint stays pinned by
    # source_deployment_sha256 below. Absent (v2-v5 artifacts): no model gate.
    npu_model: str | None = None
    timing_contract: _TimingContract
    # The old audit label and reference8's fresh measurement are the same
    # underlying observation protocol: 8x (warmup5/repeats20, profiler).
    contract_reconciliation: str = Field(min_length=1)
    # Target identity: patched runtime files (relative to the triton package)
    # and the service source file whose timing lookup consumes this policy.
    target_patched_runtime_files: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1
    )
    target_service_source_sha256: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1
    )
    evidence: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _declared_paths_are_absolute(self):
        for name in (
            "source_cache_dir", "source_index_path", "source_state_dir"
        ):
            value = getattr(self, name)
            if not value.startswith("/") or ".." in Path(value).parts:
                raise ValueError(f"{name} must be absolute without traversal")
        return self


class ValidatedTimingReuse:
    """Verified view over one approved source -> target timing transition."""

    def __init__(self, policy: TimingReusePolicy, policy_sha256: str, index: CacheIndex):
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self.index = index

    def source_cache_for(self, device_id: str) -> ReferenceCache:
        # state_dir carries the historical sentinel's monotone invalidation
        # markers: drift must stick for the whole source generation.
        return ReferenceCache(
            Path(self.policy.source_cache_dir),
            self.index.deployment,
            device_id,
            entries=self.index.entries,
            state_dir=Path(self.policy.source_state_dir),
        )

    def reuse_timings(
        self,
        *,
        config,
        reference_source: str,
        reference_device_policy: str,
        device_id: str,
        case_indices: list[int],
        input_groups: list,
    ) -> dict[int, CaseManifest] | None:
        """Return {case_idx: manifest} when every selected case matches exactly.

        None (no reuse, original execution) on any miss or protocol mismatch.
        Raises TimingReuseError only for pinned-evidence inconsistency that a
        fresh measurement cannot repair silently (index vs policy identity).
        """
        if device_id is None or device_id not in self.policy.source_devices:
            return None
        if self.policy.npu_model is not None and (
            current_npu_model() != self.policy.npu_model
        ):
            # Same-model binding: a different NPU model never borrows sealed
            # timing evidence; the engine measures fresh.
            return None
        from .reference_cache import official_source_metadata

        if official_source_metadata(reference_source) is None:
            # Only pinned official references ever had audited timings.
            return None
        cache = self.source_cache_for(device_id)
        try:
            cache.check_measurement(config)
        except ReferenceCacheError:
            # The request's measurement protocol differs from the validated
            # historical contract: measure fresh instead of aliasing.
            return None
        try:
            reused: dict[int, CaseManifest] = {}
            for case_idx, inputs in zip(case_indices, input_groups):
                if self.policy.cross_card_scope:
                    # Cross-card scope: the unique sealed entry for this
                    # workload on any declared card. The engine still measures
                    # a fresh sentinel on the current card before serving it.
                    reused[case_idx] = cache.load_case_cross_card(
                        reference_source, case_idx, inputs,
                        reference_device_policy=reference_device_policy,
                    )
                else:
                    reused[case_idx] = cache.load_case(
                        reference_source, case_idx, inputs,
                        reference_device_policy=reference_device_policy,
                    )
        except ReferenceCacheMiss:
            return None
        except ReferenceCacheError as error:
            # Pinned old evidence is unreadable: refuse reuse, but do not turn
            # an optional optimization into an evaluation failure.
            raise TimingReuseError(
                f"validated timing reuse source is corrupt: {error}"
            ) from error
        return reused


def current_patched_runtime_sha256(relative_name: str) -> str | None:
    """Hash an installed triton runtime file; None when it cannot be resolved.

    Fail-closed: an unresolvable or unreadable file never authorizes reuse.
    """
    try:
        import triton
    except Exception:
        return None
    path = Path(triton.__file__).parent / relative_name
    try:
        if not path.is_file():
            return None
        return sha256(path.read_bytes())
    except OSError:
        return None


def current_service_source_sha256(relative_name: str) -> str | None:
    """Hash a service source file relative to this package (engine identity)."""
    path = Path(__file__).resolve().parent / relative_name
    try:
        return sha256(path.read_bytes())
    except OSError:
        return None


def _target_identity_matches(policy: TimingReusePolicy) -> bool:
    for name, expected in policy.target_patched_runtime_files.items():
        if current_patched_runtime_sha256(name) != expected:
            return False
    for name, expected in policy.target_service_source_sha256.items():
        if current_service_source_sha256(name) != expected:
            return False
    return True


def load_validated_timing_reuse() -> ValidatedTimingReuse | None:
    """Load and fully verify the configured policy; None when not configured.

    Raises TimingReuseError when a configured/pinned artifact is inconsistent
    (wrong bytes, wrong index identity, contradictory timing contract) — the
    engine surfaces that as an infrastructure failure, never as a cache hit.
    A configured policy whose target identity does not match this runtime is
    not an error: it simply does not authorize reuse here.
    """
    present = {
        key: os.environ[key]
        for key in (POLICY_ENV, POLICY_SHA_ENV)
        if key in os.environ
    }
    if not present:
        return None
    if len(present) != 2:
        raise TimingReuseError(
            "timing reuse policy and pinned SHA256 must be configured together"
        )
    policy_path = Path(present[POLICY_ENV])
    if not policy_path.is_absolute():
        raise TimingReuseError("timing reuse policy path must be absolute")
    payload = policy_path.read_bytes()
    actual = sha256(payload)
    if actual != present[POLICY_SHA_ENV]:
        raise TimingReuseError("timing reuse policy differs from pinned SHA256")
    policy = TimingReusePolicy.model_validate_json(payload)
    index_path = Path(policy.source_index_path)
    index_payload = index_path.read_bytes()
    if sha256(index_payload) != policy.source_index_sha256:
        raise TimingReuseError(
            "timing reuse source index differs from the policy declaration"
        )
    index = CacheIndex.model_validate_json(index_payload)
    if index.deployment.digest != policy.source_deployment_sha256:
        raise TimingReuseError(
            "timing reuse source index deployment differs from the policy"
        )
    contract = policy.timing_contract
    measurement = index.deployment.measurement
    if (
        contract.method != measurement.method
        or contract.warmup != measurement.warmup
        or contract.repeats != measurement.repeats
    ):
        raise TimingReuseError(
            "timing reuse contract contradicts the source deployment measurement"
        )
    if not _target_identity_matches(policy):
        # The approved transition does not describe this runtime: no reuse.
        return None
    return ValidatedTimingReuse(policy, actual, index)


def current_device_id() -> str | None:
    """Physical card identity using the reference cache's own convention."""
    card = os.environ.get(DEVICE_ENV, os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""))
    return card if card.isdigit() else None
