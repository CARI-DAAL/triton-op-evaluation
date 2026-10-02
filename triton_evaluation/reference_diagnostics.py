"""Retain bounded private reference observations before attempt cleanup.

This is diagnostic telemetry, never a candidate verdict or public API payload.
The source is inside the candidate boundary: validate it as untrusted data and
never follow a candidate-created link into host files.
"""

import json
import logging
import os
import stat
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .models import AttemptLease

logger = logging.getLogger(__name__)
_MAX_BYTES = 1024 * 1024
_Device = Annotated[str, Field(pattern=r"^[A-Za-z0-9_:.-]{1,64}$")]


class _PrivateModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _OutputDevice(_PrivateModel):
    output_index: Annotated[int, Field(ge=0)]
    device: _Device


class _CaseReference(_PrivateModel):
    case_idx: Annotated[int, Field(ge=1)]
    selected_reference_device: _Device
    reference_completed: bool
    output_devices: list[_OutputDevice]
    reference_cache_status: Literal["hit", "miss"] | None = None
    reference_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    reference_binding_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    reference_output_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class _PerformanceCache(_PrivateModel):
    deployment_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    device_id: Annotated[str, Field(pattern=r"^[0-9]+$")]
    status: Literal["hit", "miss", "remeasured"]
    manifests: list[Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    reference_memory_measured: bool
    sentinel_case_idx: Annotated[int, Field(ge=1)] | None = None
    sentinel_cached_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    sentinel_measured_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    sentinel_relative_drift: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class _ReferenceTrace(_PrivateModel):
    reference_device_policy: Literal["cpu-first", "npu-strict"]
    cases: list[_CaseReference]
    performance_cache: _PerformanceCache | None = None


def _read_trace(path: Path, policy: str) -> dict:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {"status": "missing", "observations": None}
    try:
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                raise ValueError("reference telemetry must be a bounded regular file")
            data = stream.read(_MAX_BYTES + 1)
            if len(data) > _MAX_BYTES:
                raise ValueError("reference telemetry exceeds size limit")
        trace = _ReferenceTrace.model_validate_json(data)
        if trace.reference_device_policy != policy:
            raise ValueError("reference telemetry policy mismatch")
        return {"status": "recorded", "observations": trace.model_dump(exclude_none=True)}
    except (OSError, ValueError):
        return {"status": "invalid", "observations": None}


def retain_reference_diagnostics(
    settings: Settings,
    lease: AttemptLease,
    work_dir: Path,
    physical_card: int | None,
) -> None:
    """Best-effort, attempt-bound telemetry; missing evidence stays unknown."""
    if settings.reference_diagnostics_dir is None or "correctness" not in lease.requested_checks:
        return
    try:
        try:
            observation = _read_trace(
                work_dir / "verify_reference.json",
                settings.reference_device_policy,
            )
        except OSError:
            observation = {"status": "invalid", "observations": None}
        record = {
            "evaluation_id": lease.evaluation_id,
            "attempt_id": lease.attempt_id,
            "request_digest": lease.request_digest,
            "evaluator_build": lease.pinned.evaluator_build,
            "correctness_algorithm": lease.pinned.algorithms.correctness,
            "configured_reference_device_policy": settings.reference_device_policy,
            "physical_card": physical_card,
            **observation,
        }
        destination = settings.reference_diagnostics_dir / lease.evaluation_id
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = destination / f"{lease.attempt_id}.json"
        # Unique attempt identity owns this path; never overwrite previous evidence.
        with target.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError):
        logger.warning("reference diagnostics could not be retained for %s", lease.attempt_id)
