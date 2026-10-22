"""Own trusted baseline measurement, advisory reuse and candidate attribution.

Both standalone timing and timing+trace prepare the same evidence. Only an
isolated reference process may populate the cache. Cache/producer faults fall
back to fresh timing; lease loss and deadline expiry always interrupt execution.
"""

import json
import logging
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ..api.contract import EvaluationRequest, PerformanceParams
from ..runtime.baseline_cache import BaselineCache
from .base import (
    LeaseLostError,
    ScriptOutcome,
    materialize_reference,
    read_json,
    remaining_seconds,
    safe_module_stem,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineEvidence:
    """Per-invocation evidence shared by engine arguments and result validation."""

    _comparison: bool
    _selected: list[int] | None
    _timings: dict[str, float] | None = None
    _hits: frozenset[int] = frozenset()

    def command_options(self) -> list[str]:
        if self._timings is None:
            return []
        return ["--baseline-timings-json", json.dumps(self._timings, allow_nan=False)]

    def bind(self, result: dict) -> None:
        """Validate the comparison denominator and stamp only trusted cache hits.

        Single-implementation results pass through without comparison evidence.
        Invalid candidate output raises ValueError; it never updates the cache.
        """
        if not self._comparison:
            return
        timings, hits, selected = self._timings, self._hits, self._selected
        cases = result.get("per_shape_results")
        if not isinstance(cases, list):
            if timings is None:
                # Existing shaping owns completeness for fresh combined results.
                return
            raise ValueError("benchmark cases missing")
        if any(not isinstance(case, dict) for case in cases):
            raise ValueError("benchmark case record invalid")
        indices = [case.get("case_idx") for case in cases]
        if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("benchmark case identities invalid")
        expected = (
            {int(index) for index in timings}
            if timings is not None
            else None
            if selected is None
            else {index + 1 for index in selected}
        )
        progress = result.get("progress")
        if progress is not None:
            if not isinstance(progress, dict):
                raise ValueError("benchmark progress invalid")
            selected_progress = progress.get("selected_case_indices")
            if not isinstance(selected_progress, list) or any(
                type(index) is not int for index in selected_progress
            ):
                raise ValueError("benchmark progress selection invalid")
            declared = {index + 1 for index in selected_progress}
            if not set(indices).issubset(declared) or (
                expected is not None and declared != expected
            ):
                raise ValueError("benchmark case coverage mismatch")
        elif expected is not None and set(indices) != expected:
            raise ValueError("benchmark case coverage mismatch")
        for case in cases:
            framework = case.get("framework")
            if framework is None:
                if case.get("status") == "pass":
                    raise ValueError("benchmark framework measurement missing")
                continue
            if not isinstance(framework, dict):
                raise ValueError("benchmark framework measurement invalid")
            index = case["case_idx"]
            if timings is not None:
                latency = framework.get("avg_latency_ms")
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or abs(latency - timings[str(index)]) > 0.000051
                ):
                    raise ValueError("benchmark denominator differs from trusted baseline")
            framework["baseline_cache_hit"] = index in hits


def prepare_baseline(
    request: EvaluationRequest,
    *,
    cache: BaselineCache | None,
    engine_dir: str,
    runner: Callable[..., ScriptOutcome],
    device_id: int,
    deadline_at: datetime,
    lease_valid: Callable[[], bool] | None = None,
) -> BaselineEvidence:
    """Resolve trusted timing or fresh fallback; propagate lease/deadline failures."""
    params = request.performance or PerformanceParams()
    comparison = request.implementation == "comparison"
    fresh = BaselineEvidence(comparison, params.case_indices)
    if not comparison or cache is None or not params.use_baseline_cache:
        return fresh
    try:
        timings, hits = _measure(
            request, params, cache, engine_dir, runner, device_id, deadline_at, lease_valid,
        )
    except (LeaseLostError, TimeoutError):
        raise
    except Exception as error:  # noqa: BLE001 — advisory producer falls back fresh
        logger.warning("baseline producer unavailable (%s); measuring fresh", type(error).__name__)
        return fresh
    return BaselineEvidence(comparison, params.case_indices, timings, frozenset(hits))


def _fence(deadline_at, lease_valid):
    if lease_valid is not None and not lease_valid():
        raise LeaseLostError("baseline measurement lost its lease")
    if remaining_seconds(deadline_at) <= 0:
        raise TimeoutError("baseline measurement deadline expired")


def _measure(request, params, cache, engine_dir, runner, device_id, deadline_at, lease_valid):
    key = cache.key_for(
        request.reference_code,
        method=params.method,
        warmup=params.warmup,
        repeats=params.repeats,
    )
    _fence(deadline_at, lease_valid)
    snapshot = cache.read(key)
    selected = (
        None
        if params.case_indices is None
        else sorted(index + 1 for index in params.case_indices)
    )
    if snapshot is not None and selected is not None:
        if any(index > snapshot.total_cases for index in selected):
            snapshot = None
    if selected is None and snapshot is not None:
        selected = list(range(1, snapshot.total_cases + 1))
    existing = {} if snapshot is None else snapshot.latencies
    hits = set() if selected is None else set(selected) & set(existing)
    missing = None if selected is None else [index for index in selected if index not in hits]
    measured = {}
    if missing is None or missing:
        with TemporaryDirectory(prefix="triton-baseline-") as directory:
            root = Path(directory)
            materialize_reference(request, root)
            output = root / "reference_timings.json"
            producer = [
                sys.executable, str(Path(engine_dir) / "benchmark.py"),
                "--op_name", safe_module_stem(request),
                "--verify_dir", str(root),
                "--triton_impl_name", request.triton_impl_name,
                "--warmup", str(params.warmup),
                "--repeats", str(params.repeats),
                "--measurement-method", params.method,
                "--disable-reference-cache",
            ]
            if missing is not None:
                producer += ["--case-indices", ",".join(map(str, missing))]
            producer += ["--reference-only", "--output", str(output)]
            _fence(deadline_at, lease_valid)
            outcome = runner(
                producer,
                str(root),
                device_id,
                deadline_at,
                lease_valid,
                budget_seconds=max(1, math.ceil(remaining_seconds(deadline_at))),
            )
            if outcome.outcome == "fenced":
                raise LeaseLostError("baseline engine lost its lease")
            _fence(deadline_at, lease_valid)
            if outcome.timed_out:
                raise TimeoutError("baseline measurement timed out")
            if outcome.exit_code != 0:
                raise ValueError("reference-only engine failed")
            envelope = read_json(output, root=root)
            total, measured = _validate_reference(envelope, missing, params)
            if snapshot is not None and snapshot.total_cases != total:
                raise ValueError("baseline inventory count changed")
            if selected is None:
                selected = list(range(1, total + 1))
            _fence(deadline_at, lease_valid)
            cache.publish(key, total, measured)
    _fence(deadline_at, lease_valid)
    return {
        str(index): (existing[index] if index in hits else measured[index])
        for index in selected
    }, hits


def _validate_reference(envelope, selected, params):
    if not isinstance(envelope, dict):
        raise ValueError("reference-only output missing")
    total = envelope.get("total_cases")
    if type(total) is not int or not 1 <= total <= 128:
        raise ValueError("invalid reference inventory count")
    expected = list(range(1, total + 1)) if selected is None else selected
    if any(index > total for index in expected) or envelope.get("case_indices") != expected:
        raise ValueError("reference-only case selection mismatch")
    for name, value in (
        ("measurement_method", params.method),
        ("warmup", params.warmup),
        ("repeats", params.repeats),
    ):
        if envelope.get(name) != value or type(envelope.get(name)) is not type(value):
            raise ValueError("reference-only measurement protocol mismatch")
    values = envelope.get("reference_latencies_ms")
    if not isinstance(values, dict) or set(values) != {str(index) for index in expected}:
        raise ValueError("reference-only timing coverage mismatch")
    measured = {}
    for index in expected:
        value = values[str(index)]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("reference-only latency is invalid")
        measured[index] = float(value)
    return total, measured
