"""Performance check runner: engine benchmark in the correctness work dir.

benchmark.py's L1 gate reads the verify_result.json written by the
correctness check, so the orchestrator only invokes this after a succeeded
and correct subject outcome. A gate rejection (exit 2), a timeout, or missing
output is an infrastructure failure, never a subject verdict.
"""

import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from ..api.contract import EvaluationRequest, PerformanceParams
from .base import (
    EngineCheckResult,
    LeaseLostError,
    engine_failure,
    engine_timeout,
    log_engine_crash,
    materialize_reference,
    read_json,
    remaining_seconds,
    run_script,
    safe_module_stem,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    def __init__(self, engine_dir: str, timeout_seconds: int = 600, baseline_cache=None):
        self.engine_dir = engine_dir
        self.timeout_seconds = timeout_seconds
        self.baseline_cache = baseline_cache

    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at,
        lease_valid=None,
    ) -> EngineCheckResult:
        deadline_at = min(
            deadline_at, datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
        )
        if lease_valid is not None and not lease_valid():
            raise LeaseLostError("benchmark attempt lost its lease")
        remaining = remaining_seconds(deadline_at)
        if remaining <= 0:
            return engine_timeout("performance", "resource deadline expired before performance ran")
        params = request.performance or PerformanceParams()
        if params.case_indices is not None and not params.case_indices:
            return engine_failure("performance", "performance case selection is empty")
        effective_method = params.method
        out_path = Path(work_dir) / "perf_result.json"
        cmd = [
            sys.executable,
            str(Path(self.engine_dir) / "benchmark.py"),
            "--op_name",
            safe_module_stem(request),
            "--verify_dir",
            str(work_dir),
            "--triton_impl_name",
            request.triton_impl_name,
            "--warmup",
            str(params.warmup),
            "--repeats",
            str(params.repeats),
            "--measurement-method",
            effective_method,
        ]
        if params.case_indices is not None:
            # Public indices are zero-based; the engine takes one-based labels.
            cmd += [
                "--case-indices",
                ",".join(str(index + 1) for index in params.case_indices),
            ]
        cmd += ["--disable-reference-cache"]
        timings = None
        hits = set()
        if self.baseline_cache is not None and params.use_baseline_cache:
            try:
                timings, hits = self._baseline(
                    request, cmd, params, device_id, deadline_at, lease_valid
                )
            except LeaseLostError:
                raise
            except TimeoutError:
                return engine_timeout("performance", "baseline measurement exceeded the deadline")
            except Exception as error:  # noqa: BLE001 — advisory producer falls back fresh
                logger.warning(
                    "baseline producer unavailable (%s); measuring fresh", type(error).__name__
                )
                timings, hits = None, set()
            if timings is not None:
                cmd += ["--baseline-timings-json", json.dumps(timings, allow_nan=False)]
        cmd += ["--output", str(out_path)]
        outcome = run_script(
            cmd,
            str(work_dir),
            device_id,
            deadline_at,
            lease_valid,
            budget_seconds=max(
                1, min(self.timeout_seconds, math.ceil(remaining_seconds(deadline_at)))
            ),
        )
        if outcome.outcome == "fenced" or (lease_valid is not None and not lease_valid()):
            raise LeaseLostError("benchmark engine lost its lease")
        if outcome.timed_out or remaining_seconds(deadline_at) <= 0:
            return engine_timeout("performance", "performance exceeded the resource deadline")
        if outcome.exit_code == 2:
            return engine_failure("performance", "benchmark verify gate rejected the run")
        result = read_json(out_path, root=work_dir)
        if outcome.exit_code == 0 and isinstance(result, dict):
            try:
                self._stamp_result(result, timings, hits, params.case_indices)
            except (ValueError, TypeError, KeyError) as error:
                return engine_failure(
                    "performance", f"invalid benchmark baseline evidence: {error}"
                )
            return EngineCheckResult(status="succeeded", raw=result, error=None)
        log_engine_crash("benchmark", outcome, result_file_exists=isinstance(result, dict))
        return engine_failure(
            "performance",
            f"benchmark engine produced no usable result (engine exit={outcome.exit_code})",
        )

    @staticmethod
    def _fence(deadline_at, lease_valid):
        if lease_valid is not None and not lease_valid():
            raise LeaseLostError("baseline measurement lost its lease")
        if remaining_seconds(deadline_at) <= 0:
            raise TimeoutError("baseline measurement deadline expired")

    def _baseline(self, request, command, params, device_id, deadline_at, lease_valid):
        cache = self.baseline_cache
        key = cache.key_for(
            request.reference_code,
            method=params.method,
            warmup=params.warmup,
            repeats=params.repeats,
        )
        self._fence(deadline_at, lease_valid)
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
                producer = list(command)
                producer[producer.index("--verify_dir") + 1] = str(root)
                if "--case-indices" in producer:
                    offset = producer.index("--case-indices")
                    del producer[offset : offset + 2]
                if missing is not None:
                    producer += ["--case-indices", ",".join(map(str, missing))]
                producer += ["--reference-only", "--output", str(output)]
                self._fence(deadline_at, lease_valid)
                outcome = run_script(
                    producer,
                    str(root),
                    device_id,
                    deadline_at,
                    lease_valid,
                    budget_seconds=max(1, math.ceil(remaining_seconds(deadline_at))),
                )
                if outcome.outcome == "fenced":
                    raise LeaseLostError("baseline engine lost its lease")
                self._fence(deadline_at, lease_valid)
                if outcome.timed_out:
                    raise TimeoutError("baseline measurement timed out")
                if outcome.exit_code != 0:
                    raise ValueError("reference-only engine failed")
                envelope = read_json(output, root=root)
                total, measured = self._validate_reference(envelope, missing, params)
                if snapshot is not None and snapshot.total_cases != total:
                    raise ValueError("baseline inventory count changed")
                if selected is None:
                    selected = list(range(1, total + 1))
                self._fence(deadline_at, lease_valid)
                cache.publish(key, total, measured)
        self._fence(deadline_at, lease_valid)
        return {
            str(index): (existing[index] if index in hits else measured[index])
            for index in selected
        }, hits

    @staticmethod
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

    @staticmethod
    def _stamp_result(result, timings, hits, selected):
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
        if expected is not None and set(indices) != expected:
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
