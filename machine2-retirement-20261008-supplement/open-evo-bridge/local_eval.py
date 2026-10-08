#!/usr/bin/env python3
"""Run one evaluation on a locally claimed NPU, using the service's own
orchestrator so trial-side measurements are byte-comparable with grade-side.

Runs INSIDE the execution container (``cc``). The device is already selected by
``claim_and_run.sh`` (flock pool) and exported as ``ASCEND_RT_VISIBLE_DEVICES``;
this script never picks a device itself.

The evaluator orchestration is imported from the service checkout rather than
reimplemented: decoy pre-check, compile gating, correctness -> performance ->
profile ordering, shaping and provenance all come from the same code the grade
path runs. Only the execution *site* differs (this process instead of a service
worker), which is exactly the split the measurement design allows.

Output is the same EvaluationResult document the evaluation API returns, so the
caller can render observations with one renderer for both paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE_ROOT = os.environ.get(
    "LOCAL_EVAL_SERVICE_ROOT", "/home/e84381970/triton-op-evaluation"
)
sys.path.insert(0, SERVICE_ROOT)

from triton_evaluation.api.contract import EvaluationRequest  # noqa: E402
from triton_evaluation.config import Settings  # noqa: E402
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator  # noqa: E402
from triton_evaluation.execution_environment import (  # noqa: E402
    build_execution_environment,
)
from triton_evaluation.models import AlgorithmVersions, PinnedProvenance  # noqa: E402

TRITON_IMPL_NAME = "triton_ascend_impl"


class _Sink:
    """Minimal ProgressSink: local runs need no fencing."""

    def begin(self, check):
        return True

    def save(self, check, outcome):
        return True


def _load_env_file() -> None:
    """Apply the deployment env file the service itself runs with.

    ``Settings`` reads process environment variables, and the service is
    normally started from a shell that sourced this file; a standalone
    invocation must do the same or the provenance build id silently falls back
    to its placeholder.
    """
    path = os.environ.get("TRITON_EVAL_ENV_FILE")
    if not path or not os.path.isfile(path):
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _claimed_device() -> int:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    first = raw.split(",")[0].strip()
    if not first:
        raise SystemExit(
            "ASCEND_RT_VISIBLE_DEVICES is empty: local_eval.py must run under "
            "claim_and_run.sh so a device is claimed for its whole duration"
        )
    return int(first)


async def _run(args) -> dict:
    device_id = _claimed_device()
    _load_env_file()
    settings = Settings.from_env()
    engine_dir = str(Path(SERVICE_ROOT) / "triton_evaluation" / "engine")

    reference = Path(args.reference).read_text(encoding="utf-8")
    kernel = Path(args.kernel).read_text(encoding="utf-8")
    checks = [name for name in args.checks.split(",") if name]

    request = EvaluationRequest.model_validate(
        {
            "schema_version": "2.1",
            "op_name": args.op_name,
            "reference_code": reference,
            "kernel_code": kernel,
            "triton_impl_name": TRITON_IMPL_NAME,
            "checks": checks,
            "non_compute": bool(args.non_compute),
            "deadline_seconds": int(args.timeout),
            "performance": json.loads(args.performance) if args.performance else None,
            "profile": json.loads(args.profile) if args.profile else None,
        }
    )
    pinned = PinnedProvenance.for_request(
        request,
        evaluator_build=settings.evaluator_build,
        algorithms=AlgorithmVersions(
            correctness="correctness-v1",
            decoy="decoy-v1",
            benchmark="geometric-mean-v1",
            profiler="profiler-v1",
        ),
    )
    evaluator = EvaluationEvaluator(engine_dir, settings)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    deadline_at = datetime.now(timezone.utc) + timedelta(seconds=int(args.timeout))

    result = evaluator.run(
        request,
        work_dir,
        device_id,
        deadline_at,
        lambda: True,
        build_execution_environment(settings),
        pinned,
        _Sink(),
        precheck=None,
    )
    return result.model_dump(mode="json", exclude_none=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checks", default="correctness")
    parser.add_argument("--performance", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--non-compute", action="store_true")
    parser.add_argument("--timeout", type=int, default=840)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    document = asyncio.run(_run(args))
    Path(args.output).write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
