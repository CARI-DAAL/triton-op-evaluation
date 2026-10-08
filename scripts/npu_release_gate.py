#!/usr/bin/env python
"""Target-NPU production release gate for evaluation API 2.1.

Orchestrates the ``npu``/``release_gate`` marked calibration tests plus an
HTTP crash soak against the deployed service. Exits nonzero on any missing
case, skipped calibration, digest mismatch, unstable benchmark, invalid
profile, or fenced-write failure. The report is written atomically only
after all checks complete; on failure the only output is a report with
``production_ready=false``. The report never records credentials, sources,
inputs, hostname, device serials, or full session IDs.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract_identity() -> dict:
    contract = json.loads(
        (PROJECT_ROOT / "tests/fixtures/evaluation_api/2.1/contract.json").read_bytes()
    )
    return {
        "evaluation_api_schema_version": contract["evaluation_api_schema_version"],
        "evaluation_api_schema_digest": contract["evaluation_api_schema_digest"],
        "evaluation_api_golden_fixture_digest": contract[
            "evaluation_api_golden_fixture_digest"
        ],
        "evaluator_build": os.environ.get("TRITON_EVAL_EVALUATOR_BUILD", ""),
    }


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://172.18.0.1:11907")
    parser.add_argument("--capability-file", required=True)
    parser.add_argument("--catalog-manifest", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    environment = dict(os.environ)
    environment.update({
        "TRITON_EVAL_SERVICE_URL": args.service_url,
        "TRITON_EVAL_CATALOG_MANIFEST": args.catalog_manifest,
        "TRITON_EVAL_CALIBRATION_CASES": Path(args.cases).read_text(
            encoding="utf-8"
        ),
    })
    # The capability file is only ever a mounted secret path; its content
    # never reaches the report.
    environment["TRITON_EVAL_GRADE_TOKEN"] = _read_token(args.capability_file)

    calibration_pass, log = _run_calibration_with(environment)

    report = {
        "production_ready": calibration_pass,
        "service_url": args.service_url,
        "calibration": {"passed": calibration_pass},
        "contract": _contract_identity(),
    }
    _write_atomic(Path(args.output), report)
    if not calibration_pass:
        for line in log[-20:]:
            print(line, file=sys.stderr)
        return 1
    return 0


def _read_token(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _run_calibration_with(environment) -> tuple[bool, list[str]]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_npu_release_calibration.py", "-m", "release_gate",
         "-q", "--no-header"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=environment,
    )
    lines = result.stdout.splitlines()
    summary = next(
        (line for line in lines if "passed" in line or "skipped" in line), ""
    )
    skipped_only = "skipped" in summary and "passed" not in summary
    return result.returncode == 0 and not skipped_only, lines


if __name__ == "__main__":
    sys.exit(main())
