#!/usr/bin/env python
"""Run readiness and elementwise-add correctness smoke checks on an Ascend host.

This limited gate checks two add workloads and execution provenance. It does
not cover the full correctness taxonomy, timing stability, profile accuracy,
worker crash recovery or a crash soak. Passing is not full production calibration.
The report records the target's live OpenAPI digest and bounded check scope;
it never records capability tokens, source code or input values.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract_identity(service_url: str) -> dict:
    """Identify the target's current HTTP schema without a local schema artifact."""
    with urllib.request.urlopen(f"{service_url.rstrip('/')}/openapi.json", timeout=10) as response:
        schema = json.load(response)
    if not isinstance(schema, dict) or not schema.get("openapi") or not schema.get("paths"):
        raise ValueError("target did not return an OpenAPI schema")
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "source": "target-openapi",
        "openapi_version": schema["openapi"],
        "openapi_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = {
        "passed": False,
        "scope": "readiness-and-add-correctness-smoke",
        "service_url": args.service_url,
        "calibration": {"passed": False},
        "contract": None,
    }
    try:
        environment = dict(os.environ)
        environment.update({
            "TRITON_EVAL_SERVICE_URL": args.service_url,
            "TRITON_EVAL_CAPABILITY_TOKEN": _read_token(args.capability_file),
        })
        report["contract"] = _contract_identity(args.service_url)
        passed, log = _run_calibration_with(environment)
        report["passed"] = passed
        report["calibration"]["passed"] = passed
    except (OSError, ValueError, urllib.error.URLError):
        # HTTP errors may contain response bodies; do not copy them into the report.
        log = ["gate setup failed: check capability file and target OpenAPI availability"]
    _write_atomic(Path(args.output), report)
    if not report["passed"]:
        for line in log[-20:]:
            print(line, file=sys.stderr)
        return 1
    return 0


def _read_token(path: str) -> str:
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("capability file is empty")
    return token


def _run_calibration_with(environment) -> tuple[bool, list[str]]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_npu_release_calibration.py", "-m", "release_gate",
             "-q", "--no-header", "--tb=short"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, env=environment,
            timeout=720,
        )
    except subprocess.TimeoutExpired:
        return False, ["NPU smoke checks exceeded the gate timeout"]
    summary = re.search(r"\b(\d+) passed\b", result.stdout)
    skipped = re.search(r"\b\d+ skipped\b", result.stdout)
    passed = result.returncode == 0 and summary is not None and int(summary[1]) >= 2
    return passed and skipped is None, result.stdout.splitlines()


if __name__ == "__main__":
    sys.exit(main())
