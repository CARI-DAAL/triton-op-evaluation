#!/usr/bin/env python3
"""Run the official 51-case BatchMatmul gate through the public HTTP API."""

# Direct execution needs the checkout before project imports.
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_eval_service import http_json
from triton_evaluation.api.contract import EvaluationResource
from triton_evaluation.api.operator import OperatorRequest
from triton_evaluation.evaluators.operator import to_evaluation_request

FIXTURE = ROOT / "tests/fixtures/npu_kernelbench/batchmatmul"
OFFICIAL_DIGEST = "57e902365bd9b2fa437cc500cf0d862989b5881f439b2157de7341b127126c73"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture_check(fixture=FIXTURE):
    manifest = json.loads((fixture / "manifest.json").read_text())
    for name, pin in manifest["files"].items():
        data = (fixture / name).read_bytes()
        if sha(data) != pin["sha256"] or len(data) != pin["size_bytes"]:
            raise ValueError(f"fixture integrity failed: {name}")
    definition_bytes = (fixture / "definition.json").read_bytes()
    workload_bytes = (fixture / "workload.jsonl").read_bytes()
    digest = sha(definition_bytes + b"\0" + workload_bytes)
    definition = json.loads(definition_bytes)
    workloads = [json.loads(line) for line in workload_bytes.splitlines()]
    if digest != OFFICIAL_DIGEST or digest != manifest["official_digest"]["value"]:
        raise ValueError("official combined digest mismatch")
    if manifest["case_count"] != 51 or len(workloads) != 51:
        raise ValueError("official case count must be 51")
    if list(definition["inputs"]) != ["A", "B"] or manifest["input_order"] != ["A", "B"]:
        raise ValueError("official input order must be A, B")
    return manifest, definition, workloads


def build_body(definition, workloads, stage, deadline_seconds, fixture=FIXTURE):
    body = {
        "definition": definition,
        "workloads": workloads,
        "triton_code": (fixture / "candidate.py").read_text()
        + "\n\ndef run(A, B):\n    return _bmm(A, B)\n",
        "seed": 200,
        "custom_inputs_code": None,
        "deadline_seconds": deadline_seconds,
    }
    if stage == "correctness":
        body["checks"] = ["correctness"]
    elif stage.startswith("performance_"):
        body["checks"] = ["correctness", "performance"]
        body["performance"] = {
            "method": stage.removeprefix("performance_"),
            "use_baseline_cache": False,
        }
    else:
        body["checks"] = ["profile"]
        body["profile"] = {"mode": stage.removeprefix("profile_"), "case_indices": list(range(51))}
    OperatorRequest.model_validate(body)
    return body


def reference_digest(body):
    request = to_evaluation_request(OperatorRequest.model_validate(body))
    return sha(request.reference_code.encode())


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def validate_result(resource, stage):
    if resource.state != "completed" or resource.result is None:
        return "failed"
    result = resource.result
    if stage == "correctness" or stage.startswith("performance_"):
        outcome = result.correctness
        if (
            outcome is None
            or outcome.status != "succeeded"
            or not outcome.data.passed
            or outcome.data.decoy
            or outcome.data.total_cases != 51
            or outcome.data.passed_cases != 51
            or outcome.data.failures
        ):
            return "failed"
    if stage.startswith("performance_"):
        outcome = result.performance
        if outcome is None or outcome.status != "succeeded":
            return "failed"
        data = outcome.data
        tag = "npu_event_v1" if stage.endswith("event") else "profiler_kernel_sum_v1"
        if (
            data.selected_cases != 51
            or data.measured_cases != 51
            or data.coverage != 1
            or data.unmeasured_cases
            or data.measurement_method != tag
            or {case.case_idx for case in data.per_case} != set(range(51))
        ):
            return "failed"
        if any(
            not positive(case.implementation_latency_ms)
            or not positive(case.reference_latency_ms)
            or case.measurement_method != tag
            or case.baseline_cache_hit
            for case in data.per_case
        ):
            return "failed"
    if stage.startswith("profile_"):
        outcome = result.profile
        if outcome is None or outcome.status != "succeeded":
            return "failed"
        data = outcome.data
        mode = stage.removeprefix("profile_")
        if (
            data.mode != mode
            or data.counts.selected != 51
            or data.counts.failed
            or {case.case_idx for case in data.cases} != set(range(51))
        ):
            return "failed"
        if data.counts.partial:
            return "partial"
        if data.counts.complete != 51:
            return "failed"
        for case in data.cases:
            if case.status != "complete" or case.error:
                return "failed"
            if mode == "op":
                if not case.kernels or any(not positive(kernel.time_ms) for kernel in case.kernels):
                    return "failed"
            elif (
                not case.target_kernel
                or case.bound is None
                or case.missing_metrics
                or not any(value is not None for value in case.ratios.model_dump().values())
            ):
                return "failed"
    return "complete"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def redact(value, token):
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, dict):
        return {redact(key, token): redact(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    return value


def save_report(output, report, token=""):

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(redact(report, token)), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(output)


def run_gate(
    url,
    token,
    output,
    *,
    deadline_seconds=3600,
    job_timeout=19000,
    transport=http_json,
    sleep=time.sleep,
    fixture=FIXTURE,
    check_only=False,
):
    report = {
        "passed": False,
        "scope": "fixture-integrity-only"
        if check_only
        else "HTTP engine official 51-case BatchMatmul",
        "hardware_verified": False,
        "checks": {},
    }
    save_report(output, report, token)
    try:
        manifest, definition, workloads = fixture_check(fixture)
        report.update(
            official_digest=OFFICIAL_DIGEST,
            official_commit=manifest["official_source"]["commit"],
            fixture_hashes=manifest["files"],
            case_count=51,
        )
        if check_only:
            report["passed"] = True
            return report
        report.update(
            seed=200,
            generator_policy="official typed HTTP input generation",
            device="npu",
            deployment_requirement="npu-strict reference device policy",
        )
        headers = {"Authorization": "Bearer " + token, "X-Session-ID": uuid4().hex}
        stop = time.monotonic() + job_timeout
        for stage in (
            "correctness",
            "performance_event",
            "performance_profiler",
            "profile_op",
            "profile_deep",
        ):
            if time.monotonic() >= stop:
                raise TimeoutError("gate job timeout expired")
            body = build_body(definition, workloads, stage, deadline_seconds, fixture)
            expected = reference_digest(body)
            route = (
                "/verify"
                if stage == "correctness"
                else "/benchmark"
                if stage.startswith("performance_")
                else "/profile"
            )
            stage_headers = {**headers, "Idempotency-Key": stage + "-" + uuid4().hex}
            entry = {
                "status": "submitting",
                "expected_reference_digest": expected,
                "session_id": headers["X-Session-ID"],
                "idempotency_key": stage_headers["Idempotency-Key"],
                "route": route,
                "deadline_seconds": deadline_seconds,
            }
            report["checks"][stage] = entry
            save_report(output, report, token)
            for attempt in range(2):
                if time.monotonic() >= stop:
                    raise TimeoutError("gate job timeout expired before POST replay")
                try:
                    raw = transport(
                        "POST",
                        url.rstrip("/") + route,
                        body=body,
                        headers=stage_headers,
                        timeout=min(15, stop - time.monotonic()),
                    )
                    break
                except urllib.error.HTTPError:
                    raise  # A server response is not an ambiguous transport loss.
                except (urllib.error.URLError, OSError):
                    if attempt == 1:
                        raise
                    entry["post_retries"] = 1
                    save_report(output, report, token)
                    sleep(min(1, max(0, stop - time.monotonic())))
            # Preserve the admitted handle before parsing the complete typed envelope.
            entry.update(evaluation_id=raw.get("evaluation_id"), resource=raw, status="accepted")
            save_report(output, report, token)
            while True:
                resource = EvaluationResource.model_validate_json(json.dumps(raw))
                if resource.reference_digest != expected:
                    raise ValueError("deployed service reference/input protocol digest differs")
                if resource.evaluation_id != entry["evaluation_id"]:
                    raise ValueError("poll returned a different evaluation identity")
                if resource.state in ("completed", "failed", "timed_out"):
                    break
                if time.monotonic() >= stop:
                    raise TimeoutError("gate job timeout expired; admitted handle saved")
                sleep(min(1, max(0, stop - time.monotonic())))
                raw = transport(
                    "GET",
                    url.rstrip("/") + "/evaluations/" + resource.evaluation_id,
                    headers=headers,
                    timeout=max(0.01, min(15, stop - time.monotonic())),
                )
                entry["resource"] = raw
                save_report(output, report, token)
            provenance = resource.result.provenance
            if provenance.execution_environment is None:
                raise ValueError("server omitted execution environment evidence")
            if ":npu-strict-v1" not in provenance.correctness_algorithm:
                raise ValueError("server does not declare the required npu-strict reference policy")
            if (
                report.get("evaluator_build", provenance.evaluator_build)
                != provenance.evaluator_build
            ):
                raise ValueError("evaluator build changed across gate stages")
            report["evaluator_build"] = provenance.evaluator_build
            report["execution_environment"] = provenance.execution_environment.model_dump(
                mode="json"
            )
            report["declared_reference_policy"] = "npu-strict-v1"
            entry["status"] = validate_result(resource, stage)
            entry["evaluator_build"] = resource.result.provenance.evaluator_build
            save_report(output, report, token)
            if entry["status"] != "complete":
                break
        report["passed"] = len(report["checks"]) == 5 and all(
            entry["status"] == "complete" for entry in report["checks"].values()
        )
        report["hardware_verified"] = report["passed"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report = redact(report, token)
        save_report(output, report, token)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11907")
    parser.add_argument("--capability-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    parser.add_argument("--job-timeout", type=int, default=19000)
    args = parser.parse_args()
    if not 10 <= args.deadline_seconds <= 3600 or args.job_timeout <= 0:
        parser.error("deadline-seconds must be 10..3600 and job-timeout positive")
    if not args.check_only and not args.capability_file:
        parser.error("--capability-file is required for HTTP execution")
    token = args.capability_file.read_text().strip() if args.capability_file else ""
    if not args.check_only and not token:
        parser.error("capability file must contain a token")
    report = run_gate(
        args.url,
        token,
        args.output,
        deadline_seconds=args.deadline_seconds,
        job_timeout=args.job_timeout,
        check_only=args.check_only,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
