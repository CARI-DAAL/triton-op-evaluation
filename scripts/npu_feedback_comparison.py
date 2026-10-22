#!/usr/bin/env python3
"""Compare separate timing/trace requests with one joint request on an NPU service.

No NPU code executes locally. Run against a single-card instance to keep device
conditions comparable. Both strategies verify once, use fresh Torch baselines,
and collect trace with the same warmup=5/repeats=20 protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.test_eval_service import build_request  # noqa: E402

TERMINAL = {"completed", "failed", "timed_out"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def requests_for(body, method, indices=None):
    base = {key: copy.deepcopy(body[key]) for key in (
        "definition", "workloads", "triton_code", "seed", "custom_inputs_code", "deadline_seconds",
        "non_compute",
    ) if key in body}
    if not base.get("triton_code") or not base["definition"].get("reference"):
        raise ValueError("comparison requires reference and Triton source")
    performance = {"method": method, "warmup": 5, "repeats": 20, "use_baseline_cache": False}
    profile = {"mode": "trace"}
    if indices is not None:
        selected = sorted(indices)
        if (not selected or len(set(selected)) != len(selected) or min(selected) < 0
                or max(selected) >= len(base["workloads"])):
            raise ValueError("case selection is outside the workload inventory")
        performance["case_indices"] = selected
        profile["case_indices"] = selected
    return {
        "timing": {**base, "checks": ["correctness", "performance"], "performance": performance},
        "trace": {**base, "checks": ["profile"], "profile": profile},
        "combined": {
            **base, "checks": ["correctness", "performance", "profile"],
            "performance": performance, "profile": profile,
        },
    }


def validate_result(resource, checks, expected_cases, method, total_cases):
    if resource.get("state") != "completed":
        raise ValueError("evaluation did not complete")
    result = resource.get("result") or {}
    provenance = result.get("provenance") or {}
    if not provenance.get("evaluator_build") or not provenance.get("execution_environment"):
        raise ValueError("result has no execution provenance")
    for check in checks:
        outcome = result.get(check) or {}
        if outcome.get("status") != "succeeded":
            raise ValueError(f"{check} did not produce successful evidence")
    if "correctness" in checks:
        data = result["correctness"]["data"]
        if (data.get("outcome") != "correct" or data.get("passed") is not True
                or data.get("total_cases") != total_cases
                or data.get("passed_cases") != total_cases):
            raise ValueError("whole-task correctness did not pass")
    if "performance" in checks:
        data = result["performance"]["data"]
        expected_method = "npu_event_v1" if method == "event" else "profiler_kernel_sum_v1"
        cases = data.get("per_case", [])
        if (data.get("measurement_method") != expected_method
                or {case.get("case_idx") for case in cases} != set(expected_cases)
                or len(cases) != len(expected_cases)
                or data.get("unmeasured_cases")):
            raise ValueError("timing method or coverage differs from the request")
        if any(case.get("baseline_cache_hit") is not False for case in cases):
            raise ValueError("comparison did not use a fresh Torch baseline")
        for case in cases:
            for field in ("implementation_latency_ms", "reference_latency_ms"):
                value = case.get(field)
                if (type(value) not in {float, int} or not math.isfinite(value) or value <= 0):
                    raise ValueError("timing contains invalid latency")
    if "profile" in checks:
        data = result["profile"]["data"]
        cases = data.get("cases", [])
        if (data.get("mode") != "trace" or data.get("total_cases") != total_cases
                or data.get("measurement_method") != "profiler_kernel_sum_v1"
                or {case.get("case_idx") for case in cases} != set(expected_cases)
                or len(cases) != len(expected_cases)
                or data.get("counts") != {
                    "selected": len(expected_cases), "complete": len(expected_cases),
                    "partial": 0, "failed": 0,
                }
                or any(case.get("status") != "complete" for case in cases)):
            raise ValueError("trace evidence is incomplete or has different coverage")
        for case in cases:
            kernels = case.get("kernels") or []
            names = [kernel.get("op_name") for kernel in kernels]
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("trace has missing or duplicate kernel groups")
            times = [kernel.get("time_ms") for kernel in kernels]
            if any(type(value) not in {float, int} or not math.isfinite(value) or value < 0
                   for value in times) or sum(times) <= 0:
                raise ValueError("trace has invalid kernel durations")
    return result


class Service:
    def __init__(self, url, token, report, output, poll_seconds):
        self.url = url.rstrip("/")
        self.token = token
        self.report, self.output, self.poll_seconds = report, output, poll_seconds
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, body=None, *, key=None):
        headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}
        if key is not None:
            headers["Idempotency-Key"] = key
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        # Replaying a POST retains the exact key and body.
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=15) as response:
                    return json.load(response)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def run(self, body, record):
        key = "feedback-" + uuid4().hex
        record.update(idempotency_key=key, url=self.url)
        save_report(self.output, self.report)
        started = time.monotonic()
        resource = self.request("POST", "/evaluations", body, key=key)
        evaluation_id = resource["evaluation_id"]
        record["evaluation_id"] = evaluation_id
        save_report(self.output, self.report)
        deadline = started + body.get("deadline_seconds", 3600) + 60
        while resource["state"] not in TERMINAL:
            if time.monotonic() >= deadline:
                raise TimeoutError("client wait expired; recover the recorded evaluation handle")
            if "performance" in body["checks"] and "time_to_performance_seconds" not in record:
                progress = self.request("GET", f"/evaluations/{evaluation_id}/progress")
                evidence = progress.get("completed_checks", {}).get("performance")
                if evidence is not None:
                    record["time_to_performance_seconds"] = time.monotonic() - started
                    record["performance_attempt_id"] = progress.get("attempt_id")
                    record["performance_progress_revision"] = progress.get("progress_revision")
            time.sleep(self.poll_seconds)
            resource = self.request("GET", f"/evaluations/{evaluation_id}")
        record["wall_seconds"] = time.monotonic() - started
        # Do not attribute a replaced attempt's early result to the final
        # attempt's first-feedback latency.
        if "performance_attempt_id" in record:
            final_progress = self.request("GET", f"/evaluations/{evaluation_id}/progress")
            record["final_progress_revision"] = final_progress["progress_revision"]
            record["final_attempt_id"] = final_progress.get("attempt_id")
            record["feedback_attempt_replaced"] = (
                record["final_attempt_id"] != record["performance_attempt_id"]
            )
            if record["feedback_attempt_replaced"]:
                record["time_to_performance_seconds"] = record["wall_seconds"]
        if "performance" in body["checks"]:
            record.setdefault("time_to_performance_seconds", record["wall_seconds"])
        record["resource"] = resource
        save_report(self.output, self.report)
        return resource


def summarize(rounds):
    measurements = {"separate": [], "combined": []}
    first_feedback = {"separate": [], "combined": []}
    for pair in rounds:
        if pair["warmup"] or any(
            pair[strategy].get("accepted") is not True for strategy in measurements
        ):
            continue
        for strategy in measurements:
            run = pair[strategy]
            if run.get("accepted") is not True:
                continue
            measurements[strategy].append(run["wall_seconds"])
            first_feedback[strategy].append(run["time_to_performance_seconds"])
    result = {}
    for strategy, values in measurements.items():
        if values:
            result[strategy] = {
                "n": len(values), "wall_seconds": values,
                "median_wall_seconds": statistics.median(values),
                "median_time_to_performance_seconds": statistics.median(first_feedback[strategy]),
            }
    if len(result) == 2:
        result["separate_over_combined_wall_ratio"] = (
            result["separate"]["median_wall_seconds"] / result["combined"]["median_wall_seconds"]
        )
    return result


def compare_measurements(pair, max_relative_drift=None):
    """Report measurement changes independently of service wall-clock savings."""
    separate = pair["separate"]["requests"]
    combined = pair["combined"]["requests"][0]["resource"]["result"]
    timing = separate[0]["resource"]["result"]["performance"]["data"]
    trace = separate[1]["resource"]["result"]["profile"]["data"]
    combined_timing = combined["performance"]["data"]
    combined_trace = combined["profile"]["data"]
    metrics = []
    new_cases = {case["case_idx"]: case for case in combined_timing["per_case"]}
    old_trace = {case["case_idx"]: case for case in trace["cases"]}
    new_trace = {case["case_idx"]: case for case in combined_trace["cases"]}
    for case in timing["per_case"]:
        index = case["case_idx"]
        other = new_cases[index]
        entry = {"case_idx": index}
        for field in ("implementation_latency_ms", "reference_latency_ms"):
            ratio = other[field] / case[field]
            entry[field + "_combined_over_separate"] = ratio
        entry["trace_total_ms_combined_over_separate"] = (
            sum(kernel["time_ms"] for kernel in new_trace[index]["kernels"])
            / sum(kernel["time_ms"] for kernel in old_trace[index]["kernels"])
        )
        entry["same_kernel_names"] = (
            {k["op_name"] for k in old_trace[index]["kernels"]}
            == {k["op_name"] for k in new_trace[index]["kernels"]}
        )
        ratios = [value for key, value in entry.items() if key.endswith("_over_separate")]
        entry["within_requested_drift"] = (
            None if max_relative_drift is None
            else all(abs(ratio - 1) <= max_relative_drift for ratio in ratios)
        )
        metrics.append(entry)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--capability-file", required=True, type=Path)
    parser.add_argument("--request", type=Path, help="Operator request JSON; defaults to add smoke")
    parser.add_argument("--method", choices=("event", "profiler"), default="event")
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    parser.add_argument(
        "--max-relative-drift", type=float,
        help="Optional acceptance bound for timing/trace drift; otherwise report only",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.pairs < 1 or args.warmup_pairs < 0 or not 0.05 <= args.poll_seconds <= 10
            or not 10 <= args.deadline_seconds <= 3600
            or (args.max_relative_drift is not None
                and (not math.isfinite(args.max_relative_drift) or args.max_relative_drift < 0))):
        parser.error("invalid repetitions, polling interval or deadline")
    body = json.loads(args.request.read_text()) if args.request else build_request()
    body["deadline_seconds"] = args.deadline_seconds
    bodies = requests_for(body, args.method, args.cases)
    selected = sorted(args.cases) if args.cases is not None else list(range(len(body["workloads"])))
    report = {
        "scope": "same-build-separate-versus-joint-timing-trace",
        "url": args.url, "measurement_method": args.method,
        "warmup": 5, "repeats": 20, "baseline_cache": False,
        "request_sha256": hashlib.sha256(
            json.dumps(bodies, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "poll_seconds": args.poll_seconds, "passed": False, "rounds": [],
        "max_relative_drift": args.max_relative_drift,
    }
    token = args.capability_file.read_text().strip()
    service = Service(args.url, token, report, args.output, args.poll_seconds)
    try:
        for index in range(args.warmup_pairs + args.pairs):
            pair = {"warmup": index < args.warmup_pairs, "separate": {}, "combined": {}}
            order = ["separate", "combined"] if index % 2 == 0 else ["combined", "separate"]
            pair["order"] = order
            report["rounds"].append(pair)
            for strategy in order:
                entry = pair[strategy]
                entry["requests"] = []
                started = time.monotonic()
                for stage in (["timing", "trace"] if strategy == "separate" else ["combined"]):
                    record = {"stage": stage}
                    entry["requests"].append(record)
                    resource = service.run(bodies[stage], record)
                    evidence = validate_result(
                        resource, bodies[stage]["checks"], selected, args.method,
                        len(body["workloads"]),
                    )
                    environment = evidence["provenance"]
                    identity = {
                        name: environment[name]
                        for name in ("evaluator_build", "execution_environment")
                    }
                    if report.setdefault("execution_identity", identity) != identity:
                        raise ValueError("comparison crossed evaluator or environment identities")
                    if stage in {"timing", "combined"}:
                        entry["time_to_performance_seconds"] = record["time_to_performance_seconds"]
                entry["wall_seconds"] = time.monotonic() - started
                entry["accepted"] = True
                save_report(args.output, report)
            pair["measurement_comparison"] = compare_measurements(pair, args.max_relative_drift)
            if any(item["within_requested_drift"] is False
                   for item in pair["measurement_comparison"]):
                raise ValueError("measurements exceed the caller's drift bound")
        report["summary"] = summarize(report["rounds"])
        report["passed"] = True
    except Exception as error:
        # Keep recovery IDs and results; never echo source, token or response bodies.
        report["error"] = {"type": type(error).__name__, "message": "comparison did not complete"}
        report["summary"] = summarize(report["rounds"])
    save_report(args.output, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
