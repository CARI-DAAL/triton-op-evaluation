#!/usr/bin/env python3
"""Smoke-test the durable operator endpoint end-to-end.

Submits a known-good Ascend smoke op (elementwise add: a torch reference + a
Triton kernel) once to ``POST /evaluations`` and prints a PASS/FAIL verdict.
The client polls the admitted resource until it reaches a terminal result.

Stdlib only — runs from the host against the service (the NPU work happens
inside the triton-eval container):

    python3 scripts/test_eval_service.py [--url http://127.0.0.1:11907] [--no-speed]

Exit code: 0 = correctness passed, 1 = a valid terminal non-pass result,
2 = service/transport error or malformed response.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Elementwise add with two explicit workloads, using the public run(*args) interface.
REFERENCE_CODE = """import torch

def run(x, y):
    return x + y
"""

KERNEL_CODE = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def run(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
"""


def build_request(*, measure_speed: bool = True) -> dict:
    """Build the public operator request shared by host and NPU smoke checks."""
    body = {
        "torch_code": REFERENCE_CODE,
        "triton_code": KERNEL_CODE,
        "cases": [
            {"inputs": [{"shape": [size], "dtype": "float32"} for _ in range(2)], "seed": 0}
            for size in (4096, 8192)
        ],
        "checks": ["correctness", "performance"] if measure_speed else ["correctness"],
        "deadline_seconds": 600,
    }
    if measure_speed:
        body["performance"] = {"warmup": 5, "repeats": 20}
    return body


def http_json(
    method: str,
    url: str,
    body: dict | None = None,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
):
    data = json.dumps(body).encode() if body is not None else None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def response_exit_code(resource: dict) -> int:
    """Map a terminal durable resource to the smoke-test process exit code."""
    state = resource.get("state")
    if state not in {"completed", "failed", "timed_out"}:
        return 2
    if state != "completed" or resource.get("result") is None:
        return 1
    correctness = (resource.get("result") or {}).get("correctness") or {}
    if correctness.get("status") != "succeeded" or not isinstance(
        correctness.get("data"), dict
    ):
        return 2
    return 0 if correctness["data"].get("passed") else 1


def _print_resource(resource: dict) -> None:
    result = resource.get("result") or {}
    print(f"\n[result] state={resource.get('state')} "
          f"evaluation_id={resource.get('evaluation_id')}")
    correctness = result.get("correctness") or {}
    data = correctness.get("data") or {}
    if data:
        print(f"  cases: {data.get('passed_cases')}/{data.get('total_cases')} passed")
        for failure in data.get("failures") or []:
            print(f"  [case {failure.get('case_idx')}] {failure.get('kind')}")
    performance = result.get("performance") or {}
    perf_data = performance.get("data") or {}
    if perf_data:
        print(f"  speedup_vs_reference: "
              f"{perf_data.get('overall_speedup_vs_reference')} "
              f"(coverage {perf_data.get('coverage')})")
    if correctness.get("error"):
        print(f"  error: {correctness['error'].get('message')}")


def submit_and_recover(base: str, body: dict, headers: dict, timeout: float) -> dict:
    """POST once; on no-response transport failure replay the original request.

    The evaluation_id is persisted immediately after a 200/202 body — the only
    safe replay is when NO response was obtained at all.
    """
    for attempt in range(2):
        try:
            return http_json("POST", f"{base}/evaluations", body=body,
                             timeout=timeout, headers=headers)
        except (urllib.error.URLError, OSError):
            if attempt == 1:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")


def poll_until_terminal(base: str, evaluation_id: str, headers: dict,
                        deadline_seconds: float, max_backoff: float = 5.0) -> dict:
    """GET with capped exponential backoff until a terminal resource appears."""
    backoff = 0.5
    deadline = time.monotonic() + deadline_seconds
    while True:
        resource = http_json("GET", f"{base}/evaluations/{evaluation_id}",
                             headers=headers, timeout=15.0)
        if resource.get("state") in {"completed", "failed", "timed_out"}:
            return resource
        if time.monotonic() >= deadline:
            return resource
        time.sleep(min(backoff, max_backoff))
        backoff = min(backoff * 2, max_backoff)


def main() -> int:
    import uuid

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11907",
                    help="base URL of the triton-eval service")
    ap.add_argument("--token", required=True,
                    help="caller-scoped opaque capability token (Bearer)")
    ap.add_argument("--measure-speed", dest="measure_speed", action="store_true",
                    default=True, help="also benchmark a passing kernel (default on)")
    ap.add_argument("--no-speed", dest="measure_speed", action="store_false")
    ap.add_argument("--job-timeout", type=int, default=600,
                    help="client recovery window in seconds (the resource deadline "
                         "is set by the request body)")
    ap.add_argument("--session-id", default="triton-eval-smoke",
                    help="correlation session (never authorizes anything)")
    args = ap.parse_args()

    base = args.url.rstrip("/")

    # 0) liveness/readiness
    try:
        print(f"[livez] {http_json('GET', f'{base}/livez', timeout=5.0)}")
        print(f"[readyz] {http_json('GET', f'{base}/readyz', timeout=5.0)}")
    except Exception as e:
        print(f"[health] FAILED: {e}", file=sys.stderr)
        return 2

    headers = {
        "Authorization": f"Bearer {args.token}",
        "X-Session-ID": args.session_id,
        "Idempotency-Key": f"smoke-{uuid.uuid4().hex}",
    }
    body = build_request(measure_speed=args.measure_speed)

    print(f"\n[evaluate] POST /evaluations checks={body['checks']}")
    try:
        # Client HTTP timeout covers only wait_seconds plus transport margin;
        # recovery after that goes through GET with the persisted ID.
        resource = submit_and_recover(base, body, headers, timeout=15.0)
    except Exception as e:
        print(f"[evaluate] FAILED: {e}", file=sys.stderr)
        return 2

    if not isinstance(resource, dict) or not resource.get("evaluation_id"):
        print("[evaluate] malformed response: no evaluation_id", file=sys.stderr)
        return 2
    evaluation_id = resource["evaluation_id"]
    print(f"  evaluation_id={evaluation_id} state={resource.get('state')}")

    if resource.get("state") not in {"completed", "failed", "timed_out"}:
        resource = poll_until_terminal(base, evaluation_id, headers,
                                       deadline_seconds=float(args.job_timeout))

    _print_resource(resource)
    return response_exit_code(resource)


if __name__ == "__main__":
    sys.exit(main())
