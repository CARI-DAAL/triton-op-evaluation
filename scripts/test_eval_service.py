#!/usr/bin/env python3
"""Smoke-test the synchronous evaluation v2 endpoint end-to-end.

Submits a known-good Ascend smoke op (elementwise add: a torch reference + a
Triton kernel) once to ``POST /evaluations`` and prints a PASS/FAIL verdict.
The service owns queue polling; this client waits for the terminal typed result.

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
import urllib.error
import urllib.request

# Known-good Ascend smoke case: elementwise add. reference_code defines a torch
# Model + get_init_inputs/get_input_groups; kernel_code defines a triton kernel
# + ModelNew with a matching forward(). (Same shape as tests/msprof_smoke_kernels.py.)
REFERENCE_CODE = """import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_input_groups():
    return [
        [torch.randn(4096), torch.randn(4096)],
        [torch.randn(8192), torch.randn(8192)],
    ]
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

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
"""

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


def response_exit_code(payload: dict) -> int:
    """Map a typed evaluation response to the smoke-test process exit code."""
    status = payload.get("status")
    result = payload.get("result")
    if status not in {"SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"}:
        return 2
    if not isinstance(result, dict):
        return 2
    correctness = result.get("correctness")
    if status == "SUCCEEDED":
        if not isinstance(correctness, dict) or not isinstance(
            correctness.get("passed"), bool
        ):
            return 2
        return 0 if correctness["passed"] else 1
    return 1


def _print_result(payload: dict) -> None:
    status = payload.get("status")
    result = payload.get("result") or {}
    correctness = result.get("correctness") or {}
    print(f"\n[result] status={status} task_id={payload.get('task_id')}")
    if correctness:
        print(
            "  cases: "
            f"{correctness.get('passed_cases')}/{correctness.get('total_cases')} passed"
        )
        for failure in correctness.get("failures") or []:
            detail = failure.get("kind") or failure.get("error") or "unknown failure"
            print(f"  [case {failure.get('case_idx')}] {detail}")

    performance = result.get("performance") or {}
    if performance:
        print(f"  speedup_vs_torch: {performance.get('speedup_vs_torch')}")

    if status != "SUCCEEDED":
        print(f"  error: {payload.get('error')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11907",
                    help="base URL of the triton-eval service")
    ap.add_argument("--op", default="smoke_add", help="op_name (filename-safe slug)")
    ap.add_argument("--measure-speed", dest="measure_speed", action="store_true",
                    default=True, help="also benchmark a passing kernel (default on)")
    ap.add_argument("--no-speed", dest="measure_speed", action="store_false")
    ap.add_argument(
        "--job-timeout",
        type=int,
        default=600,
        help="request-wide service timeout including queue wait (s)",
    )
    ap.add_argument(
        "--session-id",
        default="triton-eval-smoke",
        help="session/container identity used by the service rate limit",
    )
    args = ap.parse_args()

    base = args.url.rstrip("/")

    # 0) health
    try:
        print(f"[healthz] {http_json('GET', f'{base}/healthz', timeout=5.0)}")
    except Exception as e:
        print(f"[healthz] FAILED: {e}", file=sys.stderr)
        return 2

    # One synchronous request. The client timeout intentionally exceeds the server
    # timeout so it does not disconnect while the worker may still be running.
    body = {
        "op_name": args.op,
        "reference_code": REFERENCE_CODE,
        "kernel_code": KERNEL_CODE,
        "triton_impl_name": "triton_ascend_impl",
        "checks": (
            ["correctness", "performance"]
            if args.measure_speed
            else ["correctness"]
        ),
        "non_compute": False,
        "timeout": args.job_timeout,
    }
    print(f"\n[evaluate] POST /evaluations  op={args.op} checks={body['checks']}")
    try:
        response = http_json(
            "POST",
            f"{base}/evaluations",
            body=body,
            timeout=args.job_timeout + 60,
            headers={"X-Session-ID": args.session_id},
        )
    except urllib.error.HTTPError as e:
        print(f"[evaluate] HTTP {e.code}: {e.read().decode()[:500]}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[evaluate] FAILED: {e}", file=sys.stderr)
        return 2

    _print_result(response)
    code = response_exit_code(response)
    print("  ✅ evaluation PASS" if code == 0 else "  ❌ evaluation did not pass")
    return code


if __name__ == "__main__":
    sys.exit(main())
