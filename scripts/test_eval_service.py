#!/usr/bin/env python3
"""Smoke-test the triton-evaluation service end-to-end.

Submits a known-good Ascend smoke op (elementwise add: a torch reference + a
triton kernel) to POST /verify, polls GET /tasks/{task_id} to a terminal
status, and prints a PASS/FAIL verdict. Mirrors how kernel-harness's
TritonEvalClient calls the service
(kernel_agent/rewards/triton_eval_client.py): same /verify body, same poll loop.

Stdlib only — runs from the host against the service (the NPU work happens
inside the triton-eval container):

    python3 scripts/test_eval_service.py [--url http://127.0.0.1:11907] [--no-speed]

Exit code: 0 = verify passed (all cases correct), 1 = verify failed,
2 = service/transport error, 3 = poll timed out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

TERMINAL = {"SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"}


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 15.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11907",
                    help="base URL of the triton-eval service")
    ap.add_argument("--op", default="smoke_add", help="op_name (filename-safe slug)")
    ap.add_argument("--measure-speed", dest="measure_speed", action="store_true",
                    default=True, help="also benchmark a passing kernel (default on)")
    ap.add_argument("--no-speed", dest="measure_speed", action="store_false")
    ap.add_argument("--job-timeout", type=int, default=600, help="per-job timeout (s)")
    ap.add_argument("--poll", type=float, default=2.0, help="poll interval (s)")
    ap.add_argument("--max-wait", type=float, default=900.0, help="max wall-clock wait (s)")
    args = ap.parse_args()

    base = args.url.rstrip("/")

    # 0) health
    try:
        print(f"[healthz] {http_json('GET', f'{base}/healthz', timeout=5.0)}")
    except Exception as e:
        print(f"[healthz] FAILED: {e}", file=sys.stderr)
        return 2

    # 1) submit /verify
    body = {
        "op_name": args.op,
        "reference_code": REFERENCE_CODE,
        "kernel_code": KERNEL_CODE,
        "triton_impl_name": "triton_ascend_impl",
        "non_compute": False,
        "measure_speed": args.measure_speed,
        "timeout": args.job_timeout,
    }
    print(f"\n[submit] POST /verify  op={args.op} measure_speed={args.measure_speed}")
    try:
        resp = http_json("POST", f"{base}/verify", body=body, timeout=15.0)
    except urllib.error.HTTPError as e:
        print(f"[submit] HTTP {e.code}: {e.read().decode()[:500]}", file=sys.stderr)
        return 2
    task_id = resp.get("task_id")
    print(f"[submit] task_id={task_id}")

    # 2) poll to terminal
    deadline = time.monotonic() + args.max_wait
    last: dict | None = None
    while time.monotonic() < deadline:
        last = http_json("GET", f"{base}/tasks/{task_id}", timeout=10.0)
        if last.get("status") in TERMINAL:
            break
        time.sleep(args.poll)
    else:
        st = last.get("status") if last else None
        print(f"[poll] TIMEOUT after {args.max_wait}s; last status={st}", file=sys.stderr)
        return 3

    # 3) verdict
    result = last.get("result") or {}
    print(f"\n[result] status={last.get('status')}")
    if last.get("status") == "SUCCEEDED":
        total = result.get("total_cases")
        passed = result.get("passed_cases")
        print(f"  cases: {passed}/{total} passed")
        if isinstance(total, int) and isinstance(passed, int) and total > 0 and passed == total:
            print("  ✅ verify PASS (kernel matches reference)")
            if args.measure_speed:
                bench = result.get("benchmark") or {}
                print(f"  speedup_vs_torch: {bench.get('speedup_vs_torch')}")
            return 0
        for f in (result.get("failures") or [])[:5]:
            print(f"  [case {f.get('case_idx')}] {f.get('error_type')}: {f.get('error_msg')}")
        print("  ❌ verify FAIL (output mismatch)")
        return 1

    # FAILED / TIMEOUT / REJECTED
    print(f"  error: {last.get('error') or result.get('error')}")
    for f in (result.get("failures") or [])[:5]:
        print(f"  [case {f.get('case_idx')}] {f.get('error_type')}: {f.get('error_msg')}")
    print("  ❌ verify did not pass")
    return 1


if __name__ == "__main__":
    sys.exit(main())
