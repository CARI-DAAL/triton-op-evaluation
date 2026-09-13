#!/usr/bin/env python3
"""THROWAWAY spike: on-card failure classification via the npu0 smoke service.

POSTs each failure-corpus kernel with checks=["correctness"] and records the
terminal outcome, to be compared against the cardless stage (criterion 3).
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import MODULES  # noqa: E402

# Reference contract (engine/verify.py requires get_init_inputs). Arity must
# match the candidate kernel's forward: fail_backend is 2-arg, the other
# failure modules are 1-arg.
REFERENCE_2ARG = """import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_input_groups():
    return [[torch.randn(4096), torch.randn(4096)]]
"""

REFERENCE_1ARG = """import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return x

def get_init_inputs():
    return []

def get_input_groups():
    return [[torch.randn(4096)]]
"""

REFERENCES = {
    "fail_backend": REFERENCE_2ARG,
    "fail_syntax": REFERENCE_1ARG,
    "fail_import_dep": REFERENCE_1ARG,
    "fail_frontend": REFERENCE_1ARG,
}


def http_json(method, url, body, headers, timeout=15.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def classify_one(base, headers, name, source, reference, deadline=600) -> dict:
    body = {
        "schema_version": "2.1",
        "purpose": "trial",
        "op_name": f"spike_{name}",
        "reference_code": reference,
        "kernel_code": source,
        "checks": ["correctness"],
        "deadline_seconds": deadline,
    }
    post_headers = {**headers, "Idempotency-Key": f"spike-{name}-{uuid.uuid4()}"}
    resource = http_json("POST", f"{base}/evaluations?wait_seconds=0",
                         body, post_headers)
    eval_id = resource["evaluation_id"]
    end = time.monotonic() + deadline + 60
    while resource.get("state") not in ("completed", "failed", "timed_out"):
        if time.monotonic() > end:
            return {"oncard_outcome": "client_gave_up"}
        time.sleep(2.0)
        resource = http_json("GET", f"{base}/evaluations/{eval_id}", None, headers)
    data = ((resource.get("result") or {}).get("correctness") or {}).get("data") or {}
    return {"state": resource.get("state"), "oncard_outcome": data.get("outcome"),
            "evaluation_id": eval_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11908")
    ap.add_argument("--token", required=True, help="trial capability Bearer token")
    ap.add_argument("--report", default="spike-report/oncard_classify.json")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.token}",
               "X-Session-ID": "spike-classify"}
    prior = {}
    prior_path = Path("spike-report/cardless.json")
    if prior_path.exists():
        prior = {n: m.get("stage")
                 for n, m in json.loads(prior_path.read_text())["modules"].items()}

    report = {"modules": {}}
    for name, entry in MODULES.items():
        if entry["expect"] == "ok":
            continue
        reference = REFERENCES.get(name, REFERENCE_1ARG)
        try:
            row = classify_one(args.url, headers, name, entry["source"], reference)
        except Exception as exc:  # one transient HTTP error must not kill the sweep
            row = {"oncard_outcome": "client_error", "error": str(exc)[:500]}
        row["cardless_stage"] = prior.get(name, "unknown")
        # consistent := service agrees the kernel is not runnable; the exact
        # outcome→stage mapping is documented in the spike report (spec §8
        # criterion 3 is a documentation requirement, not a forced match).
        row["consistent"] = row.get("oncard_outcome") in (
            "compile_error", "runtime_error", "wrong")
        report["modules"][name] = row

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
