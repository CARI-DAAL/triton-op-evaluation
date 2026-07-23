import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--op_name", required=True)
p.add_argument("--verify_dir", default=".")
p.add_argument("--triton_impl_name", default="triton_ascend_impl")
p.add_argument("--timeout", type=int, default=900)
p.add_argument("--output", default=None)
p.add_argument("--non-compute", action="store_true")
args = p.parse_args()

mode = os.environ.get("FAKE_VERIFY_MODE", "pass")  # pass | fail | crash
out = args.output or os.path.join(args.verify_dir, "verify_result.json")

if mode == "crash":
    sys.stderr.write("boom\n")
    sys.exit(1)

result = {
    "op_name": args.op_name,
    "total_cases": 2,
    "passed_cases": 2 if mode == "pass" else 1,
    "failed_cases": 0 if mode == "pass" else 1,
    "failures": [] if mode == "pass" else [{"case_idx": 2, "error_type": "AccuracyError"}],
}
with open(out, "w") as f:
    json.dump(result, f)
sys.exit(0 if mode == "pass" else 1)
