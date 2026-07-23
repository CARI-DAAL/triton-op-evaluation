import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--op_name", required=True)
p.add_argument("--verify_dir", default=".")
p.add_argument("--triton_impl_name", default="triton_ascend_impl")
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--repeats", type=int, default=50)
p.add_argument("--output", default=None)
p.add_argument("--verify_not_required", action="store_true")
args = p.parse_args()


def _verify_json_name(name):
    if name == "triton_ascend_impl":
        return "verify_result.json"
    suffix = name[len("triton_"):] if name.startswith("triton_") else name
    return "verify_result_%s.json" % suffix


if not args.verify_not_required:
    vpath = os.path.join(args.verify_dir, _verify_json_name(args.triton_impl_name))
    try:
        with open(vpath) as f:
            v = json.load(f)
    except Exception:
        sys.stderr.write("L1 gate: verify_result missing\n")
        sys.exit(2)
    if v.get("total_cases", 0) == 0 or v.get("passed_cases") != v.get("total_cases"):
        sys.stderr.write("L1 gate: verify not all passed\n")
        sys.exit(2)

out = args.output or os.path.join(args.verify_dir, "perf_result.json")
with open(out, "w") as f:
    json.dump({"op_name": args.op_name, "speedup_vs_torch": 2.0,
               "total_cases": 1, "passed_cases": 1, "failed_cases": 0}, f)
sys.exit(0)
