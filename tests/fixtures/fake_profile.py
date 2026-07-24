import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--op_name", required=True)
parser.add_argument("--verify_dir", required=True)
parser.add_argument("--triton_impl_name", required=True)
parser.add_argument("--profile-mode", required=True)
parser.add_argument("--case-idx", type=int)
parser.add_argument("--launch-count", type=int)
parser.add_argument("--timeout", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

payload = {
    "profile_mode": args.profile_mode,
    "requested_case_idx": args.case_idx,
    "total_cases": 2,
    "selected_cases": 1 if args.case_idx is not None else 2,
    "completed_cases": 1 if args.case_idx is not None else 2,
    "partial_cases": 0,
    "failed_cases": 0,
    "remaining_case_indices": [],
    "timed_out": False,
    "summary_text": "fake profile",
    "warnings": [],
    "wallclock_sec": 0.01,
    "per_case_results": [],
    "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
}
with open(args.output, "w", encoding="utf-8") as handle:
    if os.environ.get("FAKE_PROFILE_BAD_JSON") == "1":
        handle.write("{")
    else:
        json.dump(payload, handle)

raise SystemExit(int(os.environ.get("FAKE_PROFILE_EXIT", "0")))
