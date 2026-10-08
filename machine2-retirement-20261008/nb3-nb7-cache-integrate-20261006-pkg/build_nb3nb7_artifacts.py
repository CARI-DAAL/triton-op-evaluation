#!/usr/bin/env python3
"""Build the nb3-nb7-cache-integrate artifacts from the sealed cache (PREP).

Same existing mechanisms as NB7's build_model_bound.py recipe and the
v5->v6 policy precedent (documented in nb7 UNAPPLIED-PACKAGE.md and the
nb7-cache-validity review):

1. model-bound-cache-index-v2prime.json: rekeys the sealed 4c165dfc index
   (1046 card-bound entries, never edited) under ModelBoundBinding
   identities. Target deployment re-pins algorithm_files (verify.py,
   benchmark.py, _common_utils.py) to THIS merged branch's engine bytes;
   runtime_versions/runtime_files stay the sealed fingerprint;
   npu_model = Ascend910_9362. binding key does not include engine file
   hashes, so sealed manifests resolve unchanged.
2. validated-timing-reuse-policy-v7.json: v6 content + target pins
   re-derived on this branch's bytes. Per the review, only
   engine/benchmark.py differs between the merged tree and v6's pins
   (683cc5fb -> d65b54d4, the c03fd80 npu-strict loud-failure guard,
   measurement-neutral). v6 artifact untouched.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent / "worktree"
sys.path.insert(0, str(WT))

from triton_evaluation.reference_cache import (  # noqa: E402
    CacheIndex, build_model_bound_index, sha256,
)

SEAL = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/state/sealed-cache-index.json")
SEAL_SHA = "4c165dfca2675951c7dabf3caeabbef22f54d69a44e1540e95110b470db4fa87"
CACHE_DIR = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/cache")
V6 = Path("/root/.paseo/worktrees/0u9809pj/nb7-model-bound-cache-20261006/.scratch/nb7-model-bound-cache-20261006/validated-timing-reuse-policy-v6.json")
V6_SHA = "6369ff3756a5b4004aa7972e27080294256f1ddc7d0758fe55adcfc73664d8dc"
STABLE_STATE_DIR = "/home/e84381970/open-evo-work/runtime-artifacts/worktree-detach-20261006/c-v3-service-reference8/timing-reuse-delivery/state"
MODEL = "Ascend910_9362"

payload = SEAL.read_bytes()
assert sha256(payload) == SEAL_SHA, "sealed index drift"
source = CacheIndex.model_validate_json(payload)

engine = WT / "triton_evaluation/engine"
algorithm_files = dict(source.deployment.algorithm_files)
algorithm_files.update({
    name: sha256((engine / name).read_bytes())
    for name in ["verify.py", "benchmark.py", "_common_utils.py"]
})
target = source.deployment.model_copy(update={
    "generation": source.deployment.generation + "-model-bound-nb3nb7",
    "algorithm_files": algorithm_files,
    "npu_model": MODEL,
})
model_index, report = build_model_bound_index(source, target, cache_root=CACHE_DIR)
assert report == {
    "source_entries": 1046, "model_bound_entries": 1046,
    "ambiguous_excluded": 0,
    "sealed_cards": sorted(str(card) for card in range(16)),
}, report
index_bytes = json.dumps(model_index.model_dump(mode="json"), indent=1).encode()
out_index = HERE / "model-bound-cache-index-v2prime.json"
out_index.write_bytes(index_bytes)
index_sha = sha256(index_bytes)

# --- policy v7 (v5->v6 precedent: re-derive target pins on branch bytes) ---
v6_bytes = V6.read_bytes()
assert sha256(v6_bytes) == V6_SHA, "policy v6 drift"
policy = json.loads(v6_bytes)
old_pins = dict(policy["target_service_source_sha256"])
old_state_dir = policy["source_state_dir"]
policy["source_state_dir"] = STABLE_STATE_DIR
policy["target_service_source_sha256"] = {
    "engine/benchmark.py": sha256((WT / "triton_evaluation/engine/benchmark.py").read_bytes()),
    "evaluators/sandbox.py": sha256((WT / "triton_evaluation/evaluators/sandbox.py").read_bytes()),
    "reference_diagnostics.py": sha256((WT / "triton_evaluation/reference_diagnostics.py").read_bytes()),
    "reference_timing_reuse.py": sha256((WT / "triton_evaluation/reference_timing_reuse.py").read_bytes()),
    "reference_cache.py": sha256((WT / "triton_evaluation/reference_cache.py").read_bytes()),
}
changed_pins = {k: (old_pins[k], v) for k, v in policy["target_service_source_sha256"].items() if old_pins[k] != v}
policy["evidence"]["nb3nb7_integration_2026_10_07"] = (
    "Review nb7-cache-validity-review-20261006 (YES, acc 3/3): merged candidate "
    "c03fd80+f5e43f1 has no actual measurement inequivalence; the only blockers were "
    "mechanical pins. Human ruling: 其实就是 硬件 + 软件是一样就可以复用 — same NPU "
    "model + measurement-relevant software equivalence + same workload/input reuses the "
    "sealed cache; measurement-neutral whole-file repairs only change provenance. This v7 "
    "re-pins target_service_source_sha256 on the merged branch bytes per the v5->v6 "
    "precedent; only engine/benchmark.py changed vs v6 (npu-strict loud-failure guard, "
    "measurement-neutral). source_state_dir rebinds to the migration stable path exactly "
    "as the live v5-rebound 428ea024 precedent (old C worktree archived). npu_model gate, "
    "cross_card_scope, timing_contract (profiler/w5/r20/8 samples/median), source "
    "index/deployment digests, sentinel rules and budgets are unchanged."
)
policy["evidence"]["supersedes_policy"] = (
    "validated-timing-reuse-policy-v6 (6369ff3756a5b4004aa7972e27080294256f1ddc7d0758fe55adcfc73664d8dc); "
    "v6 artifact untouched; v6 superseded v5 1b128616… untouched"
)
policy_bytes = json.dumps(policy, indent=1, sort_keys=True).encode()
out_policy = HERE / "validated-timing-reuse-policy-v7.json"
out_policy.write_bytes(policy_bytes)
policy_sha = sha256(policy_bytes)

tip = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
receipt = {
    "branch": "nb3-nb7-cache-integrate",
    "tip": tip,
    "parents": ["c03fd80 (nb3-official-npu-repair-20261006)",
                "f5e43f1b777a747636c02939ffb6e65ea7d90223 (nb7-model-bound-cache-20261006)"],
    "model_bound_index_v2prime": {"path": str(out_index), "sha256": index_sha,
                                  "npu_model": MODEL,
                                  "source_index_sha256": SEAL_SHA,
                                  "source_deployment_sha256": source.deployment.digest,
                                  "target_deployment_sha256": target.digest,
                                  "report": report},
    "timing_reuse_policy_v7": {"path": str(out_policy), "sha256": policy_sha,
                               "delta_vs_v6_6369ff37": {
                                   "target_service_source_sha256_changed": changed_pins,
                                   "source_state_dir_changed": {"from": old_state_dir, "to": STABLE_STATE_DIR,
                                                                 "precedent": "live v5-rebound 428ea024 (migration worktree-detach-20261006)"},
                                   "evidence_keys_added": ["nb3nb7_integration_2026_10_07"],
                                   "supersedes_policy_updated": True,
                                   "everything_else_unchanged": True,
                               },
                               "pins_asserted_equal_to_branch_bytes": True,
                               "no_old_worktree_dependency": True},
    "algorithm_files_repin": {
        k: v for k, v in algorithm_files.items()
        if source.deployment.algorithm_files.get(k) != v
    },
}
(HERE / "build-nb3nb7-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps({"index_sha256": index_sha, "policy_v7_sha256": policy_sha,
                  "changed_pins": changed_pins, "tip": tip,
                  "index_report": report}, indent=1))
