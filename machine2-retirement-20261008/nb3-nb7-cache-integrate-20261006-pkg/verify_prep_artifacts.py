#!/usr/bin/env python3
"""Verify the prep artifacts: v2prime index parse + v7 policy load chain.

Simulates the pinned runtime environment (patched triton files with the
declared digests, service sources = merged branch bytes) so
load_validated_timing_reuse() runs its FULL validation chain on v7:
policy sha pin, source index digest, deployment digest, contract vs
sealed deployment measurement, target identity. Asserts it authorizes
(validated reuse object) instead of None/raise. Also parses the v2prime
index as CacheIndexV2 and re-verifies entry counts + sealed binding keys.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent / "worktree"
sys.path.insert(0, str(WT))

import triton_evaluation.reference_timing_reuse as rtr
from triton_evaluation.reference_cache import CacheIndexV2, sha256
from triton_evaluation import reference_cache as rc

# --- v2prime index ---
idx_bytes = (HERE / "model-bound-cache-index-v2prime.json").read_bytes()
idx = CacheIndexV2.model_validate_json(idx_bytes)
assert len(idx.entries) == 1046
assert idx.source_deployment_sha256 == "84154dfb56d4a06a818cee7e110589d27817fce784881edaa864210412932f5d"
assert idx.deployment.npu_model == "Ascend910_9362"
print("v2prime index: parse OK, 1046 entries, npu_model OK, source deployment OK")

# --- v7 policy: full load chain with pinned runtime simulated ---
pol_path = HERE / "validated-timing-reuse-policy-v7.json"
pol_bytes = pol_path.read_bytes()
pins = rtr.TimingReusePolicy.model_validate_json(pol_bytes).target_service_source_sha256

real_src = rtr.current_service_source_sha256
real_rt = rtr.current_patched_runtime_sha256

def fake_service_source(name):
    assert name in pins, f"unexpected service source probe: {name}"
    return pins[name]

staged = Path("/home/e84381970/open-evo-work/runtime-artifacts/worktree-detach-20261006/"
              "a-v3-cache-cutover-prep/staged-triton")
rt_pins = rtr.TimingReusePolicy.model_validate_json(pol_bytes).target_patched_runtime_files

def fake_patched_runtime(name):
    assert name in rt_pins, f"unexpected runtime probe: {name}"
    return sha256((staged / name).read_bytes())

rtr.current_service_source_sha256 = fake_service_source
rtr.current_patched_runtime_sha256 = fake_patched_runtime
import os
os.environ["TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY"] = str(pol_path)
os.environ["TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256"] = sha256(pol_bytes)
try:
    validated = rtr.load_validated_timing_reuse()
    assert validated is not None, "v7 failed to authorize with pinned identity simulated"
    print("v7 policy: full load chain OK -> ValidatedTimingReuse authorized")
    print("  policy_sha256:", validated.policy_sha256[:16], "…")
    print("  npu_model gate:", validated.policy.npu_model,
          "cross_card:", validated.policy.cross_card_scope)
finally:
    rtr.current_service_source_sha256 = real_src
    rtr.current_patched_runtime_sha256 = real_rt
    os.environ.pop("TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY", None)
    os.environ.pop("TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256", None)

# --- v7 pins == merged branch bytes (the substantive assertion) ---
branch_pins = {
    "engine/benchmark.py": sha256((WT / "triton_evaluation/engine/benchmark.py").read_bytes()),
    "evaluators/sandbox.py": sha256((WT / "triton_evaluation/evaluators/sandbox.py").read_bytes()),
    "reference_diagnostics.py": sha256((WT / "triton_evaluation/reference_diagnostics.py").read_bytes()),
    "reference_timing_reuse.py": sha256((WT / "triton_evaluation/reference_timing_reuse.py").read_bytes()),
    "reference_cache.py": sha256((WT / "triton_evaluation/reference_cache.py").read_bytes()),
}
assert pins == branch_pins, f"v7 pin drift: {pins} vs {branch_pins}"
v6_pins = rtr.TimingReusePolicy.model_validate_json(
    Path("/root/.paseo/worktrees/0u9809pj/nb7-model-bound-cache-20261006/.scratch/"
         "nb7-model-bound-cache-20261006/validated-timing-reuse-policy-v6.json").read_bytes()
).target_service_source_sha256
diff = {k for k in pins if pins[k] != v6_pins[k]}
assert diff == {"engine/benchmark.py"}, diff
print("v7 pins == merged branch bytes; delta vs v6 == {engine/benchmark.py} only")

# --- sealed manifests still resolve byte-identical through v2prime (spot) ---
import json
cache_root = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/cache/cases")
checked = 0
for key, entry in idx.entries.items():
    if checked >= 3:
        break
    p = cache_root / f"{entry.sealed_binding_key}.json"
    assert p.is_file() and sha256(p.read_bytes()) == entry.manifest_sha256, f"sealed manifest drift: {p}"
    checked += 1
print(f"v2prime -> sealed manifest resolution spot-check OK ({checked} entries)")
