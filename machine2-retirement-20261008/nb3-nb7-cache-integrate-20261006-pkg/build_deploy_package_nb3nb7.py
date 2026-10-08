#!/usr/bin/env python3
"""Build the nb3-nb7-cache-integrate deploy package anchors (path-parameterized).

Mirrors NB3's build_anchors.py + NB7's build_deploy_package.py on the CURRENT
live post-migration anchors (worktree-detach-20261007-*, d2521b41/2d75d31e):

- sandbox manifest: LIVE manifest with the engine ro-bind swapped to THIS
  branch's engine dir (stable paths for A staged-triton kept byte-identical);
- runtime overlay: LIVE overlay with the cache binding pair repointed at the
  model-bound index v2prime, the timing-reuse pair at policy v7, and the
  cache state dir at a fresh dedicated dir under /dev/shm;
- BUILD formula identical to service_fleet_preflight_nb2.py (commit +
  manifest sha + verdict_cache + npu-strict).

PROVISIONAL when run with scratch paths: the engine path string is part of
the manifest bytes, so final hashes are re-derived at deploy time after the
worktree moves to its stable runtime-artifacts home. Pass --engine-dir and
--pkg-dir explicitly for the final build.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LIVE_MANIFEST = Path(
    "/root/.paseo/worktrees/0u9809pj/nb3-verify-input-materialization-fix-20261006/"
    ".scratch/nb3-verify-input-materialization-fix-20261006b/"
    "worktree-detach-20261007-sandbox-manifest.json"
)
LIVE_OVERLAY = LIVE_MANIFEST.parent / "worktree-detach-20261007-runtime-overlay.json"
LIVE_ENGINE = "/root/.paseo/worktrees/0u9809pj/nb3-verify-input-materialization-fix-20261006/triton_evaluation/engine"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True,
                    help="this branch's triton_evaluation/engine (stable path for final build)")
    ap.add_argument("--pkg-dir", required=True,
                    help="directory holding index v2prime + policy v7 (stable path for final build)")
    ap.add_argument("--state-dir", default="/dev/shm/nb3nb7-state-20261006")
    ap.add_argument("--tag", default="provisional")
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir).resolve()
    pkg_dir = Path(args.pkg_dir).resolve()
    assert (engine_dir / "benchmark.py").is_file(), f"bad engine dir: {engine_dir}"
    assert engine_dir != LIVE_ENGINE

    index_sha = sha((pkg_dir / "model-bound-cache-index-v2prime.json").read_bytes())
    policy_sha = sha((pkg_dir / "validated-timing-reuse-policy-v7.json").read_bytes())

    # --- manifest: single delta -> this branch's engine dir ---
    manifest = json.loads(LIVE_MANIFEST.read_bytes())
    mounts = [(s, d) for s, d in manifest["readonly_mounts"]]
    assert (LIVE_ENGINE, LIVE_ENGINE) in [tuple(x) for x in mounts], "live engine mount missing"
    manifest["readonly_mounts"] = [
        (str(engine_dir), str(engine_dir)) if (s, d) == (LIVE_ENGINE, LIVE_ENGINE) else (s, d)
        for (s, d) in mounts
    ]
    assert (str(engine_dir), str(engine_dir)) in [tuple(x) for x in manifest["readonly_mounts"]]
    manifest_bytes = json.dumps(manifest, indent=1).encode()
    manifest_out = pkg_dir / "deploy-sandbox-manifest.json"
    manifest_out.write_bytes(manifest_bytes)
    manifest_sha = sha(manifest_bytes)

    # --- overlay: v2prime + v7 + fresh state dir ---
    overlay = json.loads(LIVE_OVERLAY.read_bytes())
    env = overlay["environment"]
    assert env["TRITON_EVAL_REFERENCE_CACHE_SHA256"] == "c1acb240ea8224629f34a45cf402bcc0fbc8e69457cacf59f342b035733e8d6b"
    assert env["TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256"] == "428ea024c10afbe8b40ed042bb23ffd2f9ff81988086257aebf0ffc61f050f97"
    env["TRITON_EVAL_REFERENCE_CACHE_BINDING"] = str(pkg_dir / "model-bound-cache-index-v2prime.json")
    env["TRITON_EVAL_REFERENCE_CACHE_SHA256"] = index_sha
    env["TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY"] = str(pkg_dir / "validated-timing-reuse-policy-v7.json")
    env["TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256"] = policy_sha
    env["TRITON_EVAL_REFERENCE_CACHE_STATE_DIR"] = args.state_dir
    overlay_bytes = json.dumps(overlay, indent=1).encode()
    overlay_out = pkg_dir / "deploy-runtime-overlay.json"
    overlay_out.write_bytes(overlay_bytes)
    overlay_sha = sha(overlay_bytes)

    # --- BUILD formula (service_fleet_preflight_nb2.py) ---
    commit = subprocess.check_output(
        ["git", "-C", str(engine_dir.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    build = "sha256:" + hashlib.sha256(json.dumps({
        "git_commit": commit,
        "sandbox_manifest_sha256": manifest_sha,
        "verdict_cache_enabled": False,
        "reference_device_policy": "npu-strict",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    receipt = {
        "tag": args.tag,
        "branch": "nb3-nb7-cache-integrate",
        "git_commit": commit,
        "engine_dir": str(engine_dir),
        "sandbox_manifest_path": str(manifest_out),
        "sandbox_manifest_sha256": manifest_sha,
        "delta_vs_live_d2521b41": [f"engine ro-bind {LIVE_ENGINE} -> {engine_dir}"],
        "runtime_overlay_path": str(overlay_out),
        "runtime_overlay_sha256": overlay_sha,
        "delta_vs_live_2d75d31e": [
            f"cache binding -> {pkg_dir / 'model-bound-cache-index-v2prime.json'} ({index_sha})",
            f"timing policy -> {pkg_dir / 'validated-timing-reuse-policy-v7.json'} ({policy_sha})",
            f"cache state dir -> {args.state_dir}",
        ],
        "evaluator_build": build,
        "model_bound_index_v2prime_sha256": index_sha,
        "timing_reuse_policy_v7_sha256": policy_sha,
        "cache_dir_unchanged": "/dev/shm/v3-ocp-payload-run-20261004-a2/cache",
        "profiler_runtime_unchanged": {
            env["TRITON_EVAL_PROFILER_RUNTIME_MANIFEST"]: env["TRITON_EVAL_PROFILER_RUNTIME_SHA256"],
        },
        "note": ("PROVISIONAL hashes when built from scratch paths; final build re-runs this "
                 "script with the stable runtime-artifacts engine/pkg paths at deploy time "
                 "(after OWNERSHIP-GRANTED + worktree move)."),
    }
    out = pkg_dir / f"deploy-package-receipt-{args.tag}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in (
        "tag", "git_commit", "sandbox_manifest_sha256", "runtime_overlay_sha256",
        "evaluator_build")}, indent=1))


if __name__ == "__main__":
    main()
