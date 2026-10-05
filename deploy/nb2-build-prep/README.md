# nb2-build-prep deployment configuration (CANN 9.1.1 runtime/sandbox)

Branch `nb2-build-prep` (base `212b832` = the deployed native-facade service
revision). This directory is the operator-owned runtime/sandbox configuration
for the NB2 freeze-window service build. Nothing here is applied by committing;
deployment, restart and re-admission happen in the NB2 maintenance window
through the package delivered with this branch (see the task report
`nb2-build-prep-20261005`).

## What each file is

- `sandbox-manifest.json` — base engine sandbox manifest for this build.
  Same shape as the known-good phase2 base manifest `7eff529b…`
  (`…/v3-cache-cutover-prep/formal-staging-v1-patched-triton/new-base-sandbox-manifest.json`),
  with exactly two NB2 deltas:
  1. the engine read-only bind now points at this build's deployment checkout
     `/home/e84381970/open-evo-work/service-nb2-buildprep/triton_evaluation/engine`
     (the phase2 manifest bound the retired fb90951 engine path);
  2. the eight vendor CANN environment entries are the pinned CANN 9.1.1
     values captured by the service profiler-runtime CLI
     (`cann-environment.json` `b3f45fcf…`: `cann_root=/usr/local/Ascend/cann-9.1.1`,
     `set_env.sh` `be40277c…`) instead of the deleted cann-8.5.2 paths.
  Everything else is unchanged from the proven phase2 manifest: `.venv-p0`
  (torch_npu 2.9.0.post8, the CANN 9.1.1-paired build), the SHA-pinned staged
  triton patch binds (`driver.py` `3942188b…`, `npu_utils.cpp` `6bd908f6…`,
  bound over the venv triton paths), `/usr`, `/sys`, the `/etc` files, the
  interpreter alias binds and the `/usr/bin/ld.bfd → /etc/alternatives/ld`
  alias, and the shared Ascend control devices.
- `engine-runtime-overlay.json` — byte copy of the phase2
  `overlay-uncached-profiler-only-v1.json` (`3b446566…`): exactly the
  `TRITON_EVAL_PROFILER_RUNTIME_MANIFEST/_SHA256` pair pointing at the
  service-owned profiler runtime
  `/home/e84381970/open-evo-work/baseline-v3/diagnostics/PROFILER-RUNTIME-212b832-attempt3/profiler-runtime/manifest.json`
  (`870732c5…`). Through `evaluators/sandbox.py::apply_runtime_overlay` this
  pair derives the pinned current CANN 9.1.1 environment for every engine
  child (`derive_cann_environment`), adds the cann-9.1.1 + driver read-only
  mounts, and binds the executing service package. This is the exact
  mechanism of the known-good phase2 staging evaluation
  `ev_e04caf2e5693456c8823ee6229543655` (correctness 51/51 under CANN 9.1.1).
  NB7's reference-cache/timing-reuse overlay groups are deliberately NOT
  included; see "Merging later work" below.

## Worker/API environment wiring (deployment-time)

Set on BOTH the API and worker units (values = the frozen artifacts above):

```text
TRITON_EVAL_ENGINE_SANDBOX_CONFIG=/home/e84381970/open-evo-work/service-nb2-buildprep/deploy/nb2-build-prep/sandbox-manifest.json
TRITON_EVAL_ENGINE_SANDBOX_SHA256=<sha256 of sandbox-manifest.json at freeze>
TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_CONFIG=/home/e84381970/open-evo-work/service-nb2-buildprep/deploy/nb2-build-prep/engine-runtime-overlay.json
TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_SHA256=<sha256 of engine-runtime-overlay.json at freeze>
TRITON_EVAL_ENGINE_DIR=/home/e84381970/open-evo-work/service-nb2-buildprep/triton_evaluation/engine
PYTHONPATH=/home/e84381970/open-evo-work/service-nb2-buildprep:<cann site-packages>:<tbe>
TRITON_EVAL_EVALUATOR_BUILD=sha256:<see identity formula>
```

All other anchors keep their current production values: `TRITON_EVAL_VERDICT_CACHE_ENABLED=false`,
`TRITON_EVAL_REFERENCE_DEVICE_POLICY=npu-strict`, `TRITON_EVAL_REFERENCE_DIAGNOSTICS_DIR`
unchanged, `TRITON_EVAL_CORRECTNESS_ALGORITHM_VERSION=correctness-v1`,
algorithms `correctness-v1:npu-strict-v1` / `decoy-v1` / `geometric-mean-v1` / `profiler-v1`.

The parent environment must NOT additionally export the profiler/cache
runtime variables (`TRITON_EVAL_PROFILER_RUNTIME_*`, `TRITON_EVAL_REFERENCE_CACHE*`,
`TRITON_EVAL_REFERENCE_TIMING_REUSE_*`): they reach `--clearenv` engine children
only through this overlay (`isolate_command` refuses a parent declaration the
overlay does not carry).

## Deployment identity (evaluator_build)

Same formula as every prior cutover (`deploy_service_result_collection.py`):

```python
import hashlib, json
identity = {
    "git_commit": "<full sha of the deployed nb2-build-prep checkout>",
    "sandbox_manifest_sha256": "<sha256 of sandbox-manifest.json>",
    "verdict_cache_enabled": False,
    "reference_device_policy": "npu-strict",
}
evaluator_build = "sha256:" + hashlib.sha256(
    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
```

`evaluator_build` changes versus production (`sha256:d3232fdf…`, fb90951 +
50ad3b6e) because both the source commit (212b832-based build, which already
contains the d3 `EvalWorker`/`EvalEngine` proctitle renames via `3a7079a`)
and the sandbox manifest change. This is the expected NB2 re-admission
trigger ("d3 使 evaluator_build 变化").

## Merging later work (recorded, not done here)

- `nb3-facade-serialization-fix` (API facade `kernel_eval.py`, commit
  `e9c711b`): touches files this branch does not touch; merge/rebase on top
  of this branch before the window deploy if approved, then recompute the
  identity above for the merged tip.
- NB7 phase2 timing-reuse engine changes + staging/card8: those live in the
  `v3-service-reference8` worktree (engine/benchmark.py `66ee3307…`,
  sandbox.py `8b92a0df…` + repairs). They merge by (a) applying their source
  diff to this branch and (b) extending `engine-runtime-overlay.json` with
  their complete overlay groups (cache quartet + timing-reuse pair, exactly
  the delta in the NB7 `LAUNCH-CONFIG-DELTA.md`). Do not mix that extension
  into the NB2 window without the NB7 owner's reconciled pins.
