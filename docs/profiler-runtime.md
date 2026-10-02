# Deploy the pinned profiler SQLite alias

Use this compatibility artifact when the deployed CANN profiler consumer requires
`libsqlite3.so`, but the existing system SQLite library has SONAME
`libsqlite3.so.0`. The service copies the existing, explicitly pinned ELF bytes
into a private read-only alias. It does not install packages or replace CANN,
torch, system libraries, loader configuration, or the six deployment anchors.

The original `torch_npu` profiler and synchronous trace callback remain in use.
The callback must expose its existing `async_mode=False` default. The engine
holds a private loader prefix through profile creation, collection and callback
completion, then restores the process's original `LD_LIBRARY_PATH`. This runs in
the main thread of one engine check process; ordinary worker concurrency uses
different processes. Nested and concurrent captures in the same process fail
closed. Audit case children are a separate execution boundary.

## Prepare a fresh immutable artifact

Run the tracked CLI in the trusted deployment preparation step. It performs
static `readelf` checks of ELF class/data/machine, target SONAME, consumer
DT_NEEDED and all required SQLite exports. It neither imports the NPU stack nor
loads either inspected ELF. The worker subsequently checks hashes only and does
not require `readelf` or access to deployment preparation tools.

For the observed machine2 files, verify these pins again before preparation:

```bash
python scripts/prepare_profiler_runtime.py \
  --output /absolute/new-service-owned-runtime \
  --sqlite-library /usr/lib64/libsqlite3.so.0.8.6 \
  --sqlite-library-sha256 79452f7f5066ef2b0e888fb6a564c1b041ce57d0221bda3756184e14943078b1 \
  --sqlite-consumer /usr/local/Ascend/cann-9.1.1/tools/profiler/profiler_tool/analysis/lib64/msprof_analysis.so \
  --sqlite-consumer-sha256 663b11fadb84f95258af643ee9b6d76bc8244205f223cfc3d8b85de0833de7a0 \
  --msprof /usr/local/Ascend/cann-9.1.1/tools/profiler/bin/msprof \
  --msprof-sha256 75541a2535f1dc9b39484b833b0454a002d0eabd68d7673006f52cb00e73dc82
```

Use the existing deployment Python environment. The output directory must not
already exist. Keep unsuccessful preparation artifacts with their attempt
identity. The CLI prints the absolute `manifest_path` and its actual
`manifest_sha256`. The resulting `profiler-runtime-v1` manifest binds the alias,
source library, consumer, msprof and retained ELF observations by absolute path
and SHA. Artifact files have mode `0444`, and their directories have mode `0555`.

## Deploy the pair and read-only visibility

Add both non-anchor variables to the API and worker runtime configuration:

```text
TRITON_EVAL_PROFILER_RUNTIME_MANIFEST=/absolute/new-service-owned-runtime/manifest.json
TRITON_EVAL_PROFILER_RUNTIME_SHA256=<actual CLI output SHA>
```

Missing halves, changed files, writable artifacts or unexpected resolved
`msprof` paths fail closed. Do not prefix the host shell, API or worker parent
with the alias directory. The engine derives that prefix only while executing
the original profiler context. An unconfigured historical deployment retains
its previous behavior; that compatibility does not authorize omitting this
pair from an experiment that requires it.

Keep the original sandbox manifest bytes and SHA unchanged. Its `--clearenv`
boundary does not forward new parent variables. Prepare a separate sparse
overlay after sealing its referenced dependencies:

```bash
python scripts/prepare_engine_runtime_overlay.py \
  --output /absolute/new-runtime-overlay.json \
  --base-manifest /absolute/original-sandbox.json \
  --base-sha256 <unchanged sandbox anchor SHA> \
  --profiler-manifest /absolute/new-service-owned-runtime/manifest.json \
  --profiler-sha256 <actual profiler manifest SHA>
```

Deploy the printed file path and SHA as the non-anchor pair
`TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_CONFIG` and
`TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_SHA256`. The strict
`engine-runtime-overlay-v1` file contains only `environment`. It accepts the
complete profiler pair and/or all four F-63 variables; arbitrary environment
keys, mounts and source paths are rejected. For cache-on execution add the CLI
options `--cache-dir`, `--cache-index`, `--cache-sha256`, `--cache-state-dir`.
The dedicated invalidation directory must exist and contain only the declared
deployment/card invalidation namespace. Cache-off and cache-on diagnostics need
separately pinned overlays and consistent parent configuration.

The service derives read-only visibility from the pinned artifact paths, keeps
the entire dedicated profiler directory read-only with its real `0555` modes,
and exposes the source SQLite/consumer/msprof files at their declared paths.
F-63 exposes only its `cases` and `blobs` directories, the index and pinned
runtime files, plus one exact dedicated writable invalidation directory. It
never adds a host home or state root. The executing service's own package path
is derived from the trusted sandbox module and mounted read-only; the effective
child `PYTHONPATH` starts with that package's parent. This permits the current
GitHub checkout to run while preserving an anchored manifest that names the
old live engine. A mismatching engine directory or external script fails
closed; regular scripts inside the validated attempt directory are allowed and
SHA-recorded for trusted fleet probes.
The existing reference-device-policy anchor retains its value and is explicitly
carried into the child: verification receives it through its CLI, while the
benchmark cache key reads the environment. The overlay cannot supply or change
that anchor value.

Each prepared invocation retains an `engine-runtime-boundary-*.json` artifact
with original/overlay pins, actual source-file SHAs, effective argv/environment
digests and derived mounts. Its `executed=false` value deliberately records
preparation only. Device acceptance must establish actual bwrap execution,
loaded helper path/SHA and host-canary denial separately.

## Reseal before device acceptance

Regenerate the snapshot and cache deployment/index. Preserve all six deployment
anchor keys and their existing values, including `TRITON_EVAL_EVALUATOR_BUILD`.
Record the actual service Git/source SHA, helper SHA and effective cache
algorithm identity as separate frozen evidence. Bind the
actual runtime manifest, alias, ELF observations, source SQLite ELF, consumer,
resolved msprof and complete CANN/torch_npu parser/native dependency closure in
`runtime_files`. Bind the changed benchmark and fixed service-owned helper in
`algorithm_files`. The only permitted sibling key is
`../profiler_runtime.py`, resolving to
`triton_evaluation/profiler_runtime.py` within the validated service root;
other traversal, absolute keys and symlink escape are refused. This is a fixed
deployment dependency and does not expand candidate H/K materials. The three
historically required engine files remain required and do not by themselves
describe the complete measurement implementation.

The numerical cache deployment must pin the original sandbox and its actual
derivation/helper source, but must not include the overlay file that contains
the cache index SHA: that would create a digest cycle. Seal the numerical
deployment and index first. Then seal the sparse overlay and bind its raw SHA,
index SHA and numerical deployment SHA independently in final admission/freeze,
checking actual execution against its derived conditions. Excluding the cyclic
overlay file from `runtime_files` does not close that final admission binding.

Every configured profiler invocation writes `profiler-runtime.json` inside its
profile directory after the original callback completes. It records actual
helper, manifest, alias, library, consumer and msprof SHAs, process identity,
prefix and environment-value SHAs, callback completion and restoration. This
proves that the measurement entered the configured context; an environment pair
alone does not. Bind this artifact and the raw profiler outputs to device
acceptance observations.

Run the original sharp NPU probe from an exact GitHub checkout in a fresh
attempt directory with one idle physical card selected before Python starts:

```bash
source /usr/local/Ascend/cann/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=8 python scripts/probe_profiler_latency.py \
  --sandbox --device 8 --output /absolute/new-profiler-runtime-probe
```

Both runtime variables and the new overlay pair must already be exported by the
deployment environment, alongside the six unchanged anchors. `--sandbox` uses
the worker's actual `run_script` boundary, preserving physical-card assignment
while Ascend uses logical zero inside the namespace. It retains child stdout,
stderr, the actual native report and its SHA. A direct host probe omits this
flag and cannot establish worker isolation.
The probe keeps profiler method, five warmups, twenty repeats and the original
`kernel_details.csv` last-twenty formula. Green requires eight finite positive
latencies, their median, actual NPU input/output proof and eight matching
completed-callback runtime artifacts. It stops on the first failed observation
and retains raw evidence. CANN export rows, CPU loader tests and the probe itself
are diagnostic evidence; they do not count as independent Samples or fleet
admission. Resume full reference auditing only after this probe and the actual
worker sandbox path satisfy their separate gates.
