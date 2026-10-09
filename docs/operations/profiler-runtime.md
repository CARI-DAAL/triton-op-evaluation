# Deploy the pinned profiler SQLite alias

Use this compatibility artifact when the deployed CANN profiler consumer requires
`libsqlite3.so`, but the existing system SQLite library has SONAME
`libsqlite3.so.0`. The service copies the existing, explicitly pinned ELF bytes
into a private read-only alias. It does not install packages or replace CANN,
torch, system libraries, loader configuration, or the host installation.

The original `torch_npu` profiler and synchronous trace callback remain in use.
The callback must expose its existing `async_mode=False` default. The engine
holds a private `LD_LIBRARY_PATH` prefix and prepends the pinned alias to
`LD_PRELOAD` through profile creation, collection and callback completion, then
restores both original variables, including their absent state. This runs in
the main thread of one engine check process; ordinary worker concurrency uses
different processes. Nested and concurrent captures in the same process fail
closed. Audit case children are a separate execution boundary.

## Prepare a fresh immutable artifact

Run the tracked CLI in the trusted deployment preparation step. It performs
static `readelf` checks of ELF class/data/machine, target SONAME, consumer
DT_NEEDED and all required SQLite exports. It neither imports the NPU stack nor
loads either inspected ELF. Runtime pin checks still require no `readelf` or
access to deployment preparation tools. Before a configured capture, the engine
also runs a bounded fresh-child loader probe with the same interpreter: it loads
the pinned consumer and verifies the selected SQLite symbol's eager relocation
and provider path. A global symbol lookup alone cannot establish that the
consumer actually bound to the alias.

The probe uses glibc binding diagnostics and `dladdr`; failure to establish the
binding stops capture. The probe owns a separate process group and cleans up
its descendants on success, failure and timeout. `LD_PRELOAD` addresses the observed legacy interpreter
`DT_RPATH` precedence over a directory prefix. See the Linux
[loader rules](https://man7.org/linux/man-pages/man8/ld.so.8.html) and
[symbol lookup semantics](https://man7.org/linux/man-pages/man3/dlsym.3.html).
This is a same-interpreter preflight, not proof of every vendor export child;
actual NPU capture and export still need the acceptance checks below.

Set the paths and SHA256 values from the actual target host before preparation:

```bash
python scripts/prepare_profiler_runtime.py \
  --output /absolute/new-service-owned-runtime \
  --sqlite-library "$SQLITE_LIBRARY" \
  --sqlite-library-sha256 "$SQLITE_LIBRARY_SHA256" \
  --sqlite-consumer "$SQLITE_CONSUMER" \
  --sqlite-consumer-sha256 "$SQLITE_CONSUMER_SHA256" \
  --msprof "$MSPROF" \
  --msprof-sha256 "$MSPROF_SHA256" \
  --cann-set-env "$CANN_SET_ENV" \
  --cann-set-env-sha256 "$CANN_SET_ENV_SHA256"
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
with the alias directory or preload. The engine derives both only while executing
the original profiler context. An unconfigured historical deployment retains
its previous behavior; that compatibility does not authorize omitting this
pair from an experiment that requires it.

Keep the original sandbox manifest bytes and SHA unchanged. Its environment
reset (`--clearenv`, or the legacy explicit reset) does not forward new parent variables. Prepare a separate sparse
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
complete profiler pair and/or all four reference-cache variables; arbitrary environment
keys, mounts and source paths are rejected. For cache-on execution add the CLI
options `--cache-dir`, `--cache-index`, `--cache-sha256`, `--cache-state-dir`.
The dedicated invalidation directory must exist and contain only the declared
deployment/card invalidation namespace. Cache-off and cache-on diagnostics need
separately pinned overlays and consistent parent configuration.

The service derives read-only visibility from the pinned artifact paths, keeps
the entire dedicated profiler directory read-only with its real `0555` modes,
and exposes the source SQLite/consumer/msprof files at their declared paths.
The reference cache exposes only its `cases` and `blobs` directories, the index and pinned
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

After a [runtime module migration](recovery.md#runtime-module-migration), prepare
new pinned artifacts for the new evaluator build; keep active deployments intact.

Regenerate the cache deployment/index for the implementation being deployed.
Record the actual service revision, helper SHA and effective cache algorithm
identity. Assign a new evaluator build whenever the execution environment or
measurement implementation changes. Bind the
actual runtime manifest, alias, ELF observations, source SQLite ELF, consumer,
resolved msprof and complete CANN/torch_npu parser/native dependency closure in
`runtime_files`. Bind the changed benchmark and fixed service-owned helper in
`algorithm_files`. The only permitted sibling key is
`../runtime/profiler_runtime.py`, resolving to
`triton_evaluation/runtime/profiler_runtime.py` within the validated service root;
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
loader prefix/preload and their environment-value SHAs, verified alias-binding
preflight evidence, callback completion and restoration of both loader variables.
The replay export tool applies and probes the same loader environment only for
its export child, recording the verified provider and environment differences.
This proves that the measurement entered the configured context; an environment pair
alone does not. Bind this artifact and the raw profiler outputs to device
acceptance observations.

Run the original sharp NPU probe from an exact GitHub checkout in a fresh
attempt directory with one idle physical card selected before Python starts:

```bash
source /usr/local/Ascend/cann/set_env.sh
ASCEND_RT_VISIBLE_DEVICES="$CARD" python scripts/probe_profiler_latency.py \
  --sandbox --device "$CARD" --output /absolute/new-profiler-runtime-probe
```

Both runtime variables and the new overlay pair must already be exported by the
deployment environment, alongside the pinned base configuration. `--sandbox` uses
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

## Derive the existing CANN environment without changing the base manifest

A sandbox manifest may still contain paths for a removed historical CANN
installation. Selecting the pinned `msprof` alone does not fix its runtime,
compiler, OPP or Python paths. The preparation command above includes the pinned
`--cann-set-env` and `--cann-set-env-sha256` options for the target
installation. Supply both when the child must derive that CANN environment.

The trusted preparation step checks that this source, the pinned profiler and
its consumer belong to one canonical versioned CANN directory. It runs that
existing vendor script once in a clean environment with system utility paths,
the existing driver paths and empty Python paths. It captures only
`ASCEND_AICPU_PATH`, `ASCEND_HOME_PATH`, `ASCEND_OPP_PATH`,
`ASCEND_TOOLKIT_HOME`, `TOOLCHAIN_HOME`, `PATH`, `LD_LIBRARY_PATH` and
`PYTHONPATH`. This step does not import Torch or execute NPU work. The script
is not sourced in workers.

The immutable `cann-environment.json` artifact records those eight values,
required and optional directory observations, the command's return code and
stderr digest, actual version metadata and the resolved bytes of the four
required runtime libraries. The profiler manifest binds it and the original
vendor script through the complete quartet `cann_set_env_path`,
`cann_set_env_sha256`, `cann_environment_path`, `cann_environment_sha256`.
Workers check the artifact, source, libraries and directory conditions before
using them. A missing required runtime, compiler, OPP, Python or driver directory
fails closed. Optional profiler/compiler-adjacent tool directories may be absent;
their observed existence and resolved paths remain explicit.

The overlay still accepts only the profiler pair and the four cache variables.
Its implementation derives the five CANN scalar values and three path lists
from the pinned artifact. It removes old CANN entries only within the same
installation parent and the vendor's `cann`, `cann-` or `cann_` directory
boundary. It preserves anchored non-CANN entries, driver paths, language values
and the current service package's Python path. It does not change the original
manifest's bytes, SHA or unrelated base settings. The fixed canonical CANN and
driver roots are exposed read-only when the original runtime mounts do not
already cover them; arbitrary overlay environment and mounts remain forbidden.

Both the prepared boundary and the actual helper record the effective eight-key
CANN map and its SHA, source/artifact pins and directory observations. The
helper verifies actual child scalar values, required path entries and the
resolved profiler before measuring. Each of the eight sharp-probe callback
artifacts must match that actual map. Prepared evidence remains
`executed=false`; the native report and callback artifacts establish execution.
The source, environment artifact, metadata and runtime-library pins belong in
the numerical deployment's `runtime_files`. The index-dependent overlay remains
an independent final admission/freeze binding and is not added to that
numerical closure. Historical manifests without the quartet remain readable;
a deployment claiming the derived CANN environment must require the quartet
and verify it in the actual child.
