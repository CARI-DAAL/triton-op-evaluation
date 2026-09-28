# Isolating candidate execution

Every compile, correctness, performance and profile engine invocation passes
through `evaluators.base.run_script`. Workers that set
`TRITON_EVAL_ENGINE_SANDBOX_CONFIG=/absolute/path/manifest.json` run that invocation
inside Bubblewrap. Also set `TRITON_EVAL_ENGINE_SANDBOX_SHA256` to the manifest's
64-character lowercase SHA-256 hex digest, computed at deployment time. Every
invocation compares the exact manifest bytes to that pinned digest before
starting candidate code. A missing digest or changed manifest fails closed.
Do not recompute the expected digest for each invocation: that would permit
execution conditions to drift under the same evaluator build.
Absent sandbox settings preserve the existing host execution
mode. A configured but missing, invalid or unusable sandbox fails the invocation;
it never retries candidate code directly on the host.

This setting is a worker **process environment** contract, read at the common
subprocess boundary. It is not a `Settings` constructor field: passing a value
to an API or evaluator object does not configure worker subprocess execution.
Export it in each worker's launch environment, including separately launched
CPU workers.

The operator-owned manifest names immutable runtime dependencies, selected
runtime environment variables and Ascend control devices. It is deployment
configuration, never request or candidate input. Keep it outside attempt
directories and version its contents with the evaluator build.

```json
{
  "readonly_mounts": [
    ["/usr", "/usr"],
    ["/opt/python-runtime", "/opt/python-runtime"],
    ["/opt/evaluator/.venv", "/opt/evaluator/.venv"],
    ["/opt/evaluator/triton_evaluation/engine", "/opt/evaluator/triton_evaluation/engine"],
    ["/sys", "/sys"],
    ["/etc/ld.so.cache", "/etc/ld.so.cache"],
    ["/etc/ascend_install.info", "/etc/ascend_install.info"]
  ],
  "environment": {
    "ASCEND_HOME_PATH": "/usr/local/Ascend/cann",
    "ASCEND_OPP_PATH": "/usr/local/Ascend/cann/opp"
  },
  "shared_npu_devices": [
    "/dev/davinci_manager",
    "/dev/devmm_svm",
    "/dev/hisi_hdc"
  ]
}
```

Replace these illustrative paths with the deployed Python, engine and CANN
paths, including Python symlink targets and necessary runtime loader paths.
Resolve compiler aliases as well: on systems where `/usr/bin/ld` points to
`/etc/alternatives/ld`, mount the selected linker at that individual alias.
Mount individual runtime files or directories. Do not mount the service
checkout, historical jobs, other attempts, credential directories, host `/tmp`,
host `/run`, or the Docker socket. The manifest rejects whole host state roots;
the operator must also review narrower mounts for unintended data.

The engine sees its own attempt directory read/write, runtime mounts read-only,
private PID/network/IPC namespaces and temporary `/tmp` and `/run`. It receives
an allowlist of runtime variables, a private home/cache and no inherited service
credentials. NPU invocations receive only their assigned physical device plus
the shared Ascend control devices. CPU-only invocations receive no NPU devices.
Ascend renumbers the one exposed physical card to logical device zero inside
the sandbox. The worker lease and device provenance retain the physical card
number; `ASCEND_RT_VISIBLE_DEVICES=0` selects that card within the engine.
Capabilities are dropped and the root mount is read-only. Invocation deadline,
lease loss and process termination retain the existing cleanup path; the private
PID namespace also kills descendants that create a new session.

The host's result collection is part of this boundary. Correctness, benchmark
and profile outputs are read relative to the trusted attempt directory using
directory descriptors and `O_NOFOLLOW` for every parent and the final file.
Parent traversal, symlinked roots or parents, symlinked results and nonregular
files are unusable engine output. The host never follows a candidate-created
result link into another attempt or a host file. Descriptor-relative traversal
also prevents replacing an already checked parent from redirecting the read.
Restaging the frozen reference and implementation for a later check writes fresh
files and replaces entries relative to the same trusted directory descriptor.
Existing symbolic links or hard links therefore cannot redirect a host write or
modify another file when the modules are restored.

Ascend devices can retain ownership by a live mount namespace after the last
device file descriptor closes. A historical idle container can therefore make
a fresh engine sandbox fail with `EBUSY` at `/dev/davinciN` while host execution
still succeeds. Coordinate exclusive device ownership and release obsolete
containers; do not fix this by exposing host state or granting candidates
privileged device administration.

Before admitting work, verify both sides of the boundary: an intentionally
placed host canary is unreadable through the actual evaluation API, and a fixed
known-good kernel passes device correctness and the declared timing method.
Also verify timeout cleanup and record the manifest digest, evaluator build,
physical device assignment and test results. A passing CPU isolation test does
not establish NPU or profiler compatibility.
