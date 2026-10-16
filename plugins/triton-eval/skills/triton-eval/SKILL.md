---
name: triton-eval
description: Measure or profile PyTorch or Triton operators, compare Triton against PyTorch for correctness and speedup, or recover an evaluation from a deployed triton-evaluation service.
---

# Triton evaluation

Use the bundled [client](scripts/triton_eval.py) with Python 3.10+; it needs only
the standard library. The HTTP service is deployed separately. Obtain its URL
and the caller capability-file path from the user's existing configuration:
`EVALUATION_URL` and `EVALUATION_CAPABILITY_FILE`, or the equivalent `--url` and
`--capability-file` flags. Resolve the script relative to this `SKILL.md`, then
invoke its absolute path from the directory containing the user's inputs.

## Submit an operator

Use the official definition JSON and workloads JSONL files unchanged for a
comparison. A JSON array of workload objects is also accepted. The definition
contains `axes`, ordered named `inputs`, `outputs`, and optional `reference`
Python source; `name` is optional. Each workload contains `uuid`, `axes` and
named `inputs`. Random inputs carry `type: "random"`, `shape` and `dtype`.
The request seed defaults to 200; set `--seed` to reproduce a different seed.

Choose `verify` with a reference in the definition and a candidate Triton
source. `benchmark` measures one implementation, or compares correctness and
speedup when both are present. `profile` accepts one or both implementations.
Each implementation defines `run` with arguments matching definition input
order. With `TRITON_EVAL_CLI` set to the resolved client path:

```bash
python3 "$TRITON_EVAL_CLI" verify \
  --definition definition.json --workloads workloads.jsonl \
  --triton triton_impl.py --save-handle evaluation-handle.json
```

For Torch alone, use the reference in the definition without `--triton`.
For Triton alone, use `--triton-only` to remove the reference from the submitted
definition while preserving the official file on disk:

```bash
python3 "$TRITON_EVAL_CLI" benchmark --definition definition.json --workloads workloads.jsonl
python3 "$TRITON_EVAL_CLI" profile --definition definition.json --workloads workloads.jsonl \
  --triton triton_impl.py --triton-only
```

When workloads use custom input generators, pass their UTF-8 Python source with
`--custom-inputs custom_inputs.py`; it becomes `custom_inputs_code` in the request.

For custom workloads, declare `custom_inputs_entrypoint` in the definition.
With `--triton-only`, the client preserves the original reference source as
`custom_inputs_code` for that factory; the submitted definition omits its
mathematical reference. An explicit `--custom-inputs` file takes precedence.

The client reads the capability from its file and sends it only as the
Authorization credential. Keep credentials out of source files, workloads, output
artifacts and pasted logs. The supplied definition, workloads and source files are sent to the
configured evaluation service; importing them locally is unnecessary.

For two-source `benchmark`, Torch baseline timing is cached by default. Use
`--no-baseline-cache` for a fresh comparison baseline without reading or writing
the cache. Candidate timing is always measured. Preserve per-case
`baseline_cache_hit` when reporting comparison evidence. Single-source benchmarks
always measure fresh and never read or write this cache; accepting the flag does
not establish a cache hit.

Use subcommand `--help` for the available options. Benchmarks default to Event
timing. For per-kernel latency and pipe-utilization
feedback, choose `benchmark --method profiler`. Inspect
`result.performance.data.per_case[].kernel_metrics`, keyed by implementation
kernel name. Ratios are fractions in `[0, 1]`; missing counters are `null`. These
counters come from the benchmark capture without an extra `profile` request;
they do not establish op/deep profiling coverage. Event and older results omit
the field. Candidate counters remain fresh when Torch baseline timing is cached.

The client normally
polls for up to 960 seconds with a 1-second interval; `--timeout`,
`--poll-interval` and `--no-wait` control that client behavior. An elapsed client
wait does not cancel accepted work.

## Combine checks and control coverage

Use `evaluate` for explicit combinations. For correctness, event speedup and a
kernel-duration trace in one request:

```bash
python3 "$TRITON_EVAL_CLI" evaluate \
  --definition definition.json --workloads workloads.jsonl --triton triton_impl.py \
  --checks correctness performance profile --profile-mode trace --profile-cases 0 \
  --warmup 5 --repeats 20 --measurement-method event \
  --wait-for performance --save-handle evaluation-handle.json
```

`--wait-for performance` returns its progress snapshot as soon as timing evidence
is available; recover the later trace with `get` using the saved handle.
Timing is published before later trace capture. `trace` supplies kernel durations;
use `op` or `deep` for hardware counters. For a deep capture, select cases and
`--metric-groups`, optionally `--no-per-core`, according to the evidence needed.
`--max-kernels-per-case` limits returned groups, not capture work.

`verify --case-indices 0,3 --stop-on first_failure` requests bounded correctness
coverage. Read `coverage` and `outcome`: `incomplete` is a subset observation,
not a whole-task pass. The CLI returns status 2 for incomplete correctness.
For `evaluate`, use `--correctness-cases`, `--performance-cases`, and
`--profile-cases` to express independent selections.

An unchanged candidate can use `benchmark --correctness-from EVALUATION_ID` to
reference its original complete correctness observation. The service validates
owner, sources, environment, coverage and freshness; results mark `reused_from`.
An explicit refusal requires the caller to choose a fresh verification. Reused
evidence cannot be chained or used for a grade.

Use `progress EVALUATION_ID` to retrieve completed checks while other checks are
running. Track `attempt_id` and `progress_revision`; an attempt replacement
invalidates its predecessor's partial observations. A progress command is a
snapshot and does not cancel or wait for the remaining work.

## Identify the service and this client

`--version` reports this client's own version offline. `version` asks a
deployment for its service version, the API contracts it implements, its build
and whether it verified its own release:

```bash
python3 "$TRITON_EVAL_CLI" version --url SERVICE_URL --capability-file "$EVALUATION_CAPABILITY_FILE"
```

Report both versions when an evaluation outcome depends on the deployment, and
report `release.verified: false` as an unverified deployment identity.

## Recover and interpret results

Once an `evaluation_id` is known, use `get` on its original service URL:

```bash
python3 "$TRITON_EVAL_CLI" get EVALUATION_ID --url ORIGINAL_SERVICE_URL
```

`--save-handle` records `url`, `evaluation_id`, `idempotency_key` and `session_id`.
Retain these values for recovery; a missing session is recorded as null. If a POST gets
no response, retry only with the same definition, workloads, source bytes, options, session and
`--idempotency-key`. A different key creates new work. After bounded retries or
a service outage, report the recoverable handle or original request identity;
resume recovery when the service is available.

Stdout is one JSON result; stderr contains progress and request identity. Read
the returned state and per-check evidence. A pending resource remains pending,
and `completed` alone does not mean the candidate passed. Report a failed
correctness verdict as a candidate outcome; report transport or infrastructure
failure separately, without inventing a correctness score. Preserve available
measurement method and profile evidence when summarizing results.

Single-implementation results identify `implementation: torch` or `triton` and
`correctness_verified: false`. Performance reports per-case `latency_ms` without
reference speedup; profiling reports its own evidence. A zero CLI exit status
for those operations means measurement success, not verified correctness. Keep
that limitation in the user-facing result even when timing or profiling succeeds.
Syntax/import prechecks also do not establish successful device JIT compilation.

Preserve partial-profile warnings. PyTorch profiling can include process
initialization and has bounded launch coverage; its observations do not establish
complete profiling of only the supplied `run` function.
