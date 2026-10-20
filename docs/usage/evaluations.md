# Submit and recover evaluations

Send `definition` and `workloads` to `POST /evaluations`, with at least one of
`definition.reference` (Torch) or `triton_code`. This is the only public input
protocol. The named workload format follows NPUKernelBench's pinned
[definition](https://gitcode.com/Ascend/ascendc-kernelgen-data/blob/0607a29d231cde384c9dfdb8a39f54f91837c327/npu-kernelbench/src/npu_kernelbench/data/definition.py)
and [workload](https://gitcode.com/Ascend/ascendc-kernelgen-data/blob/0607a29d231cde384c9dfdb8a39f54f91837c327/npu-kernelbench/src/npu_kernelbench/data/workload.py)
models. The service does not require the upstream package, a task catalog or task
registration. Supplied code and input generators execute only in worker attempts.

## Describe a task and its workloads

```json
{
  "definition": {
    "name": "add_one",
    "axes": {"N": {"type": "var"}},
    "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
    "outputs": {},
    "reference": "def run(x): return x + 1"
  },
  "triton_code": "... actual source defining run(x) ...",
  "workloads": [
    {"uuid": "small", "axes": {"N": 1024}, "inputs": {"x": {"type": "random"}}}
  ],
  "seed": 200
}
```

The two implementations receive positional arguments in `definition.inputs`
order. Preserve that order when serializing the definition. Workloads are ordered,
with result `case_idx` values starting at zero. UUIDs are retained when supplied;
otherwise the service derives `case-1`, `case-2`, etc. Source, workload order,
input order, generation code, custom source and seed are bound into request and
reference identities. Reordering definition inputs can change argument and RNG
order, so it is not an interchangeable request.

A workload descriptor overrides its input's definition shape/dtype. Definitions
can use constant, variable or arithmetic-expression axes. Axis evaluation is
bounded metadata processing, not Python `eval`. As upstream, expressions are
evaluated once in definition order (an expression may use only axes defined
before it) with `+ - * / // % **`; the only added bounds refuse integers wider
than 1024 bits and resolved values beyond ±(2³¹−1). A missing workload input uses
definition defaults; if neither default shape nor dtype exists, it supplies
`None`, matching the official protocol rather than a Python function default.
Definitions may have empty inputs for zero-argument operators. There are at most
128 workloads and 128 named inputs; `/docs` describes other validation bounds.

| Input kind | Meaning |
| --- | --- |
| `random` | Concrete or inherited shape/dtype, optional distribution parameters, integer `range`, and special `fill` |
| `scalar` | JSON scalar, list or explicit null; nested values must be finite JSON |
| `tensor_list` | One argument containing tensors of a common dtype and per-tensor `shapes` |
| `custom` | A named value returned by the explicitly declared custom generator |

Generation resets Python, Torch and NPU RNGs for every workload, using top-level
`seed` (default `200`), and produces tensors on the assigned NPU. For ordinary
floating inputs without a distribution override, all inputs in a workload share
one normal/uniform family choice. Explicit `mixed` selects per tensor. Normal
means/stddevs, BF16 generation/casting and RNG draw order follow the pinned
[official generator](https://gitcode.com/Ascend/ascendc-kernelgen-data/blob/0607a29d231cde384c9dfdb8a39f54f91837c327/npu-kernelbench/src/npu_kernelbench/evaluator/generator.py).
There is no legacy CPU standard-normal input mode.

Distributions are `mixed`, `randn`, `uniform`, `normal`, `zeros`, `ones` and
`uniform_signed`. Optional `low`/`high`, `mu_low`/`mu_high` and
`sigma_low`/`sigma_high` tune their ranges. `fill` accepts `inf`, `-inf`, `nan`
and `zero` without putting nonfinite numbers into JSON. Integer `range` uses an
exclusive upper bound; as upstream, `range` on float and bool inputs is ignored.
Admission rejects only bounds that upstream refuses for every seed: an explicit
`uniform`, `uniform_signed` or `normal` whose bounds torch cannot sample, or an
integer `range` torch cannot draw. Every other combination reaches the generator
and behaves exactly as upstream, including its seed-dependent failures. Such a
failure, or a Python-level error, missing callback or non-dictionary result in
custom input code, ends the check with non-retryable `INPUT_GENERATION_FAILED`
naming the workload; the card is not suspected and the job is not requeued.
Torch `RuntimeError`s and memory exhaustion keep the retryable engine-failure
path, because the device can cause those too.
Device support is still required for the selected dtype;
the request cannot manufacture unavailable NPU capabilities.

The supported named dtypes are float16, bfloat16, float32, the official FP8/FP4
names, int8/int16/int32/int64 and bool. Upstream has no float64 random branch;
float64 is accepted only with `fill`. uint8 is not in the official dtype map.
Unsupported combinations fail validation or device execution explicitly, without
silently casting to another dtype.

## Custom inputs and unsupported resources

Declare `definition.custom_inputs_entrypoint` as a function name, e.g.
`make_inputs`. By default the function is read from `definition.reference`;
`custom_inputs_code` can supply an explicit replacement module or support a
Triton-only request. It receives `(resolved_axes, torch.device)` and must return
a dictionary keyed by input name. It executes only when a custom workload is
materialized in the worker, in a fresh namespace with the official seed schedule.
A workload cannot mix custom and non-custom input descriptors.

The service never interprets a custom entrypoint as a local file path.
`safetensors` input paths are rejected before admission: this HTTP protocol has
no uploaded-artifact transport. Do not send paths from a client filesystem and
expect the worker to resolve them.

A workload that declares `tolerance` is judged by the pinned official check:
the declared fields, with unset ones taken from its dtype standard (atol/rtol,
a 0.99 matched ratio and an absolute-error cap widened to 32 ULP), equal output
dtypes, no NaN/Inf except matching `-inf` under `allow_negative_inf`, and no
all-zero output. Workloads without a declared tolerance keep this service's own
accuracy policy, and `non_compute` still requires binary equality. All 105
official problems at the pinned commit can be submitted unchanged. As upstream,
workload `perf_baseline` metadata and inputs the definition does not declare are
ignored. Apart from declared tolerances, input generation compatibility does not
mean the service uses the upstream framework's accuracy policy, scheduler or
profiling implementation; this service retains its own correctness and
measurement algorithms.

## Submit

Load existing official files directly; the JSONL file becomes a `workloads` array:

```bash
python3 - <<'PYREQUEST'
import json
from pathlib import Path
body = {
    "definition": json.loads(Path("definition.json").read_text()),
    "workloads": [json.loads(line) for line in Path("workload.jsonl").read_text().splitlines()
                  if line.strip()],
    "triton_code": Path("triton_impl.py").read_text(),
    "seed": 200,
}
Path("request.json").write_text(json.dumps(body))
PYREQUEST

curl -sS -i "$EVALUATION_URL/evaluations?wait_seconds=0" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: example-evaluation-1' \
  --data-binary @request.json
```

For Torch-only measurement, omit `triton_code`. For Triton-only measurement,
omit `definition.reference`. The bundled client provides `--triton-only` to
remove that source from a loaded definition without modifying its file.

`checks` defaults to correctness and performance with both sources, or performance
with one. `/verify` requires both sources; `/benchmark` and `/profile` also support
single implementations. A single-source result reports `correctness_verified:
false` and no comparison speedup. Compile checks remain syntax/import checks;
they do not generate inputs or establish device correctness.

`Idempotency-Key` and `X-Session-ID` are optional. A key enables recovery-safe
retries on the same instance; without a key each submission is new work. A session
groups quotas and is not an authorization credential.

## Select checks and retrieve evidence early

The caller selects checks, cases, stopping rules and budgets. The service does
not decide whether to keep or optimize a candidate. Existing shortcuts remain
available; `POST /evaluations` combines requested capabilities.

For correctness, optional `correctness.case_indices` selects original zero-based
cases and `correctness.stop_on` is `complete` (default) or `first_failure`.
Selected runs return `coverage.selected_case_indices`, `evaluated_case_indices`
and `stop_reason`. Unexecuted cases are not failures. Passing a subset yields
`outcome: incomplete`, `passed: false`; only a complete task pass yields `correct`.
A proven mismatch can yield `wrong` before other selected cases run. Partial
correctness cannot authorize comparative performance or grade eligibility.

`GET /evaluations/{evaluation_id}/progress` uses the same read authorization as
resource recovery. It returns `attempt_id`, `active_check`, `progress_revision`
and validated `completed_checks`, including results obtained while later checks
are still running. Replacement attempts invalidate earlier progress and advance
the progress revision. The existing resource endpoint keeps its terminal-only
`result` contract. Clients should retain the evaluation ID and poll either
endpoint; observing partial evidence does not cancel accepted work.

For timing plus a kernel-duration trace, request:

```json
{
  "checks": ["correctness", "performance", "profile"],
  "performance": {"method": "event", "warmup": 5, "repeats": 20},
  "profile": {"mode": "trace", "case_indices": [0]}
}
```

Add the ordinary definition, workloads and source fields. One candidate engine
process handles timing and trace. Event timing finishes before profiling starts,
and its checkpoint becomes visible through progress. Profiler timing reuses its
existing captures for overlapping trace cases. Other trace cases are collected
explicitly; neither selection silently widens the other. Later trace failure
does not discard already valid timing. Reference-only baseline production keeps
its isolated process. Standalone trace uses warmup 5/repeats 20; combined trace
uses the requested timing warmup/repeats.

`trace` reports kernel durations from `torch_npu.profiler`, labeled
`profiler_kernel_sum_v1`. It does not supply `msprof op` or deep hardware counters.
`op` and `deep` retain their collection methods. Explicit multi-case profiling
uses one outer engine invocation; native msprof capture processes remain separate.
For deep profiling, `metric_groups` selects any nonempty subset of
`PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `MemoryUB`,
`L2Cache`, and `ResourceConflictRatio`; `include_per_core: false` omits the
sampling pass. Results expose the requested scope and collected raw metrics.
`max_kernels_per_case` bounds returned kernel groups, not collection launches;
truncation produces partial evidence with an explanatory error.

## Reuse a completed correctness observation

Set `correctness_from` to a completed evaluation ID while requesting correctness
and performance for the same unchanged candidate. Reuse is explicit and
same-owner: the source must be an original complete `correct` observation, with
the same candidate, reference/input-generation bytes, seed, tolerances,
`non_compute`, evaluator build and correctness/decoy algorithms. The worker
rechecks environment, recorded physical card and freshness inside its lease.
Freshness is bounded by `max_recovery_age_seconds`, measured conservatively from
the source execution start; a later profile cannot refresh the observation.

The destination may wait for the source's recorded card under its ordinary
deadline. Missing or expired evidence becomes a nonretryable error instead of
silently triggering a fresh verification. Already admitted idempotent replays
remain recoverable. The returned correctness result identifies `reused_from`;
chains of reused evidence and grade reuse are rejected. The engine receives a
bound evidence artifact, not a synthetic fresh `verify_result.json`.

## Profiler benchmark feedback

Set `performance.method="profiler"` to receive the implementation's kernel latency
and pipe-utilization counters from the existing benchmark capture. The client
supports `benchmark --method profiler`; Event timing remains the default.

Each measured `result.performance.data.per_case` item can contain `kernel_metrics`,
keyed by the kernel names observed in that implementation's capture:

```json
{
  "kernel_metrics": {
    "_bmm_kernel": {
      "latency_ms": 0.02,
      "aic_cube_ratio": 0.42,
      "aiv_vec_ratio": null
    }
  }
}
```

Each entry includes mean kernel latency in milliseconds and available AIC/AIV
Cube/MAC, vector, scalar, MTE and Fixpipe ratios. Ratios are fractions in `[0, 1]`;
explicit percentage columns are normalized. `cube_utilization` is separate from
Cube/MAC pipe ratios. Missing, nonnumeric, nonfinite or out-of-range counters stay
`null`. Cube and MAC vendor columns retain distinct names.

Counters use the same final active-repeat rows as the existing kernel-duration
calculation. They describe individual kernels in this case, without averaging
across cases or making a whole-device utilization claim. All implementation
kernels are retained, including native kernels called by candidate code.

Event responses and older persisted results omit `kernel_metrics`. Single-source
profiler benchmarks return counters for the selected Torch or Triton implementation.
Comparison benchmarks return fresh candidate counters even when Torch timing
comes from the baseline cache. No additional profiling pass is launched. This
feedback does not establish the broader evidence returned by `profile_op` or
`profile_deep`.

## Reuse Torch baseline timings

Comparison performance defaults to `performance.use_baseline_cache=true`.
A hit reuses Torch latency; a miss measures and stores it. Candidate timing
executes; correctness executes unless an explicit validated `correctness_from`
is supplied. Only latency scalars are retained, not reference outputs.

Set `performance.use_baseline_cache=false` for a fresh baseline used only by this
evaluation, leaving existing cache entries untouched. Use a new idempotency key
for a new evaluation; replaying an accepted request does not rerun it.

Reference source, normalized workloads, input generator code, custom source and
seed affect cache identity, along with method/warmup/repeats. Changing only the
candidate can reuse the baseline. `performance.case_indices` can incrementally
fill selected timings. Each measured case reports `baseline_cache_hit`; older
stored results may omit it. Single-source measurements always run fresh.
Cache/producer failures fall back to ordinary measurement, while deadlines and
lease loss still stop execution. See [deployment scope](../operations/deployment.md#torch-baseline-cache).

## Recover results

`wait_seconds=0..60` bounds the HTTP wait. HTTP 202 returns an admitted resource
and `Location`; HTTP 200 returns a terminal resource when available. Disconnects
and wait expiry do not cancel accepted work.

```bash
curl -sS -i "$EVALUATION_URL/evaluations/$EVALUATION_ID" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")"
```

Pending resources include `Retry-After: 1`; unknown or invisible IDs return 404.
Terminal states are `completed`, `failed` and `timed_out`. Inspect each check:
completed work can establish that the candidate is wrong, while infrastructure
failure establishes no correctness verdict.

When an engine exits without usable output, its `EVALUATOR_EXECUTION_FAILED`
message includes the exit code and available stderr/stdout tails. Stderr is
listed first. The excerpts are collapsed to one line and share the error's
512-byte UTF-8 budget with the leading explanation; they are not full logs.
They provide diagnostic evidence, not an attribution of fault to the candidate.

If valid benchmark output contains no usable measurements, performance fails
with retryable `INTERNAL` and bounded per-case diagnostics. Earlier successful
correctness evidence remains intact, and a subsequent profile check is skipped
with `PRIOR_CHECK_FAILED`. Malformed engine output still reports non-retryable
`INVALID_ENGINE_OUTPUT`.

The old public `torch_code` and positional `cases` body is retired. Existing
internal v2.1 records remain recoverable; they do not enable legacy submissions.
Follow the [upgrade procedure](../operations/recovery.md#upgrading) and deploy
matching API, worker and client builds. Native `/openapi.json` and `/docs` are the
live schemas; no separately generated schema is required to start the service.
