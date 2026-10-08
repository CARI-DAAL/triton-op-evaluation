# Submit and recover evaluations

Submit `torch_code`, `triton_code` and `cases` to `POST /evaluations`. Both code
strings define `run(*args)`. The service builds its engine interface from these
sources and cases; callers do not define `Model`, input factories, task IDs or
catalog manifests. Candidate and reference code execute only in worker attempts.

## Describe cases

`cases` is a nonempty list. Each case contains `inputs`, an ordered list of
positional arguments, and an optional `seed` (default `0`). Each input is either
a tensor descriptor (`shape`, optional `dtype` and `range`) or a JSON scalar:
integer, finite float, boolean, string or null. For example:

```json
{"inputs": [{"shape": [32, 64], "dtype": "float32"}, 2.0, true, null], "seed": 0}
```

The implementations receive four positional arguments for that case. `dtype`
defaults to `float32`; supported types are `float16`, `bfloat16`, `float32`,
`float64`, `int8`, `int16`, `int32`, `int64`, `uint8` and `bool`. `range` is
`[low, high]` with an inclusive lower and exclusive upper bound. Integer ranges
must have integral endpoints within the dtype's bounds; boolean tensors do not
accept ranges.

Without a range, floating tensors use a normal distribution, signed integers
use `[-5, 5)`, unsigned integers use `[0, 5)`, and booleans use random false/true
values. Generation is reproducible for the seed. `/docs` describes the validation
limits and optional execution controls.

## Submit

Put your implementations in `torch_impl.py` and `triton_impl.py`, each exporting
`run` with arguments matching the cases. Create the minimal request:

```bash
python - <<'PYREQUEST'
import json
from pathlib import Path

request = {
    "torch_code": Path("torch_impl.py").read_text(encoding="utf-8"),
    "triton_code": Path("triton_impl.py").read_text(encoding="utf-8"),
    "cases": [{"inputs": [{"shape": [1024], "dtype": "float32"}]}],
}
Path("request.json").write_text(json.dumps(request), encoding="utf-8")
PYREQUEST

# Set EVALUATION_URL and EVALUATION_CAPABILITY_FILE for your instance.
curl -sS -i "$EVALUATION_URL/evaluations?wait_seconds=0" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: example-evaluation-1' \
  --data-binary @request.json
```

Authentication uses the capability registry for every evaluation endpoint.
`Idempotency-Key` and `X-Session-ID` are optional. Provide a key for submissions
that may need retrying; repeat the same request and headers against the same
instance. Without a key, every submission is new work. With a key but no session
header, the service uses a stable default session; an explicit session groups
related requests and applies the session quota. A session grants no authority.

## Choose checks

`checks` defaults to `["correctness", "performance"]` on `/evaluations`.

| Intent | `checks` |
| --- | --- |
| Correctness | `["correctness"]` |
| Correctness and timing | `["correctness", "performance"]` |
| Profiling | `["profile"]` |
| Syntax/import precheck | `["compile"]` |

[Operation shortcuts](kernel-eval.md) select these checks through `/verify`,
`/benchmark` or `/profile`. Timing defaults to `event`; optional performance
settings can select `profiler`. Results identify the measurement method. Keep
methods separate when comparing measurements. A syntax/import precheck cannot
prove successful device compilation for every case.

Admission persists identity, quotas and deadline before any supplied code runs.
Compile-only requests use CPU workers. Requests with device checks use one NPU
worker for the whole attempt, including an explicitly requested compile check.

## Reuse Torch baseline timings

Performance checks default to `performance.use_baseline_cache=true`. A cache
hit reuses the Torch timing; a miss measures it and saves it for later
evaluations. Correctness and candidate timing still execute. Reference outputs,
input tensors and candidate results are not stored in this cache.

For a fresh baseline used only by this evaluation, submit:

```json
{"performance": {"use_baseline_cache": false}}
```

Include this option alongside the usual `torch_code`, `triton_code` and `cases`.
This bypasses both cache reads and writes, leaving any existing baseline intact.
Use a new idempotency key (or omit it) when requesting a new evaluation; replaying
an accepted request recovers its existing resource rather than rerunning it.

The cache identity includes Torch source, the complete normalized cases with
seeds, and the measurement method, warmup and repeat count. Changing only the
Triton candidate can reuse the baseline. Changing code, cases or timing settings
produces a different identity. Whitespace changes in source also cause a miss.
`performance.case_indices` chooses which cases to measure and can incrementally
fill the same task's cache; changing the full cases list creates a new task.

Each performance case reports `baseline_cache_hit`: `true` means its Torch timing
came from an existing cache entry, and `false` means it was measured for this
evaluation. Older stored results can omit the field. Redis cache errors fall
back to fresh measurement. If a reference-only measurement cannot produce usable
timings, performance falls back to the normal per-case measurement flow without
caching that output. The cache assumes a fixed instance environment; see
[deployment scope](../operations/deployment.md#torch-baseline-cache).

## Recover results

`wait_seconds=0..60` bounds the HTTP wait. HTTP 202 returns an admitted resource
and `Location`; HTTP 200 returns a terminal resource when the wait observes one.
The response retains the internal v2.1 durable resource format. An HTTP timeout
or client disconnect does not cancel accepted work.

```bash
# Set EVALUATION_ID from the submission response; poll the original instance.
curl -sS -i "$EVALUATION_URL/evaluations/$EVALUATION_ID" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")"
```

GET returns HTTP 200 with the current resource. Pending states include
`Retry-After: 1`; unknown or invisible IDs return 404. States are `queued`,
`running`, `completed`, `failed` and `timed_out`. For `completed`, inspect each
check result: a completed evaluation can establish that the candidate is wrong.
Infrastructure failure does not establish a correctness verdict.

`/livez` reports process liveness. `/readyz` checks Redis, credentials and
compatible CPU/NPU workers. Admission checks the capacity needed for the request.
`/internal/metrics` requires an observer capability.

## Validation and compatibility

FastAPI validates bodies, headers and query arguments before admission. Missing
fields, unsupported values or invalid check combinations return 422
`INVALID_REQUEST`. The 4 MiB body limit returns 413 `LIMIT_EXCEEDED`.
Authentication failures return 401; authorization failures return 403. Errors
include a request ID and a redacted explanation.

Python models and routes define the interface; `/openapi.json` and `/docs` are
generated from them. Ordinary requests do not first fetch the schema. Canonical
JSON is an internal identity requirement, not an HTTP whitespace requirement.

The former task-ID facade and public `reference_code`/`kernel_code` request
shape are retired. Existing internal durable records remain recoverable; this
does not make the old submission shape valid. Follow the
[upgrade procedure](../operations/recovery.md#upgrading) before switching clients.
