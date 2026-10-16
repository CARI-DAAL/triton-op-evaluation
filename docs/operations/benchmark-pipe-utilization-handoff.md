# Server-agent handoff: benchmark pipe utilization

Task: pull `feat/benchmark-pipe-utilization` from
`https://github.com/CARI-DAAL/triton-op-evaluation.git`, validate on the server,
and return a compact result. The implementation is based on `312bbc7`.
Keep raw requests, responses, profiler files and test logs on the server.

## Design decisions already made

Implement these semantics as written; do not redesign them during validation.

- Benchmark capture settings, warmups/repeats, latency calculation, correctness
  thresholds, case selection and speedup aggregation are unchanged. No extra
  NPU capture is added. Event remains the default; counters require profiler timing.
- The response field is
  `result.performance.data.per_case[].kernel_metrics[kernel_name]`.
  It contains `latency_ms` plus available AIC/AIV Cube/MAC, scalar, vector,
  MTE and Fixpipe ratios, and a separate `cube_utilization` value.
- Ratios are fractions in `[0, 1]`. Only explicitly percentage-labelled CSV
  columns are divided by 100. Missing/invalid counters are `null`, never zero.
  Cube/MAC vendor names are kept separate, not guessed to be aliases.
- Metrics use the same final active-repeat rows as latency, per kernel and case.
  No averages across cases or whole-device utilization claims. All kernels
  observed in the implementation capture are included, including native kernels.
- Comparison results contain fresh **candidate** counters. Torch baseline caching
  still stores latency scalars only. Single-source profiler results contain that
  implementation's counters. Event/older results omit the new field.
- The CLI supports `benchmark --method profiler` and passes JSON counters to the
  agent. Op/deep profile endpoints retain their separate evidence and behavior.
- The public schema adds optional response fields; old golden response bytes
  remain unchanged. `measure_single` keeps three-value unpacking/indexing while
  attaching metadata to its measurement record.

## Prepare the server checkout

Preserve existing changes; use a clean worktree if necessary. Fetch the feature
branch and record its actual HEAD. Use the existing validated NPU Python/CANN
environment rather than reinstalling the device stack. Update API and workers
together, and stamp the actual evaluator build. Renew source pins through the
normal upgrade procedure where required; do not bypass runtime validation.
Use an isolated validation instance/card, not an unrelated production service.

```bash
git fetch origin feat/benchmark-pipe-utilization
git switch feat/benchmark-pipe-utilization
git pull --ff-only origin feat/benchmark-pipe-utilization
git rev-parse HEAD
python -m ruff check .
python -m compileall -q triton_evaluation scripts plugins
python -m pytest -m 'not npu' -q > /tmp/pipe-counter-cpu-tests.log 2>&1
git diff --check
```

If the local feature branch does not exist, use
`git switch --track origin/feat/benchmark-pipe-utilization` for the switch step.
The root agent's Windows checks cover CSV parsing, invalid/missing counters,
Event exclusion, cached-reference candidate metrics, standalone output, HTTP
serialization, CLI method selection/pass-through and contract compatibility.
Full CPU validation requires Linux/Torch; complete that on this server.
The existing Linux permission (`0600`), SIGINT recovery and descriptor-relative
staging tests cannot run faithfully on Windows; do not weaken those checks.
Final local targeted run: **168 passed, 20 skipped, 2 deselected** (Windows,
95.01 s). The two exclusions were the Linux permission/SIGINT CLI checks.
Ruff, compilation and all regenerated contract-artifact byte checks passed.
This is not a full Linux suite or device-validation result.

## Small NPU smoke before any full gate

Avoid rerunning the roughly two-hour five-stage gate just to check this feature.
Prepare one workload from the tracked official BatchMatmul fixture; selecting
one performance case from a 51-workload request would still verify all 51.

```bash
PIPE_CHECK_DIR=$(mktemp -d /tmp/pipe-counter-smoke.XXXXXX)
export PIPE_CHECK_DIR
python - <<'PY'
import json, os
from pathlib import Path
source = Path('tests/fixtures/npu_kernelbench/batchmatmul')
target = Path(os.environ['PIPE_CHECK_DIR'])
(target / 'definition.json').write_bytes((source / 'definition.json').read_bytes())
workloads = [json.loads(line) for line in (source / 'workload.jsonl').read_text().splitlines()]
(target / 'one-workload.jsonl').write_text(json.dumps(workloads[0]) + '\n')
(target / 'candidate.py').write_text(
    (source / 'candidate.py').read_text() + '\n\ndef run(A, B):\n    return _bmm(A, B)\n'
)
PY
TRITON_EVAL_CLI=plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py
python "$TRITON_EVAL_CLI" benchmark --method profiler --no-baseline-cache \
  --definition "$PIPE_CHECK_DIR/definition.json" \
  --workloads "$PIPE_CHECK_DIR/one-workload.jsonl" --triton "$PIPE_CHECK_DIR/candidate.py" \
  --timeout 1200 > "$PIPE_CHECK_DIR/profiler.json" 2> "$PIPE_CHECK_DIR/profiler-client.log"
python "$TRITON_EVAL_CLI" benchmark --method event --no-baseline-cache \
  --definition "$PIPE_CHECK_DIR/definition.json" \
  --workloads "$PIPE_CHECK_DIR/one-workload.jsonl" --triton "$PIPE_CHECK_DIR/candidate.py" \
  --timeout 1200 > "$PIPE_CHECK_DIR/event.json" 2> "$PIPE_CHECK_DIR/event-client.log"
```

Set `EVALUATION_URL` and `EVALUATION_CAPABILITY_FILE` to the private test instance
before these commands; do not put tokens in logs or the handoff. Default repeats
remain 5 warmups/50 active iterations. Inspect actual check outcomes, build,
method and coverage, not just HTTP/resource terminal state.

For profiler: correctness must pass, coverage must be 1, both latencies positive,
and candidate `kernel_metrics` must contain observed kernel names. At least one
finite utilization counter is needed to claim hardware counter availability;
a real measured zero is valid. For Event: `kernel_metrics` must be absent.
Optionally repeat profiler with caching enabled twice: candidate counters must
remain fresh when `baseline_cache_hit` becomes true. Do not clear shared caches.

If hardware returns all-null counters, keep that truthful result. Report the
local CSV header names and runtime versions; do not fabricate values, guess units,
loosen accuracy/CV thresholds, or switch the timing backend to force a pass.
If the original capture was cleaned up, use the existing local profiler probe
with retained artifacts to inspect the CSV. No raw archive needs to leave the server.

## Return only a small summary

Send: commit/build; Python/Torch/torch_npu/CANN/card identity; full CPU test result;
profiler/Event correctness, coverage and wall seconds; one kernel name/latency
and its available ratios; cache-hit status if checked; any failure plus the
relevant last 20 log lines. Keep the complete JSON, CSVs and logs at the printed
`PIPE_CHECK_DIR` path. Do not claim full 51-case/op/deep gate validation from this
one-case smoke. Architecture, field names and timing semantics remain decisions
for the root agent; the server agent owns setup, execution and local evidence.
