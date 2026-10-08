# Tests

Run tests from the repository root. Compile and tensor tests need CPU Torch;
`.[test,dev]` provides NumPy for the tensor/cache bridge. Every Python version
in CI installs both:

```bash
python -m pip install -e '.[test,dev]'
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -m 'not npu' -q
```

To focus on tensor generation and both `run(...)` implementations:

```bash
python -m pytest tests/test_operator_inputs.py -q
```

The suite exercises real HTTP handling, operator input adaptation and durable state changes
with fakeredis. Process and NPU adapters are replaced where hardware is not the
subject of the test. Real local subprocess, sandbox and HTTP/TLS tests run when
their dependencies are available.

## Choose a focused check

| Area | Test files |
| --- | --- |
| Portable client plugin | `test_plugin_package.py`, `test_cli*` |
| Generated HTTP documentation | `test_openapi.py` |
| Admission, identity, durable state | `test_admission*`, `test_broker.py`, `test_idempotency.py`, `test_fencing.py` |
| Workers and recovery | `test_worker*`, `test_recovery.py`, `test_card_health*`, `test_engine_failure_requeue.py` |
| Evidence and scoring | `test_evaluation_evaluator.py`, `test_shaping.py`, `test_profile_evidence.py` |
| Host smoke and NPU-gate orchestration (CPU) | `test_smoke_script.py`, `test_npu_release_gate.py` |
| Operator requests and transport | `test_operator*`, `test_api*`, `test_durable_http.py` |
| Torch baseline timing reuse | `test_baseline_cache.py`, `test_baseline_engine.py`, `test_baseline_evaluator.py` |
| Engine, runtime and cache | `test_verify*`, `test_benchmark*`, `test_profile*`, `test_reference*`, `test_engine*` |

Test modules share helpers through `conftest.py` and narrowly scoped helper files.
The current flat test layout preserves those imports; module names identify the
behavior under test.

## Fixtures

`fixtures/engine/` contains private engine observations.
`fixtures/evaluation_api/2.1/` contains durable-record compatibility examples and
their schemas; these are regression data, not deployment inputs.
[`fixtures/legacy_candidate_defaults.py.txt`](fixtures/legacy_candidate_defaults.py.txt)
is a fixed reference consumed by active engine input-default regression tests.
It does not provide a dataset importer. Preserve hash-pinned fixture bytes when
changing these tests. Retired tool envelopes and bridge archives remain in
[Git history](../docs/history.md).

## External consumer and NPU checks

See the [NPU validation plan](../docs/operations/npu-validation.md) for the current
hardware coverage, proposed functional and recovery matrix, load/soak stages,
and acceptance criteria. It distinguishes existing commands from tests still
to be implemented.

The optional consumer checkout must support `torch_code`, `triton_code` and
`cases`. The previous source/factory submission interface is no longer accepted.

```bash
EVALUATION_CONSUMER_ROOT=/path/to/kernel-harness \
  python -m pytest tests/test_consumer_integration.py -q

# Against a configured Ascend deployment; this performs real device work.
python scripts/test_eval_service.py --url "$EVALUATION_URL" \
  --token "$(cat "$EVALUATION_CAPABILITY_FILE")" --no-speed
python scripts/npu_release_gate.py --service-url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" --output /tmp/npu-smoke-report.json
```

The consumer suite needs the actual client checkout. The `npu` marker requires
an Ascend host; `release_gate` checks are orchestrated by the release-gate script.
The current gate checks liveness/readiness, two float32 add shapes (4096 and 8192),
passed correctness and execution provenance. It polls HTTP 202 responses until a
terminal resource arrives; skipped checks fail the gate. It does not run a full
failure matrix, crash soak, timing-stability or profiling calibration.

The report records `passed`, the scope `readiness-and-add-correctness-smoke`, and
the SHA-256 of the target's live `/openapi.json` (fetched with a 10-second timeout).
It does not label this limited check `production_ready`. There is no catalog
manifest or separate grade token; the subprocess uses the caller capability via
`TRITON_EVAL_CAPABILITY_TOKEN`. The smoke client measures performance by default;
`--no-speed` selects correctness only, and its exit status reflects correctness.
Passing or skipped CPU checks are not evidence of NPU correctness, isolation
recovery, latency or cache coverage.

For baseline-cache changes, also exercise a fresh two-source comparison on the
deployed NPU:
submit `benchmark` twice with different idempotency keys and check that per-case
`baseline_cache_hit` changes from false to true, then submit with
`--no-baseline-cache` and verify false. Submit once more with caching enabled to
check that bypass did not overwrite the retained Torch times. Correctness and
candidate timings must be present on every successful run. Repeat with a changed
case shape or seed to confirm a miss. These hardware checks are separate from
the readiness/correctness release gate above.

Single-source NPU benchmarks should report fresh per-case latency and
`correctness_verified: false` for both cache-flag values, without creating or
changing a comparison baseline entry. They should not report correctness or
reference speedup. These are separate hardware checks, not evidence established
by the CPU CLI tests.
