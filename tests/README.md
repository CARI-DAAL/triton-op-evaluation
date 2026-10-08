# Tests

Run tests from the repository root after installing `.[test,dev]`.

```bash
python -m pytest -m 'not npu' -q
```

The suite exercises real HTTP handling, catalog loading and durable state changes
with fakeredis. Process and NPU adapters are replaced where hardware is not the
subject of the test. Real local subprocess, sandbox and HTTP/TLS tests run when
their dependencies are available.

## Choose a focused check

| Area | Test files |
| --- | --- |
| Generated HTTP documentation | `test_openapi.py` |
| Admission, identity, durable state | `test_admission*`, `test_broker.py`, `test_idempotency.py`, `test_fencing.py` |
| Workers and recovery | `test_worker*`, `test_recovery.py`, `test_card_health*`, `test_engine_failure_requeue.py` |
| Evidence and scoring | `test_evaluation_evaluator.py`, `test_shaping.py`, `test_profile_evidence.py` |
| Catalog binding and task lookup | `test_catalog.py` |
| Offline official-task conversion | `test_official_task.py`, `test_register_official_tasks.py` |
| Tool operations and transport | `test_kernel_evaluation.py`, `test_kernel_eval_api.py`, `test_kernel_eval_budget.py` |
| Engine, runtime and cache | `test_verify*`, `test_benchmark*`, `test_profile*`, `test_reference*`, `test_engine*` |

Test modules share helpers through `conftest.py` and narrowly scoped helper files.
The current flat test layout preserves those imports; module names identify the
behavior under test.

## Fixtures

`fixtures/engine/` contains private engine observations. `fixtures/kernel_eval/`
pins tool envelopes and observation text, including non-executable legacy bridge
bytes and their provenance. `fixtures/evaluation_api/2.1/` contains compatibility
examples and their schemas. These are regression data, not deployment inputs.
Preserve hash-pinned reference bytes and licenses when moving fixtures.

## External consumer and NPU checks

```bash
EVALUATION_CONSUMER_ROOT=/path/to/kernel-harness \
  python -m pytest tests/test_consumer_integration.py -q

python scripts/npu_release_gate.py --help
```

The consumer suite needs the actual client checkout. The `npu` marker requires
an Ascend host; `release_gate` checks are orchestrated by the release-gate script.
Skipped external or hardware checks are not evidence of device correctness,
isolation recovery, latency or cache coverage.
