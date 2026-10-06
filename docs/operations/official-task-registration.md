# Register and replay official npu-kernelbench tasks

The service assembles `definition.json` and `workload.jsonl` directly with
`triton_evaluation.official_task`. The assembler has no Torch, NPU, bridge, or
example-package dependency. It embeds the official reference string verbatim,
then appends the versioned input loader and `Model` interface. Registration
compiles the result without importing or executing official reference code.
Device execution and whole-case admission remain separate checks.

Use an explicit B1 selection: exactly ten level3 and ten level4 tasks. A minimal
selection is a JSON list of `{"level": "level3", "name": "BatchMatmul"}` objects.
The Open Evo `official-b1-v1` selection document is also accepted; its declared
`problem_id`, official digest, and case count are checked against current input
files. `MsaSparseAttnFwd` and `PagedAttentionV2` are excluded. This tool discovers
no replacement tasks and infers no approval from a successful compilation.

```bash
python scripts/register_official_tasks.py \
  --official-root ~/open-evo-data/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator \
  --catalog-root ~/open-evo-work/svc-validation-p0/catalog \
  --selection ~/open-evo/examples/evo/npu-baseline/b1-tasks.json

python scripts/register_official_tasks.py \
  --official-root ~/open-evo-data/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator \
  --catalog-root ~/open-evo-work/svc-validation-p0/catalog \
  --selection ~/open-evo/examples/evo/npu-baseline/b1-tasks.json \
  --replay
```

The catalog root must already exist. Registration adds
`references/official_l3_<Name>.py` and `references/official_l4_<Name>.py` plus each
file's `.PROVENANCE.json` sidecar, under manifest
`npukernelbench-official-v1`. Existing catalog entries and files are preserved.
Different bytes at an existing manifest/problem binding or artifact path are
rejected. The entire staged catalog passes `ReleaseCatalog.load` before the
active manifest is atomically replaced; a failed publication rolls back newly
created files and restores the previous manifest. Do not run unrelated catalog
writers during registration. The registrar's advisory lock serializes instances
of this script and detects changes to the active manifest during staging.

Provenance records the official combined digest, separate definition/workload
SHA256 digests, adapter version, reference SHA256, operation, case count, source
repository, and task directory relative to the official source root. Relative
source paths allow replay on another machine. `--replay` is read-only: it
reassembles all twenty references and sidecars, requires exact byte equality,
and checks catalog loading and official-task resolution. It does not restart a
service or update its environment. Freeze and admission must bind the generated
reference digests before evaluation begins.

`ReleaseCatalog.resolve_official_task(task_id, manifest_id=...)` validates the
current reference file on each lookup, checks sidecar binding fields and the
reference header's official digest/adapter version, and returns reference code
with a grade context. Missing, ambiguous, or inconsistent bindings are rejected.
Serving the registered task does not require the official source checkout;
checking the raw official inputs is the registrar/replay tool's responsibility.

Materialization is fixed for this adapter version. Integer ranges include both
endpoints; signed integers without a range use `[-5, 5)` and uint8 uses `[0, 5)`.
Floating tensors use a uniform range when declared and `randn` otherwise. Bool
uses `rand > 0.5`. Scalar/custom values pass through, including explicit `None`.
An absent parameter uses a distinct sentinel so `run()` applies its own Python
default. The engine controls the ambient device and seed. No uint8 inputs occur
in the current B1 selection; its policy is covered by a CPU structural test.
The optional `get_input_group(zero_based_index)` interface lets a case audit
materialize preceding cases in order and release them as it advances, preserving
the full inventory's RNG consumption while keeping memory bounded by a few
cases. It is checked against seed-0 `get_input_groups()` on CPU.

Run controlled CPU checks with:

```bash
OFFICIAL_TASK_ROOT=/path/to/kernel_generator \
  python -m pytest tests/test_official_task.py tests/test_register_official_tasks.py tests/test_catalog.py
```

These checks include cross-process assembly under multiple `PYTHONHASHSEED`
values, missing versus explicit-None arguments, catalog immutability and failed
publication rollback. The six optional real-task checks compile source only.
They provide no evidence of NPU correctness, latency, or device acceptance.
