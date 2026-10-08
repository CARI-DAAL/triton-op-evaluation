# Register and replay official npu-kernelbench tasks

This optional offline importer assembles `definition.json` and `workload.jsonl`
with [`scripts/adapters/official_task.py`](../../scripts/adapters/official_task.py).
The running service accepts ordinary catalog references without this dataset,
its naming rules or provenance sidecars; see the
[catalog task example](../usage/kernel-eval.md#add-a-catalog-task).
The assembler has no Torch, NPU, bridge, or example-package dependency. It embeds
the official reference string verbatim,
then appends the versioned input loader and `Model` interface. Registration
compiles the result without importing or executing official reference code.
Device execution and whole-case admission remain separate checks.

This import script retains the B1 selection policy: exactly ten level3 and ten
level4 tasks. A minimal selection is a JSON list of `{"level": "level3", "name": "BatchMatmul"}` objects.
The Open Evo `official-b1-v1` selection document is also accepted; its declared
`problem_id`, official digest, and case count are checked against current input
files. `MsaSparseAttnFwd` and `PagedAttentionV2` are excluded. This tool discovers
no replacement tasks and infers no approval from a successful compilation.

```bash
python scripts/register_official_tasks.py \
  --official-root /data/kernel_generator \
  --catalog-root /srv/triton-eval/catalog \
  --selection /etc/triton-eval/selection.json

python scripts/register_official_tasks.py \
  --official-root /data/kernel_generator \
  --catalog-root /srv/triton-eval/catalog \
  --selection /etc/triton-eval/selection.json \
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
and checks catalog loading and generic task resolution. It does not restart a
service or update its environment. Freeze and admission must bind the generated
reference digests before evaluation begins.

Source-format and provenance checks belong to registration and replay. They
compare generated reference and sidecar bytes to the selected official inputs;
the service does not read those sidecars or require their reference headers.
`ReleaseCatalog.resolve_task(task_id, manifest_id=...)` checks the same generic
manifest binding and reference bytes for imported and ordinary tasks. Serving a
registered task needs neither the source checkout nor the offline adapter.
Moving the adapter preserves generated reference source, digests and the
four-field catalog format. The tool layer now uses `problem_id` for `op_name`;
follow the [upgrade procedure](../usage/kernel-eval.md#upgrade-from-official-only-task-lookup)
when replacing an active official-task deployment.

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
