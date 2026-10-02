# F-63 audited reference cache

The service can reuse seed-zero official reference outputs and the median of eight
reference measurements. Candidate execution, the existing accuracy algorithms and
the decoy pre-check remain required. This implementation needs native NPU A/B and
fleet admission before the 320-slot experiment starts. Controlled CPU tests do not
establish NPU equality, full task admission, cache coverage or a throughput benefit.

## Produce and seal evidence

The official assembler source is the reference identity. Audit workers must use
its `get_input_group(zero_based_index)` route after resetting both CPU and NPU seeds
to zero, with the same ambient NPU device and convolution precision policy as the
engine. The random-access adapter consumes preceding-case random draws while
releasing previous tensors. Verification, benchmarking and profile runners use
this route without retaining the full official input set. There is currently one
engine subprocess per check with a whole-check watchdog; this does **not** provide
one subprocess per case. Audit child isolation is a separate fact.

Create one `CacheDeployment` JSON for all workers in the frozen generation:

- `generation`: the frozen measurement window identity.
- `runtime_versions`: actual distribution versions checked at reuse.
- `runtime_files`: absolute CANN/driver version-file paths and their SHA256 hashes,
  checked at reuse. A label alone does not satisfy this binding.
- `algorithm_files`: engine basenames and their SHA256 hashes, including at least
  `verify.py`, `benchmark.py`, `_common_utils.py`. A configured
  [profiler runtime](profiler-runtime.md) additionally requires the fixed key
  `../profiler_runtime.py`; no other sibling traversal is allowed. Its actual
  manifest/dependency files also belong in the immutable runtime closure. With
  the canonical CANN environment quartet configured, this includes the vendor
  source, captured environment artifact, observed version metadata and actual
  resolved runtime-library pins. Missing any configured pin rejects cache reuse.
- `device_conditions`: physical card IDs mapped to the operator's admitted hardware
  and measurement conditions. Admission validates this declaration; the cache does
  not independently discover hardware or room conditions.
- `measurement`: `method`, `warmup`, `repeats`, `cv_threshold`, `clear_l2_cache`.

The audit-only writer is:

```python
cache = ReferenceCache(cache_root, deployment, str(physical_card))
key, manifest_sha = cache.write_case(
    reference_source, case_idx, original_inputs, outputs,
    repeated_outputs=independently_replayed_output_tree,
    reference_device_policy="npu-strict",
    selected_reference_device="npu:0",
    latency_samples_ms=eight_measured_positive_latencies,
)
```

`case_idx` is one based. `repeated_outputs` is one complete output tree, including
all outputs, and must equal `outputs` bit for bit. It is not a list of repeated
runs. Additional independent replays belong in the audit evidence. Keep original
input tensors unchanged; run references on separate copies when they can mutate
inputs. The writer fingerprints input tensor bytes, dtype, shape, stride, storage
offset, device type, scalar values and the official absent-input sentinel.

Artifacts are immutable JSON manifests and raw tensor bytes. No pickle is loaded.
Tuple/list/dictionary output structure, dtype and shape are preserved, including
bfloat16 and empty tensors. Loaded outputs own fresh CPU storage. The manifest
binds reference SHA, official digest, adapter version, case, input seed and bytes,
physical card, reference policy and deployment digest. It records all eight timings
and their median. A later attempt cannot overwrite existing evidence.

After all audit writers finish, seal and review the index:

```bash
python scripts/reference_cache_index.py \
  --cache-root /absolute/audited-cache \
  --deployment /absolute/deployment.json \
  --output /absolute/cache-index.json \
  --coverage-output /absolute/cache-coverage.json
```

This validates manifest identity and every output blob before pinning manifest
SHA256s in one index. Coverage lists observed case indices by source and physical
card. `coverage_complete` remains unknown until compared with the frozen catalog
and all intended cards. Eight-card task allocation does not prove sixteen-card
cache coverage: fill every desired task/card pair with the same reference-only
audit procedure to obtain broad reuse. A partial index remains valid evidence.

## Enable the frozen cache

Add these four settings without changing the six existing deployment anchors:

```text
TRITON_EVAL_REFERENCE_CACHE_DIR=/absolute/audited-cache
TRITON_EVAL_REFERENCE_CACHE_BINDING=/absolute/cache-index.json
TRITON_EVAL_REFERENCE_CACHE_SHA256=<SHA256 of exact index bytes>
TRITON_EVAL_REFERENCE_CACHE_STATE_DIR=/absolute/cache-invalidations
```

All four must be present together. The effective correctness protocol identity
includes the full index SHA so admission and workers agree on the exact deployment.
Freeze the four settings, index, deployment conditions and evaluator build. In an
engine sandbox, mount cache artifacts, the index and runtime files read only;
include the settings and the reference-device policy in the manifest environment.
Set `reference_cache_state_dir` to a separate dedicated writable state directory.
The sandbox preserves the host physical card identity while Ascend uses logical
zero. Keep invalidation state for the whole frozen generation.

Nonofficial admission canaries use the original reference. An official task/card
with no pinned matching entry also runs its complete live reference and records
`miss`; this is never reported as cache reuse. A changed pinned manifest, blob,
runtime version/file, algorithm or measurement configuration raises an infrastructure
error. The cache never uses that evidence to judge the candidate.

## Measurement and retained observations

A fully covered task first measures its frozen first-case reference sentinel eight
times and compares the median with the cached median. This happens on every
performance evaluation. A relative shift strictly above 3% writes a monotone
physical-card/deployment invalidation marker. That evaluation and subsequent ones
in that generation remeasure every reference case eight times and use the median.
They do not overwrite or repin the audit evidence. A cache miss also uses live
median-of-eight denominators. This can be considerably slower than the old live
single-measurement route; measure actual coverage and wall time before predicting
batch duration. A one-case task still spends eight measurements on its sentinel,
so reuse does not imply reduced cost for that task.

`verify_reference.json`, retained by the existing attempt-bound diagnostics, records
hit/miss, output/manifest/binding SHA and whether a live reference actually ran.
The performance record includes card, deployment, sentinel observations,
hit/miss/remeasurement and manifest SHA identities. Cached framework peak memory
is not measured in that attempt: the raw engine compatibility value is zero and
`reference_memory_measured=false` records that limitation. Do not interpret that
zero as a measured peak. Diagnostics contain identities and observations, never
input/output tensor values, and are not public verdicts.

## Machine2 acceptance and reporting

Have the machine2 Pi agent pull the reviewed GitHub revision. Rerun service unit
checks, byte-identical catalog replay and the complete official-case audit; a
first-four-case probe does not admit a task. Verify actual `get_input_group` input
hash equality between audit and real engine paths, with peak memory evidence on
FFN, BatchMatmul and other large workloads. Preserve the final index and coverage.

Run controlled cache-off/cache-on evaluations with identical official source,
candidate bytes, seed, assigned card, stack and measurement settings. Include correct
and wrong candidates, multioutput tasks, relevant dtype/accuracy paths and a proven
decoy. Compare the complete verdict and case counts. Record reference executions,
cache status, SHA, end-to-end wall time and sentinel overhead. Exercise a missing
card entry, changed manifest/blob and forced sentinel drift: misses execute the
reference, corrupt evidence cannot judge, drift causes full-case remeasurement.
Report partial coverage and unknown observations explicitly.

These A/B probes are diagnostics. After the frozen build, index and settings pass
new fleet admission, obtain serial and concurrent f3 evidence under the new
conditions before granting the run license. The 320 formal slots are twenty tasks
by four frameworks by four independent samples; infrastructure retries remain new
Attempts of their original Sample. Retain completed executions by identity and
resolve unfinished ones before retrying. Full-round and overlapping-task cross-round
reports must disclose the CANN, official task-set, harness/model, concurrency,
bridge retirement and cache-protocol changes. They measure the combined evolution;
unknown outcomes are not success, failure or proof of no execution.
