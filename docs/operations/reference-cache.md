# Pinned reference caches

For automatic Torch timing reuse in two-source operator comparisons, see
[baseline timing reuse](../usage/evaluations.md#reuse-torch-baseline-timings).
Comparison benchmarks use that cache and bypass the pinned timing paths described
below. Single-source benchmarks always measure fresh and read or write neither
baseline nor pinned timing caches. This page covers retained reference-output
and standalone engine tooling.

The service can reuse seed-zero official reference outputs and the median of eight
reference measurements. Candidate execution, the existing accuracy algorithms and
the decoy pre-check remain required. Validate native NPU cache-off/cache-on behavior and intended device coverage
before enabling reuse. CPU tests establish neither device equality nor a throughput benefit.

This guide describes retained compatibility for existing pinned reference-cache
deployments. The operator input API does not register datasets or generalize
these cache identities. Arbitrary reference code and workloads do not gain cache
coverage through submission; validate a matching deployment before reuse.

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
  `../runtime/profiler_runtime.py`; no other sibling traversal is allowed. Its actual
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

Configure all four settings as one deployment change:

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
median-of-eight denominators. Since `geometric-mean-v1:reference8-v1` an
**uncached** grade (no cache environment configured, or a non-official reference)
also measures every benchmarked reference case eight times instead of once; one
grade therefore yields eight warmup/repeats reference samples per case without
eight separate requests. This can be considerably slower than the old live
single-measurement route (roughly eight times the reference-side measurement
cost); measure actual coverage and wall time before predicting batch duration.
A one-case task still spends eight measurements on its sentinel, so reuse does
not imply reduced cost for that task.

Each measured case in the public `PerformanceResult.per_case` carries the real
reference evidence behind its `reference_latency_ms` (which stays the median):
`reference_latency_samples_ms` — exactly eight finite positive latencies — and
`reference_latency_samples_source`, either `fresh_measurements` (eight samples
measured in this grade) or `cached_manifest` (the eight historical samples
recorded in the pinned cache manifest, returned only on a sentinel-validated
cache hit). Samples are never fabricated around a cached scalar, and a case with
a legacy or unbound reference scalar carries no samples and no source label.
Failed or partial reference measurement stays a failed/unmeasured case; it never
becomes a successful eight-sample result.

Case identity convention: engine and cache/audit labels are one-based; the
public `PerformanceCase.case_idx` and the optional request selection
`PerformanceParams.case_indices` are zero-based (audit case N ↔ public N-1), the
same convention as `ProfileParams.case_indices`. A request with
`performance.case_indices` benchmarks only those original cases for performance
(correctness always covers the full task); the engine result reports the
original one-based `case_indices` it measured and never renumbers them, so
`PerformanceResult.selected_cases`/`coverage` describe the selected subset. A
subset performance result is diagnostic evidence; full-coverage consumers keep
their own fail-closed coverage checks. An empty or out-of-range selection is
rejected (invalid request or failed performance check), never silently widened.

## Opt-in reuse of validated historical timings (timing only)

A deployment may pin ONE owner-adopted policy artifact that authorizes reusing
the OLD reference cache's timing evidence across one explicitly approved
runtime transition (for example original -> patched-Triton), configured like
the other pinned anchors:

```text
TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY=/absolute/validated-timing-reuse-policy.json
TRITON_EVAL_REFERENCE_TIMING_REUSE_SHA256=<SHA256 of exact policy bytes>
```

The policy declares the source deployment digest, cache root and sealed index
SHA256, the covered physical cards, the 8×(warmup5/repeats20, profiler) timing
contract (reconciled explicitly with reference8's identical per-sample
protocol, never silently aliased), the target patched runtime file hashes, the
consuming service source hash, and the sampled evidence that justified adoption.
`target_service_source_sha256` paths are relative to the service package root.
Runtime helper entries therefore use the `runtime/` prefix; a module move also
changes source hashes. Reissue these policies under the
[runtime migration procedure](recovery.md#runtime-module-migration).
Record the scope of the supporting measurements explicitly. Sampled evidence
does not establish individual validation of every case in the source inventory.

At the real timing lookup, reuse requires an exact `CaseBinding` match — same
reference source, official digest, case, seed-zero inputs, physical device and
reference policy — and serves the manifest's REAL eight historical samples and
their median, labeled `historical_validated` with the policy digest as public
proof (`reference_timing_policy_sha256`). The existing sentinel contract is
retained for historical timings: before serving reuse, the engine measures the
first selected case fresh (the same eight-sample warmup/repeats protocol) and
compares it with the historical median using the same >3% rule and monotone
invalidation markers (written under the policy's dedicated state directory,
keyed by source deployment and card). Drift or an existing marker forces the
original fresh measurement path for every case (trace `timing_reuse_drift`);
the fresh sentinel observation is recorded in the trace separately from the
historical per-case samples. Timings are never borrowed across devices or
cases: any miss (unrepresented card, case, input or protocol) leaves the engine
on its original execution path (fresh measurement), and a partially covered
selection is measured fresh for every case. A tampered policy (bytes ≠ pinned
SHA256), a contradictory index or a mismatched target identity is an
infrastructure failure or a denial — never a hit. Reuse is timing evidence only:
correctness, quality and admission gates are unchanged, and the source cache,
index and manifests are never modified or rekeyed.

### Sandbox delivery of the policy pair

`--clearenv` sandbox children receive the policy pair ONLY through the pinned
engine runtime overlay: `TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY` and
`..._SHA256` form one complete group in `RUNTIME_ENV` and in the overlay's
allowed environment (both-or-neither everywhere; a state mount without the
pair is refused too). The overlay verifies the policy bytes against the pinned
SHA, the declared index and deployment identities, and binds exactly the
declared artifacts read-only (policy file, sealed index, the cache's `cases`
and `blobs` namespaces, pinned runtime files) plus one dedicated writable
state directory for the sentinel's drift markers. A worker declaring the pair
without a pinned overlay is refused rather than silently dropped by
`--clearenv`, and no other parent environment is inherited.

`verify_reference.json`, retained by the existing attempt-bound diagnostics, records
hit/miss, output/manifest/binding SHA and whether a live reference actually ran.
The performance record includes card, deployment, sentinel observations,
hit/miss/remeasurement and manifest SHA identities. Cached framework peak memory
is not measured in that attempt: the raw engine compatibility value is zero and
`reference_memory_measured=false` records that limitation. Do not interpret that
zero as a measured peak. Diagnostics contain identities and observations, never
input/output tensor values, and are not public verdicts.

## Validate a deployment

Run cache-off/cache-on evaluations with identical official source, candidate,
seed, assigned card, stack and measurement settings. Include correct and wrong
candidates, multiple outputs, relevant dtypes and a proven decoy. Compare complete
verdicts and case counts; record reference executions, cache decisions, source
hashes, wall time and sentinel overhead.

Exercise a missing cache entry, changed manifest/blob and forced sentinel drift.
A miss executes the reference; corrupt pinned evidence cannot judge a candidate;
drift forces fresh measurements. Compare coverage with the actual catalog and
intended cards. Retain the tested build, cache index and measurement conditions
outside the repository, and report unknown or partial observations explicitly.
