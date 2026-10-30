# Changelog

## Unreleased

## 0.2.0 — 2026-10-30

### Commits

- 30b71f8 fix(engine): cpu-first runs with a CPU ambient device
- f510178 feat(api): per-request reference_device_policy for trial correctness
- 8bb4aca fix(engine): isolate correctness inputs via .to(device, copy=True)
- afe97ff fix(shaping): keep head+tail in bounded error details; redact quoted paths
- 2095825 fix: retain completed benchmark measurements on timeout
- f4e1351 fix: preserve host Python runtimes in deployment test sandboxes

## 0.1.1 — 2026-10-20

### Commits

- c32cadd fix: preserve actionable engine failure diagnostics

## 0.1.0 — 2026-10-18

### Commits

- 689896f feat: automate reviewed service releases through GitHub Actions
- fae385f refactor: centralize baseline measurement and evidence
- fce85a7 refactor: give operator input one module that validates once
- d4c7369 fix: report matched non-finite pairs with the zero diff they rank by
- 6db69ae docs: record the EVV-01 implementation commit
- 2cd6107 feat: publish a verifiable release identity
- f2d08cb fix: detect source-installed Triton runtime version
- 79cecb6 docs: record first NPU feedback measurements and remaining acceptance
- f54b9a0 feat: accelerate evaluation feedback and document NPU acceptance
- 33cd86c feat: return benchmark pipe utilization to agents
- aa8622f fix: reject deep shard settings that skip or stall the gate
- ec373c4 feat: fan profile_deep out as parallel per-case shard evaluations
- 7f8d3ce fix: read sample-based per-core data from any pinned device dir
- caa3f1b fix: classify deep bound across vector-core generations
- 31dc0be fix: cache profile case discovery and give it a dedicated timeout
- b9c2235 test: stop the CLI timeout tests racing interpreter startup
- a92b061 feat: judge declared workload tolerances by the pinned official check
- 6a94375 fix: report refused workloads as request errors, not card faults
- bfcb95d docs: sync the NPU validation plan with upstream admission parity
- 6f8e7a7 fix: admit exactly what the pinned upstream generator admits
- 7f6bffe fix: address independent review of the official input migration
- dc9de06 fix: let sandbox argv tests run without a host bwrap
- 72542c4 feat!: adopt the official definition/workloads input protocol
- 8f526f2 fix: address NPU import, profiler loader and sandbox compatibility failures
- cbdad07 docs: publish NPU functional and stability validation plan
- 0ed156b refactor: consolidate Docker deployment assets
- ae2eb68 refactor: assemble runtime dependencies explicitly
- d817110 feat: measure single Torch or Triton implementations without comparison claims
- b614622 feat: reuse Torch baseline timings through an advisory Redis cache
- db2407b docs: clarify local and remote evaluation service URLs
- f3d7522 docs: show repository layout as an annotated directory tree
- 1a59104 docs: organize README around setup, usage and contribution
- acb0372 refactor: group runtime modules and package the evaluation client plugin
- e8e3760 fix(ci): install CPU Torch and NumPy across the test matrix
- 7f2cdfb refactor: evaluate operators from code and explicit cases
- 5d8baf7 refactor: decouple catalog tools from official task datasets
- 7aea42a refactor: organize evaluation modules and operational entrypoints
- 40bd31f docs: organize current service guidance and retire historical implementation material
- f360ca6 refactor(api): use native FastAPI models and remove runtime contract artifacts
- 51ab527 refactor: give kernel tool operations one owner outside HTTP
- 1d3c84f fix(profile): validate observations before combining evidence
- be7e889 fix: bound opt-in engine retries and gate recovery on card health
- 482f254 test: recover canary JIT source regression coverage
- 46acaca Requeue retryable pre-subject engine failures under the shared attempt budget
- f2373e3 Resolve PR-introduced cpu-gate findings: ruff (F401/E731/UP037) and reference_source trace assertion
- e74b1f5 Adapt model-bound engine test double to the bounded ambient-wait signature
- c03fd80 Fail loud when npu-strict benchmark loses the ambient device
- d175a65 Fix ambient-device detection in the NPU RNG state snapshot
- f5e43f1 Sandbox overlay accepts the pinned model-bound index v2
- 903d27f Bind reference caches to the NPU model; enable correctness-from-cache
- 1d45e74 Derive the default ambient-device wait from the reference policy env
- cc0b22c Wait bounded for ambient NPU recovery before the npu-strict guard
- 75cdbba Skip the real-bwrap overlay tests when bwrap is absent on the host
- 7b008d2 Test doubles: tensors derive from their owning generator only
- 5cce7ef Capture NPU RNG state only when the ambient device is NPU
- 10ef9c1 Materialize official inputs once per case; fail fast when npu-strict loses the ambient device
- 4859e6c Make cpu-gate pass: resolve ruff findings, skip env-dependent tests
- 910f3f2 chore: remove retired legacy bridge and improve README
- af31c25 Make reference caching a pure optional accelerator (ruling 2026-10-06)
- 1405439 nb7-cross-card: card-agnostic timing-reuse lookup under policy v4 cross_card_scope
- 5c2f597 nb7-integration: carry the live-validated NB7 reference-caching changes onto the production build line
- 0812546 Add nb2-build-prep CANN 9.1.1 engine sandbox and runtime overlay config
- e9c711b fix: render facade kernel_eval from JCS-restabilized contract document
- 212b832 fix: emit native boolean profiler probe flags
- b3bff1f Derive pinned current CANN environment in native engine sandbox
- 72b538f Pin service-owned profiler runtime and sparse sandbox overlay
- f15584d Add SHA-pinned private SQLite alias to offline profiler diagnostic
- 0244939 Add pinned offline profiler export replay with retained process evidence
- 1dd9f78 Preserve failed profiler captures when retention is requested
- c1519b4 Test profiler feedback probe refusal and raw artifact retention
- 7cc7f5f Add isolated profiler latency feedback probe preserving raw captures
- 629e9b5 refactor: archive retired kernel-eval bridge without executable entry
- 419cb9b fix: preserve official omitted parameter defaults for candidate execution
- 41a2883 test: cover worker version provenance with Ascend distribution fallback
- 3a7079a feat: serve official kernel tools natively with trial budgets and sealed reference cache
- 74caf53 Publish controlled round2 official catalog replay evidence
- 430109f Materialize official audit cases without retaining whole task inputs
- 900a75a Support immutable official npu-kernelbench task registration
- fb90951 Constrain engine result collection and source staging to each attempt
- a1f6369 Freeze strict NPU reference policy with retained device observations
- 44b31bf Isolate engine invocations with a pinned runtime manifest
- 243068d fix: reload persisted canonical bytes with the serializer's number rule
- f5d55d9 feat(engine): port the sandbox golden-CPU-first policy
- 9ac8384 fix(engine): invoke models by their declared signature
- 2f1d440 style: ruff fix the new validator regression test
- 1ad300d fix: restore the bare-call and rng guards lost in validator vendoring
- 2a31d43 refactor: one ambient-device invariant, one implementation
- 83a25b0 fix: guard the ambient-device probe in the generated profile runner
- 61b17a5 fix(engine): make the NPU the ambient device like the sandbox did
- 7921b29 fix: keep the unattributed-failure message inside ErrorInfo's byte cap
- 1183692 fix: survive CANN stdout noise in the decoy count snippet
- aa19606 fix: count NPU-only references in the decoy fast path
- 668188a fix: guard engine-metrics tests for the torch-free CI gate
- c9a15c6 fix: emit quant-path max_abs_diff as a double too
- 97298d7 fix(engine): share the NPU input fallback with benchmark and profile
- 06e3193 fix(engine): construct reference inputs in the NPU context
- 602c8b6 fix: project unattributed case failures as retryable INTERNAL
- 27223e0 fix: emit integer-compute max_abs_diff as a double
- 4745c74 fix: re-register canary source in linecache after health refactor
- e757f6a feat(contract): op-profile kernels carry msprof utilization ratios

