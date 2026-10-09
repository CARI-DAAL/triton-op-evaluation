# Operator input and durable execution

The public `OperatorRequest` contains an official-shaped `definition` and
`workloads`, optional `triton_code`, optional custom-generator source, and a seed.
`definition.reference` supplies Torch code; either source may be omitted for
single-implementation measurements. Both implementations expose `run(*args)` in
the declared input order. HTTP validation resolves bounded input metadata and
axes without executing supplied source or importing Torch.

The operator adapter embeds the audited input generator source in a private
namespace inside the reference module. Its code bytes, ordered input names,
normalized workloads, seed and custom source therefore participate in durable
reference identity and the baseline-cache key. Candidate source stays separate.
Random values are generated only inside a leased device attempt, using the
assigned NPU and the pinned official seed/distribution rules.

The resulting internal request follows the same `EvaluationInstance` admission,
lease, execution and recovery path. Stored v2.1 records remain recoverable; the
retired positional public request is not accepted. There is no benchmark catalog
or task-registration endpoint.

```mermaid
flowchart LR
    HTTP[OperatorRequest] --> Adapter[Engine input adapter]
    Adapter --> Instance[EvaluationInstance submit / get]
    Instance --> Store[Fenced durable evaluations]
    Store --> Worker[Leased worker attempt]
    Worker --> Engine[Correctness / timing / profile]
    Engine --> Evidence[Typed check evidence]
```

`/evaluations`, `/verify`, `/benchmark` and `/profile` use one capability registry
and one durable resource response. Shortcuts select the checks; the general
endpoint accepts an explicit selection. Caller identity owns visibility and
quotas. An optional session groups work, and an optional idempotency key makes
retry safe within its caller/session scope. Omitted keys submit fresh work.
Neither operation requires catalog lookup, a dataset adapter or a tool budget.

Worker execution support lives under `triton_evaluation/runtime/`, including
sandbox isolation, profiler integration, reference caches and diagnostics.
Evaluators own check orchestration; shared configuration and canonical
serialization remain at the package root. Moving these implementations requires
[new deployment source pins](../operations/recovery.md#runtime-module-migration).

Process entrypoints assemble dependencies explicitly. The API gives its instance
and durable store the same metrics adapter. The worker gives its store, lease
keeper and card-health module the same metrics adapter, and passes independent
`BaselineCache` and `VerdictCache` adapters to their consumers. Advisory caches
and metrics use a separate Redis client with bounded timeouts and no retries;
the durable store retains its own client and transition semantics.

`EvaluationEvaluator` receives already assembled correctness, benchmark and
profile adapters. It owns check ordering and evidence shaping, not runtime
configuration. All checks, including precheck, use the same subprocess runner
and a startup environment snapshot built from validated `Settings`. Each
invocation copies that snapshot and selects its own device. Precheck uses the
configured engine directory just like the device checks. Tests replace check
adapters through the constructor rather than overriding private factories.

Workers execute supplied code after admission. CPU workers own compile-only
checks; one NPU worker owns all requested device checks in an attempt. Evidence
publication is fenced so an older attempt cannot replace a newer result.
Missing evidence cannot establish success, and infrastructure failures remain
separate from candidate correctness outcomes.

Tests exercise public requests, source adaptation, Redis-backed admission and
resource recovery with CPU test doubles at the device boundary. Real NPU
correctness, timing and profiling require separate device validation. Existing
engine/reference-cache compatibility remains scoped to its pinned deployments;
the new input interface does not establish cache coverage for arbitrary operators.

Single-implementation timing has a distinct result with `implementation`,
`correctness_verified: false` and per-case latency, without reference speedup.
Single-implementation profiling carries the same non-verification marker. The
caller retains those semantics when interpreting a completed resource; success
of a measurement is not evidence of equality with another implementation.
