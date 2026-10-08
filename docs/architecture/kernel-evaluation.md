# Operator input and durable execution

The public `OperatorRequest` describes `torch_code`, `triton_code` and `cases`.
Both implementations export `run(*args)`; cases describe positional tensors and
scalars with deterministic seeds. HTTP validation establishes the input shape.
It does not import or execute supplied code.

The operator adapter converts this input into the engine's existing reference
and candidate interfaces. It supplies the `Model` and input factories internally.
The resulting request enters `EvaluationInstance` and its ordinary durable
admission, lease, execution and recovery path. Internal v2.1 record models remain
available for persisted resources; they are not an alternative public input.

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
