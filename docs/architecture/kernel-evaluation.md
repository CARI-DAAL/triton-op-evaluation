# Kernel evaluation caller intent

`KernelEvaluation.evaluate(action, body, session_id=..., operation_id=...)`
owns official-task resolution, request translation, durable trial turns,
operation identity, admission retry, waiting and optional diagnostics. It
returns a tool envelope or raises `ToolInputError` with an input rejection
kind. It imports no HTTP framework and never executes candidate code.

```mermaid
flowchart LR
    HTTP[HTTP adapter] --> Intent[KernelEvaluation]
    Direct[Direct caller / tests] --> Intent
    Intent --> Budget[Durable trial turns]
    Intent --> Instance[EvaluationInstance submit / get]
    Instance --> Store[Fenced durable evaluations]
    Intent --> Renderer[Tool observation]
```

The HTTP adapter owns authentication, JSON decoding, selection of the session
from headers/body or the legacy remote-address fallback, and mapping rejection
kinds to HTTP status codes. It does not construct Redis dependencies, resolve
catalog entries or manage polling. The application composition root injects the
same owning instance and a budget registry using the same Redis namespace.
The registry's maximum is fixed at construction, like other deployment policy.

The interface preserves two different identities:

- An explicit operation ID reserves one trial turn and derives a stable
  evaluation idempotency key. A conflicting request cannot consume another turn.
- Without an operation ID, each call reserves a new turn. Identical evaluation
  input still derives the same evaluation identity and may replay durable work.

Admission failures consume the reserved turn. Rate and inflight limits retry
the same idempotent admission with bounded backoff. Exhausting the wait budget
never cancels or terminalizes accepted work; the same operation can recover it.
Optional case-zero diagnostics reuse the primary wait deadline and do not
consume a second turn. Diagnostic failures preserve the primary observation.
Rendering still uses canonical contract bytes and the existing frozen fixtures.

The instance and Redis budget are existing modules, not new abstract repository
ports. The injected monotonic clock and asynchronous wait are a real seam for
deterministic deadline/quota tests. Policy tests cross the same `evaluate`
interface as callers using a real catalog, fakeredis, and fenced worker results;
HTTP tests cover transport behavior and end-to-end compatibility.

ADR 0001 and ADR 0002 remain in force. This refactor adds no shared cross-host
state, alternate execution path, result cache or scoring policy.
