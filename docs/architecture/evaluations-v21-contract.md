# Evaluation contract cutover

The supported public interface is `POST /evaluations` and
`GET /evaluations/{evaluation_id}` using `schema_version: "2.1"`.
Requests, resources, result provenance, and error envelopes use that same
version. The release contains one strict model and one canonical serializer.
Unknown fields and versions are rejected before admission.

Execution is selected through `checks`, ordered as `compile`, `correctness`,
`performance`, then `profile`. Performance requires correctness and defaults
to event timing; callers may explicitly request profiler timing. `performance`
may carry zero-based `case_indices` (same shape as profile selection) to
benchmark only original cases; without it performance covers the whole task.
Each measured performance case reports `reference_latency_ms` (the median) plus,
when sample evidence exists, `reference_latency_samples_ms` (eight samples) and
`reference_latency_samples_source` (`fresh_measurements` or `cached_manifest`).
Profile-only and compile-only evaluations remain valid. `grade_context` belongs
only to `purpose: "grade"`; trial and rollout traffic cannot carry it.

Caller facades map intent into checks before submission:

| Intent | Checks |
| --- | --- |
| Verify | `['correctness']` |
| Benchmark | `['correctness', 'performance']` |
| Profile | `['profile']` |
| Compile | `['compile']` |

These are caller operations, not additional HTTP routes. Every submission
requires the capability credential, `X-Session-ID`, and `Idempotency-Key`.
Retries keep the original request and idempotency key. A submitted evaluation
continues independently of an HTTP wait timeout or disconnected caller;
recovery uses its resource ID on the owning deployment.

## Publication and release identity

`contracts/evaluation-api/2.1` contains the sole active JSON schema, a complete
41-fixture inventory, and recomputable digests. The fixtures cover lifecycle
states, correctness domain outcomes, compile checks, both timing methods,
profile-only evaluations, grade and rollout requests, poison quarantine,
and admission/transport errors. The portable consumer tests validate the
published bytes independently of the server models.

This cutover **rebuilds the previously incomplete 2.1 publication**: its old
error and provenance literals were 2.0, and the schema admitted both request
versions. Those literals now consistently identify 2.1 and the fixture index
includes the full interface. Schema and fixture digests consequently change.
Consumers must rebuild against the new `contract.json` digests; manifests or
release catalogs pinned to earlier digests must be regenerated. The version
label alone does not establish release compatibility.

There is no active 2.0 artifact tree, frozen model snapshot, version-stripping
serializer, sunset timer, or adapter for `/verify`, `/profile`, and `/tasks`.
Old interface requests fail rather than being reinterpreted. Tests verify
that stale route-enabling environment settings cannot restore removed routes.

The default Redis namespace is `triton-eval:v2.1`. Upgrade by draining the
previous deployment and switching migrated consumers to the new release.
The new process does not interpret old resources or delete old durable state.
In-flight work must finish and its results be collected from the deployment
that admitted it before that deployment is retired.

## Compile evidence

An explicit compile check runs the syntax/import precheck within a leased
attempt. Admission never executes candidate code. `compiled` means that the precheck passed;
it does not claim that backend JIT compilation will succeed for every input
shape. Backend compilation failures remain correctness domain outcomes.

A compile-only request is durably queued for a compatible CPU worker. Its first
POST normally returns 202; `wait_seconds` can observe completion and return 200.
It needs a CPU worker heartbeat but no NPU card. A definitive precheck failure carries
`compile_error`; an unavailable or inconclusive precheck produces a failed
compile check with retryable `COMPILE_PRECHECK_UNAVAILABLE`. GET and idempotent
replay recover the same terminal resource and do not repeat the check.

For mixed requests, one NPU worker executes all requested checks in the same
attempt. Compile progress is fenced and persisted before remaining checks run.
Inconclusive compile evidence does not prevent independent on-card checks.
Correctness always comes from actual verification; syntax/import evidence has
no measured case count and cannot substitute for it. Requests without `compile`
do not perform an extra syntax/import precheck.

A replacement attempt begins with fresh check evidence. Recovery preserves
valid evidence from the current attempt, records the ending failure on the
first missing check, and marks later missing checks skipped. Normal completion
with missing evidence fails closed with `INVALID_ENGINE_OUTPUT`; it never
infers a successful check. Grade binding survives every terminal path, but
grade eligibility requires complete scoring evidence and execution provenance.

See [ADR 0002](../adr/0002-admission-before-execution.md) for the execution and
evidence module design. These changes retain the published v2.1 wire schema.
