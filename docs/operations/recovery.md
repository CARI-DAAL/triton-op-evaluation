# Recovery, retries and upgrades

Synchronous Redis AOF EXEC acknowledgement is the durability boundary. HTTP
timeout or disconnect never cancels an accepted evaluation. Persist its
handle and recover by ID; without an ID, replay exactly the same POST and
idempotency key against the same instance.

Card health gates NPU claims and heartbeats. Canaries and bounded recovery restore
claim eligibility after suspicious endings. Repeated suspicious deaths with
attempt-owned card provenance produce `SUSPECTED_POISON_CANDIDATE`. Lease
fencing prevents stale attempts from publishing results.

The precheck cache stores syntax/import verdicts. Atomic publication owns
negative precedence and TTL. Reuse requires `TRITON_EVAL_CANN_FINGERPRINT`
and is scoped by Python and installed toolchain versions. Missing fingerprints
or Redis errors disable reuse; an inconclusive explicit compile check reports
`COMPILE_PRECHECK_UNAVAILABLE`. Prechecks parse and import candidate code; they
do not call reference input factories or synthesize correctness case results.

A positive precheck skips another precheck. New logical evaluations still
execute their requested checks and measurements. Backend JIT failures are not reused
across inputs because they can depend on shape and dtype. Reference outputs and reference timings have a separate optional, pinned
[reference-cache policy](reference-cache.md); candidate evaluation results are not cached.

Engine-failure retries are disabled by default. See [card health](../architecture/card-health.md)
for the opt-in policy, shared attempt budget and same-card recovery behavior.

## Upgrading

Before switching to operator inputs, stop old admissions and collect in-flight
results from their original instance. Update clients to send `cases` and at
least one of `torch_code` or `triton_code`; each supplied implementation must
export `run(*args)`. Correctness requests require both sources.
The former task-ID facade, separate tool token and public low-level evaluation
request are no longer accepted. Use the capability registry for all endpoints.

Start new operations with new session and idempotency keys. Replaying an old
payload across this interface change is not a migration mechanism. Existing
internal v2.1 records remain recoverable by ID on their owning instance; keep
their handles and preserve Redis. Retained models support recovery, not old
submission bodies or task-catalog resolution.

The Redis namespace remains `triton-eval:v2.1`. Old records are not deleted.
Rollback requires the matching clients, image and durable state. Never use
`docker compose down -v`. Engine and reference-cache policies retain their
existing deployment bounds; the new input interface does not establish device
correctness or cache coverage without the corresponding validation.

Single-source timing and profiling extend the operator interface while preserving
the two-source comparison mode. Clients must accept the single-performance result
and its `correctness_verified: false` marker; a latency is not a correctness
verdict. Changes to execution support require a new evaluator build and renewed
source/artifact pins as described below.

## Runtime module migration

Execution support now lives in `triton_evaluation/runtime/`: profiler handling,
reference cache, timing reuse, diagnostics and sandbox isolation join the worker
and card-health modules. Shared configuration and serialization remain at the
package root; evaluators orchestrate checks through the runtime layer.

This move changes source paths and hashes used by pinned deployments. Use a new
evaluator build and regenerate the profiler/sandbox manifests, cache deployment
and index, runtime overlays and timing-reuse policies that bind those files.
The profiler cache algorithm key is now `../runtime/profiler_runtime.py`, resolved
from the engine directory; the old helper key is rejected. Timing-reuse policy
`target_service_source_sha256` keys remain relative to the `triton_evaluation/`
root, so pinned helpers now use `runtime/sandbox.py`,
`runtime/reference_timing_reuse.py`, `runtime/reference_cache.py` and
`runtime/profiler_runtime.py` as applicable. Engine source hashes also change
where imports change. Follow the [profiler resealing procedure](profiler-runtime.md#reseal-before-device-acceptance)
and [cache policy guidance](reference-cache.md) for artifact ordering and source
closure. Preserve deployed manifests and hashes as historical evidence; prepare
new artifacts instead of rewriting active pins to bypass a mismatch.

CPU tests establish the moved modules' behavior and validation boundaries.
Device acceptance, cache coverage and profiling accuracy must be re-established
on the target deployment before claiming compatibility with its measurements.
