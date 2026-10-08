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
measure correctness and performance. Backend JIT failures are not reused
across inputs because they can depend on shape and dtype. Reference outputs and reference timings have a separate optional, pinned
[reference-cache policy](reference-cache.md); candidate evaluation results are not cached.

Engine-failure retries are disabled by default. See [card health](../architecture/card-health.md)
for the opt-in policy, shared attempt budget and same-card recovery behavior.

## Upgrading

Before switching to operator inputs, stop old admissions and collect in-flight
results from their original instance. Update clients to send `torch_code`,
`triton_code` and `cases`; both implementations must export `run(*args)`.
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
