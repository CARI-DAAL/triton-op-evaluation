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

The service uses the `triton-eval:v2.1` namespace. Legacy routes, task polling,
old dispatch queues, and the retired `kernel_eval` bridge are no longer shipped.
Stop old admissions, collect all in-flight results from
their original owner, then switch consumers and instances together. Old records
are neither read nor deleted. Preserve the Redis volume; rollback requires
the matching consumer and original namespace. Never use `docker compose down -v`.
