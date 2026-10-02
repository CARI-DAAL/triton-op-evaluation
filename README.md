# triton-evaluation

Durable Triton-Ascend evaluation over one HTTP interface. The service owns
admission, leases, recovery, execution and typed evidence. The consumer owns
session, instance selection and scoring. Workers run separate evaluation
subprocesses: CPU workers for compile-only requests and one NPU worker per
configured card for all other evaluations.

## Interface

The durable evaluation protocol is **v2.1**:

- `POST /evaluations?wait_seconds=0..60` requires `Authorization: Bearer`,
  `X-Session-ID` and `Idempotency-Key`. HTTP 200 contains a terminal resource;
  HTTP 202 contains an admitted resource and `Location`.
- `GET /evaluations/{evaluation_id}` on the original instance returns HTTP 200
  with its current state, including pending states. Pending resources include
  `Retry-After: 1`. Unknown and invisible IDs return 404.
- `/livez`: process liveness. `/readyz` requires the contract, Redis, catalog,
  credentials and compatible CPU and NPU workers. Submissions require only
  the execution capacity needed by their requested checks.
- `/internal/metrics`: observer-authenticated metrics.

The optional [native kernel_eval facade](docs/kernel-eval-native.md) serves
`POST /verify`, `/benchmark` and `/profile` for task-agent tools. It resolves
official task IDs from the service catalog and uses one `KERNEL_EVAL_TOKEN`.

| Caller intent | `checks` |
| --- | --- |
| Correctness | `["correctness"]` |
| Correctness and timing | `["correctness", "performance"]` |
| Profiling | `["profile"]` |
| Syntax/import precheck | `["compile"]` |

Timing uses the `performance` check in v2.1. `performance.method` defaults to
`event`; callers may explicitly select `profiler`. Results identify the measurement
method; consumers must not mix methods silently. Compile-only precheck does
not prove that every input shape can compile on the device.

Admission validates and persists the request without executing candidate code.
Compile-only requests enter a CPU queue and normally return HTTP 202; they can
return 200 when an explicit wait observes completion. Every check executes
within a leased attempt under the resource deadline. Mixed requests stay on
one NPU worker, including an explicitly requested compile check.

```json
{
  "schema_version": "2.1",
  "purpose": "trial",
  "op_name": "add",
  "reference_code": "<trusted reference source>",
  "kernel_code": "<candidate source>",
  "checks": ["correctness", "performance"],
  "deadline_seconds": 900,
  "performance": {"warmup": 5, "repeats": 50, "method": "profiler"}
}
```

Resources have `queued`, `running`, `completed`, `failed` or `timed_out`
states. Completed evaluations may contain a wrong kernel or domain failure.
Infrastructure failure is not a numeric correctness failure. Results never
return hidden reference source or test inputs.

Trial and rollout use trial capabilities; grade requires a trusted capability
and manifest-bound `grade_context`. Caller identity owns visibility, primary
quotas and idempotency. Session IDs correlate work and enforce a secondary
quota; they grant no authority.

[The published contract](contracts/evaluation-api/2.1/contract.json) includes
41 fixtures and their digests. [The cutover note](docs/architecture/evaluations-v21-contract.md)
describes the publication rebuild and removed interfaces.

## One or multiple hosts

Run one independent instance per host, each with its own API, local durable
Redis and card list. The image contains the declared Python stack; CANN and
driver come from the host. Bind each instance to an explicit private IPv4
address. The driver reaches that address; task-agent containers reach only
the driver's session facade.

```bash
cp docker.env.example docker.env
# Set private bind, immutable image/build, CANN root, cards, secret and catalog.
set -a
source docker.env
set +a
export TRITON_EVAL_ENV_FILE="$PWD/docker.env"
scripts/compose.sh up -d --force-recreate
scripts/compose.sh logs -f
```

Preflight validates the exact configuration being launched, pinned images,
private bind, mounted capability/catalog and local Redis AOF durability.
Export the env file before invoking the wrapper, as above. Use `up` to apply
configuration changes; `start` and `restart` reuse existing containers.
`TRITON_EVAL_CARDS` selects worker processes; the privileged Ascend container
has host device access, so this list is scheduling rather than a container
security boundary. Restrict network access to trusted drivers. For access
outside the private network, use a separately managed TLS gateway and put its
HTTPS URL in the consumer pool.

For a single-card smoke deployment, use this same Compose file with
`TRITON_EVAL_CARDS=0`. Card lists are unique nonnegative integers, such as
`0,1,3`. Redis uses the local server on port 6379, with database 7 by default.
`TRITON_EVAL_CPU_WORKERS` defaults to 1 and accepts 0–32. Setting it to 0
disables compile-only capacity unless separate `run_worker.py --cpu` processes
join the same instance. CPU workers do not claim cards or run card recovery.

Set `EVALUATION_POOL_FILE` on the consumer driver to a JSON file:

```json
{"endpoints": [
  {"id": "ascend-a", "url": "http://10.0.0.11:11907", "capability_file": "/run/secrets/eval-a"},
  {"id": "ascend-b", "url": "http://10.0.0.12:11907", "capability_file": "/run/secrets/eval-b"}
]}
```

Each consumer capability file contains one opaque token; the service secret
contains its caller registry. These are different formats, both kept outside
agent containers. A single endpoint uses `EVALUATION_URL` and
`EVALUATION_CAPABILITY_FILE` instead.

The client uses rendezvous hashing of session ID and stable endpoint ID.
Submission, ambiguous retries and GET recovery retain the original instance.
Unavailable owners produce recoverable infrastructure failures; accepted
work is never automatically resubmitted onto another host. Pool changes affect
new sessions. Keep old owners available while their evaluation handles exist.
Quotas, card health and caches are per instance. Never share one namespace
between machines with overlapping card IDs.

For host execution, install `.[npu]` in an isolated environment, configure
`.env.example`, then use `scripts/start.sh` / `scripts/stop.sh`. Supply Redis
with `--appendonly yes --appendfsync always`, as Compose does. The engine is
bundled and needs no external checkout.

Candidate code needs an operating-system boundary when evaluations must not
read other attempts or service credentials. Subprocesses alone do not provide
that boundary. Set `TRITON_EVAL_ENGINE_SANDBOX_CONFIG` on each worker to an
operator-owned runtime manifest, and pin its digest with
`TRITON_EVAL_ENGINE_SANDBOX_SHA256`, to enable the fail-closed Linux engine sandbox.
See [engine isolation](docs/architecture/engine-sandbox.md) for deployment and
device requirements. Bind changes to this execution environment to a new
evaluator build; do not combine its measurements with a previous environment
without declaring the change.

For a deployed CANN profiler with a missing `libsqlite3.so` name, see
[the pinned profiler runtime](docs/profiler-runtime.md). Preparation, immutable
file binding and the actual worker/device gates are separate steps.

Correctness defaults to the existing CPU-first reference with NPU fallback for
reference execution errors. To declare an NPU reference experiment, set
`TRITON_EVAL_REFERENCE_DEVICE_POLICY=npu-strict` on the API and every worker.
This mode always selects the assigned NPU and never changes reference device
after seeing a candidate outcome. It adds `:npu-strict-v1` to the published
correctness algorithm identity automatically. Bind the policy to a new evaluator
build and experiment; numerical rounding can differ even though the comparison
formula and tolerances are unchanged. See [reference policy and observations](docs/architecture/reference-device-policy.md)
for provenance, retention and admission checks.

## Recovery and caching

Synchronous Redis AOF EXEC acknowledgement is the durability boundary. HTTP
timeout or disconnect never cancels an accepted evaluation. Persist its
handle and recover by ID; without an ID, replay exactly the same POST and
idempotency key against the same instance.

Card health gates NPU claims and heartbeats. Canaries and bounded recovery restore
claim eligibility after suspicious endings. Repeated suspicious deaths with
attempt-owned card provenance produce `SUSPECTED_POISON_CANDIDATE`. Lease
fencing prevents stale attempts from publishing results.

The cache stores only syntax/import precheck verdicts. Atomic publication owns
negative precedence and TTL. Reuse requires `TRITON_EVAL_CANN_FINGERPRINT`
and is scoped by Python and installed toolchain versions. Missing fingerprints
or Redis errors disable reuse; an inconclusive explicit compile check reports
`COMPILE_PRECHECK_UNAVAILABLE`. Prechecks parse and import candidate code; they
do not call reference input factories or synthesize correctness case results.

A positive precheck skips another precheck. New logical evaluations still
measure correctness and performance. Backend JIT failures are not reused
across inputs because they can depend on shape and dtype. There is no promised
cross-request executable-artifact or successful-result cache. See
[the architecture decision](docs/adr/0001-evaluation-instance-ownership.md).
The [admission and evidence decision](docs/adr/0002-admission-before-execution.md)
explains attempt ownership, CPU scheduling and terminal evidence rules.

## Cutover

This release removes old routes, task polling, version stripping, sunset
switches, old dispatch queues and unscoped card metadata. The namespace is
`triton-eval:v2.1`. Stop old admissions, collect all in-flight results from
their original owner, then switch consumers and instances together. Old records
are neither read nor deleted. Preserve the Redis volume; rollback requires
the matching consumer and original namespace. Never use `docker compose down -v`.

Historical design/spike documents remain evidence of prior decisions. Current
contract and architecture notes describe supported behavior. Host-specific
throwaway Compose files are retired; the single deployment configuration uses
the declared image stack and explicit host inputs.

## Validation

```bash
pip install -e ".[test,dev]"
python -m pytest -m 'not npu' -q
ruff check .
python scripts/generate_contract_artifacts.py --check
python -m compileall -q triton_evaluation scripts
EVALUATION_CONSUMER_ROOT=/path/to/kernel-harness \
  python -m pytest tests/test_consumer_integration.py -q
```

The cross-repository gate uses the actual consumer against the real HTTP
application, durable store and CPU Worker evaluator. Real NPU calibration,
canary recovery and timing run separately via `scripts/npu_release_gate.py`.
See also [card health](docs/architecture/card-health.md) and
[attempt provenance](docs/architecture/attempt-card-provenance.md).
