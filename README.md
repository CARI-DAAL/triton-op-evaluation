# triton-evaluation

An HTTP service for evaluating Triton-Ascend kernels: correctness, performance,
profiling, and syntax/import prechecks. Accepted evaluations are persisted in
Redis and can be recovered after a client disconnect or worker restart.

The service owns admission, execution, leases, recovery, and typed results.
The consumer owns sessions, instance selection, and scoring. Each host runs
one API, local durable Redis, CPU workers for compile-only requests, and one
NPU worker per configured card. The evaluation engine is bundled.

## Repository layout

| Path | Responsibility |
| --- | --- |
| [`triton_evaluation/`](triton_evaluation/) | API, durable scheduling, workers, evaluators, and bundled engine |
| [`contracts/evaluation-api/2.1/`](contracts/evaluation-api/2.1/) | Published schema, golden fixtures, and digests; required at API startup |
| [`scripts/`](scripts/) | Deployment, contract generation, smoke checks, and release gates |
| [`tests/`](tests/) | CPU tests, contract checks, consumer integration, and NPU tests |
| [`docker-compose.yml`](docker-compose.yml), [`docker/`](docker/) | Single-host deployment, reused independently on each host |
| [`docs/`](docs/) | Architecture, decisions, specifications, and historical plans |

`contracts/` is generated from [`api/contract.py`](triton_evaluation/api/contract.py)
and [`api/contract_artifacts.py`](triton_evaluation/api/contract_artifacts.py).
The API and host launcher verify its bytes; Docker copies it into the image,
and CI and consumers check its digests. After a contract change, regenerate it
with `python scripts/generate_contract_artifacts.py --write` and review and
commit the generated changes together. Deleting it prevents API startup.

## Local development

Run commands from the repository root. Python 3.10+ is required; CI checks
Python 3.10–3.12. The core CPU tests do not require an Ascend device.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,dev]"
python scripts/generate_contract_artifacts.py --check
python -m pytest -m 'not npu' -q
python -m ruff check .
python -m compileall -q triton_evaluation scripts
```

For the actual consumer's HTTP integration tests, point to a checkout containing
`kernel_agent/rewards/triton_eval_client.py`:

```bash
EVALUATION_CONSUMER_ROOT=/path/to/kernel-harness \
  python -m pytest tests/test_consumer_integration.py -q
```

These tests exercise the real HTTP application, durable store, and a Worker
with a stub evaluator. They are skipped when the consumer checkout is not
configured. Real NPU calibration, card recovery, and timing run separately via
[`scripts/npu_release_gate.py`](scripts/npu_release_gate.py).

## Deploy on an Ascend host

Run one independent instance per host, each with its own API, local durable
Redis and card list. The image contains the declared Python stack; CANN and
driver come from the host. Bind each instance to an explicit private IPv4
address. The driver reaches that address; task-agent containers reach only
the driver's session facade.

Copy [`docker.env.example`](docker.env.example) to `docker.env` and replace
its placeholders before starting:

| Setting | Value to supply |
| --- | --- |
| `TRITON_EVAL_API_HOST` | This host's private IPv4 address; API port is `11907` |
| `TRITON_EVAL_IMAGE`, `TRITON_EVAL_EVALUATOR_BUILD` | Immutable image reference and its matching `sha256:…` digest |
| `CANN_HOST`, `TRITON_EVAL_WORKER_*` | Host toolkit path and actual hardware/toolchain inventory |
| `TRITON_EVAL_CARDS` | Unique physical card IDs, e.g. `0` or `0,1,3` |
| `TRITON_EVAL_CAPABILITY_SECRET_PATH` | Caller registry JSON, permissions no broader than `0600` |
| `TRITON_EVAL_CATALOG_ROOT`, `TRITON_EVAL_CATALOG_MANIFEST_PATH` | Release directory and catalog manifest |

The file formats are defined in [capability loading](triton_evaluation/security.py)
and [catalog loading](triton_evaluation/catalog.py). After editing `docker.env`:

```bash
set -a
source docker.env
set +a
export TRITON_EVAL_ENV_FILE="$PWD/docker.env"
scripts/compose.sh pull
scripts/compose.sh up -d --force-recreate
scripts/compose.sh logs -f
```

Preflight validates the exact configuration being launched, locally available
pinned images, private bind, mounted capability/catalog and Redis AOF settings.
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

For host execution without Docker, install `.[npu]` in an isolated environment
(with the Ascend wheel index if needed), configure [`.env.example`](.env.example)
as `.env`, then use `bash scripts/start.sh` / `bash scripts/stop.sh`. Supply Redis
with `--appendonly yes --appendfsync always`, as Compose does. The engine is
bundled and needs no external checkout.

## Submit and recover an evaluation

The only supported evaluation protocol is **v2.1**:

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

| Caller intent | `checks` |
| --- | --- |
| Correctness | `["correctness"]` |
| Correctness and timing | `["correctness", "performance"]` |
| Profiling | `["profile"]` |
| Syntax/import precheck | `["compile"]` |

There is no benchmark HTTP verb. `performance.method` defaults to `event`;
callers may explicitly select `profiler`. Results identify the measurement
method; consumers must not mix methods silently. Compile-only precheck does
not prove that every input shape can compile on the device.

Admission validates and persists the request without executing candidate code.
Compile-only requests enter a CPU queue and normally return HTTP 202; they can
return 200 when an explicit wait observes completion. Every check executes
within a leased attempt under the resource deadline. Mixed requests stay on
one NPU worker, including an explicitly requested compile check.

Create `request.json` from your reference and candidate source files:

```bash
python - <<'PYREQUEST'
import json
from pathlib import Path

request = {
    "schema_version": "2.1",
    "purpose": "trial",
    "op_name": "add",
    "reference_code": Path("reference.py").read_text(encoding="utf-8"),
    "kernel_code": Path("kernel.py").read_text(encoding="utf-8"),
    "checks": ["correctness", "performance"],
    "deadline_seconds": 900,
    "performance": {"warmup": 5, "repeats": 50, "method": "event"},
}
Path("request.json").write_text(json.dumps(request), encoding="utf-8")
PYREQUEST

# Set EVALUATION_URL and EVALUATION_CAPABILITY_FILE for your instance.
# Use a new key for new work; keep the same key and payload for retries.
curl -sS -i "$EVALUATION_URL/evaluations?wait_seconds=0" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")" \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: example-session-1' \
  -H 'Idempotency-Key: example-evaluation-1' \
  --data-binary @request.json

# Set EVALUATION_ID to the returned evaluation_id; poll the original instance.
curl -sS -i "$EVALUATION_URL/evaluations/$EVALUATION_ID" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")"
```

Resources have `queued`, `running`, `completed`, `failed` or `timed_out`
states. For `completed` evaluations, inspect each check result: the kernel may be
wrong or have a domain failure.
Infrastructure failure is not a numeric correctness failure. Results never
return hidden reference source or test inputs.

Trial and rollout use trial capabilities; grade requires a trusted capability
and manifest-bound `grade_context`. Caller identity owns visibility, primary
quotas and idempotency. Session IDs correlate work and enforce a secondary
quota; they grant no authority.

[The published contract](contracts/evaluation-api/2.1/contract.json) includes
41 fixtures and their digests. [The cutover note](docs/architecture/evaluations-v21-contract.md)
describes the publication rebuild and removed interfaces.

## Multiple hosts

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
cross-request executable-artifact or successful-result cache.

## Upgrading from retired protocols

The service uses the `triton-eval:v2.1` namespace. Legacy routes, task polling,
old dispatch queues, and the retired `kernel_eval` bridge are no longer shipped.
Stop old admissions, collect all in-flight results from
their original owner, then switch consumers and instances together. Old records
are neither read nor deleted. Preserve the Redis volume; rollback requires
the matching consumer and original namespace. Never use `docker compose down -v`.

Historical design/spike documents remain evidence of prior decisions. Current
contract and architecture notes describe supported behavior. Host-specific
throwaway Compose files are retired; the single deployment configuration uses
the declared image stack and explicit host inputs.

## Further reading

- [Domain vocabulary](CONTEXT.md)
- [Instance ownership and caching](docs/adr/0001-evaluation-instance-ownership.md)
- [Admission, CPU scheduling, and evidence](docs/adr/0002-admission-before-execution.md)
- [Card health](docs/architecture/card-health.md) and [attempt provenance](docs/architecture/attempt-card-provenance.md)
