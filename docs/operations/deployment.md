# Deploy an evaluation instance

Run all commands from the repository root.

Run one independent instance per host, each with its own API, local durable
Redis and card list. The image contains the declared Python stack; CANN and
driver come from the host. Bind each instance to an explicit private IPv4
address. The driver reaches that address; task-agent containers reach only
the driver's session facade.

Copy [`docker.env.example`](../../docker.env.example) to `docker.env` and replace
its placeholders before starting:

| Setting | Value to supply |
| --- | --- |
| `TRITON_EVAL_API_HOST` | This host's private IPv4 address; API port is `11907` |
| `TRITON_EVAL_IMAGE`, `TRITON_EVAL_EVALUATOR_BUILD` | Immutable image reference and its matching `sha256:…` digest |
| `CANN_HOST`, `TRITON_EVAL_WORKER_*` | Host toolkit path and actual hardware/toolchain inventory |
| `TRITON_EVAL_CARDS` | Unique physical card IDs, e.g. `0` or `0,1,3` |
| `TRITON_EVAL_CAPABILITY_SECRET_PATH` | Caller registry JSON, permissions no broader than `0600` |
| `TRITON_EVAL_CATALOG_ROOT`, `TRITON_EVAL_CATALOG_MANIFEST_PATH` | Release directory and catalog manifest |

The file formats are defined in [capability loading](../../triton_evaluation/security.py)
and [catalog loading](../../triton_evaluation/catalog.py). After editing `docker.env`:

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
(with the Ascend wheel index if needed), configure [`.env.example`](../../.env.example)
as `.env`, then use `bash scripts/start.sh` / `bash scripts/stop.sh`. Supply Redis
with `--appendonly yes --appendfsync always`, as Compose does. The engine is
bundled and needs no external checkout.

Candidate code needs an operating-system boundary when evaluations must not
read other attempts or service credentials. Subprocesses alone do not provide
that boundary. Set `TRITON_EVAL_ENGINE_SANDBOX_CONFIG` on each worker to an
operator-owned runtime manifest, and pin its digest with
`TRITON_EVAL_ENGINE_SANDBOX_SHA256`, to enable the fail-closed Linux engine sandbox.
See [engine isolation](../architecture/engine-sandbox.md) for deployment and
device requirements. Bind changes to this execution environment to a new
evaluator build; do not combine its measurements with a previous environment
without declaring the change.

For a deployed CANN profiler with a missing `libsqlite3.so` name, see
[the pinned profiler runtime](profiler-runtime.md). Preparation, immutable
file binding and the actual worker/device gates are separate steps.

Correctness defaults to the existing CPU-first reference with NPU fallback for
reference execution errors. To declare an NPU reference experiment, set
`TRITON_EVAL_REFERENCE_DEVICE_POLICY=npu-strict` on the API and every worker.
This mode always selects the assigned NPU and never changes reference device
after seeing a candidate outcome. It adds `:npu-strict-v1` to the published
correctness algorithm identity automatically. Bind the policy to a new evaluator
build and experiment; numerical rounding can differ even though the comparison
formula and tolerances are unchanged. See [reference policy and observations](../architecture/reference-device-policy.md)
for provenance, retention and admission checks.

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
