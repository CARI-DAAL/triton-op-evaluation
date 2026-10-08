# Native kernel_eval tools

The optional facade serves `POST /verify`, `POST /benchmark` and `POST /profile`
on the evaluation API listener. It obtains task references from
`ReleaseCatalog` and renders the established `[kernel_eval]` observation.
It has no bridge process, configuration, second token, runtime adapter import
or upstream HTTP dependency. `/evaluations` keeps its v2.1 contract.

## Enable the facade

Prepare a catalog manifest and reference files, then add these settings to the
protected service environment. `TRITON_EVAL_KERNEL_EVAL_MANIFEST_ID` optionally
restricts lookup to one manifest; the example below uses `demo-v1`:

```dotenv
TRITON_EVAL_KERNEL_EVAL_ENABLED=true
KERNEL_EVAL_TOKEN=<sandbox tool secret>
TRITON_EVAL_KERNEL_EVAL_MANIFEST_ID=demo-v1
```

Disabled is the default. An empty token also disables the three tools and
`/health`, returning HTTP 404 (tools retain the existing v2.1 `NOT_FOUND`
envelope while disabled). The Pydantic `SecretStr`
masks printable configuration. `KERNEL_EVAL_TOKEN` has precedence over the
compatibility alias `TRITON_EVAL_KERNEL_EVAL_TOKEN`; other fields retain the
`TRITON_EVAL_` prefix. No existing deployment-attestation setting changes.

## Add a catalog task

Each entry has four fields: `manifest_id`, `problem_id`, `reference_digest` and
`reference_path`. The digest is SHA-256 of the exact UTF-8 file bytes. References
must be regular files inside the catalog root; absolute paths, `..` traversal
and escaping symlinks are rejected. A `(manifest_id, problem_id)` pair is unique.
Lookup without a manifest restriction also requires a unique matching task.
Task IDs need no dataset prefix, reference header or provenance sidecar.

For example, create a small reference and its manifest in a new directory:

```bash
python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

root = Path("catalog-demo")
root.mkdir(exist_ok=False)
reference = root / "add.py"
reference.write_text("""import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_inputs():
    return [torch.ones(16), torch.ones(16)]
""", encoding="utf-8")
manifest = {"schema_version": "1.0", "entries": [{
    "manifest_id": "demo-v1",
    "problem_id": "vector_add",
    "reference_digest": hashlib.sha256(reference.read_bytes()).hexdigest(),
    "reference_path": "add.py",
}]}
(root / "catalog.json").write_text(json.dumps(manifest), encoding="utf-8")
PYTHON
export TRITON_EVAL_CATALOG_ROOT="$PWD/catalog-demo"
export TRITON_EVAL_CATALOG_MANIFEST_PATH="$PWD/catalog-demo/catalog.json"
```

Use these catalog settings alongside the remaining
[instance configuration](../operations/deployment.md). Catalog validation checks
file identity and encoding; successful loading does not establish that reference
execution will succeed on the target device. Publish new reference bytes under
a new manifest binding. Dataset conversion, if needed, happens before serving;
the [official-task adapter](../operations/official-task-registration.md) is one
optional importer.

## Native HTTPS for the platform verifier

Harbor's platform verifier requires HTTPS even when its destination is
loopback. Start the service's own two listeners in one `python -m scripts.run_api` process:

```dotenv
TRITON_EVAL_API_HOST=0.0.0.0
TRITON_EVAL_API_PORT=12907
TRITON_EVAL_API_TLS_HOST=127.0.0.1
TRITON_EVAL_API_TLS_PORT=12909
TRITON_EVAL_API_TLS_CERTFILE=/absolute/service-tls/server.crt
TRITON_EVAL_API_TLS_KEYFILE=/absolute/service-tls/server.key
```

Set the service unit's working directory to the service checkout and its
`ExecStart` to the existing Python interpreter followed by `-m scripts.run_api`.
The `uvicorn scripts.run_api:app` CLI serves
only one port and does not activate this mode. Migrate the certificates and
CA bundle into a service-owned directory, preserving certificate validity
and restricting the private key. The lane-A verifier pins
`https://127.0.0.1:12909`, allowed origin port 12909 and loopback CIDR; its
`SSL_CERT_FILE` points to the migrated CA bundle. Tool clients use HTTP 12907
on the admitted container gateway.

Both listeners share one AppContext, evaluation instance, catalog, Redis
connection and ASGI lifespan. Startup storage checks and maintenance
execute once. TLS configuration and both sockets are validated before serving;
TLS bind/key failure leaves no HTTP-only service. One shutdown signal or owner
cancellation joins both server tasks before lifespan maintenance cleanup.
The owner explicitly manages tasks for deployed Python 3.10 compatibility.
The listener tests perform real local HTTP/TLS requests and verify port closure
and maintenance shutdown; each deployment must validate its certificate and single-process listener setup. No certificate/key bytes belong in the repository.

The token authenticates the fixed `kernel-eval` trial caller only on the facade;
it grants no `/evaluations` or observer access. Submission and recovery go
through `EvaluationInstance.submit/get`. Readiness, durable admission,
caller/session quotas, idempotency and visibility apply to both tools and
internal diagnostics. A task lookup requires an exact catalog `problem_id`,
optionally restricted to the configured manifest. The reference bytes are
rechecked on every lookup. Missing, ambiguous or invalid reference bindings
return `403 {"error":"task_id mismatch"}` before budget use. Sidecars are not
read during startup or request resolution.

## Tool calls

```json
{
  "task_id": "vector_add",
  "kernel_code": "<candidate source>",
  "session": "<trial execution identity>"
}
```

Use `Authorization: Bearer <KERNEL_EVAL_TOKEN>`. `session` or `X-Session-ID`
identifies the trial; both, when supplied, must agree. IDs match
`[A-Za-z0-9._:-]{1,128}`. Old clients omitting both retain per-client-IP budgets.
Trials sharing an IP must supply distinct trial identities.

`verify` requests correctness. `benchmark` requests correctness plus profiler
timing (five warmups, twenty repeats by default). The prefixed
`KERNEL_EVAL_PERFORMANCE_WARMUP/REPEATS` settings may change before freezing.
`profile` accepts `profile_mode` (`op`/`deep`, default `op`), nonnegative integer
`case_idx`, and `profile_launch_count` (integer 1–100, `op` only). As previously,
launch count is validated but not forwarded to the evaluator.

FastAPI validates body and header models directly. Invalid JSON, missing required
fields, forbidden reference/extra fields and invalid profile options return 422
`INVALID_REQUEST`. Valid requests with missing or invalid credentials return 401
`UNAUTHENTICATED`. These errors use the shared error envelope and include a
request ID. Body/header session disagreement remains a 400 tool-input error;
catalog mismatch is 403 and operation conflict is 409. Global body limits apply.
Successful tool responses are validated and serialized through their response
models, while observation text and optional-field presence are preserved.

## Budget and recovery

The three actions share eighteen turns per trial. Redis transactions persist
reservations across concurrent API processes and restarts, with no memory
fallback. Failed admission consumes its reserved turn; auth/schema/catalog
rejection consumes none. `TRITON_EVAL_KERNEL_EVAL_MAX_TURNS` defaults to 18;
`-1` is unlimited. Records expire after the existing durable retention window,
refreshed by each newly reserved turn.

Invocation and evaluation identities differ. Without a client
`Idempotency-Key`, each HTTP tool call uses a new turn while identical
session/action/source calls replay the same durable evaluation. With an explicit
key, transport retries reuse both the turn and evaluation. Reusing a key with
different canonical request bytes returns 409 without another reservation.
Keys are scoped by session so independent samples never share evaluations.

Durable rate/inflight contention (429/413 decisions) retries up to eight
admissions with the same key and bounded backoff. Polling retains ownership.
The wait cap defaults to 840 seconds
(`TRITON_EVAL_KERNEL_EVAL_MAX_WAIT_SECONDS`). Timeout reports the evaluation ID
and that it may finish later; it neither cancels nor terminalizes execution.
An identical retry recovers the original durable execution.

Correct verify/benchmark outcomes attach an internal case-zero `op` profile by
default. It has a separate durable key, follows the same admission/ownership
path, and consumes no extra tool turn. Diagnostic failure only omits metrics.
Set `TRITON_EVAL_KERNEL_EVAL_AUTO_DIAGNOSTICS=false` before freezing to disable
it. `/health` reports actual whole-instance readiness.

## Upgrade from official-only task lookup

The four-field catalog format and `/verify`, `/benchmark`, `/profile` request
and response shapes are unchanged. Existing official catalog entries and their
reference bytes and hashes remain valid. Tool translation now sets the internal
evaluation `op_name` to `problem_id`, rather than a sidecar's operation name.
That can change canonical request identity for an existing official task.

Before upgrading, stop new admissions and collect in-flight results from the
original instance. Start new operations with new session and idempotency keys.
An identical old HTTP payload is not necessarily the same internal request after
this change; do not use replay across the upgrade as a migration mechanism.
Keep any outstanding evaluation handles for recovery on their original version.
The engine's existing official-reference compatibility and pinned reference-cache
policies remain in place; this change does not generalize those caches to every
catalog reference. See [reference caching](../operations/reference-cache.md).

## Verification boundary

```bash
.venv/bin/pytest -q tests/test_kernel_eval_renderer.py tests/test_kernel_eval_budget.py tests/test_kernel_eval_api.py
```

The 23 historical fixtures replay against canonical envelope/observation
SHA-256 values without importing historical bridge code. Synthetic tests add
benchmark unavailable, correctness skipped, profile rejected and evaluation
failed. Historical pretty-printed files contain a final newline; their MANIFEST
hashes canonical unindented envelopes and observations without that newline.

Route tests use an ordinary reference catalog, Redis transactions, durable
admission, lease claims, fenced terminal writes and owned polling. A test worker
commits captured outcomes in place of NPU computation. These establish CPU
protocol behavior. Live admission, container/token delivery, device execution,
native-model gates and fleet cutover require validation on the target host.
