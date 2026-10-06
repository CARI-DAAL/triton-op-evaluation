# Native kernel_eval tools

The optional facade serves `POST /verify`, `POST /benchmark` and `POST /profile`
on the evaluation API listener. It obtains official references from
`ReleaseCatalog` and renders the established `[kernel_eval]` observation.
It has no bridge process, configuration, second token, runtime adapter import
or upstream HTTP dependency. `/evaluations` keeps its v2.1 contract.

## Enable the facade

Register the official catalog and `.PROVENANCE.json` sidecars, then add these
settings to the protected service environment:

```dotenv
TRITON_EVAL_KERNEL_EVAL_ENABLED=true
KERNEL_EVAL_TOKEN=<sandbox tool secret>
TRITON_EVAL_KERNEL_EVAL_MANIFEST_ID=npukernelbench-official-v1
```

Disabled is the default. An empty token also disables the three tools and
`/health`, returning HTTP 404 (tools retain the existing v2.1 `NOT_FOUND`
envelope while disabled). The Pydantic `SecretStr`
masks printable configuration. `KERNEL_EVAL_TOKEN` has precedence over the
compatibility alias `TRITON_EVAL_KERNEL_EVAL_TOKEN`; other fields retain the
`TRITON_EVAL_` prefix. No existing deployment-attestation setting changes.

## Native HTTPS for the platform verifier

Harbor's platform verifier requires HTTPS even when its destination is
loopback. Start the service's own two listeners in one `run_api.py` process:

```dotenv
TRITON_EVAL_API_HOST=0.0.0.0
TRITON_EVAL_API_PORT=12907
TRITON_EVAL_API_TLS_HOST=127.0.0.1
TRITON_EVAL_API_TLS_PORT=12909
TRITON_EVAL_API_TLS_CERTFILE=/absolute/service-tls/server.crt
TRITON_EVAL_API_TLS_KEYFILE=/absolute/service-tls/server.key
```

Set the service unit's `ExecStart` to its existing Python interpreter followed
by the service checkout's `run_api.py`. The `uvicorn run_api:app` CLI serves
only one port and does not activate this mode. Migrate the certificates and
CA bundle into a service-owned directory, preserving certificate validity
and restricting the private key. The lane-A verifier pins
`https://127.0.0.1:12909`, allowed origin port 12909 and loopback CIDR; its
`SSL_CERT_FILE` points to the migrated CA bundle. Tool clients use HTTP 12907
on the admitted container gateway.

Both listeners share one AppContext, evaluation instance, catalog, Redis
connection and ASGI lifespan. Startup contract/storage checks and maintenance
execute once. TLS configuration and both sockets are validated before serving;
TLS bind/key failure leaves no HTTP-only service. One shutdown signal or owner
cancellation joins both server tasks before lifespan maintenance cleanup.
The owner explicitly manages tasks for deployed Python 3.10 compatibility.
The listener tests perform real local HTTP/TLS requests and verify port closure
and maintenance shutdown; machine2 still validates its certificate and single
PID environment admission. No certificate/key bytes belong in the repository.

The token authenticates the fixed `kernel-eval` trial caller only on the facade;
it grants no `/evaluations` or observer access. Submission and recovery go
through `EvaluationInstance.submit/get`. Readiness, durable admission,
caller/session quotas, idempotency and visibility apply to both tools and
internal diagnostics. A task lookup requires an exact official `problem_id`,
optionally restricted to the configured manifest. Missing, ambiguous or mutated
references/sidecars return `403 {"error":"task_id mismatch"}` before budget use.

## Tool calls

```json
{
  "task_id": "official_l3_MatmulBothTrans",
  "kernel_code": "<candidate source>",
  "session": "<trial execution identity>"
}
```

Use `Authorization: Bearer <KERNEL_EVAL_TOKEN>`. `session` or `X-Session-ID`
identifies the trial; both, when supplied, must agree. IDs match
`[A-Za-z0-9._:-]{1,128}`. Old clients omitting both retain per-client-IP budgets.
Trials sharing an IP must supply distinct identities; the 320-slot experiment
uses the Trial execution identity.

`verify` requests correctness. `benchmark` requests correctness plus profiler
timing (five warmups, twenty repeats by default). The prefixed
`KERNEL_EVAL_PERFORMANCE_WARMUP/REPEATS` settings may change before freezing.
`profile` accepts `profile_mode` (`op`/`deep`, default `op`), nonnegative integer
`case_idx`, and `profile_launch_count` (integer 1–100, `op` only). As previously,
launch count is validated but not forwarded to the evaluator.

Authentication precedes JSON/schema checks. Credentials return the legacy
401 `unauthorized` error; invalid JSON/non-object bodies return 400 `invalid
json`. Missing required strings, forbidden reference/extra fields and invalid
profile options retain the existing 400 messages. Global API body limits apply.

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

## Verification boundary

```bash
.venv/bin/pytest -q tests/test_kernel_eval_renderer.py tests/test_kernel_eval_budget.py tests/test_kernel_eval_api.py
```

The 23 historical fixtures replay against canonical envelope/observation
SHA-256 values without importing historical bridge code. Synthetic tests add
benchmark unavailable, correctness skipped, profile rejected and evaluation
failed. Historical pretty-printed files contain a final newline; their MANIFEST
hashes canonical unindented envelopes and observations without that newline.

Route tests use the actual assembler/catalog, Redis transactions, durable
admission, lease claims, fenced terminal writes and owned polling. A test worker
commits captured outcomes in place of NPU computation. These establish CPU
protocol behavior. Live admission, container/token delivery, device execution,
native-model gates and fleet cutover require machine2 validation.
