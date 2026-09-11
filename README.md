# triton-evaluation

Self-contained FastAPI + Redis service for evaluating Triton-Ascend kernels.
The vendored evaluation engine provides correctness verification, optional
in-job benchmarking, and standalone msprof diagnostics. Jobs run as isolated
subprocesses on workers pinned one-per-NPU-card.

## Run

Two ways: **directly on the host** (uses your existing conda env + CANN) or **in Docker**
(see below). Both need a running **Redis**.

### On the host — one-time setup

Either reuse a conda env that already has the Ascend stack, or create one —
the versions are declared once in `pyproject.toml` (`[project.optional-dependencies].npu`):

```bash
conda create -n triton-eval python=3.11 && conda activate triton-eval
pip install -e ".[npu]" --extra-index-url https://<ascend-wheel-index>/simple
#   (the [npu] extra carries the pinned Ascend stack — see pyproject.toml;
#    the extra index is only needed because torch_npu / triton-ascend are
#    not on PyPI)
redis-server --daemonize yes       # or point TRITON_EVAL_REDIS_URL at an existing Redis
```

The engine is bundled, so `TRITON_EVAL_ENGINE_DIR` is not needed. `scripts/start.sh` must
run with this conda env active (or pass `PYTHON_BIN=/path/to/env/bin/python`).

### Quick start (script)

```bash
cp .env.example .env        # then edit .env: set TRITON_EVAL_CARDS (and TRITON_EVAL_REDIS_URL)
bash scripts/start.sh       # starts the API + one worker per card in TRITON_EVAL_CARDS
bash scripts/stop.sh        # stops them (reads the PID file start.sh wrote to LOG_DIR)
```

`scripts/start.sh` sources `.env`, optionally sources `CANN_SET_ENV_PATH`, then launches
`uvicorn run_api:app` and one `run_worker.py --card <n>` per card (each pins
`ASCEND_RT_VISIBLE_DEVICES`). Logs go to `LOG_DIR/` (`api.log`, `worker_<card>.log`).
Ctrl-C stops a foreground run; `scripts/stop.sh` stops a backgrounded one. For production,
supervise with systemd/docker and restart on crash.

### Manual

```bash
export TRITON_EVAL_REDIS_URL=redis://localhost:6379/0
# engine is bundled; TRITON_EVAL_ENGINE_DIR is optional (override only)

python3 run_api.py                 # API on 127.0.0.1:10907
python3 run_worker.py --card 0 &   # one worker per physical NPU card
python3 run_worker.py --card 1 &
```

See `.env.example` for all environment variables.

## Run in Docker (Ascend NPU host)

The image is **self-contained for the Python stack**: on the official `python:3.11`
(aarch64) base it installs the SAME declared versions the host path uses
(`pip install ".[npu]"` — see `pyproject.toml`). Network access is needed at build time;
when torch_npu / triton-ascend are not reachable from PyPI, pass
`--build-arg PIP_EXTRA_INDEX_URL=https://<ascend-wheel-index>/simple`.

Host-coupled pieces — the **CANN toolkit, NPU driver, and `/dev/davinci*` devices** — are
NOT baked into the image; they are bind-mounted at runtime because they must match the
host's driver. Redis is the **host's** (via `--network host`), not bundled. One container
runs the API + one worker per card.

### Quick path — scripts

```bash
docker build -t triton-eval:1.0 .     # add --build-arg PIP_EXTRA_INDEX_URL=... if needed
cp .env.example .env                  # set CANN_HOST, TRITON_EVAL_CARDS, redis url, ...
bash scripts/docker_run.sh 1.0        # mounts CANN/driver/devices/secret/catalog, runs all
curl 127.0.0.1:10907/healthz          # -> {"redis":"ok"}
```

`scripts/docker_run.sh` reads `.env` (same file the host path uses), mounts
`${REPO_ROOT}/.capability-secret.json` (mode 0600 caller capabilities) and
`${REPO_ROOT}/catalog/` (release catalog) read-only, and honors
`TRITON_EVAL_CARDS` (one `--device /dev/davinciN` per card).

### With docker compose

```bash
cp docker.env.example docker.env      # set TRITON_EVAL_CARDS (must match the mounted chip) + redis
docker compose --env-file docker.env build
docker compose --env-file docker.env up -d
docker compose --env-file docker.env logs -f
```

For multiple chips, list one `- /dev/davinciN` per card under `devices:` in the compose file.

Card health (spec §10): workers run a canary kernel before claiming after any
suspicious attempt death and at startup. A failed canary drains the worker (no
heartbeats — readiness drops to 503 once no claim-capable worker remains) and
drives a recovery ladder: in-process cleanup → `TRITON_EVAL_CARD_RESET_COMMAND`
("{card}" substituted) → process restart. Host reboot is never automatic. An
evaluation whose attempts die suspiciously twice is quarantined (terminal
`WORKER_LOST` with a poison message; never retried by the service).

Compile verdict cache (spec §8): deterministic compile failures (syntax/import
class via an admission-time precheck subprocess; all compile-class via
worker-observed engine outcomes) are cached by a toolchain-scoped digest for
`TRITON_EVAL_VERDICT_CACHE_TTL_SECONDS` (default 7 d) and short-circuited at
admission into a terminal identical in shape to an on-card compile failure —
no queue slot, no card time. Set `TRITON_EVAL_CANN_FINGERPRINT` to your CANN
version so toolchain upgrades invalidate stale verdicts. The precheck runs in
the API process environment, so API and worker must share one image/conda env
or import-class verdicts diverge from on-card reality.

## Endpoints (evaluation API v2 — durable)

- `POST /evaluations?wait_seconds=0..60` — authenticate with a caller-scoped opaque
  capability (`Authorization: Bearer <token>`), always send `X-Session-ID`
  (correlation + secondary quota; never authorizes) and `Idempotency-Key`
  (scoped to the authenticated caller). Returns `200` with a terminal
  `EvaluationResource`, or `202` with the nonterminal resource plus
  `Location: /evaluations/{evaluation_id}`. Same key + same canonical request
  replays the same resource; a different request with the same key is `409`.
- `GET /evaluations/{evaluation_id}` — recover a visible resource by ID; unknown and
  invisible IDs are the same `404`. Nonterminal responses carry `Retry-After: 1`.
- `GET /livez` / `GET /readyz` — process liveness / dependency-aware readiness.
- `GET /internal/metrics` — observer-authenticated OpenMetrics.

All non-2xx responses are one redacted `ErrorEnvelope` shape. The wire contract is
frozen and published as reproducible artifacts — the only cross-repository contract:

- `contracts/evaluation-api/2.0/contract.json` (schema + fixture-index digests)
- `contracts/evaluation-api/2.0/schema.json` (entrypoints
  `#/$defs/EvaluationRequest`, `#/$defs/EvaluationResource`, `#/$defs/ErrorEnvelope`)
- `contracts/evaluation-api/2.0/fixtures/index.json` (+ 29 golden fixtures)
- `contracts/evaluation-api/2.1/contract.json`, `schema.json`, `fixtures/index.json`
  (+ 4 golden fixtures) — the additive v2.1 tree, described below

Current digests (pin these in consumer contract tests):

- 2.0 schema: `5c81434931f7178ee9c1a4f2220404a42764c076f64b0a1b9f40bb88b87ca502`
- 2.0 fixture index: `50c226aa68c926284e3bd130145056e3b542bb5eb7090e9fb513b5a4ee5f0d52`
- 2.1 schema: `f6ba908315753b5f5f70281e0ea23c824cf355e3b4bdf5f91955159e588305ac`
- 2.1 fixture index: `bb30accda364ffeee8bb2552429bb501fffa891d809fdb7b36f50223af3975a9`

The 2.0 tree is byte-frozen: its artifacts generate from a frozen model snapshot, so
runtime extensions can never move its digest-pinned bytes. Both trees are verified
byte-for-byte at startup (API lifespan, container entrypoint, and `scripts/start.sh`);
any drift fails closed before serving.

### Contract v2.1 (additive; 2.0 stays valid until its sunset)

A request selects the contract version with `schema_version: "2.0" | "2.1"` (default
behavior for omitted fields is unchanged). v2.1 adds, and v2.0 requests are rejected
at validation if they carry any of:

- `checks: ["compile", ...]` — the compile check. `["compile"]` alone is legal and runs
  at admission (never queues, never touches a card); a compile failure short-circuits to
  a terminal shaped like an on-card compile_error. When requested alongside other
  checks, `result.compile` carries the typed outcome
  (`{outcome: "compiled"|"compile_error", compiler_stage?, error_excerpt?, duration_ms}`).
  The published schemas cannot express the per-version rule that 2.0 allows at most
  3 checks — that rejection is runtime-only; validate 2.0 requests against the 2.0 schema.
- `purpose: "rollout"` — server-declared RL traffic; admitted like a trial (same
  capability scope, no catalog binding) and resolved to the rollout quota class.
- `performance.method: "event" | "profiler"` — selects the measurement path. The
  effective default is `profiler` for 2.0 requests and `event` for 2.1 requests.
  Results then carry `measurement_method: "npu_event_v1" | "profiler_kernel_sum_v1"`
  on the overall performance result and every measured case. **Cross-method absolute
  latencies are not comparable** (event timing is stream-ordered device time; the
  profiler path sums kernel CSV rows) — the field exists so datasets never mix methods
  silently; compare speedups only within one method.
- Poison quarantine surfaces the typed check code `SUSPECTED_POISON_CANDIDATE`
  (retryable `false`) on 2.1 resources; 2.0 keeps the frozen `WORKER_LOST` shape.

2.0 sunset: `2027-03-31T00:00:00Z` (published instant, same time-box pattern as the
legacy routes). Deployments arm enforcement by setting `TRITON_EVAL_V20_SUNSET_AT` to
the exact instant; an armed deployment rejects 2.0 requests with
`400 INVALID_REQUEST` once the clock passes the instant. Arming requires the exact
published value, at most 90 days before it, and never after it — anything else fails
startup. Unset means unarmed (pre-sunset development).

Deploy note: this release widens the runtime request model, so the canonical
bytes of a 2.0 request now include `performance.method: null`. Idempotency
records admitted before the deploy will not byte-match a replay after it —
replays of in-flight 2.0 submissions return 409 (IDEMPOTENCY_CONFLICT) for up
to `retention_seconds` instead of 200. Clients should treat a 409 after the
deploy as "resubmit as a new evaluation".

Callers hold static capabilities from a mounted secret file: `trial:submit/read` for
interactive harnesses (trial traffic only), `grade:submit/read` for the trusted hub
(grade traffic, manifest-bound reference), and `observer:read` for global read-only
observability. Identity, visibility, idempotency, and primary quotas key on the
authenticated `caller_id` — never the session header.

Durability: Redis runs `--appendonly yes --appendfsync always`; EXEC acknowledgement
is the first-2xx durability boundary. The API binds the preflight-validated IPv4
gateway of Docker network `kernel-agent-sandbox-net` on port `11907` — never loopback
or wildcard. `scripts/compose.sh` runs `scripts/deployment_preflight.py` before
`up|start|restart` and refuses to start otherwise. Rollback rule: keep the
`triton-eval-redis-v2` volume (never `down -v`); the prior build can read the same
v2 keyspace.

Client HTTP timeouts cover only `wait_seconds` plus transport margin — a wait-budget
expiry or disconnect never cancels or terminalizes the evaluation. Persist the
`evaluation_id` from any 200/202 body immediately and recover through GET; replay the
original POST (same key and body) only when no response was obtained at all.

Legacy compatibility (`POST /verify`, `POST /profile`, `GET /tasks/{task_id}`) is
time-boxed: enabled only with `legacy_routes_enabled=true` and the exact published
sunset instant `2026-10-15T00:00:00Z`, with a window of at most 90 days. Enabling
without that instant, after it, or wider fails startup. The routes require the same
authorization/session/idempotency headers and trial capabilities as v2.

Example correctness + performance request:

```bash
curl -X POST "http://127.0.0.1:11907/evaluations?wait_seconds=30" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <trial-capability-token>' \
  -H 'X-Session-ID: example-shell' \
  -H 'Idempotency-Key: example-run-1' \
  --data '{
    "schema_version": "2.0",
    "purpose": "trial",
    "op_name": "add",
    "reference_code": "import torch ...",
    "kernel_code": "import torch ...",
    "checks": ["correctness", "performance"],
    "deadline_seconds": 900,
    "performance": {"warmup": 5, "repeats": 50}
  }'
```

The response is an `EvaluationResource` with a monotonic `revision`, `state` in
`queued|running|completed|failed|timed_out`, and (once terminal) a typed
`result.correctness`, `result.performance`, `result.profile` plus `provenance`.
It never returns reference code or test inputs.

## Test

The [card health module](docs/architecture/card-health.md) documents the worker's
health gate, recovery invariants, and CPU/hardware test seams.
The [attempt card provenance](docs/architecture/attempt-card-provenance.md) note
documents claim ownership, advisory failure handling, and storage compatibility.

```bash
pip install -e ".[test]"
python -m pytest -v
```
