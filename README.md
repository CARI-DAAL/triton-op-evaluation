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

Current digests (pin these in consumer contract tests):

- schema: `5c81434931f7178ee9c1a4f2220404a42764c076f64b0a1b9f40bb88b87ca502`
- fixture index: `50c226aa68c926284e3bd130145056e3b542bb5eb7090e9fb513b5a4ee5f0d52`

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


```bash
pip install -e ".[test]"
python -m pytest -v
```
