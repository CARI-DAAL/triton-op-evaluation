# triton-evaluation

Self-contained FastAPI + Redis service for evaluating Triton-Ascend kernels.
The vendored evaluation engine provides correctness verification, optional
in-job benchmarking, and standalone msprof diagnostics. Jobs run as isolated
subprocesses on workers pinned one-per-NPU-card.

## Run

Two ways: **directly on the host** (uses your existing conda env + CANN) or **in Docker**
(see below). Both need a running **Redis**.

### On the host — one-time setup

Use the conda env that already has `torch_npu` + `triton-ascend`, add the service's light
deps, and make sure Redis is up:

```bash
conda activate <your-env>          # the env you run these kernels with
pip install -e . && pip install pandas     # service deps (fastapi/uvicorn/redis/...) + engine's pandas
redis-server --daemonize yes       # or point TRITON_EVAL_REDIS_URL at an existing Redis
```

`pip install -e .` does not touch torch / torch_npu / triton-ascend. The engine is bundled,
so `TRITON_EVAL_ENGINE_DIR` is not needed. `scripts/start.sh` must run with this conda env
active (or pass `PYTHON_BIN=/path/to/env/bin/python`).

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

Reuses the proven **vllm-ascend** manual device-mount scheme (no ascend-docker-runtime).
The base image `quay.io/ascend/vllm-ascend` carries **CANN 9.0.0 + torch_npu (arm64)** matching
the host driver, so the container mounts only the host **driver + NPU devices** — not CANN.
The base is py3.12, so the image builds a **py3.11 conda env** mirroring the working
triton-ascend stack (torch 2.9.0 / torch_npu 2.9.0 / triton 3.5.0 / triton-ascend 3.2.1).
Redis is the **host's** (via `--network host`), not bundled. One container runs the API +
one worker per card.

### With docker compose

```bash
cp docker.env.example docker.env      # set TRITON_EVAL_CARDS (must match the mounted chip) + redis
docker compose --env-file docker.env build
docker compose --env-file docker.env up -d
docker compose --env-file docker.env logs -f
```

For multiple chips, list one `- /dev/davinciN` per card under `devices:` in the compose file.

### Or plain docker run (single chip, e.g. npu:14)

```bash
docker build -t triton-evaluation .
docker run -d --name triton-eval --network host --shm-size 32g \
  --device /dev/davinci14 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD/logs":/opt/triton-eval/logs \
  -e TRITON_EVAL_CARDS=14 \
  -e TRITON_EVAL_REDIS_URL=redis://127.0.0.1:6379/0 \
  -e TRITON_EVAL_API_PORT=10907 \
  triton-evaluation

curl 127.0.0.1:10907/healthz          # -> {"redis":"ok"}
```

`TRITON_EVAL_CARDS` must match the mounted `/dev/davinciN`. Smoke-test one real Triton kernel
after bring-up to confirm triton-ascend links `bishengir-compile` from the image's CANN.

## Endpoints

- `POST /evaluations` synchronously waits for one terminal, typed evaluation result.
  Select `correctness`, `performance`, and/or `profile` with `checks`. Performance
  requires correctness and only runs when correctness passes; profile is independent.
  Every request requires an `X-Session-ID` header so the service can enforce one
  in-flight evaluation per session/container.
- `GET /healthz` → Redis health.

The API submits work to the Redis queue and polls it internally; clients never receive
or poll a non-terminal task. `timeout` covers the whole server-side request, including
queue wait and all checks. Set the client HTTP timeout higher than the request timeout
(the smoke client uses 60 seconds of grace), and do not automatically retry an HTTP read
timeout because the worker may still be running. Server timeouts do not cancel workers.

All checks share one FIFO Redis queue. Whichever configured card worker becomes free
first claims the next evaluation. Once claimed, all requested checks stay on that card.
“Free” is worker state, not an `npu-smi` probe; cards configured for this service are
expected to be dedicated.

Example correctness + performance request:

```bash
curl --max-time 960 -X POST http://127.0.0.1:10907/evaluations \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: example-shell' \
  --data '{
    "op_name": "add",
    "reference_code": "import torch ...",
    "kernel_code": "import torch ...",
    "checks": ["correctness", "performance"],
    "timeout": 900,
    "performance": {"warmup": 5, "repeats": 50}
  }'
```

The response always has terminal `status` plus typed `result.correctness`,
`result.performance`, and `result.profile` fields. It never returns reference code or
test inputs. The API binds to loopback by default; sandbox consumers should reach it
through the managed kernel-harness gateway/relay rather than exposing the port remotely.

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```
