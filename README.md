# triton-evaluation

Self-contained FastAPI + Redis service for evaluating Triton-Ascend kernels. The
evaluation **engine** (correctness `verify` + performance `benchmark`) is vendored
inside the package at `triton_evaluation/engine/` and run per-job as an isolated
subprocess on a pinned NPU card. Evaluators are pluggable (verify, benchmark, later profile).

## Run

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

python3 run_api.py                 # API on 0.0.0.0:8000
python3 run_worker.py --card 0 &   # one worker per physical NPU card
python3 run_worker.py --card 1 &
```

See `.env.example` for all environment variables.

## Run in Docker (Ascend NPU host)

No Ascend base image is required — the container mounts the host's CANN toolkit,
driver, and NPU devices at runtime, and only bakes Python + the service + the ML
stack (`torch` / `torch_npu` / `triton-ascend` / `pandas`). Redis runs as a
bundled container. Target host arch: **aarch64**.

```bash
cp docker.env.example docker.env
# Edit docker.env: set TORCH_SPEC / TORCH_NPU_SPEC / TRITON_ASCEND_SPEC to the exact
# versions from your host env, plus CANN_PATH and TRITON_EVAL_CARDS.
# Then edit docker-compose.yml `worker.devices` to list one /dev/davinciN per card.

docker compose --env-file docker.env build     # build on the NPU host (aarch64)
docker compose --env-file docker.env up -d      # redis + api + workers
docker compose --env-file docker.env logs -f worker
docker compose --env-file docker.env down
```

The `worker` container sources `CANN_PATH/ascend-toolkit/set_env.sh` and runs one
`run_worker.py --card N` per card in `TRITON_EVAL_CARDS`; the `api` container serves
uvicorn on the published port. See `docker.env.example` for how to read the ML
versions off your host.

## Endpoints

- `POST /verify` → `{task_id}`
- `POST /benchmark` (requires `verify_task_id` or `verify_not_required`) → `{task_id}`
- `GET /tasks/{task_id}` → status + skill result JSON
- `GET /healthz`

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```
