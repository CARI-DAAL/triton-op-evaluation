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

python3 run_api.py                 # API on 0.0.0.0:10907
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

- `POST /verify` → `{task_id}`. Pass `measure_speed: true` to run the internal
  benchmark after every verify case passes; performance is returned under
  `result.benchmark`.
- `POST /profile` → `{task_id}`. `profile_mode` is `"op"` (default) or
  `"deep"`. Omit `case_idx` to profile every input case, or pass a zero-based
  index to profile one case. Profile does not require a verify task.
- `GET /tasks/{task_id}` → task status and direct evaluator result JSON.
- `GET /healthz` → Redis health.

All verify/profile jobs share one Redis queue. Whichever configured card worker
becomes free first may claim a profile job. Once claimed, all cases and metric
passes in that job stay on that card. “Free” is worker state, not an
`npu-smi` probe; cards configured for this service are expected to be
dedicated.

The following creates a complete two-case profile request, submits the default
full-case op profile, polls its result once, and then submits a deep profile for
case 1:

```bash
python3 - <<'PY' >/tmp/triton-profile-request.json
import json

reference_code = """import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_input_groups():
    return [
        [torch.randn(4096), torch.randn(4096)],
        [torch.randn(8192), torch.randn(8192)],
    ]
"""

kernel_code = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        return out
"""

print(json.dumps({
    "op_name": "add",
    "reference_code": reference_code,
    "kernel_code": kernel_code,
    "profile_mode": "op",
}))
PY

TASK_ID=$(
  curl -sS -X POST http://127.0.0.1:10907/profile \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/triton-profile-request.json |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)
curl -sS "http://127.0.0.1:10907/tasks/${TASK_ID}"

python3 - <<'PY'
import json

path = "/tmp/triton-profile-request.json"
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
payload.update({"profile_mode": "deep", "case_idx": 1, "timeout": 7200})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

curl -X POST http://127.0.0.1:10907/profile \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/triton-profile-request.json
```

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```
