# triton-evaluation on Ascend — docker deploy + repo sync guide

This host runs the triton-evaluation service in **docker**, but using the **host's own Python
environment** (bind-mounted), not the image's conda env (which diverged and is missing the
CANN-TBE / triton-ascend runtime deps). This doc contains the **exact repo file contents** to
sync to another machine, plus the per-host setup that lives outside the repo.

Production: docker compose, 1 eval + 1 redis, cards 8-15, port **11907**, redis db7,
`restart: unless-stopped`. Smoke test (correctness **and** speed benchmark) passes.

> Host-specific paths below (`/home/n84449292/miniconda3`, `/home/n84449292/.local`,
> `/home/canada_group_folder/CANN/9.0.0.0430`, card ids 8-15) are for THIS host. On another
> machine, edit them to match (the env vars + the volume `devices`/paths).

---

## 1. Repo files to sync (full content)

### `docker-compose.yml` (rewrite)
```yaml
# triton-evaluation on an Ascend host — runs the HOST's proven Python env inside the
# container (Plan A / "copy host env to docker"). The image supplies the base + /opt/triton-eval
# source + entrypoint; the actual Python stack (the kernelgym conda env + ~/.local user-site,
# where torch 2.9.0 / torch_npu / triton + ascend backend live) and the host CANN are
# bind-mounted, and PYTHON_BIN points the service at the kernelgym python.
#
# This sidesteps the image's own conda env, which diverged from the host and is missing the
# CANN-TBE + triton-ascend runtime deps (decorator/scipy/attrs/psutil/pybind11/...). Replicating
# the host env is the only reliable way to make the Ascend triton path work in-container today.
#
# Configured for THIS host: cards 8-15, port 11907, redis db7.
# To change the card set, edit TWO places together: the `devices:` davinci list AND
# TRITON_EVAL_CARDS (here or in docker.env).
#
#   docker compose --env-file docker.env build     # one-time (base + source only)
#   docker compose --env-file docker.env up -d
#   docker compose --env-file docker.env logs -f
#   python3 scripts/test_eval_service.py --url http://127.0.0.1:11907
#   docker compose --env-file docker.env down
#
# Runs as root: NPU device access (rtSetDefaultDeviceId / get-device-count) fails for non-root
# in the container even with privileged + group_add, but works as root. The bind-mounted
# kernelgym env / ~/.local / CANN are owned by host uid 1001; root reads them fine (torch_npu
# emits an owner-mismatch WARNING, harmless).

services:
  triton-eval:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        # Only relevant if you rebuild AND stop using the host env below. The host-env
        # bind-mount bypasses the image's conda env at runtime regardless of these.
        BASE_IMAGE: ${BASE_IMAGE:-quay.io/ascend/vllm-ascend:v0.22.1rc1-a3}
        TORCH_SPEC: ${TORCH_SPEC:-torch==2.9.0}
        TORCH_NPU_SPEC: ${TORCH_NPU_SPEC:-torch_npu==2.9.0}
        TRITON_SPEC: ${TRITON_SPEC:-triton==3.5.0}
        TRITON_ASCEND_SPEC: ${TRITON_ASCEND_SPEC:-triton-ascend==3.2.0}
        PIP_EXTRA_INDEX_URL: ${PIP_EXTRA_INDEX_URL:-}
    image: triton-evaluation:latest
    command: ["all"]
    restart: unless-stopped
    depends_on: [redis]          # start redis first (host network; eval tolerates a brief gap)
    network_mode: host           # reach Redis at 127.0.0.1:6379
    privileged: true             # required for NPU device access in-container (see note above)
    shm_size: "32gb"             # torch_npu needs a large /dev/shm
    environment:
      # Run the service with the HOST kernelgym python (replicate host env in docker).
      PYTHON_BIN: ${PYTHON_BIN:-/home/n84449292/miniconda3/envs/kernelgym/bin/python3}
      # user-site = ~/.local (torch 2.9.0 / torch_npu / triton + ascend backend), decoupled from
      # HOME so HOME can point at a writable dir for ascend/triton compile caches.
      PYTHONUSERBASE: ${PYTHONUSERBASE:-/home/n84449292/.local}
      HOME: /tmp
      TRITON_EVAL_REDIS_URL: ${TRITON_EVAL_REDIS_URL:-redis://127.0.0.1:6379/7}
      TRITON_EVAL_API_HOST: 0.0.0.0
      TRITON_EVAL_API_PORT: ${TRITON_EVAL_API_PORT:-11907}
      TRITON_EVAL_CARDS: ${TRITON_EVAL_CARDS:-8,9,10,11,12,13,14,15}
      CANN_SET_ENV_PATH: ${CANN_SET_ENV_PATH:-/home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh}
      ASCEND_DRIVER_PATH: /usr/local/Ascend/driver                     # mounted from host
    devices:
      # shared Ascend control devices
      - /dev/davinci_manager
      - /dev/devmm_svm
      - /dev/hisi_hdc
      # one davinci device per physical card — keep in sync with TRITON_EVAL_CARDS
      - /dev/davinci8
      - /dev/davinci9
      - /dev/davinci10
      - /dev/davinci11
      - /dev/davinci12
      - /dev/davinci13
      - /dev/davinci14
      - /dev/davinci15
    volumes:
      # The host's working Python stack — kernelgym conda env + user-site (~/.local).
      - /home/n84449292/miniconda3:/home/n84449292/miniconda3:ro
      - /home/n84449292/.local:/home/n84449292/.local:ro
      # Host CANN (canada 9.0.0.0430, matches driver 25.5.0). Mounted at the same host path so
      # set_env.sh resolves. The base image's baked-in CANN is too old (rtGetDevMsg unsupported).
      - /home/canada_group_folder/CANN/9.0.0.0430:/home/canada_group_folder/CANN/9.0.0.0430
      - /usr/local/dcmi:/usr/local/dcmi
      # Full host driver (lib64/device/kernel/...). GE op-store init needs more than lib64.
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver
      - /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
      - /etc/ascend_install.info:/etc/ascend_install.info
      # Fixed entrypoint: PYTHON_BIN + unset baked-in CANN vars + add arch lib64 (libhccl.so).
      # (A future image rebuild that COPYs this entrypoint would let you drop this mount.)
      - ./docker/entrypoint.sh:/usr/local/bin/entrypoint.sh
      - ./logs:/opt/triton-eval/logs

  # Redis: the eval service's task queue + result store. Compose-managed so it comes up with
  # the service and survives reboot (restart: unless-stopped). Data is ephemeral (task queue
  # with TTL), so no AOF/RDB persistence. Binds 127.0.0.1:6379 on the host network — the same
  # address TRITON_EVAL_REDIS_URL points at.
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    network_mode: host
    command: ["redis-server", "--bind", "127.0.0.1", "--port", "6379", "--save", "", "--appendonly", "no"]
```

### `docker/entrypoint.sh` (3 fixes: PYTHON_BIN, unset baked-in CANN, arch lib64)
```bash
#!/usr/bin/env bash
# Container entrypoint: wire up the mounted host driver + baked CANN, then run the service.
#
#   entrypoint.sh all       -> API + one worker per card in TRITON_EVAL_CARDS (default)
#   entrypoint.sh api       -> uvicorn run_api:app only
#   entrypoint.sh workers   -> workers only
#   entrypoint.sh <cmd...>  -> exec cmd (escape hatch, e.g. `bash`)
#
# Python interpreter: defaults to the image's py3.11 conda env python3 (on PATH from the
# Dockerfile). Override with PYTHON_BIN to run a different interpreter — e.g. a bind-mounted
# host conda env (Plan A), so the engine subprocess (sys.executable) shares that env too.
set -euo pipefail
: "${PYTHON_BIN:=python3}"

# --- Host driver libs (mounted from host) ---
DRIVER_LIB="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64"
if [[ -d "${DRIVER_LIB}" ]]; then
  export LD_LIBRARY_PATH="${DRIVER_LIB}:${DRIVER_LIB}/driver:${LD_LIBRARY_PATH:-}"
fi

# --- CANN toolkit env (host CANN mounted over the base image's baked-in copy) ---
# The base image pre-sets ASCEND_TOOLKIT_HOME / ASCEND_OPP_PATH / ASCEND_HOME_PATH to its
# baked-in /usr/local/Ascend CANN. That build is too old for this driver: GEInitialize /
# OpCompileProcessor init fail (ACL_ALLOW_HF32 error 500001). CANN's set_env.sh will NOT
# overwrite those vars when already set, so unset them first and let the HOST CANN's
# set_env.sh repopulate them from its own install path.
if [[ -n "${CANN_SET_ENV_PATH:-}" && -r "${CANN_SET_ENV_PATH}" ]]; then
  unset ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH ASCEND_OPP_PATH \
        ASCEND_AICPU_PATH ASCEND_SLOG_PATH DAVINCIHOME 2>/dev/null || true
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV_PATH}"
else
  echo "WARN: CANN_SET_ENV_PATH ('${CANN_SET_ENV_PATH:-unset}') not readable; " \
       "torch_npu / the engine will likely fail to import." >&2
fi

# libhccl.so (loaded by torch_npu) lives under <toolkit>/aarch64-linux/lib64, which
# set_env.sh does not add — without it `import torch` fails
# ("libhccl.so: cannot open shared object file"). Add the arch lib dir if present.
if [[ -n "${CANN_SET_ENV_PATH:-}" ]]; then
  _cann_toolkit_dir="$(dirname "$(readlink -f "${CANN_SET_ENV_PATH}")")"
  for _archlib in "${_cann_toolkit_dir}/aarch64-linux/lib64" "${_cann_toolkit_dir}/lib64"; do
    if [[ -d "${_archlib}" && ":${LD_LIBRARY_PATH:-}:" != *":${_archlib}:"* ]]; then
      export LD_LIBRARY_PATH="${_archlib}:${LD_LIBRARY_PATH:-}"
    fi
  done
  unset _cann_toolkit_dir _archlib
fi

cd /opt/triton-eval

run_api() {
  exec "${PYTHON_BIN}" -m uvicorn run_api:app \
    --host "${TRITON_EVAL_API_HOST:-0.0.0.0}" \
    --port "${TRITON_EVAL_API_PORT:-10907}"
}

run_workers() {
  local pids=("$@")   # any pre-existing pids to also wait on
  IFS=',' read -ra cards <<< "${TRITON_EVAL_CARDS:-0}"
  for c in "${cards[@]}"; do
    c="${c//[[:space:]]/}"
    [[ -z "${c}" ]] && continue
    echo "Starting worker on NPU card ${c}" >&2
    "${PYTHON_BIN}" run_worker.py --card "${c}" &
    pids+=("$!")
  done
  [[ ${#pids[@]} -eq 0 ]] && { echo "No cards in TRITON_EVAL_CARDS" >&2; exit 2; }
  # shellcheck disable=SC2317
  stop() { kill "${pids[@]}" 2>/dev/null || true; }
  trap stop TERM INT
  wait
}

case "${1:-all}" in
  api)
    run_api
    ;;
  workers)
    run_workers
    ;;
  all)
    "${PYTHON_BIN}" -m uvicorn run_api:app \
      --host "${TRITON_EVAL_API_HOST:-0.0.0.0}" \
      --port "${TRITON_EVAL_API_PORT:-10907}" &
    api_pid=$!
    run_workers "${api_pid}"
    ;;
  *)
    exec "$@"
    ;;
esac
```

### `scripts/start.sh` (host-mode fallback — one block added after the CANN `source`)
The only change vs the original is the block below, inserted immediately after the
`source "${CANN_SET_ENV_PATH}"` block (mirrors the entrypoint's libhccl.so fix):
```bash
# CANN's set_env.sh adds <toolkit>/lib64, but on this install libhccl.so (loaded by
# torch_npu) lives under <toolkit>/aarch64-linux/lib64. Without it `import torch` fails
# ("libhccl.so: cannot open shared object file"). Add the arch lib dir if it exists.
if [[ -n "${CANN_SET_ENV_PATH:-}" ]]; then
  _cann_toolkit_dir="$(dirname "$(readlink -f "${CANN_SET_ENV_PATH}")")"
  for _archlib in "${_cann_toolkit_dir}/aarch64-linux/lib64" "${_cann_toolkit_dir}/lib64"; do
    if [[ -d "${_archlib}" && ":${LD_LIBRARY_PATH:-}:" != *":${_archlib}:"* ]]; then
      export LD_LIBRARY_PATH="${_archlib}:${LD_LIBRARY_PATH:-}"
    fi
  done
  unset _cann_toolkit_dir _archlib
fi
```
(`scripts/start.sh` is only used for host-mode; production is docker. Kept for parity.)

### `docker-compose.npu0-test.yml` (NEW — isolated single-card smoke-test harness)
Card 0 / port 11908 / redis db8, same host-env recipe, for validating without touching the
8-card prod service.
```yaml
# Plan A — isolated single-card smoke test. Bind-mount host kernelgym env + ~/.local + host
# CANN, run as root, point PYTHON_BIN at the kernelgym python. card 0 / port 11908 / redis db8.
#
#   docker compose -f docker-compose.npu0-test.yml up -d
#   python3 scripts/test_eval_service.py --url http://127.0.0.1:11908
#   docker compose -f docker-compose.npu0-test.yml down

services:
  triton-eval-npu0:
    image: triton-evaluation:latest
    command: ["all"]
    network_mode: host
    privileged: true
    shm_size: "32gb"
    environment:
      PYTHON_BIN: /home/n84449292/miniconda3/envs/kernelgym/bin/python3
      PYTHONUSERBASE: /home/n84449292/.local
      HOME: /tmp
      TRITON_EVAL_REDIS_URL: redis://127.0.0.1:6379/8
      TRITON_EVAL_API_HOST: 0.0.0.0
      TRITON_EVAL_API_PORT: "11908"
      TRITON_EVAL_CARDS: "0"
      CANN_SET_ENV_PATH: /home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh
      ASCEND_DRIVER_PATH: /usr/local/Ascend/driver
    devices:
      - /dev/davinci_manager
      - /dev/devmm_svm
      - /dev/hisi_hdc
      - /dev/davinci0
    volumes:
      - /home/n84449292/miniconda3:/home/n84449292/miniconda3:ro
      - /home/n84449292/.local:/home/n84449292/.local:ro
      - /home/canada_group_folder/CANN/9.0.0.0430:/home/canada_group_folder/CANN/9.0.0.0430
      - /usr/local/dcmi:/usr/local/dcmi
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver
      - /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
      - /etc/ascend_install.info:/etc/ascend_install.info
      - ./docker/entrypoint.sh:/usr/local/bin/entrypoint.sh
      - ./logs:/opt/triton-eval/logs
```

---

## 2. Local deploy config (NOT in repo — host-specific; create from the examples)

### `docker.env` (from `docker.env.example`) — used by `docker compose --env-file docker.env`
```
TRITON_EVAL_CARDS=8,9,10,11,12,13,14,15
TRITON_EVAL_REDIS_URL=redis://127.0.0.1:6379/7
TRITON_EVAL_API_PORT=11907
CANN_SET_ENV_PATH=/home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh
BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
TORCH_SPEC=torch==2.9.0
TORCH_NPU_SPEC=torch_npu==2.9.0
TRITON_SPEC=triton==3.5.0
TRITON_ASCEND_SPEC=triton-ascend==3.2.0
PIP_EXTRA_INDEX_URL=
```

### `.env` (from `.env.example`) — only used by `scripts/start.sh` (host-mode fallback)
Add this line (host default python3 is py3.13 with no torch):
```
PYTHON_BIN=/home/n84449292/miniconda3/envs/kernelgym/bin/python3
```
Plus the usual: `TRITON_EVAL_API_PORT=11907`, `TRITON_EVAL_CARDS=8,9,...,15`,
`TRITON_EVAL_REDIS_URL=redis://localhost:6379/7`,
`CANN_SET_ENV_PATH=/home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh`.

> The repo templates `.env.example` and `docker.env.example` now document `PYTHON_BIN` as a
> commented line (with the why), so deployers know to set it.

---

## 3. Per-host system setup (NOT in repo — do once on each host; needs sudo)

### a) Docker daemon proxy (so `docker pull`/`build` reach the registry)
`/etc/systemd/system/docker.service.d/http-proxy.conf` had a dead proxy. Set the working one:
```ini
[Service]
Environment="HTTP_PROXY=http://80.253.19.172:6688"
Environment="HTTPS_PROXY=http://80.253.19.172:6688"
Environment="NO_PROXY=127.0.0.1,localhost,::1"
Environment="http_proxy=http://80.253.19.172:6688"
Environment="https_proxy=http://80.253.19.172:6688"
Environment="no_proxy=127.0.0.1,localhost,::1"
```
Then `sudo systemctl daemon-reload && sudo systemctl restart docker`.
(`~/.docker/config.json` already has this proxy; the daemon override took precedence.)
Backup saved as `http-proxy.conf.bak`. **Restarting docker stops containers with restart=no
(e.g. vllm) — restart them manually after.**

### b) Profiler owner fix (so the speed benchmark works in docker)
The container runs as root but host CANN is owned by uid 1001; `torch_npu.profiler` (msprof)
rejects owner mismatch (`ERR04012`) → `speedup_vs_torch=None`. Chown ONLY the profiler tooling
(rest of CANN + host basic usage unaffected; only host-side profiling would fail):
```bash
CANN=/home/canada_group_folder/CANN/9.0.0.0430/cann-9.0.0
sudo chown -R root:root "$CANN/tools/profiler"
sudo chown root:root "$CANN/bin/msprof"*
# revert: sudo chown -R 1001:1002 "$CANN/tools/profiler" "$CANN/bin/msprof"*
```
Why not run the container as uid 1001 instead? Device access (`rtGetDeviceCount`) fails for
non-root in this manual-mount container with `Resource_Busy (EL0005)` — the driver needs root
in-container (no ascend-docker-runtime). So root + chowned-profiler-tooling is the way.

### c) Pull redis + build the image
```bash
docker pull redis:7-alpine                       # needs (a)
docker compose --env-file docker.env build       # one-time
```

### d) Host prerequisites
- A conda env (here `kernelgym`) with torch 2.9.0 / torch_npu 2.9.0 / triton + ascend backend,
  reachable at `PYTHON_BIN`. `~/.local` must hold torch 2.9.0 / torch_npu (set `PYTHONUSERBASE`).
- Host CANN matching the driver (here canada 9.0.0.0430); set `CANN_SET_ENV_PATH` + the CANN
  volume to its path.

---

## 4. Operate
```
docker compose --env-file docker.env up -d        # start eval + redis
docker compose --env-file docker.env logs -f
docker compose --env-file docker.env down
python3 scripts/test_eval_service.py --url http://127.0.0.1:11907   # smoke (correctness + speed)
```

## 5. Notes / future work
- Container depends on the host `kernelgym` env + `~/.local` (less portable than a
  self-contained image). A self-contained image rebuild is **not done** — blocked on the
  triton-ascend "0 active drivers" issue with the image's triton 3.5.0 (host works on 3.2.0).
- Bake the fixed `docker/entrypoint.sh` into the image (then drop the entrypoint bind-mount).
- `docker-compose.npu0-test.yml` is optional (test harness); safe to drop for a minimal repo.
