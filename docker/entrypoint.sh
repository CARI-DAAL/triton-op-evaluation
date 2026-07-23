#!/usr/bin/env bash
# Container entrypoint: wire up the mounted host driver + baked CANN, then run the service.
#
#   entrypoint.sh all       -> API + one worker per card in TRITON_EVAL_CARDS (default)
#   entrypoint.sh api       -> uvicorn run_api:app only
#   entrypoint.sh workers   -> workers only
#   entrypoint.sh <cmd...>  -> exec cmd (escape hatch, e.g. `bash`)
#
# python3 / uvicorn resolve to the py3.11 conda env (on PATH from the Dockerfile),
# so the engine subprocess (sys.executable) also runs under py3.11 + torch_npu.
set -euo pipefail

# --- Host driver libs (mounted from host) ---
DRIVER_LIB="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64"
if [[ -d "${DRIVER_LIB}" ]]; then
  export LD_LIBRARY_PATH="${DRIVER_LIB}:${DRIVER_LIB}/driver:${LD_LIBRARY_PATH:-}"
fi

# --- CANN toolkit env (baked into the base image) ---
if [[ -n "${CANN_SET_ENV_PATH:-}" && -r "${CANN_SET_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV_PATH}"
else
  echo "WARN: CANN_SET_ENV_PATH ('${CANN_SET_ENV_PATH:-unset}') not readable; " \
       "torch_npu / the engine will likely fail to import." >&2
fi

cd /opt/triton-eval

run_api() {
  exec python3 -m uvicorn run_api:app \
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
    python3 run_worker.py --card "${c}" &
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
    python3 -m uvicorn run_api:app \
      --host "${TRITON_EVAL_API_HOST:-0.0.0.0}" \
      --port "${TRITON_EVAL_API_PORT:-10907}" &
    api_pid=$!
    run_workers "${api_pid}"
    ;;
  *)
    exec "$@"
    ;;
esac
