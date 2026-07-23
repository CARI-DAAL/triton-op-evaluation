#!/usr/bin/env bash
# Container entrypoint: wire up the mounted CANN/driver, then run the API or workers.
#
#   entrypoint.sh api       -> uvicorn run_api:app
#   entrypoint.sh workers   -> one run_worker.py --card N per card in TRITON_EVAL_CARDS
#   entrypoint.sh <cmd...>   -> exec cmd (escape hatch, e.g. `bash`)
set -euo pipefail

# --- Ascend driver libs (mounted from host) ---
DRIVER_LIB="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64"
if [[ -d "${DRIVER_LIB}" ]]; then
  export LD_LIBRARY_PATH="${DRIVER_LIB}:${DRIVER_LIB}/driver:${LD_LIBRARY_PATH:-}"
fi

# --- CANN toolkit env (mounted from host) ---
if [[ -n "${CANN_SET_ENV_PATH:-}" && -r "${CANN_SET_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV_PATH}"
else
  echo "WARN: CANN_SET_ENV_PATH ('${CANN_SET_ENV_PATH:-unset}') not readable; " \
       "torch_npu / the engine will likely fail to import." >&2
fi

cd /app

case "${1:-api}" in
  api)
    exec python3 -m uvicorn run_api:app \
      --host "${TRITON_EVAL_API_HOST:-0.0.0.0}" \
      --port "${TRITON_EVAL_API_PORT:-8000}"
    ;;
  workers)
    pids=()
    # shellcheck disable=SC2317
    stop() { kill "${pids[@]}" 2>/dev/null || true; }
    trap stop TERM INT
    IFS=',' read -ra cards <<< "${TRITON_EVAL_CARDS:-0}"
    for c in "${cards[@]}"; do
      c="${c//[[:space:]]/}"
      [[ -z "${c}" ]] && continue
      echo "Starting worker on NPU card ${c}" >&2
      python3 run_worker.py --card "${c}" &
      pids+=("$!")
    done
    [[ ${#pids[@]} -eq 0 ]] && { echo "No cards in TRITON_EVAL_CARDS" >&2; exit 2; }
    wait
    ;;
  *)
    exec "$@"
    ;;
esac
