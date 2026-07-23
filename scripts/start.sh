#!/usr/bin/env bash
# Start the triton-evaluation service: one API + one worker per NPU card.
# Style mirrors kernelgym-npu/scripts/start_service.sh (require .env, preflight, then launch).
#
# Usage:
#   cp .env.example .env   # then edit .env
#   bash scripts/start.sh
#
# Override the env file with TRITON_EVAL_ENV_FILE=/path/to/env bash scripts/start.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${TRITON_EVAL_ENV_FILE:-${ROOT_DIR}/.env}"

if [[ ! -r "${ENV_FILE}" ]]; then
  echo "Missing readable environment file: ${ENV_FILE}" >&2
  echo "Create it from ${ROOT_DIR}/.env.example and set site-specific values." >&2
  exit 2
fi

# Load .env (export every KEY=VALUE; comments/blank lines are ignored by the shell)
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# Optionally source the CANN toolkit env so torch_npu and the skill scripts resolve
if [[ -n "${CANN_SET_ENV_PATH:-}" && -r "${CANN_SET_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV_PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
API_HOST="${TRITON_EVAL_API_HOST:-0.0.0.0}"
API_PORT="${TRITON_EVAL_API_PORT:-8000}"
CARDS="${TRITON_EVAL_CARDS:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

# Preflight: the skill scripts dir must exist (workers subprocess-call it)
if [[ -z "${TRITON_EVAL_SKILL_SCRIPTS_DIR:-}" || ! -d "${TRITON_EVAL_SKILL_SCRIPTS_DIR}" ]]; then
  echo "TRITON_EVAL_SKILL_SCRIPTS_DIR is unset or not a directory: '${TRITON_EVAL_SKILL_SCRIPTS_DIR:-}'" >&2
  echo "Point it at cannbot-skills/ops/triton-op-verifier/scripts" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

pids=()
cleanup() {
  echo "Shutting down..." >&2
  kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM

# API — uvicorn serves run_api.py's module-level `app`, so host/port come from env
echo "Starting API on ${API_HOST}:${API_PORT} (log: ${LOG_DIR}/api.log)" >&2
"${PYTHON_BIN}" -m uvicorn run_api:app --host "${API_HOST}" --port "${API_PORT}" \
  >"${LOG_DIR}/api.log" 2>&1 &
pids+=("$!")

# One worker per card; run_worker.py pins ASCEND_RT_VISIBLE_DEVICES to --card
worker_count=0
IFS=',' read -ra CARD_ARR <<< "${CARDS}"
for card in "${CARD_ARR[@]}"; do
  card="${card//[[:space:]]/}"
  [[ -z "${card}" ]] && continue
  echo "Starting worker on NPU card ${card} (log: ${LOG_DIR}/worker_${card}.log)" >&2
  "${PYTHON_BIN}" run_worker.py --card "${card}" \
    >"${LOG_DIR}/worker_${card}.log" 2>&1 &
  pids+=("$!")
  worker_count=$((worker_count + 1))
done

echo "Started API + ${worker_count} worker(s). Logs in ${LOG_DIR}/. Ctrl-C to stop." >&2
wait
