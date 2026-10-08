#!/usr/bin/env bash
# Stop the triton-evaluation service started by scripts/start.sh.
# Reads the PID file written by start.sh; falls back to matching process names.
#
# Usage:
#   bash scripts/stop.sh
# Override the env file with TRITON_EVAL_ENV_FILE=/path/to/env bash scripts/stop.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${TRITON_EVAL_ENV_FILE:-${ROOT_DIR}/.env}"

# Load .env best-effort, only to resolve LOG_DIR (where start.sh wrote the PID file)
if [[ -r "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
PID_FILE="${LOG_DIR}/triton-eval.pids"

# TERM the given pids, wait up to ~5s, then KILL any stragglers.
term_wait_kill() {
  local pids=("$@")
  [[ ${#pids[@]} -eq 0 ]] && return 0
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    local alive=()
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi
    done
    [[ ${#alive[@]} -eq 0 ]] && return 0
    sleep 0.5
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
}

if [[ -f "${PID_FILE}" ]]; then
  mapfile -t recorded < "${PID_FILE}"
  live=()
  for p in "${recorded[@]}"; do
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then live+=("$p"); fi
  done
  if [[ ${#live[@]} -gt 0 ]]; then
    echo "Stopping ${#live[@]} process(es) from ${PID_FILE}: ${live[*]}" >&2
    term_wait_kill "${live[@]}"
  else
    echo "No live processes recorded in ${PID_FILE}." >&2
  fi
  rm -f "${PID_FILE}"
else
  echo "No PID file at ${PID_FILE}; matching process names instead." >&2
  pkill -TERM -f 'uvicorn scripts[.]run_api:app' 2>/dev/null || true
  pkill -TERM -f 'scripts[.]run_worker (--cpu|--card)([[:space:]]|$)' 2>/dev/null || true
fi

echo "Stopped." >&2
