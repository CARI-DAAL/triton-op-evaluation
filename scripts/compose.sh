#!/usr/bin/env bash
# Compose wrapper: startup actions (up/start/restart) run deployment preflight
# first and refuse to touch Compose when it fails. Read/control actions
# (down/logs/ps/config) skip startup validation by design.
#
# Never uses eval; secrets never print.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT_PYTHON="${TRITON_EVAL_PYTHON:-python3}"

compose_file="${TRITON_EVAL_COMPOSE_FILE:-$REPO_ROOT/docker-compose.yml}"
env_file="${TRITON_EVAL_ENV_FILE:-/dev/null}"
# Preflight and the final Compose command must render the same files even when
# this wrapper is launched outside the repository. Deployment variables must
# already be exported; the wrapper never executes the env file as shell code.
export TRITON_EVAL_COMPOSE_FILE="$compose_file"
export TRITON_EVAL_ENV_FILE="$env_file"

action="${1:-}"
if [[ -z "$action" ]]; then
  echo "usage: $0 {up|start|restart|down|logs|ps|config|build} [compose args...]" >&2
  exit 2
fi
shift || true

case "$action" in
  up|start|restart)
    result="$("$PREFLIGHT_PYTHON" "$REPO_ROOT/scripts/deployment_preflight.py")"
    api_host="$(printf '%s' "$result" | sed -n 's/.*"api_host"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    if [[ -z "$api_host" ]]; then
      echo "deployment preflight returned no api_host; refusing to start" >&2
      exit 1
    fi
    export TRITON_EVAL_API_HOST="$api_host"
    printf '%s\n' "$api_host"
    exec docker compose --env-file "$env_file" -f "$compose_file" "$action" "$@"
    ;;
  down|stop|logs|ps|config|build|pull)
    # Read/control commands: no startup validation claimed.
    exec docker compose --env-file "$env_file" -f "$compose_file" "$action" "$@"
    ;;
  *)
    echo "unknown compose action: $action" >&2
    exit 2
    ;;
esac
