#!/usr/bin/env bash
# Compose wrapper: startup actions (up/start/restart) run deployment preflight
# first and refuse to touch Compose when it fails. Read/control actions
# (down/logs/ps/config) skip startup validation by design.
#
# Never uses eval; secrets never print.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT_PYTHON="${TRITON_EVAL_PYTHON:-python3}"

action="${1:-}"
if [[ -z "$action" ]]; then
  echo "usage: $0 {up|start|restart|down|logs|ps|config|build} [compose args...]" >&2
  exit 2
fi
shift || true

case "$action" in
  up|start|restart)
    result="$("$PREFLIGHT_PYTHON" "$REPO_ROOT/scripts/deployment_preflight.py")"
    gateway="$(printf '%s' "$result" | sed -n 's/.*"gateway"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    if [[ -z "$gateway" ]]; then
      echo "deployment preflight returned no gateway; refusing to start" >&2
      exit 1
    fi
    export TRITON_EVAL_API_HOST="$gateway"
    printf '%s\n' "$gateway"
    exec docker compose "$action" "$@"
    ;;
  down|stop|logs|ps|config|build|pull)
    # Read/control commands: no startup validation claimed.
    exec docker compose "$action" "$@"
    ;;
  *)
    echo "unknown compose action: $action" >&2
    exit 2
    ;;
esac
