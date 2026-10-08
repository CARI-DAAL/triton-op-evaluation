#!/usr/bin/env bash
set -euo pipefail
deploy=/home/e84381970/triton-eval-machine2
repo=/home/e84381970/triton-op-evaluation
case "${1:-status}" in
  status)
    python3 - <<'PY'
import json, urllib.request
from pathlib import Path
p = Path('/home/e84381970/triton-eval-machine2')
url = json.loads((p / 'client.json').read_text())['url']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url + '/readyz', timeout=10) as response:
    print(response.status, response.read().decode())
PY
    ;;
  start)
    if [[ -e "${deploy}/logs/triton-eval.pids" ]]; then
      echo 'Service PID file exists; inspect status or stop this deployment first.' >&2
      exit 1
    fi
    docker start triton-eval-redis-machine2 triton-agent-cpu >/dev/null
    docker exec -d -e TRITON_EVAL_ENV_FILE="${deploy}/service.env" -w "${repo}" cc \
      bash -c 'exec bash scripts/start.sh > /home/e84381970/triton-eval-machine2/launcher.log 2>&1'
    echo "Starting; inspect ${deploy}/launcher.log and run $0 status."
    ;;
  stop)
    if [[ ! -s "${deploy}/logs/triton-eval.pids" ]]; then
      echo 'No deployment PID file; refusing the upstream broad process-name fallback.' >&2
      exit 1
    fi
    docker exec -e TRITON_EVAL_ENV_FILE="${deploy}/service.env" -w "${repo}" cc bash scripts/stop.sh
    ;;
  *) echo "Usage: $0 {status|start|stop}" >&2; exit 2 ;;
esac
