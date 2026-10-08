#!/usr/bin/env bash
# Initialize the mounted host CANN/driver and run the image's declared Python stack.
#
#   entrypoint.sh all       -> API + CPU workers + one NPU worker per configured card
#   entrypoint.sh api       -> uvicorn scripts.run_api:app only
#   entrypoint.sh workers   -> workers only
#   entrypoint.sh <cmd...>  -> exec cmd (escape hatch, e.g. `bash`)
#
# PYTHON_BIN selects the interpreter shared by the service and engine subprocesses.
set -euo pipefail
: "${PYTHON_BIN:=python3}"

# --- Host driver libs (mounted from host) ---
DRIVER_LIB="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64"
if [[ -d "${DRIVER_LIB}" ]]; then
  export LD_LIBRARY_PATH="${DRIVER_LIB}:${DRIVER_LIB}/driver:${LD_LIBRARY_PATH:-}"
fi

# CANN's set_env.sh preserves existing values, so clear inherited toolkit paths
# before initializing the host installation mounted into this container.
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

# --- Fail closed before API/workers: artifacts, secret/catalog, build equality, bind ---
# The host preflight owns image/digest verification (no docker socket is mounted in the
# container by design); this boundary re-verifies everything reachable in-process.
verify_readiness() {
  "${PYTHON_BIN}" - <<'PYREADINESS'
import sys

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    build_artifacts,
    verify_artifacts,
)
from triton_evaluation.service.catalog import ReleaseCatalog
from triton_evaluation.service.security import CapabilityRegistry

problems = verify_artifacts(CONTRACT_ROOT, build_artifacts())
if problems:
    sys.exit(f"contract artifacts failed verification: {len(problems)} file(s)")
import os

from triton_evaluation.config import Settings

settings = Settings.from_env()
try:
    CapabilityRegistry.from_secret_file(settings.capability_secret_path)
except Exception:
    sys.exit("capability secret failed validation")
try:
    ReleaseCatalog.load(settings.catalog_root, settings.catalog_manifest_path)
except Exception:
    sys.exit("release catalog failed validation")

build = os.environ.get("TRITON_EVAL_EVALUATOR_BUILD", "")
image = os.environ.get("TRITON_EVAL_IMAGE", "")
git_sha = build.removeprefix("git:")
reviewed_git = (
    build.startswith("git:") and len(git_sha) == 40
    and all(char in "0123456789abcdef" for char in git_sha)
)
if build != settings.evaluator_build:
    sys.exit("evaluator build disagrees with the configured setting")
if image and build != image.rsplit("@", 1)[-1] and not reviewed_git:
    sys.exit("evaluator build must equal the immutable image's sha256 digest")

host = os.environ.get("TRITON_EVAL_API_HOST", "127.0.0.1")
allow_wildcard = os.environ.get("TRITON_EVAL_ALLOW_WILDCARD_BIND", "") == "1"
if not allow_wildcard and (host in ("0.0.0.0", "::") or ":" in host):
    sys.exit(
        "API host must be a single IPv4 address, never a wildcard "
        "(0.0.0.0 needs TRITON_EVAL_ALLOW_WILDCARD_BIND=1)"
    )
PYREADINESS
}

if [[ "${TRITON_EVAL_SKIP_READINESS:-}" != "1" ]]; then
  verify_readiness
fi

run_api() {
  exec "${PYTHON_BIN}" -m uvicorn scripts.run_api:app \
    --host "${TRITON_EVAL_API_HOST:-127.0.0.1}" \
    --port "${TRITON_EVAL_API_PORT:-11907}"
}

run_workers() {
  local pids=("$@")   # any pre-existing pids to also wait on
  local cpu_count="${TRITON_EVAL_CPU_WORKERS:-1}"
  if [[ ! "${cpu_count}" =~ ^([0-9]|[12][0-9]|3[0-2])$ ]]; then
    echo "TRITON_EVAL_CPU_WORKERS must be an integer from 0 to 32" >&2
    exit 2
  fi
  for ((i=0; i<cpu_count; i++)); do
    echo "Starting CPU worker ${i}" >&2
    "${PYTHON_BIN}" -m scripts.run_worker --cpu &
    pids+=("$!")
  done
  IFS=',' read -ra cards <<< "${TRITON_EVAL_CARDS:-0}"
  for c in "${cards[@]}"; do
    c="${c//[[:space:]]/}"
    [[ -z "${c}" ]] && continue
    echo "Starting worker on NPU card ${c}" >&2
    "${PYTHON_BIN}" -m scripts.run_worker --card "${c}" &
    pids+=("$!")
  done
  [[ ${#pids[@]} -eq 0 ]] && { echo "No workers configured" >&2; exit 2; }
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
    "${PYTHON_BIN}" -m uvicorn scripts.run_api:app \
      --host "${TRITON_EVAL_API_HOST:-127.0.0.1}" \
      --port "${TRITON_EVAL_API_PORT:-11907}" &
    api_pid=$!
    run_workers "${api_pid}"
    ;;
  *)
    exec "$@"
    ;;
esac
