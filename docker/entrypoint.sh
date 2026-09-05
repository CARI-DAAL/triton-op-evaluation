#!/usr/bin/env bash
# Container entrypoint: wire up the mounted host driver + baked CANN, then run the service.
#
#   entrypoint.sh all       -> API + one worker per card in TRITON_EVAL_CARDS (default)
#   entrypoint.sh api       -> uvicorn run_api:app only
#   entrypoint.sh workers   -> workers only
#   entrypoint.sh <cmd...>  -> exec cmd (escape hatch, e.g. `bash`)
#
# Python interpreter: defaults to the image's python3 (py3.11, on PATH from the
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

# --- Fail closed before API/workers: artifacts, secret/catalog, build equality, bind ---
# The host preflight owns image/digest verification (no docker socket is mounted in the
# container by design); this boundary re-verifies everything reachable in-process.
verify_readiness() {
  "${PYTHON_BIN}" - <<'PYREADINESS'
import sys

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    CONTRACT_ROOT_V21,
    build_v20_artifacts,
    build_v21_artifacts,
    verify_artifacts,
)
from triton_evaluation.catalog import ReleaseCatalog
from triton_evaluation.security import CapabilityRegistry

problems = verify_artifacts(CONTRACT_ROOT, build_v20_artifacts())
problems += verify_artifacts(CONTRACT_ROOT_V21, build_v21_artifacts())
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
if image and build != image and not reviewed_git:
    sys.exit("evaluator build must equal the immutable image")

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
  exec "${PYTHON_BIN}" -m uvicorn run_api:app \
    --host "${TRITON_EVAL_API_HOST:-127.0.0.1}" \
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
      --host "${TRITON_EVAL_API_HOST:-127.0.0.1}" \
      --port "${TRITON_EVAL_API_PORT:-10907}" &
    api_pid=$!
    run_workers "${api_pid}"
    ;;
  *)
    exec "$@"
    ;;
esac
