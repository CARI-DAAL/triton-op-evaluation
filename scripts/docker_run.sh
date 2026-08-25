#!/usr/bin/env bash
# Run the triton-eval service container. Build the image first:
#   docker build -t triton-eval:1.0 .      (see README "Run in Docker")
#
# The image carries the declared Python stack; only the host-coupled pieces are
# mounted: CANN toolkit, NPU driver, devices, and the caller secret + release
# catalog (kept out of image layers on purpose). --network host: the API binds
# 0.0.0.0:<port> (allowed by TRITON_EVAL_ALLOW_WILDCARD_BIND) and redis stays
# reachable on 127.0.0.1.
#
# Usage: bash scripts/docker_run.sh [tag]     (default tag: 1.0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${TRITON_EVAL_ENV_FILE:-${REPO_ROOT}/.env}"

# Load site-specific configuration if present (same idiom as scripts/start.sh).
if [[ -r "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi

TAG="${1:-1.0}"
CANN_HOST="${CANN_HOST:?set CANN_HOST in .env — path to the host CANN toolkit (must match the host driver)}"

# One --device per card in TRITON_EVAL_CARDS.
devs=()
IFS=',' read -ra cards <<< "${TRITON_EVAL_CARDS:-0}"
for c in "${cards[@]}"; do
  c="${c//[[:space:]]/}"
  [[ -n "${c}" ]] && devs+=(--device "/dev/davinci${c}")
done

docker rm -f triton-eval 2>/dev/null || true
docker run -d --name triton-eval \
  --network host --privileged --shm-size 32g --restart unless-stopped \
  -e TRITON_EVAL_API_HOST="${TRITON_EVAL_API_HOST:-0.0.0.0}" \
  -e TRITON_EVAL_API_PORT="${TRITON_EVAL_API_PORT:-10907}" \
  -e TRITON_EVAL_REDIS_URL="${TRITON_EVAL_REDIS_URL:-redis://127.0.0.1:6379/0}" \
  -e TRITON_EVAL_CARDS="${TRITON_EVAL_CARDS:-0}" \
  -e TRITON_EVAL_DEVICE_HOLD_MB="${TRITON_EVAL_DEVICE_HOLD_MB:-64}" \
  -e TRITON_EVAL_ALLOW_WILDCARD_BIND=1 \
  -e TRITON_EVAL_CAPABILITY_SECRET_PATH=/opt/triton-eval/.capability-secret.json \
  -e TRITON_EVAL_CATALOG_ROOT=/opt/triton-eval/catalog \
  -e TRITON_EVAL_CATALOG_MANIFEST_PATH=/opt/triton-eval/catalog/catalog.json \
  -e CANN_SET_ENV_PATH="${CANN_HOST}/ascend-toolkit/set_env.sh" \
  -v "${CANN_HOST}:${CANN_HOST}:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "${REPO_ROOT}/.capability-secret.json:/opt/triton-eval/.capability-secret.json:ro" \
  -v "${REPO_ROOT}/catalog:/opt/triton-eval/catalog:ro" \
  -v triton-eval-triton-cache:/root/.triton \
  "${devs[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  "triton-eval:${TAG}"
echo "triton-eval up (triton-eval:${TAG}); logs: docker logs -f triton-eval"
