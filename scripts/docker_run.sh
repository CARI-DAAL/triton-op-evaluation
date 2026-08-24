#!/usr/bin/env bash
# Run the triton-eval service container from the locally built image
# (bash scripts/build_local_image.sh must have produced triton-eval:<tag>).
#
# The image owns the service source + the validated NPU python stack; only the
# host-coupled pieces are mounted: CANN toolkit, NPU driver, devices, and the
# caller secret + release catalog (kept out of image layers on purpose).
# --network host: API binds 0.0.0.0:<port>, redis reachable on 127.0.0.1.
#
# Usage: bash scripts/docker_run.sh [tag]     (default tag: 1.0)
set -euo pipefail

TAG="${1:-1.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_HOST="${CANN_HOST:-/home/canada_group_folder/CANN/9.0.0.0430}"
CARDS="${TRITON_EVAL_CARDS:-4,5,6,7}"
PORT="${TRITON_EVAL_API_PORT:-10907}"

DEVICES=""
IFS=',' read -ra CARR <<< "${CARDS}"
for c in "${CARR[@]}"; do c="${c//[[:space:]]/}"; [[ -z "$c" ]] && continue
  DEVICES="${DEVICES} --device /dev/davinci${c}"
done

docker rm -f triton-eval 2>/dev/null || true
eval docker run -d --name triton-eval \
  --network host --privileged --shm-size 32g --restart unless-stopped \
  -e TRITON_EVAL_API_HOST=0.0.0.0 \
  -e TRITON_EVAL_API_PORT="${PORT}" \
  -e TRITON_EVAL_REDIS_URL=redis://127.0.0.1:6379/0 \
  -e TRITON_EVAL_CARDS="${CARDS}" \
  -e TRITON_EVAL_DEVICE_HOLD_MB=64 \
  -e TRITON_EVAL_CAPABILITY_SECRET_PATH=/opt/triton-eval/.capability-secret.json \
  -e TRITON_EVAL_CATALOG_ROOT=/opt/triton-eval/catalog \
  -e TRITON_EVAL_CATALOG_MANIFEST_PATH=/opt/triton-eval/catalog/catalog.json \
  -e TRITON_EVAL_SKIP_READINESS=1 \
  -e CANN_SET_ENV_PATH="${CANN_HOST}/ascend-toolkit/set_env.sh" \
  -v "${CANN_HOST}:${CANN_HOST}:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "${REPO_ROOT}/.capability-secret.json:/opt/triton-eval/.capability-secret.json:ro" \
  -v "${REPO_ROOT}/catalog:/opt/triton-eval/catalog:ro" \
  -v triton-eval-triton-cache:/root/.triton \
  "${DEVICES[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  "triton-eval:${TAG}"
echo "triton-eval running from triton-eval:${TAG}; logs: docker logs -f triton-eval"
