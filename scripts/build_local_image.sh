#!/usr/bin/env bash
# Build the triton-eval image from the validated host stack — no network.
#
# Stages a docker build context:
#   env/verl-base  snapshot of the host conda env carrying the validated NPU
#                  stack (torch/torch_npu/triton/triton-ascend), with the
#                  service's own deps merged in from the repo's .venv
#   app/           the service source (triton_evaluation/, entrypoints, entrypoint)
# then runs: docker build -t triton-eval:<tag> .
#
# Usage: bash scripts/build_local_image.sh [tag]        (default tag: 1.0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-1.0}"
HOST_ENV="${HOST_ENV:-/home/canada_group_folder/miniconda3/envs/verl-base}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

for d in "${HOST_ENV}" "${VENV}"; do
  [[ -d "$d" ]] || { echo "missing: $d" >&2; exit 2; }
done

BUILD_ROOT="${BUILD_ROOT:-/home/n84449292/eason}"
CTX="$(mktemp -d "${BUILD_ROOT%/}/triton-eval-build.XXXXXX")"
trap 'rm -rf "${CTX}"' EXIT
echo "staging build context: ${CTX}"

# 1. The NPU python stack snapshot.
mkdir -p "${CTX}/env"
cp -a "${HOST_ENV}" "${CTX}/env/verl-base"

# 2. Merge the service deps (fastapi/uvicorn/redis/pydantic-settings/rfc8785/
#    setproctitle) from the repo venv into the snapshot; the venv itself is not
#    relocatable (its python symlinks back to the host env path).
VENV_SITE="${VENV}/lib/python3.11/site-packages"
ENV_SITE="${CTX}/env/verl-base/lib/python3.11/site-packages"
rsync -a --exclude '__editable__*' --exclude 'easy-install.pth' \
      --exclude 'triton_evaluation.egg-link' --exclude '__pycache__' \
      "${VENV_SITE}/" "${ENV_SITE}/"

# 3. The service source.
mkdir -p "${CTX}/app"
cp -a "${REPO_ROOT}/triton_evaluation" "${CTX}/app/"
cp "${REPO_ROOT}/run_api.py" "${REPO_ROOT}/run_worker.py" \
   "${REPO_ROOT}/pyproject.toml" "${CTX}/app/"
# CONTRACT_ROOT resolves relative to the entrypoint cwd (/opt/triton-eval).
cp -a "${REPO_ROOT}/contracts" "${CTX}/app/"
mkdir -p "${CTX}/app/docker"
cp "${REPO_ROOT}/docker/entrypoint.sh" "${CTX}/app/docker/"

cp "${REPO_ROOT}/Dockerfile" "${CTX}/Dockerfile"

echo "building triton-eval:${TAG} (context $(du -sh "${CTX}" | cut -f1))..."
docker build -t "triton-eval:${TAG}" "${CTX}"
echo "done: triton-eval:${TAG}"
