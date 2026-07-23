# triton-evaluation — runtime image for Ascend NPU hosts.
#
# No Ascend base image is used: CANN toolkit + driver are MOUNTED from the host
# at runtime (see docker-compose.yml). This image only bakes Python, the service,
# and the ML stack (torch / torch_npu / triton-ascend / pandas) whose versions
# MUST match the host CANN — pass them as build args (see docker.env.example).
#
# Target host: aarch64 (Kunpeng). Build ON the NPU host (or an aarch64 builder).
FROM ubuntu:22.04

ARG TORCH_SPEC="torch==2.1.0"
ARG TORCH_NPU_SPEC="torch_npu==2.1.0"
ARG TRITON_ASCEND_SPEC="triton-ascend"
ARG PANDAS_SPEC="pandas"
# Optional extra pip index (e.g. Ascend / torch_npu index) — leave empty to skip.
ARG PIP_EXTRA_INDEX_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        gcc g++ libnuma1 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Service + its light deps (cached layer — changes rarely).
COPY pyproject.toml README.md ./
COPY triton_evaluation ./triton_evaluation
RUN pip3 install -e . && pip3 install "${PANDAS_SPEC}"

# 2) ML stack matching the host CANN (versions via build args).
RUN pip3 install \
        ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
        "${TORCH_SPEC}" "${TORCH_NPU_SPEC}" "${TRITON_ASCEND_SPEC}"

# 3) Entrypoints.
COPY run_api.py run_worker.py ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
