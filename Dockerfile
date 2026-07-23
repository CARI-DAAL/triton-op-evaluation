# triton-evaluation on Ascend — reuses the proven vllm-ascend manual device-mount scheme.
#
# The base image carries CANN 9.0.0 + torch_npu (arm64), matching the host driver (25.5.0),
# so we mount only the host DRIVER + NPU devices at runtime (not CANN). The base is py3.12,
# but the working triton-ascend stack is py3.11 — so we mirror it in a py3.11 conda env.
#
# Build ON the aarch64 NPU host. Versions default to the validated baseline; override via args.
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
FROM ${BASE_IMAGE}

ARG TORCH_SPEC="torch==2.9.0"
ARG TORCH_NPU_SPEC="torch_npu==2.9.0"
ARG TRITON_SPEC="triton==3.5.0"
ARG TRITON_ASCEND_SPEC="triton-ascend==3.2.1"
ARG PANDAS_SPEC="pandas"
# Extra pip index if torch_npu / triton-ascend are not on PyPI (e.g. the Ascend index).
ARG PIP_EXTRA_INDEX_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 git ca-certificates && rm -rf /var/lib/apt/lists/*

# py3.11 conda env mirroring the working kernelgym/triton-ascend env
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /m.sh \
 && bash /m.sh -bfp /opt/conda && rm /m.sh \
 && /opt/conda/bin/conda create -y -n triton-eval python=3.11 \
 && /opt/conda/bin/conda clean -afy
# Put the env first so python3 / pip / uvicorn — and the engine's sys.executable — are py3.11.
ENV PATH=/opt/conda/envs/triton-eval/bin:$PATH

# ML stack (validated host baseline; overridable via build args).
RUN pip install ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
      "${TORCH_SPEC}" "${TORCH_NPU_SPEC}" "${TRITON_SPEC}" "${TRITON_ASCEND_SPEC}" "${PANDAS_SPEC}"

# The service (light deps: fastapi / uvicorn / redis / pydantic / pydantic-settings).
WORKDIR /opt/triton-eval
COPY pyproject.toml README.md ./
COPY triton_evaluation ./triton_evaluation
RUN pip install -e .
COPY run_api.py run_worker.py ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]
