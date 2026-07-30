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
ARG TRITON_ASCEND_SPEC="triton-ascend==3.2.0"
ARG PANDAS_SPEC="pandas"
# Extra pip index if torch_npu / triton-ascend are not on PyPI (e.g. the Ascend index).
ARG PIP_EXTRA_INDEX_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 git ca-certificates && rm -rf /var/lib/apt/lists/*

# py3.11 conda env mirroring the working kernelgym/triton-ascend env.
# NOTE: recent Miniconda gates `conda create` on accepting the Anaconda channel
# Terms of Service; accept them non-interactively first or the build aborts with
# CondaToSNonInteractiveError.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /m.sh \
 && bash /m.sh -bfp /opt/conda && rm /m.sh \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && /opt/conda/bin/conda create -y -n triton-eval python=3.11 \
 && /opt/conda/bin/conda clean -afy
# Put the env first so python3 / pip / uvicorn — and the engine's sys.executable — are py3.11.
ENV PATH=/opt/conda/envs/triton-eval/bin:$PATH

# Pre-downloaded large wheels (scripts/dl_wheels.sh fetches them on the host with
# resumable curl). The build proxy drops big downloads mid-stream and pip can't resume
# (IncompleteRead), so we install the big wheels from local via --find-links. Small
# transitive deps still come from PyPI (they fit under the proxy's per-connection cutoff).
COPY wheels /opt/wheels
# Two-step install so the large wheels come ONLY from local /opt/wheels. In a single step
# pip silently ignores --find-links and re-downloads them over the proxy (which drops big
# downloads mid-stream; pip can't resume). Here:
#   1) --no-index --no-deps: install exactly the big local wheels, no transitive deps.
#   2) normal install: the big wheels are already satisfied, so pip only pulls the small
#      transitive deps from PyPI (they fit under the proxy's per-connection cutoff).
# pyyaml: torch_npu imports `yaml` (npu/_memory_viz.py) but doesn't declare it as a dep.
RUN pip install --no-index --find-links=/opt/wheels --no-deps \
      "${TORCH_SPEC}" "${TORCH_NPU_SPEC}" "${TRITON_SPEC}" "${TRITON_ASCEND_SPEC}" "${PANDAS_SPEC}" pyyaml \
 && pip install --retries 10 --timeout 120 \
      ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
      "${TORCH_SPEC}" "${TORCH_NPU_SPEC}" "${TRITON_SPEC}" "${TRITON_ASCEND_SPEC}" "${PANDAS_SPEC}" pyyaml

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
