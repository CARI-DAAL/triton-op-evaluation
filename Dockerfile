# triton-evaluation on Ascend — declared-dependency build.
#
# The Python stack is fully declared: `pip install ".[npu]"` installs the
# torch / torch_npu / triton / triton-ascend versions pinned in
# pyproject.toml [project.optional-dependencies].npu — the SAME declaration the
# non-Docker path (scripts/start.sh) uses. Network access is required at build
# time and is the operator's concern; torch_npu / triton-ascend are not on
# PyPI, so point PIP_EXTRA_INDEX_URL at the Ascend wheel index when building:
#
#   docker build -t triton-eval:1.0 .
#   docker build --build-arg PIP_EXTRA_INDEX_URL=https://<ascend-index>/simple \
#                -t triton-eval:1.0 .
#
# Host-coupled pieces (CANN toolkit, NPU driver, /dev/davinci*) are NOT in the
# image on purpose: they must match the host's driver and are bind-mounted at
# runtime — see scripts/docker_run.sh. Run with `bash scripts/docker_run.sh`
# or docker compose; both mount the same host paths.
#
# BASE_IMAGE: any aarch64 glibc (>= 2.17) base with bash + apt. The default
# carries a CANN-matching userspace; override with --build-arg if you prefer
# a smaller base.
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
FROM ${BASE_IMAGE}

ARG PIP_EXTRA_INDEX_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 git ca-certificates && rm -rf /var/lib/apt/lists/*

# py3.11 conda env — the validated triton-ascend stack is py3.11.
# NOTE: recent Miniconda gates `conda create` on accepting the Anaconda channel
# Terms of Service; accept them non-interactively first or the build aborts with
# CondaToSNonInteractiveError.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /m.sh \
 && bash /m.sh -bfp /opt/conda && rm /m.sh \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && /opt/conda/bin/conda create -y -n triton-eval python=3.11 \
 && /opt/conda clean -afy
# Put the env first so python3 / pip / uvicorn — and the engine's sys.executable — are py3.11.
ENV PATH=/opt/conda/envs/triton-eval/bin:$PATH

# The service: light deps + the declared NPU stack, all from pyproject.
WORKDIR /opt/triton-eval
COPY pyproject.toml README.md ./
COPY triton_evaluation ./triton_evaluation
COPY contracts ./contracts
RUN pip install --retries 10 --timeout 120 \
      ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
      ".[npu]"

COPY run_api.py run_worker.py ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]
