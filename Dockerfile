# syntax=docker/dockerfile:1
# triton-evaluation on Ascend — declared-dependency build.
#
# The Python stack is fully declared: `pip install ".[npu]"` installs the
# versions pinned in pyproject.toml [project.optional-dependencies].npu — the
# SAME declaration the host path (scripts/start.sh) uses. Build:
#
#   docker build -t triton-eval:1.0 .
#   # torch_npu / triton-ascend not on PyPI? add
#   #   --build-arg PIP_EXTRA_INDEX_URL=https://<ascend-index>/simple
#
# Host-coupled pieces (CANN toolkit, NPU driver, /dev/davinci*) are NOT in the
# image: they must match the host driver and are bind-mounted at runtime —
# see scripts/docker_run.sh.
#
# BASE_IMAGE: any aarch64 base with bash + python3.11/pip. The official python
# image is all this build needs; override only if your site mandates a
# specific userspace.
ARG BASE_IMAGE=python:3.11
FROM ${BASE_IMAGE}

ARG PIP_EXTRA_INDEX_URL=""

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/triton-eval

# Heavy pinned deps first, in their own cacheable layer: a stub package
# satisfies the ".[npu]" install so source edits never re-download torch.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p triton_evaluation && touch triton_evaluation/__init__.py \
 && pip install --retries 10 --timeout 120 \
      ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
      ".[npu]"

# Real source + the service itself (deps already in place).
COPY triton_evaluation ./triton_evaluation
COPY contracts ./contracts
COPY run_api.py run_worker.py ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]
