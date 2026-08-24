# triton-evaluation service image — host-coupled NPU deployment.
#
# First-principles layout: the image owns exactly what the service is.
#   * the service source (/opt/triton-eval) + entrypoint
#   * a snapshot of the validated NPU Python stack (torch 2.9.0 / torch_npu 2.9.0 /
#     triton 3.5.0 / triton-ascend 3.2.1) staged from the host conda env this
#     deployment was validated against — no pip downloads at build time, no
#     version drift between build and validation, no 18GB inference-stack base.
# Host-coupled pieces are bind-mounted at runtime BY DESIGN, because they must
# match the host's NPU driver: the CANN toolkit, driver libs, /dev/davinci*,
# npu-smi. No image can be self-contained there.
#
# Build context is prepared by scripts/build_local_image.sh (stages env/ and
# app/ from this repo + the host env), then:
#   bash scripts/build_local_image.sh [tag]     # default tag: 1.0
# BASE_IMAGE: any local aarch64 glibc base (>= 2.17) with bash; overridden when
# this host's default base is absent elsewhere.
ARG BASE_IMAGE=openeuler22.03_arm:autotest
FROM ${BASE_IMAGE}

# The validated NPU Python stack, relocated to /opt/env/verl-base.
# (bin/ console scripts keep host shebangs and are unused; invoke python3 -m.)
COPY env/verl-base /opt/env/verl-base

# The service itself.
COPY app /opt/triton-eval

# Import the service package from source (replaces the host's editable install)
# and strip the host-path editable-install artifacts from the staged env.
RUN echo "/opt/triton-eval" > /opt/env/verl-base/lib/python3.11/site-packages/triton-eval.pth \
 && rm -f /opt/env/verl-base/lib/python3.11/site-packages/__editable__* \
          /opt/env/verl-base/lib/python3.11/site-packages/easy-install.pth \
          /opt/env/verl-base/lib/python3.11/site-packages/triton_evaluation.egg-link \
 ; true

ENV PATH=/opt/env/verl-base/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHON_BIN=/opt/env/verl-base/bin/python3 \
    HOME=/root

# driver libs + host CANN wiring + API/workers; see docker/entrypoint.sh
ENTRYPOINT ["bash", "/opt/triton-eval/docker/entrypoint.sh"]
CMD ["all"]
