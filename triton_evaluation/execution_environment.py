"""Worker-side execution environment for provenance (design section 4.5).

The environment describes the *actual worker*, never the API process: the
hardware/driver/runtime/toolchain fields are deployment-attested ``Settings``
values, ``hardware_count`` is fixed to 1 because each worker process exposes
exactly one card, and the torch/torch-npu/triton versions are read with
``importlib.metadata`` inside the worker process. It never includes hostname,
card serial numbers, source code, or request inputs.
"""
import importlib.metadata

from triton_evaluation.api.contract import ExecutionEnvironment
from triton_evaluation.config import Settings


def _optional_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_execution_environment(settings: Settings) -> ExecutionEnvironment:
    """Assemble the worker environment from attested settings plus local packages."""
    return ExecutionEnvironment(
        hardware_model=settings.worker_hardware_model,
        hardware_count=1,
        driver_version=settings.worker_driver_version,
        runtime_version=settings.worker_runtime_version,
        torch_version=_optional_version("torch"),
        torch_npu_version=_optional_version("torch-npu"),
        triton_version=_optional_version("triton"),
        toolchain_version=settings.worker_toolchain_version,
    )
