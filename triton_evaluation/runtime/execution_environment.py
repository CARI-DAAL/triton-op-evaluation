"""Worker-side execution environment for provenance (design section 4.5).

The environment describes the *actual worker*, never the API process: the
hardware/driver/runtime/toolchain fields are deployment-attested ``Settings``
values, ``hardware_count`` is fixed to 1 because each worker process exposes
exactly one card, and package versions are read inside the worker process.
Distribution metadata is preferred; source-only Triton installations can supply
their runtime module's version. It never includes hostname, card serial numbers,
source code, or request inputs.
"""

import importlib
import importlib.metadata

from triton_evaluation.api.contract import ExecutionEnvironment
from triton_evaluation.config import Settings


def is_complete_execution_environment(environment: ExecutionEnvironment | None) -> bool:
    """Unknown hardware/software identity cannot authorize trusted evidence reuse."""
    if environment is None or environment.hardware_count <= 0:
        return False
    return all(
        isinstance(value, str) and bool(value.strip())
        for field, value in environment.model_dump().items()
        if field != "hardware_count"
    )


def _optional_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _triton_version() -> str | None:
    """Use installed metadata, falling back to the source runtime's own version."""
    version = _optional_version("triton") or _optional_version("triton_ascend")
    if version:
        return version
    try:
        version = vars(importlib.import_module("triton")).get("__version__")
    except Exception:  # Optional identity discovery must not break fresh evaluations.
        return None
    # Missing or malformed runtime identity remains unknown; it cannot authorize reuse.
    return version.strip() if isinstance(version, str) and version.strip() else None


def build_execution_environment(settings: Settings) -> ExecutionEnvironment:
    """Assemble the worker environment from attested settings plus local packages."""
    return ExecutionEnvironment(
        hardware_model=settings.worker_hardware_model,
        hardware_count=1,
        driver_version=settings.worker_driver_version,
        runtime_version=settings.worker_runtime_version,
        torch_version=_optional_version("torch"),
        torch_npu_version=_optional_version("torch-npu"),
        triton_version=_triton_version(),
        toolchain_version=settings.worker_toolchain_version,
    )
