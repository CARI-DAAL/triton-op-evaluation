"""Execution-environment provenance version detection."""

import importlib.metadata
import sys
from types import ModuleType

import pytest

from triton_evaluation.config import Settings
from triton_evaluation.runtime.execution_environment import (
    _triton_version,
    build_execution_environment,
    is_complete_execution_environment,
)


def _serve(available: dict[str, str]):
    def version(name: str) -> str:
        if name in available:
            return available[name]
        raise importlib.metadata.PackageNotFoundError(name)

    return version


@pytest.mark.parametrize(
    "available,expected",
    [
        ({}, None),
        ({"triton": "3.4.0"}, "3.4.0"),
        ({"triton_ascend": "3.2.0"}, "3.2.0"),
        # The canonical distribution wins when both are installed.
        ({"triton": "3.4.0", "triton_ascend": "3.2.0"}, "3.4.0"),
    ],
)
def test_triton_version_prefers_canonical_then_ascend_fork(monkeypatch, available, expected):
    monkeypatch.setattr(importlib.metadata, "version", _serve(available))
    monkeypatch.setitem(sys.modules, "triton", ModuleType("triton"))
    assert _triton_version() == expected


@pytest.mark.parametrize(
    "available,expected_triton",
    [
        ({}, None),
        ({"torch": "2.9.0", "torch-npu": "2.9.0.post8", "triton_ascend": "3.2.0"}, "3.2.0"),
        ({"triton": "3.4.0", "triton_ascend": "3.2.0"}, "3.4.0"),
    ],
)
def test_worker_environment_optional_distribution_presence(monkeypatch, available, expected_triton):
    monkeypatch.setattr(importlib.metadata, "version", _serve(available))
    monkeypatch.setitem(sys.modules, "triton", ModuleType("triton"))
    environment = build_execution_environment(Settings())
    assert environment.torch_version == available.get("torch")
    assert environment.torch_npu_version == available.get("torch-npu")
    assert environment.triton_version == expected_triton


def test_source_only_triton_populates_complete_worker_identity(tmp_path, monkeypatch):
    """Source installs can execute kernels without providing .dist-info metadata."""
    package = tmp_path / "triton"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "3.2.0+source"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", _serve({
        "torch": "2.10.0", "torch-npu": "2.10.0",
    }))
    settings = Settings(
        worker_hardware_model="Ascend910", worker_driver_version="25.5.0",
        worker_runtime_version="9.0.0.0430", worker_toolchain_version="CANN 9.0.0.0430",
    )
    try:
        environment = build_execution_environment(settings)
        assert environment.triton_version == "3.2.0+source"
        assert is_complete_execution_environment(environment)
    finally:
        sys.modules.pop("triton", None)


@pytest.mark.parametrize("available", [{"triton": "3.4.0"}, {"triton_ascend": "3.2.0"}])
def test_distribution_version_never_imports_runtime(tmp_path, monkeypatch, available):
    package = tmp_path / "triton"
    package.mkdir()
    (package / "__init__.py").write_text('raise AssertionError("runtime import was unnecessary")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", _serve(available))
    assert _triton_version() == next(iter(available.values()))
    assert "triton" not in sys.modules


@pytest.mark.parametrize("version", [None, "", "   ", False, 3, ["3.2.0"]])
def test_invalid_module_version_remains_unknown(monkeypatch, version):
    module = ModuleType("triton")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "triton", module)
    monkeypatch.setattr(importlib.metadata, "version", _serve({}))
    assert _triton_version() is None


@pytest.mark.parametrize("error", ["ModuleNotFoundError", "RuntimeError"])
def test_unimportable_runtime_keeps_unknown_version(tmp_path, monkeypatch, error):
    package = tmp_path / "triton"
    package.mkdir()
    (package / "__init__.py").write_text(f'raise {error}("runtime unavailable")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", _serve({}))
    assert _triton_version() is None
