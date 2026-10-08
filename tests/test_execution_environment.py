"""Execution-environment provenance version detection."""

import importlib.metadata

import pytest

from triton_evaluation.config import Settings
from triton_evaluation.runtime.execution_environment import (
    _triton_version,
    build_execution_environment,
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
    environment = build_execution_environment(Settings())
    assert environment.torch_version == available.get("torch")
    assert environment.torch_npu_version == available.get("torch-npu")
    assert environment.triton_version == expected_triton
