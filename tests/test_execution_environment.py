"""Execution-environment provenance version detection."""

import importlib.metadata

import pytest

from triton_evaluation.runtime.execution_environment import _triton_version


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
def test_triton_version_prefers_canonical_then_ascend_fork(
    monkeypatch, available, expected
):
    monkeypatch.setattr(importlib.metadata, "version", _serve(available))
    assert _triton_version() == expected
