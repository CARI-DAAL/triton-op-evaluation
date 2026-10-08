"""The dynamic canary must expose source to Triton's inspect-based decorator."""
import inspect
import sys
from types import ModuleType, SimpleNamespace

from triton_evaluation.health import _canary_namespace


def test_canary_jit_can_read_its_source(monkeypatch):
    sources = []
    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    language.constexpr = object()

    def jit(function):
        sources.append(inspect.getsource(function))
        return function

    triton.jit = jit
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Module=object)))
    namespace = _canary_namespace()
    assert "ModelNew" in namespace
    assert len(sources) == 1
    assert "def _canary_add_kernel(" in sources[0]
    assert "tl.store(" in sources[0]
