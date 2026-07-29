import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from triton_evaluation.engine import verify


def _fake_torch_module():
    torch = ModuleType("torch")
    torch.Tensor = type("FakeTensor", (), {})
    torch.manual_seed = lambda seed: None
    torch.npu = SimpleNamespace(manual_seed=lambda seed: None)
    torch.device = lambda name: name
    return torch


def test_candidate_import_error_writes_compilation_failure(tmp_path, monkeypatch):
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )

    def fake_import_module(name):
        if name == "histc_torch":
            return reference
        if name == "histc_triton_ascend_impl":
            raise TypeError(
                "jit() got an unexpected keyword argument 'num_warps'"
            )
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    output = tmp_path / "verify_result.json"
    assert verify.verify_implementations(
        "histc",
        str(tmp_path),
        output_path=str(output),
    ) == (0, 1)

    result = json.loads(output.read_text())
    assert result["op_name"] == "histc"
    assert result["total_cases"] == 1
    assert result["passed_cases"] == 0
    assert result["failed_cases"] == 1
    assert len(result["failures"]) == 1
    failure = result["failures"][0]
    assert failure["case_idx"] == 1
    assert failure["error_type"] == "CompilationError"
    assert "TypeError" in failure["error_msg"]
    assert "unexpected keyword argument" in failure["error_msg"]


def test_reference_import_error_is_not_reclassified(tmp_path, monkeypatch):
    def fake_import_module(name):
        assert name == "histc_torch"
        raise RuntimeError("broken reference module")

    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    with pytest.raises(RuntimeError, match="broken reference module"):
        verify._load_verify_modules(
            "histc",
            str(tmp_path),
            "triton_ascend_impl",
        )
