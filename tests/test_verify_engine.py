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


def test_candidate_missing_model_new_writes_compilation_failure(
    tmp_path, monkeypatch
):
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )
    candidate = SimpleNamespace()

    def fake_import_module(name):
        if name == "histc_torch":
            return reference
        if name == "histc_triton_ascend_impl":
            return candidate
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

    failure = json.loads(output.read_text())["failures"][0]
    assert failure["error_type"] == "CompilationError"
    assert "AttributeError" in failure["error_msg"]
    assert "ModelNew" in failure["error_msg"]


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


def test_failure_divergence_computes_kind_and_sample():
    """_failure_divergence computes {kind, sample} from flat lists via the classifier."""
    reference = [1.0, 2.0, 3.0, 4.0]
    candidate = [6.0, 7.0, 8.0, 9.0]  # uniform +5 offset -> constant_offset
    out = verify._failure_divergence(reference, candidate, (4,),
                                     atol=1e-3, rtol=2 ** -13,
                                     small_value_threshold=2 ** -14,
                                     small_value_error=2 ** -30,
                                     rel_threshold=2 ** -13,
                                     small_count=0, normal_count=4)
    assert out["kind"] == "constant_offset"
    assert out["sample"] and out["sample"][0]["diff"] == 5.0


def test_shape_mismatch_failure_gets_shape_mismatch_kind():
    """The shape-mismatch helper records kind without needing tensors."""
    failure = verify._shape_mismatch_failure(case_idx=1, fw_shape=(2, 3), impl_shape=(2, 4))
    assert failure["kind"] == "shape_mismatch"
    assert failure["sample"] == []
    assert "shape" in failure["error_msg"]
