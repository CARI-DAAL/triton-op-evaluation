import json
import math
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


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeMask:
    def __init__(self, values):
        self.values = list(values)

    def __and__(self, other):
        return _FakeMask(a and b for a, b in zip(self.values, other.values, strict=True))

    def sum(self):
        return _FakeScalar(sum(self.values))

    def any(self):
        return any(self.values)


class _FakeDType:
    def __init__(self, name, *, floating=False):
        self.name = name
        self.is_floating_point = floating
        self.is_complex = False

    def __str__(self):
        return self.name


class _FakeTensor:
    def __init__(self, values, dtype, shape=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = shape or (len(self.values),)

    def flatten(self):
        return _FakeTensor(self.values, self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numel(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)

    def __getitem__(self, key):
        if isinstance(key, _FakeMask):
            return _FakeTensor(
                [value for value, keep in zip(self.values, key.values, strict=True) if keep],
                self.dtype,
            )
        return _FakeScalar(self.values[key])


def _comparison_torch(bool_dtype):
    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.bool = bool_dtype
    torch.isfinite = lambda tensor: _FakeMask(
        math.isfinite(value) for value in tensor.values
    )
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


@pytest.mark.parametrize(
    "mode",
    ["non_compute", "non_finite", "bool", "integer", "quantized", "float"],
)
def test_compare_attaches_divergence_to_every_value_mismatch_path(mode, monkeypatch):
    bool_dtype = _FakeDType("bool")
    int_dtype = _FakeDType("int64")
    float_dtype = _FakeDType("float32", floating=True)
    if mode == "bool":
        dtype = bool_dtype
    elif mode in {"integer", "quantized"}:
        dtype = int_dtype
    else:
        dtype = float_dtype
    torch = _comparison_torch(bool_dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def fail(*args, **kwargs):
        raise verify.AccuracyError("mismatch", {"category": mode})

    monkeypatch.setattr(verify, "_check_nan_inf", lambda *args: None)
    monkeypatch.setattr(verify, "_is_integer_dtype", lambda value: value is int_dtype)
    if mode == "non_compute":
        monkeypatch.setattr(verify, "_compare_binary_exact", fail)
    elif mode == "non_finite":
        monkeypatch.setattr(verify, "_check_nan_inf", fail)
    elif mode == "bool":
        monkeypatch.setattr(verify, "_compare_bool_output", fail)
    elif mode == "integer":
        monkeypatch.setattr(verify, "_compare_integer_compute", fail)
    elif mode == "quantized":
        monkeypatch.setattr(verify, "_compare_quant_fp_to_int", fail)
    else:
        monkeypatch.setattr(verify, "_compare_float_output", fail)

    reference = _FakeTensor([1.0, 2.0, 3.0, 4.0], dtype)
    candidate = _FakeTensor([6.0, 7.0, 8.0, 9.0], dtype)
    input_type = "float" if mode == "quantized" else "int"

    with pytest.raises(verify.AccuracyError) as caught:
        verify.compare(
            reference,
            candidate,
            "float32",
            verify.InputSpec(input_type=input_type),
            non_compute=mode == "non_compute",
        )

    assert caught.value.metrics["divergence"]["kind"] == "constant_offset"
    assert caught.value.metrics["divergence"]["sample"][0]["diff"] == 5.0


def test_nan_position_mismatch_is_structured_accuracy_error(monkeypatch):
    dtype = _FakeDType("float32", floating=True)
    torch = _comparison_torch(_FakeDType("bool"))
    torch.isnan = lambda tensor: _FakeMask(math.isnan(value) for value in tensor.values)
    torch.isinf = lambda tensor: _FakeMask(math.isinf(value) for value in tensor.values)
    torch.equal = lambda left, right: left.values == right.values
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(verify.AccuracyError, match="NaN"):
        verify._check_nan_inf(
            _FakeTensor([math.nan, 1.0], dtype),
            _FakeTensor([0.0, 1.0], dtype),
            2,
        )


def test_try_run_case_projects_attached_divergence_into_failure(monkeypatch):
    divergence = {
        "kind": "constant_offset",
        "sample": [
            {"idx": [0], "candidate": 6.0, "reference": 1.0, "diff": 5.0}
        ],
    }

    def fail_case(*args, **kwargs):
        raise verify.AccuracyError(
            "integer mismatch",
            {"category": "integer_compute", "divergence": divergence},
        )

    monkeypatch.setattr(verify, "_instantiate_models", lambda *args: (object(), object()))
    monkeypatch.setattr(verify, "run_single_case", fail_case)
    monkeypatch.setattr(verify, "describe_input", lambda inputs: [])
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    modules = {
        "framework_cls": object,
        "impl_cls": object,
        "get_init_inputs": lambda: [],
    }

    passed, failure = verify._try_run_case(
        modules,
        verify.CaseContext(case_idx=1, total_cases=1),
        inputs=[],
        device="npu:0",
    )

    assert passed is False
    assert failure["kind"] == "constant_offset"
    assert failure["sample"] == divergence["sample"]


def test_compare_rejects_same_numel_with_different_shapes(monkeypatch):
    dtype = _FakeDType("float32", floating=True)
    torch = _comparison_torch(_FakeDType("bool"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(verify, "_check_nan_inf", lambda *args: None)
    monkeypatch.setattr(verify, "_is_integer_dtype", lambda value: False)
    monkeypatch.setattr(verify, "_compare_float_output", lambda *args: None)

    reference = _FakeTensor(range(6), dtype, shape=(2, 3))
    candidate = _FakeTensor(range(6), dtype, shape=(3, 2))

    with pytest.raises(verify._ShapeMismatchError) as caught:
        verify.compare(reference, candidate, "float32")

    assert caught.value.fw_shape == (2, 3)
    assert caught.value.impl_shape == (3, 2)


def test_first_launch_compilation_error_is_attributed_compile(tmp_path, monkeypatch):
    """A triton CompilationError raised lazily at first kernel launch is a
    compile-phase failure, not an execute-phase runtime error (spec §8/Plan D)."""
    import contextlib

    class _FakeModel:
        def __init__(self, *_a, **_k):
            pass

        def to(self, *_a):
            return self

        def forward(self, *_inputs):
            return [17]

        __call__ = forward

    reference = SimpleNamespace(
        Model=_FakeModel,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )

    class CompilationError(Exception):
        pass

    CompilationError.__module__ = "triton.compiler.errors"

    class ModelNew:
        def __init__(self, *_a, **_k):
            pass

        def to(self, *_a):
            return self

        def forward(self, *_inputs):
            raise CompilationError("at 2:14: NameError: undefined_name")

        __call__ = forward

    candidate = SimpleNamespace(ModelNew=ModelNew)

    def fake_import_module(name):
        if name == "addmm_torch":
            return reference
        if name == "addmm_triton_ascend_impl":
            return candidate
        raise AssertionError(f"unexpected module import: {name}")

    torch = _fake_torch_module()
    torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    output = tmp_path / "verify_result.json"
    assert verify.verify_implementations(
        "addmm",
        str(tmp_path),
        output_path=str(output),
    ) == (0, 1)

    failure = json.loads(output.read_text())["failures"][0]
    assert failure["candidate_error"]["phase"] == "compile"
    assert failure["error_type"] == "_CandidateCompileError"
    assert "CompilationError" in failure["error_msg"]
