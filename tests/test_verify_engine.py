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
    torch.isfinite = lambda tensor: _FakeMask(math.isfinite(value) for value in tensor.values)
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
            raise TypeError("jit() got an unexpected keyword argument 'num_warps'")
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


def test_candidate_missing_model_new_writes_compilation_failure(tmp_path, monkeypatch):
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
    out = verify._failure_divergence(
        reference,
        candidate,
        (4,),
        atol=1e-3,
        rtol=2**-13,
        small_value_threshold=2**-14,
        small_value_error=2**-30,
        rel_threshold=2**-13,
        small_count=0,
        normal_count=4,
    )
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
        "sample": [{"idx": [0], "candidate": 6.0, "reference": 1.0, "diff": 5.0}],
    }

    def fail_case(*args, **kwargs):
        raise verify.AccuracyError(
            "integer mismatch",
            {"category": "integer_compute", "divergence": divergence},
        )

    monkeypatch.setattr(verify, "_instantiate_models", lambda *args, **kwargs: (object(), object()))
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


def test_input_groups_retry_on_npu_context_when_cpu_lacks_kernel(monkeypatch):
    """References may declare inputs with NPU-only op/dtype combinations.

    torch.randn(shape, dtype=torch.int32) has no CPU kernel; torch_npu
    implements it on the NPU. The sandbox verification ran with the NPU as the
    ambient device, so such a reference resolved its inputs there. Constructing
    on CPU must not void the whole evaluation: retry in the NPU context.
    """
    calls = []
    default_devices = []

    class _Module:
        def get_input_groups(self):
            calls.append(None)
            if len(calls) == 1:
                raise NotImplementedError("normal_kernel_cpu not implemented for 'Int'")
            return [[1], [2], [3]]

    torch = _fake_torch_module()
    torch.set_default_device = lambda name: default_devices.append(name)
    monkeypatch.setitem(sys.modules, "torch", torch)

    groups, total = verify.resolve_input_provider(_Module())

    assert total == 3
    assert groups == [[1], [2], [3]]
    assert len(calls) == 2
    assert default_devices == ["npu"]


def test_input_groups_single_attempt_when_cpu_construction_succeeds(monkeypatch):
    """The NPU retry stays a fallback: the common path runs exactly once."""
    calls = []

    class _Module:
        def get_input_groups(self):
            calls.append(None)
            return [[1], [2]]

    torch = _fake_torch_module()
    torch.set_default_device = lambda name: pytest.fail("must not switch devices")
    monkeypatch.setitem(sys.modules, "torch", torch)

    groups, total = verify.resolve_input_provider(_Module())

    assert (groups, total) == ([[1], [2]], 2)
    assert len(calls) == 1


def test_input_groups_other_errors_are_not_retried(monkeypatch):
    """Only the missing-CPU-kernel signal triggers the NPU retry."""
    calls = []

    class _Module:
        def get_input_groups(self):
            calls.append(None)
            raise ValueError("reference is malformed")

    torch = _fake_torch_module()
    torch.set_default_device = lambda name: pytest.fail("must not switch devices")
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(ValueError, match="reference is malformed"):
        verify.resolve_input_provider(_Module())

    assert len(calls) == 1


def test_ensure_npu_ambient_device_is_the_single_invariant():
    """The ambient-device invariant lives in exactly one implementation.

    Engine entry points must route through _common_utils.ensure_npu_ambient_device
    instead of inlining the probe — a diverging copy shipped a broken profile
    runner (torch.npu evaluated before torch_npu import raises AttributeError
    during argument evaluation, which hasattr does not swallow).
    """
    from triton_evaluation.engine import _common_utils, benchmark, verify

    for module in (verify, benchmark):
        source = module.__file__
        text = open(source, encoding="utf-8").read()
        assert "ensure_npu_ambient_device" in text
        assert "set_default_device" not in text.replace("ensure_npu_ambient_device", ""), (
            f"{module.__name__} must not inline the device switch"
        )

    assert hasattr(_common_utils, "ensure_npu_ambient_device")
    # The generated profile runner source imports the same helper.
    text = open(
        _common_utils.__file__.replace("_common_utils.py", "profile_runner.py"),
        encoding="utf-8",
    ).read()
    assert "from _common_utils import ensure_npu_ambient_device" in text


def test_ensure_npu_ambient_device_guards_missing_npu():
    """Attribute shapes that raised during argument evaluation historically."""
    from types import ModuleType, SimpleNamespace

    from triton_evaluation.engine._common_utils import ensure_npu_ambient_device

    bare = ModuleType("torch_without_npu")  # no `npu` attribute at all
    assert ensure_npu_ambient_device(bare) is False

    # `npu` present but is_available absent (lazy attr not yet resolved).
    partial = ModuleType("torch_partial")
    partial.npu = SimpleNamespace()
    assert ensure_npu_ambient_device(partial) is False

    # is_available present but reports unavailable.
    unavailable = ModuleType("torch_off")
    unavailable.npu = SimpleNamespace(is_available=lambda: False)
    switched = []
    unavailable.set_default_device = switched.append
    assert ensure_npu_ambient_device(unavailable) is False
    assert switched == []


def test_ensure_npu_ambient_device_switches_when_available(monkeypatch):
    from triton_evaluation.engine._common_utils import ensure_npu_ambient_device

    switched = []
    torch = _fake_torch_module()
    torch.npu.is_available = lambda: True
    torch.set_default_device = switched.append
    monkeypatch.setattr("sys.modules", __import__("sys").modules)

    assert ensure_npu_ambient_device(torch) is True
    assert switched == ["npu"]


def test_ensure_npu_ambient_device_retries_transient_unavailability(monkeypatch):
    """drvGetDevNum 瞬态失败：有界等待重试后设备出现，绝不选择 CPU。"""
    from triton_evaluation.engine import _common_utils

    monkeypatch.setattr(_common_utils, "AMBIENT_POLL_SECONDS", 0.0)
    torch = _fake_torch_module()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return calls["n"] > 2  # 两次瞬态失败后设备出现

    torch.npu.is_available = flaky
    switched = []
    torch.set_default_device = switched.append

    assert _common_utils.ensure_npu_ambient_device(torch, wait_seconds=5.0) is True
    assert calls["n"] == 3
    assert switched == ["npu"]


def test_ensure_npu_ambient_device_stops_at_bounded_deadline(monkeypatch):
    """等待耗尽仍不可用则返回 False（调用方立刻显式失败），不循环不回退。"""
    from triton_evaluation.engine import _common_utils

    monkeypatch.setattr(_common_utils, "AMBIENT_POLL_SECONDS", 0.0)
    torch = _fake_torch_module()
    calls = {"n": 0}

    def never():
        calls["n"] += 1
        return False

    torch.npu.is_available = never
    switched = []
    torch.set_default_device = switched.append

    assert _common_utils.ensure_npu_ambient_device(torch, wait_seconds=0.05) is False
    assert calls["n"] >= 2
    assert switched == []


def test_npu_strict_guard_raises_only_after_bounded_wait(tmp_path, monkeypatch):
    """npu-strict 守护保留大声 fail-fast：有界等待耗尽后仍抛同样错误。"""
    from triton_evaluation.engine import _common_utils

    monkeypatch.setattr(_common_utils, "AMBIENT_POLL_SECONDS", 0.0)
    monkeypatch.setattr(verify, "_AMBIENT_WAIT_CAP_SECONDS", 0.05)
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )

    def fake_import_module(name):
        if name == "histc_torch":
            return reference
        if name == "histc_triton_ascend_impl":
            raise TypeError("boom")
        raise AssertionError(f"unexpected module import: {name}")

    torch = _fake_torch_module()
    torch.npu.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    with pytest.raises(RuntimeError, match="npu-strict reference requires the ambient NPU device"):
        verify.verify_implementations(
            "histc",
            str(tmp_path),
            reference_device_policy="npu-strict",
            subject_budget_seconds=30,
        )


def test_npu_strict_guard_recovers_within_bounded_wait(tmp_path, monkeypatch):
    """瞬态窗口内恢复：守护等待后继续执行，不把基础设施瞬态当失败。"""
    from triton_evaluation.engine import _common_utils

    monkeypatch.setattr(_common_utils, "AMBIENT_POLL_SECONDS", 0.0)
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )

    def fake_import_module(name):
        if name == "histc_torch":
            return reference
        if name == "histc_triton_ascend_impl":
            raise TypeError("boom")
        raise AssertionError(f"unexpected module import: {name}")

    torch = _fake_torch_module()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return calls["n"] > 1  # 首次瞬态失败，重试后设备出现

    torch.npu.is_available = flaky
    torch.set_default_device = lambda _name: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    output = tmp_path / "verify_result.json"
    assert verify.verify_implementations(
        "histc",
        str(tmp_path),
        output_path=str(output),
        reference_device_policy="npu-strict",
        subject_budget_seconds=30,
    ) == (0, 1)  # CompilationError 归因 —— 引擎已通过守护继续执行
    assert calls["n"] == 2


def test_ensure_npu_ambient_device_default_wait_follows_policy(monkeypatch):
    """未显式传 wait_seconds 的调用点（benchmark / profile runner）由部署策略推导：
    npu-strict 有界等待恢复，其余（含未配置）保持单次探测。"""
    from triton_evaluation.engine import _common_utils

    monkeypatch.setattr(_common_utils, "AMBIENT_POLL_SECONDS", 0.0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return calls["n"] > 1

    torch = _fake_torch_module()
    torch.npu.is_available = flaky
    torch.set_default_device = lambda _name: None

    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    assert _common_utils.ensure_npu_ambient_device(torch) is True
    assert calls["n"] == 2  # 默认等待吸收了首次瞬态失败

    calls["n"] = 0
    torch.npu.is_available = lambda: False
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    assert _common_utils.ensure_npu_ambient_device(torch) is False
    assert calls["n"] == 0  # cpu-first：单次探测，不进入等待循环

    monkeypatch.delenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY")
    assert _common_utils.ensure_npu_ambient_device(torch, wait_seconds=0.0) is False


def test_benchmark_npu_strict_fails_loud_without_ambient_device():
    """benchmark 在 npu-strict 下同样大声失败：不静默退回 CPU 物化路径。
    （该字节变更使 timing-reuse policy 目标指纹不再匹配 → 回退 fresh。）"""
    from triton_evaluation.engine import benchmark

    text = open(benchmark.__file__, encoding="utf-8").read()
    assert "npu-strict benchmark requires the ambient NPU device" in text
    assert "wait_seconds=_ambient_wait" in text
    assert "set_default_device" not in text.replace("ensure_npu_ambient_device", "")


class _KwOnlyModel:
    """Mimics lv4 references: forward(self, x1, x2, *, s1=None, s2=None)."""

    def forward(self, x1, x2, *, s1=None, s2=None):
        return (x1, x2, s1, s2)

    __call__ = forward


class _PlainModel:
    def forward(self, x1, x2):
        return x1 + x2

    __call__ = forward


class _VarArgsModel:
    def forward(self, x1, *rest):
        return (x1, rest)

    __call__ = forward


def test_invoke_model_binds_overflow_to_keyword_only_names():
    """The engine must honor the declared signature: positional slots take the
    leading inputs, the overflow binds to keyword-only parameter names."""
    from triton_evaluation.engine._common_utils import invoke_model

    out = invoke_model(_KwOnlyModel(), [1, 2, 3, 4])
    assert out == (1, 2, 3, 4)


def test_invoke_model_plain_signature_unchanged():
    from triton_evaluation.engine._common_utils import invoke_model

    assert invoke_model(_PlainModel(), [1, 2]) == 3


def test_invoke_model_varargs_absorbs_positionally():
    from triton_evaluation.engine._common_utils import invoke_model

    assert invoke_model(_VarArgsModel(), [1, 2, 3]) == (1, (2, 3))


def test_invoke_model_count_mismatch_falls_through_naturally():
    """Overflow without matching keyword slots keeps Python's own error —
    honest failure semantics are preserved for genuinely wrong inputs."""
    import pytest as _pytest

    from triton_evaluation.engine._common_utils import invoke_model

    with _pytest.raises(TypeError):
        invoke_model(_KwOnlyModel(), [1, 2, 3, 4, 5])
