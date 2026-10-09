"""CPU parity against verbatim pinned official generator functions.

Primary source: Ascend/AscendC-KernelGen-data (Apache-2.0),
0607a29d231cde384c9dfdb8a39f54f91837c327,
npu-kernelbench/src/npu_kernelbench/evaluator/generator.py.
Only the framework model namespace/device is replaced; RNG logic is original.
"""

import random
from collections.abc import Callable
from enum import Enum
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from triton_evaluation.runtime.input_generation import generate_inputs

OFFICIAL_SOURCE = (
    "def _rand_tensor(\n"
    "    shape: list[int],\n"
    "    dtype: torch.dtype,\n"
    "    device: torch.device,\n"
    "    range_spec: Optional[list[int]] = None,\n"
    "    distribution: Optional[str] = None,\n"
    "    **dist_kwargs: Any,\n"
    ") -> torch.Tensor:\n"
    '    """Generate a random tensor with dtype-aware distributio'
    "n.\n"
    "\n"
    "    The default distribution for floating-point types is ``m"
    "ixed``\n"
    "    (50% uniform[-5,5] + 50% normal with mu~U[-5,5], sigma~U"
    "[0.1,2]).\n"
    "    Override per-tensor by setting ``distribution`` in the w"
    "orkload JSON.\n"
    "\n"
    "    Args:\n"
    "        shape: Tensor dimensions.\n"
    "        dtype: Target torch dtype.\n"
    "        device: Target device.\n"
    "        range_spec: Optional [low, high] for integer types ("
    "inclusive).\n"
    '        distribution: ``"uniform"``, ``"randn"``, ``"mixed"`'
    "`,\n"
    '            ``"normal"``, ``"zeros"``, ``"ones"``, ``"unifor'
    'm_signed"``,\n'
    '            or ``None`` (defaults to ``"mixed"``).\n'
    '            * "uniform" — uniform in [low, high] (``dist_kwa'
    "rgs``) or [-5,5].\n"
    '            * "randn"   — standard normal ``N(0,1)`` (floats'
    " only).\n"
    '            * "normal"  — ``N(mu, sigma)`` with random mu∈[-'
    "5,5],\n"
    "               sigma∈[0.1,2] (floats only).\n"
    '            * "mixed"   — 50% uniform[-5,5], 50% normal with'
    " random\n"
    "               mu∈[-5,5], sigma∈[0.1,2] (floats only).\n"
    '            * "zeros"   — all zeros tensor.\n'
    '            * "ones"    — all ones tensor.\n'
    '            * "uniform_signed" — uniform(low,high) * random '
    "sign (±1).\n"
    '            * None      — same as ``"mixed"``.\n'
    "        **dist_kwargs: Extra keyword args for distribution t"
    "uning, e.g.\n"
    "            ``low``, ``high``, ``mu_low``, ``mu_high``, ``si"
    "gma_low``,\n"
    "            ``sigma_high``.\n"
    "\n"
    "    Returns:\n"
    "        Random tensor of requested shape and dtype.\n"
    "\n"
    "    Raises:\n"
    "        ValueError: If dtype is not supported for random gen"
    "eration.\n"
    '    """\n'
    '    fill_value = dist_kwargs.get("fill")\n'
    "    if fill_value is not None:\n"
    "        fill_map = {\n"
    '            "inf": float("inf"),\n'
    '            "-inf": float("-inf"),\n'
    '            "nan": float("nan"),\n'
    '            "zero": 0.0,\n'
    "        }\n"
    "        if fill_value not in fill_map:\n"
    "            raise ValueError(\n"
    "                f\"Unsupported fill value '{fill_value}'. \"\n"
    '                f"Supported: {list(fill_map.keys())}"\n'
    "            )\n"
    "        return torch.full(shape, fill_map[fill_value], dtype"
    "=dtype, device=device)\n"
    "\n"
    "    is_float = dtype in (torch.float32, torch.float16, torch"
    ".bfloat16)\n"
    "\n"
    "    if is_float:\n"
    "        # Default: mixed (50% uniform[-5,5] + 50% normal)\n"
    "        if distribution is None:\n"
    '            distribution = "mixed"\n'
    "\n"
    '        if distribution == "uniform":\n'
    '            low = float(dist_kwargs.get("low", -5.0))\n'
    '            high = float(dist_kwargs.get("high", 5.0))\n'
    "            return torch.empty(shape, dtype=dtype, device=de"
    "vice).uniform_(low, high)\n"
    '        if distribution == "zeros":\n'
    "            return torch.zeros(shape, dtype=dtype, device=de"
    "vice)\n"
    '        if distribution == "ones":\n'
    "            return torch.ones(shape, dtype=dtype, device=dev"
    "ice)\n"
    '        if distribution == "uniform_signed":\n'
    '            low = float(dist_kwargs.get("low", 0.3))\n'
    '            high = float(dist_kwargs.get("high", 3.0))\n'
    "            g = torch.empty(shape, dtype=dtype, device=devic"
    "e).uniform_(low, high)\n"
    "            sign = torch.randint(0, 2, shape, device=device)"
    ".to(dtype) * 2 - 1\n"
    "            return g * sign\n"
    "\n"
    '        if distribution in ("normal", "mixed"):\n'
    "            def _rand_norm():\n"
    '                mu_low = float(dist_kwargs.get("mu_low", -5.'
    "0))\n"
    '                mu_high = float(dist_kwargs.get("mu_high", 5'
    ".0))\n"
    '                sigma_low = float(dist_kwargs.get("sigma_low'
    '", 0.1))\n'
    '                sigma_high = float(dist_kwargs.get("sigma_hi'
    'gh", 2.0))\n'
    "                mu = float(torch.empty(1, device=device).uni"
    "form_(mu_low, mu_high).item())\n"
    "                sigma = float(torch.empty(1, device=device)."
    "uniform_(sigma_low, sigma_high).item())\n"
    "                # NPU does not support torch.normal for bflo"
    "at16, so generate in\n"
    "                # float32 and cast down to the target dtype."
    "\n"
    "                if dtype is torch.bfloat16:\n"
    "                    return torch.normal(mu, sigma, shape, dt"
    "ype=torch.float32, device=device).to(dtype)\n"
    "                return torch.normal(mu, sigma, shape, dtype="
    "dtype, device=device)\n"
    "\n"
    '            if distribution == "mixed":\n'
    "                # 50% uniform / 50% normal, decided per tens"
    "or (statistical 50/50).\n"
    "                if random.random() < 0.5:\n"
    "                    return _rand_norm()\n"
    '                low = float(dist_kwargs.get("low", -5.0))\n'
    '                high = float(dist_kwargs.get("high", 5.0))\n'
    "                return torch.empty(shape, dtype=dtype, devic"
    "e=device).uniform_(low, high)\n"
    "            return _rand_norm()\n"
    "\n"
    "        # randn 和任何未识别的 distribution 默认走标准正态\n"
    "        return torch.randn(shape, dtype=dtype, device=device"
    ")\n"
    "\n"
    "    # Low-precision floats (FP8, FP4) — generate in f32 then"
    " clamp and cast\n"
    "    if dtype in (\n"
    "        torch.float8_e4m3fn,\n"
    "        torch.float8_e5m2,\n"
    "        torch.float4_e2m1fn_x2,\n"
    "    ):\n"
    "        t = torch.randn(shape, dtype=torch.float32, device=d"
    "evice).clamp_(-2.0, 2.0)\n"
    "        return t.to(dtype)\n"
    "\n"
    "    # Boolean\n"
    "    if dtype is torch.bool:\n"
    "        return torch.randint(0, 2, shape, dtype=torch.bool, "
    "device=device)\n"
    "\n"
    "    # Integers — range_spec is [low, high) exclusive (matche"
    "s torch.randint convention)\n"
    "    if dtype in (torch.int8, torch.int16, torch.int32, torch"
    ".int64):\n"
    "        if range_spec is not None:\n"
    "            low, high = range_spec\n"
    "        else:\n"
    "            ranges: dict[torch.dtype, tuple[int, int]] = {\n"
    "                torch.int8: (-128, 128),\n"
    "                torch.int16: (-1024, 1024),\n"
    "                torch.int32: (-1024, 1024),\n"
    "                torch.int64: (-1024, 1024),\n"
    "            }\n"
    "            low, high = ranges[dtype]\n"
    "        return torch.randint(low, high, shape, device=device"
    ", dtype=dtype)\n"
    "\n"
    '    raise ValueError(f"Unsupported random dtype: {dtype}")\n'
    "\n"
    "def gen_inputs(\n"
    "    definition: Definition,\n"
    "    workload: Workload,\n"
    "    device: str,\n"
    "    seed: int,\n"
    "    *,\n"
    "    safe_tensors: Optional[dict[str, torch.Tensor]] = None,\n"
    "    custom_inputs_fn: Optional[Callable] = None,\n"
    ") -> list[Any]:\n"
    '    """Generate input values for a workload in definition or'
    "der.\n"
    "\n"
    "    Iterates definition.inputs in order and dispatches based"
    " on the\n"
    "    corresponding workload.inputs InputSpec type.\n"
    "\n"
    "    Args:\n"
    "        definition: Kernel definition with input tensor spec"
    "s.\n"
    "        workload: Workload with concrete axis values and Inp"
    "utSpec entries.\n"
    '        device: Target device string (e.g., "npu:0").\n'
    "        seed: RNG seed for reproducibility.\n"
    "        safe_tensors: Pre-loaded safetensors dict from load_"
    "safetensors().\n"
    "        custom_inputs_fn: Optional callable that returns dic"
    "t[name, Tensor]\n"
    "            for CustomInput entries. Signature:\n"
    "            (resolved_axes: dict, device: torch.device) -> d"
    "ict[str, Tensor].\n"
    "\n"
    "    Returns:\n"
    "        List of input values (torch.Tensor or scalar) in def"
    "inition.inputs order.\n"
    "\n"
    "    Raises:\n"
    "        RuntimeError: If a required SafetensorsInput or Cust"
    "omInput is missing.\n"
    '    """\n'
    "    random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    if torch.npu.is_available():\n"
    "        torch.npu.manual_seed_all(seed)\n"
    "\n"
    "    # Per-case distribution family for float inputs without "
    "an explicit\n"
    '    # "distribution" override: the whole case shares one fam'
    "ily (normal or\n"
    "    # uniform), matching the 50/50 mix required by experimen"
    "tal_standard.\n"
    '    case_distribution = "normal" if random.random() < 0.5 el'
    'se "uniform"\n'
    "\n"
    "    shapes = definition.get_input_shapes(workload.axes)\n"
    "    scalar_inputs = workload.get_scalar_inputs()\n"
    "    dev = torch.device(device)\n"
    "    out: list[Any] = []\n"
    "\n"
    "    # Pre-compute custom tensors if a factory is provided\n"
    "    custom_tensors: Optional[dict[str, Any]] = None\n"
    "    if custom_inputs_fn is not None:\n"
    "        resolved_axes = definition.get_resolved_axes_values("
    "workload.axes)\n"
    "        custom_tensors = custom_inputs_fn(resolved_axes, dev"
    ")\n"
    "\n"
    "    for name, spec in definition.inputs.items():\n"
    "        input_spec = workload.inputs.get(name)\n"
    "\n"
    "        # --- Handle SafetensorsInput, ScalarInput, CustomIn"
    "put ---\n"
    "        if isinstance(input_spec, SafetensorsInput):\n"
    "            if safe_tensors is None or name not in safe_tens"
    "ors:\n"
    "                raise RuntimeError(\n"
    "                    f\"Missing required safetensors input '{n"
    "ame}'. \"\n"
    '                    f"Call load_safetensors() before gen_inp'
    'uts()."\n'
    "                )\n"
    "            t_cpu = safe_tensors[name]\n"
    "            out.append(t_cpu.to(device=dev, non_blocking=Tru"
    "e))\n"
    "\n"
    "        elif isinstance(input_spec, TensorListInput):\n"
    "            dtype = dtype_str_to_torch_dtype(input_spec.dtyp"
    "e)\n"
    "            tensors = [\n"
    "                _rand_tensor(list(s), dtype, dev, distributi"
    "on=case_distribution)\n"
    "                for s in input_spec.shapes\n"
    "            ]\n"
    "            out.append(tensors)\n"
    "\n"
    "        elif name in scalar_inputs:\n"
    "            out.append(scalar_inputs[name])\n"
    "\n"
    "        elif isinstance(input_spec, CustomInput):\n"
    "            if custom_tensors is None or name not in custom_"
    "tensors:\n"
    "                raise RuntimeError(\n"
    "                    f\"CustomInput for '{name}' requires cust"
    'om_inputs_fn. "\n'
    "                    f\"Definition '{definition.name}' has \"\n"
    "                    f\"custom_inputs_entrypoint='{definition."
    "custom_inputs_entrypoint}'.\"\n"
    "                )\n"
    "            val = custom_tensors[name]\n"
    "            if isinstance(val, torch.Tensor):\n"
    "                out.append(val.to(device=dev, non_blocking=T"
    "rue))\n"
    "            else:\n"
    "                out.append(val)\n"
    "\n"
    "        else:\n"
    "            # --- Optional input: not in workload and no dty"
    "pe/shape in definition ---\n"
    "            if input_spec is None and spec.dtype is None and"
    " spec.shape is None:\n"
    "                out.append(None)\n"
    "                continue\n"
    "\n"
    "            # --- Resolve dtype: RandomInput.dtype -> Defini"
    "tion default -> error ---\n"
    "            dtype_str = None\n"
    "            if (\n"
    "                input_spec is not None\n"
    '                and input_spec.type == "random"\n'
    '                and getattr(input_spec, "dtype", None) is no'
    "t None\n"
    "            ):\n"
    "                dtype_str = input_spec.dtype\n"
    "            elif spec.dtype is not None:\n"
    "                raw = spec.dtype\n"
    "                dtype_str = raw.value if isinstance(raw, DTy"
    "pe) else raw\n"
    "\n"
    "            if dtype_str is None:\n"
    "                raise ValueError(\n"
    "                    f\"Cannot determine dtype for input '{nam"
    "e}': \"\n"
    '                    f"no dtype in definition or workload"\n'
    "                )\n"
    "            dtype = dtype_str_to_torch_dtype(dtype_str)\n"
    "\n"
    "            # --- Resolve shape: RandomInput.shape -> Defini"
    "tion axes -> error ---\n"
    "            shape = None\n"
    "            if (\n"
    "                input_spec is not None\n"
    '                and input_spec.type == "random"\n'
    '                and getattr(input_spec, "shape", None) is no'
    "t None\n"
    "            ):\n"
    "                shape = tuple(input_spec.shape)\n"
    "            elif spec.shape is not None:\n"
    "                shape = shapes.get(name)\n"
    "            else:\n"
    "                raise ValueError(\n"
    "                    f\"Cannot determine shape for input '{nam"
    "e}': \"\n"
    '                    f"no shape in definition or workload"\n'
    "                )\n"
    "\n"
    "            range_spec = None\n"
    "            distribution = None\n"
    "            dist_kwargs: dict[str, Any] = {}\n"
    "            if input_spec is not None and input_spec.type =="
    ' "random":\n'
    '                if getattr(input_spec, "range", None) is not'
    " None:\n"
    "                    range_spec = input_spec.range\n"
    '                if getattr(input_spec, "distribution", None)'
    " is not None:\n"
    "                    distribution = input_spec.distribution\n"
    '                for key in ("low", "high", "mu_low", "mu_hig'
    'h", "sigma_low", "sigma_high"):\n'
    "                    val = getattr(input_spec, key, None)\n"
    "                    if val is not None:\n"
    "                        dist_kwargs[key] = float(val)\n"
    '                fill = getattr(input_spec, "fill", None)\n'
    "                if fill is not None:\n"
    '                    dist_kwargs["fill"] = fill\n'
    "\n"
    "            # Float inputs without an explicit distribution "
    "inherit the\n"
    "            # case-level family (normal/uniform 50/50, per e"
    "xperimental_standard).\n"
    "            if distribution is None and dtype in (torch.floa"
    "t32, torch.float16, torch.bfloat16):\n"
    "                distribution = case_distribution\n"
    "\n"
    "            if shape is None:\n"
    "                value = _rand_tensor((), dtype, dev, range_s"
    "pec, distribution, **dist_kwargs).item()\n"
    "            else:\n"
    "                value = _rand_tensor(list(shape), dtype, dev"
    ", range_spec, distribution, **dist_kwargs)\n"
    "\n"
    "            out.append(value)\n"
    "\n"
    "    return out"
)


# Pinned default.py callback loader, kept independent of production.
OFFICIAL_CUSTOM_LOADER_SOURCE = (
    "def _load_custom_inputs_fn(\n    definition: Definition,\n    "
    'device: torch.device,\n):\n    """Extract custom_inputs_entryp'
    "oint from definition.reference and return a callable.\n\n    R"
    "eturns a function with signature\n    (resolved_axes: dict, d"
    'evice: torch.device) -> dict[str, Tensor].\n    """\n    entry'
    "point = definition.custom_inputs_entrypoint\n    if not entry"
    'point:\n        raise RuntimeError(\n            f"Definition '
    "'{definition.name}' has CustomInput but \"\n            f\"no c"
    'ustom_inputs_entrypoint"\n        )\n\n    local_ns: dict = {}\n'
    "    exec(definition.reference, local_ns)\n\n    if entrypoint "
    'not in local_ns:\n        raise RuntimeError(\n            f"c'
    "ustom_inputs_entrypoint '{entrypoint}' not found in \"\n      "
    "      f\"reference code for definition '{definition.name}'\"\n "
    "       )\n\n    custom_fn = local_ns[entrypoint]\n\n    def wrap"
    "per(resolved_axes: dict, dev: torch.device) -> dict:\n       "
    " return custom_fn(resolved_axes, dev)\n\n    return wrapper"
)


class ScalarInput(SimpleNamespace):
    pass


class TensorListInput(SimpleNamespace):
    pass


class CustomInput(SimpleNamespace):
    pass


class SafetensorsInput(SimpleNamespace):
    pass


class DType(str, Enum):
    F32 = "float32"


def official(case, seed, monkeypatch, callback=None):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    namespace = dict(
        torch=torch,
        random=random,
        Any=Any,
        Callable=Callable,
        Optional=Optional,
        Definition=object,
        Workload=object,
        ScalarInput=ScalarInput,
        TensorListInput=TensorListInput,
        CustomInput=CustomInput,
        SafetensorsInput=SafetensorsInput,
        DType=DType,
        dtype_str_to_torch_dtype=lambda name: getattr(torch, name),
    )
    exec(OFFICIAL_SOURCE, namespace)
    specs = case["inputs"]
    definition = SimpleNamespace(
        name="test",
        custom_inputs_entrypoint=None,
        inputs={
            name: SimpleNamespace(dtype=spec.get("dtype"), shape=spec.get("shape"))
            for name, spec in specs.items()
        },
        get_input_shapes=lambda axes: {name: spec.get("shape") for name, spec in specs.items()},
        get_resolved_axes_values=lambda axes: axes,
    )
    classes = {"scalar": ScalarInput, "tensor_list": TensorListInput, "custom": CustomInput}
    inputs = {
        name: classes.get(spec["type"], SimpleNamespace)(**spec) for name, spec in specs.items()
    }
    workload = SimpleNamespace(
        axes=case["axes"],
        inputs=inputs,
        get_scalar_inputs=lambda: {
            name: spec["value"] for name, spec in specs.items() if spec["type"] == "scalar"
        },
    )
    return namespace["gen_inputs"](definition, workload, "cpu", seed, custom_inputs_fn=callback)


def case_for(*specs):
    inputs = {f"x{i}": spec for i, spec in enumerate(specs)}
    return {"axes": {"N": 17}, "inputs": inputs, "input_order": list(inputs)}


def assert_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        elif isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
            assert_equal(a, b)
        else:
            assert a == b


@pytest.mark.parametrize("seed", [0, 200])
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float16",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
)
@pytest.mark.parametrize(
    "distribution", [None, "normal", "mixed", "uniform", "randn", "uniform_signed", "zeros", "ones"]
)
def test_official_rng_parity(seed, dtype, distribution, monkeypatch):
    case = case_for(
        {"type": "random", "shape": [17, 3], "dtype": dtype, "distribution": distribution}
    )
    expected = official(case, seed, monkeypatch)
    actual = generate_inputs(case, seed, "cpu")
    assert_equal(actual, expected)


@pytest.mark.parametrize("seed", [0, 200])
def test_range_scalar_list_and_tensor_list(seed, monkeypatch):
    case = case_for(
        {"type": "random", "shape": [17], "dtype": "int16", "range": [-2, 3]},
        {
            "type": "random",
            "shape": [17],
            "dtype": "bfloat16",
            "distribution": "normal",
            "mu_low": 2,
            "mu_high": 3,
            "sigma_low": 0.2,
            "sigma_high": 0.3,
        },
        {"type": "scalar", "value": [3, 2, 1]},
        {"type": "scalar", "value": None},
        {"type": "tensor_list", "dtype": "float32", "shapes": [[2], [3]]},
        {"type": "random", "dtype": "int64", "shape": []},
    )
    assert_equal(generate_inputs(case, seed, "cpu"), official(case, seed, monkeypatch))


def test_custom_module_rng_is_before_seed_and_callback_after_family(monkeypatch):
    code = (
        "import random\nrandom.random()\ntorch.rand(3)\ndef build(axes, device):\n"
        "    return {'x0': torch.rand(axes['N'], device=device)}\n"
    )
    case = case_for({"type": "custom"}, {"type": "random", "dtype": "float32", "shape": [17]})
    ns = {"torch": torch}
    exec(code, ns)
    expected = official(case, 200, monkeypatch, ns["build"])
    for _ in range(2):
        assert_equal(generate_inputs(case, 200, "cpu", code, "unused.py::build"), expected)


def test_adapter_order_identity_and_deferred_custom_execution():
    from triton_evaluation.api.operator import OperatorRequest
    from triton_evaluation.evaluators.operator import to_evaluation_request

    payload = {
        "definition": {
            "inputs": {"z": {"shape": ["2"], "dtype": "float32"}, "a": {}},
            "reference": "def run(z, a): return z",
            "custom_inputs_entrypoint": "get_inputs",
        },
        "workloads": [
            {"inputs": {"z": {"type": "random"}, "a": {"type": "scalar", "value": 9}}},
            {"inputs": {"z": {"type": "custom"}, "a": {"type": "custom"}}},
        ],
        "triton_code": "def run(z, a): return z",
        "custom_inputs_code": "raise RuntimeError('must execute only on generation')",
    }
    request = OperatorRequest.model_validate(payload)
    first = to_evaluation_request(request)
    second = to_evaluation_request(
        request.model_copy(update={"triton_code": "def run(z, a): return z + 1"})
    )
    assert first.reference_code == second.reference_code
    assert first.kernel_code != second.kernel_code
    assert '"input_order":["z","a"]' in first.reference_code
    namespace = {}
    exec(first.reference_code, namespace)
    assert len(namespace["get_input_groups"]()) == 2
    with pytest.raises(RuntimeError, match="only on generation"):
        namespace["_operator_generate_inputs"](
            case_for({"type": "custom"}),
            200,
            "cpu",
            "raise RuntimeError('must execute only on generation')",
        )
    assert (
        first.reference_code
        != to_evaluation_request(request.model_copy(update={"seed": 0})).reference_code
    )
    assert (
        first.reference_code
        != to_evaluation_request(
            request.model_copy(update={"custom_inputs_code": None})
        ).reference_code
    )


@pytest.mark.parametrize("mode", ["torch", "triton", "comparison"])
def test_adapter_model_contracts(mode):
    from triton_evaluation.api.operator import OperatorRequest
    from triton_evaluation.evaluators.operator import to_evaluation_request

    payload = {
        "definition": {"inputs": {"x": {"dtype": "float32", "shape": ["2"]}}},
        "workloads": [{"inputs": {"x": {"type": "random"}}}],
    }
    if mode != "triton":
        payload["definition"]["reference"] = "def run(x): return x"
    if mode != "torch":
        payload["triton_code"] = "def run(x): return x"
    engine = to_evaluation_request(OperatorRequest.model_validate(payload))
    assert engine.implementation == mode
    assert ("class Model(" in engine.reference_code) == (mode != "triton")
    assert ("class ModelNew(" in engine.kernel_code) == (mode != "torch")


def test_fp4_generation_uses_float32_clamp_then_cast(monkeypatch):
    calls = []

    class Draw:
        def clamp_(self, low, high):
            calls.append(("clamp", low, high))
            return self

        def to(self, dtype):
            calls.append(("cast", dtype))
            return "packed-fp4"

    class TorchNamespace:
        def __getattr__(self, name):
            return getattr(torch, name)

        def randn(self, shape, *, dtype, device):
            calls.append(("draw", shape, dtype, device))
            return Draw()

    namespace = {
        "torch": TorchNamespace(),
        "random": random,
        "Any": Any,
        "Optional": Optional,
        "Definition": object,
        "Workload": object,
        "Callable": Callable,
    }
    exec(OFFICIAL_SOURCE, namespace)
    dtype = torch.float4_e2m1fn_x2
    assert namespace["_rand_tensor"]([4], dtype, torch.device("cpu")) == "packed-fp4"
    expected_calls = list(calls)
    calls.clear()
    from triton_evaluation.runtime.input_generation import _rand_tensor

    monkeypatch.setattr(torch, "randn", TorchNamespace().randn)
    dtype = torch.float4_e2m1fn_x2
    assert _rand_tensor([4], dtype, torch.device("cpu")) == "packed-fp4"
    assert calls == expected_calls


def test_global_family_draw_even_for_explicit_distribution(monkeypatch):
    case = case_for({"type": "random", "dtype": "float32", "shape": [3], "distribution": "mixed"})
    assert_equal(generate_inputs(case, 0, "cpu"), official(case, 0, monkeypatch))
    actual_next = random.random()
    official(case, 0, monkeypatch)
    assert actual_next == random.random()


def test_embedded_generator_preserves_reference_globals():
    from triton_evaluation.api.operator import OperatorRequest
    from triton_evaluation.evaluators.operator import to_evaluation_request

    request = OperatorRequest.model_validate(
        {
            "definition": {
                "inputs": {"x": {"shape": ["2"], "dtype": "float32"}},
                "reference": (
                    "random = 2\ngenerate_inputs = 3\n_rand_tensor = 4\ndef run(x):\n"
                    "    return x * (random + generate_inputs + _rand_tensor)"
                ),
            },
            "workloads": [{"inputs": {"x": {"type": "random"}}}],
        }
    )
    generated = to_evaluation_request(request)
    namespace = {}
    exec(generated.reference_code, namespace)
    assert namespace["random"] == 2
    assert namespace["generate_inputs"] == 3
    assert namespace["_rand_tensor"] == 4
    torch.testing.assert_close(namespace["Model"]()(torch.ones(2)), torch.full([2], 9.0))
    assert (
        namespace["_operator_generate_inputs"].__globals__
        is namespace["_operator_generator_namespace"]
    )


def test_custom_caller_workflow_fresh_module_seeded_constants(monkeypatch):
    # default.py::DefaultEvaluator.build_baseline: set_seed -> fresh exec
    # definition.reference -> generator.gen_inputs (seed again, family, callback).
    code = (
        "import random\nimport torch\nconstant = random.random() + torch.rand(()).item()\n"
        "counter = 0\ndef build(axes, device):\n"
        "    global counter\n    counter += 1\n"
        "    return {'x0': torch.rand(axes['N'], device=device) + constant + counter}\n"
    )
    case = case_for({"type": "custom"})
    for seed in [0, 200]:
        random.seed(seed)
        torch.manual_seed(seed)
        namespace = {"Definition": object, "torch": torch}
        exec(OFFICIAL_CUSTOM_LOADER_SOURCE, namespace)
        definition = SimpleNamespace(reference=code, custom_inputs_entrypoint="build")
        callback = namespace["_load_custom_inputs_fn"](definition, torch.device("cpu"))
        expected = official(case, seed, monkeypatch, callback)
        for _ in range(2):
            assert_equal(generate_inputs(case, seed, "cpu", code, "build"), expected)


def test_noncustom_workload_ignores_custom_source(monkeypatch):
    case = case_for({"type": "random", "dtype": "float32", "shape": [17]})
    expected = official(case, 200, monkeypatch)
    assert_equal(
        generate_inputs(case, 200, "cpu", "raise RuntimeError('unused')", "build"), expected
    )


def test_reference_callback_source_used_when_no_override(monkeypatch):
    from triton_evaluation.api.operator import OperatorRequest
    from triton_evaluation.evaluators.operator import to_evaluation_request

    reference = (
        "import torch\ndef build(axes, device): "
        "return {'x': torch.ones(2, device=device)}\ndef run(x): return x"
    )
    request = OperatorRequest.model_validate(
        {
            "definition": {
                "inputs": {"x": {}},
                "reference": reference,
                "custom_inputs_entrypoint": "build",
            },
            "workloads": [{"inputs": {"x": {"type": "custom"}}}],
        }
    )
    namespace = {}
    exec(to_evaluation_request(request).reference_code, namespace)
    assert namespace["_operator_custom_code"] == reference
    result = namespace["_operator_generate_inputs"](
        namespace["_operator_cases"][0], 200, "cpu", reference, "build"
    )
    torch.testing.assert_close(result[0], torch.ones(2))
