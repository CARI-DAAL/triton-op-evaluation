"""Pure validation of official workload descriptors."""

import time

import pytest
from pydantic import ValidationError

from triton_evaluation.api.operator import OperatorRequest, normalize_workloads


def body(**updates):
    data = {
        "definition": {"inputs": {"x": {}}, "reference": "def run(x): return x"},
        "workloads": [{"inputs": {"x": {"type": "random", "dtype": "float32", "shape": [2]}}}],
    }
    data.update(updates)
    return data


def test_official_definition_and_named_order():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "name": "BatchMatmul",
                "op_type": "matmul",
                "axes": {},
                "inputs": {"A": {"description": "first"}, "B": {}},
                "outputs": {"y": {}},
                "reference": "def run(A, B): return A @ B",
            },
            workloads=[
                {
                    "uuid": "BatchMatmul-001",
                    "axes": None,
                    "inputs": {
                        "B": {"type": "random", "shape": [2, 3], "dtype": "float32"},
                        "A": {"type": "random", "shape": [2, 2], "dtype": "float32"},
                    },
                }
            ],
        )
    )
    case = normalize_workloads(request)[0]
    assert list(case["inputs"]) == ["A", "B"]
    assert case["uuid"] == "BatchMatmul-001"
    assert request.seed == 200


def test_axes_defaults_override_and_optional():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "axes": {
                    "N": {"type": "var"},
                    "C": {"type": "const", "value": 2},
                    "M": {"type": "expr", "expression": "N * C + 1"},
                },
                "inputs": {"x": {"dtype": "float32", "shape": ["M"]}, "optional": {}},
                "reference": "def run(x, optional): return x",
            },
            workloads=[{"axes": {"N": 3}, "inputs": {}}],
        )
    )
    case = normalize_workloads(request)[0]
    assert case["axes"] == {"N": 3, "C": 2, "M": 7}
    assert case["inputs"]["x"]["shape"] == [7]
    assert case["inputs"]["optional"] == {"type": "scalar", "value": None}


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("true")',
        "2 ** 100000",
        "N + 1",
        "1 / 0",
        "2 **",
        "2147483647 * 2",
        "1 / 2",
    ],
)
def test_unsafe_or_invalid_axes_rejected(expression):
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(
                definition={
                    "axes": {"M": {"type": "expr", "expression": expression}},
                    "inputs": {"x": {}},
                    "reference": "def run(x): return x",
                }
            )
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "safetensors", "path": "/etc/passwd", "tensor_key": "x"},
        {"type": "random", "shape": [2], "dtype": "float64"},
        {"type": "random", "shape": [2], "dtype": "uint8"},
        {"type": "random", "shape": [-1], "dtype": "float32"},
        {"type": "random", "shape": [2], "dtype": "int8", "range": [0, 129]},
        {"type": "random", "shape": [2], "dtype": "int8", "range": [2, 2]},
    ],
)
def test_invalid_input_descriptors_rejected(spec):
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(body(workloads=[{"inputs": {"x": spec}}]))


def test_scalar_list_and_tensor_list_and_custom_axes():
    request = OperatorRequest.model_validate(
        body(
            definition={"inputs": {"x": {}, "ys": {}}, "reference": "def run(x, ys): return ys"},
            workloads=[
                {
                    "inputs": {
                        "x": {"type": "scalar", "value": [1, True, "x"]},
                        "ys": {"type": "tensor_list", "dtype": "float32", "shapes": [[2], [3, 4]]},
                    }
                }
            ],
        )
    )
    assert normalize_workloads(request)[0]["inputs"]["x"]["value"] == [1, True, "x"]
    custom = OperatorRequest.model_validate(
        body(
            custom_inputs_code='def make(axes, device): return {"x": None}',
            definition={
                "inputs": {"x": {}},
                "custom_inputs_entrypoint": "make",
                "reference": "def run(x): return x",
            },
            workloads=[{"axes": {"arbitrary": 7}, "inputs": {"x": {"type": "custom"}}}],
        )
    )
    assert normalize_workloads(custom)[0]["axes"] == {"arbitrary": 7}


@pytest.mark.parametrize(
    "updates",
    [
        {"torch_code": "def run(x): return x"},
        {"cases": [{"inputs": []}]},
        {"workloads": [{"inputs": {"x": {"type": "custom"}}}]},
        {
            "workloads": [
                {"inputs": {"x": {"type": "scalar", "value": 1}}, "tolerance": {"max_atol": 0.1}}
            ]
        },
    ],
)
def test_old_schema_and_unsupported_semantics_rejected(updates):
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(body(**updates))


def test_official_metadata_extra_inputs_and_float_range_are_ignored():
    # Official SwiGLU, Gather and Nonzero workloads carry these; upstream ignores them.
    request = OperatorRequest.model_validate(
        body(
            workloads=[
                {
                    "inputs": {
                        "x": {
                            "type": "random", "dtype": "float32", "shape": [2],
                            "range": [-10, 10],
                        },
                        "sparse_grad": {"type": "scalar", "value": True},
                    },
                    "perf_baseline": {"hardware": "Ascend910B3", "latency_ms": 0.2},
                }
            ]
        )
    )
    [case] = normalize_workloads(request)
    assert case["input_order"] == ["x"]
    assert "range" not in case["inputs"]["x"]


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_axis_chain_resolves_in_one_definition_order_pass(order):
    axes = {"a0": {"type": "const", "value": 1}}
    for i in range(1, 128) if order == "forward" else range(127, 0, -1):
        axes[f"a{i}"] = {"type": "expr", "expression": " + ".join(["1"] * 30) + f" + a{i - 1}"}
    body = {
        "definition": {
            "axes": axes,
            "inputs": {"x": {"shape": ["a127"], "dtype": "float32"}},
            "reference": "def run(x): return x",
        },
        "workloads": [{"inputs": {}}] * 128,
    }
    start = time.perf_counter()
    if order == "forward":
        cases = normalize_workloads(OperatorRequest.model_validate(body))
        assert cases[-1]["inputs"]["x"]["shape"] == [1 + 127 * 30]
    else:
        # Upstream resolves once in definition order, so a later axis is unknown.
        with pytest.raises(ValidationError, match="not resolved before it"):
            OperatorRequest.model_validate(body)
    # The old any-order loop re-parsed pending expressions: about a minute here.
    assert time.perf_counter() - start < 5


def test_single_implementation_modes():
    torch = OperatorRequest.model_validate(body())
    triton = OperatorRequest.model_validate(
        body(definition={"inputs": {"x": {}}}, triton_code="def run(x): return x")
    )
    comparison = OperatorRequest.model_validate(body(triton_code="def run(x): return x"))
    assert torch.checks == triton.checks == ["performance"]
    assert comparison.checks == ["correctness", "performance"]


def test_workload_overrides_and_distribution_controls():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "inputs": {"x": {"dtype": "int8", "shape": ["2"]}},
                "reference": "invalid source syntax left for worker",
            },
            workloads=[
                {
                    "inputs": {
                        "x": {
                            "type": "random",
                            "dtype": "float32",
                            "shape": [3],
                            "distribution": "normal",
                            "mu_low": 1.0,
                            "mu_high": 1.0,
                            "sigma_low": 0.5,
                            "sigma_high": 0.5,
                            "fill": "nan",
                        }
                    },
                    "tolerance": {"max_atol": None, "allow_negative_inf": False},
                }
            ],
        )
    )
    spec = normalize_workloads(request)[0]["inputs"]["x"]
    assert spec["dtype"] == "float32"
    assert spec["shape"] == [3]
    assert spec["fill"] == "nan"


def test_one_sided_distribution_bounds_rejected():
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(
                workloads=[
                    {
                        "inputs": {
                            "x": {
                                "type": "random", "dtype": "float32", "shape": [2],
                                "distribution": "uniform", "high": -6.0,
                            }
                        }
                    }
                ]
            )
        )


def test_zero_argument_compile_and_non_compute():
    request = OperatorRequest.model_validate(
        body(
            definition={"inputs": {}, "reference": "def run(): return 1"},
            workloads=[{"inputs": {}}],
            checks=["compile"],
            non_compute=True,
        )
    )
    assert normalize_workloads(request)[0]["inputs"] == {}
    assert normalize_workloads(request)[0]["input_order"] == []


def test_int64_range_torch_high_bound():
    valid = {"type": "random", "shape": [2], "dtype": "int64", "range": [-(2**63), 2**63 - 1]}
    OperatorRequest.model_validate(body(workloads=[{"inputs": {"x": valid}}]))
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(workloads=[{"inputs": {"x": dict(valid, range=[0, 2**63])}}])
        )


def test_axis_true_division_requires_integral_result():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "axes": {"N": {"type": "var"}, "M": {"type": "expr", "expression": "N / 2"}},
                "inputs": {"x": {"shape": ["M"], "dtype": "float32"}},
                "reference": "source",
            },
            workloads=[{"axes": {"N": 6}, "inputs": {}}],
        )
    )
    assert normalize_workloads(request)[0]["axes"]["M"] == 3
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(
                definition=request.definition.model_dump(),
                workloads=[{"axes": {"N": 5}, "inputs": {}}],
            )
        )


def test_axis_override_precedence_and_custom_parameters():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "axes": {
                    "C": {"type": "const", "value": 2},
                    "M": {"type": "expr", "expression": "C * 2"},
                },
                "inputs": {"x": {"shape": ["M"], "dtype": "float32"}},
                "reference": "source",
            },
            workloads=[{"axes": {"C": 3, "M": 99, "flag": 1}, "inputs": {}}],
        )
    )
    case = normalize_workloads(request)[0]
    assert case["axes"] == {"C": 3, "M": 6, "flag": 1}
    assert case["inputs"]["x"]["shape"] == [6]


@pytest.mark.parametrize("value", [[float("nan")], [{"x": float("inf")}], [object()]])
def test_scalar_nested_non_json_or_non_finite_rejected(value):
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(workloads=[{"inputs": {"x": {"type": "scalar", "value": value}}}])
        )


def test_unused_custom_code_rejected_and_metadata_entrypoint_allowed():
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(body(custom_inputs_code="source"))
    OperatorRequest.model_validate(
        body(
            definition={
                "inputs": {"x": {}},
                "reference": "source",
                "custom_inputs_entrypoint": "make",
            }
        )
    )


def test_official_custom_uses_reference_source_without_api_execution():
    request = OperatorRequest.model_validate(
        body(
            definition={
                "inputs": {"x": {}},
                "reference": 'raise RuntimeError("not executed at admission")',
                "custom_inputs_entrypoint": "make",
            },
            workloads=[{"inputs": {"x": {"type": "custom"}}}],
        )
    )
    assert normalize_workloads(request)[0]["inputs"]["x"] == {"type": "custom"}
    assert request.custom_inputs_code is None


def test_custom_triton_only_requires_explicit_factory_source():
    data = body(
        definition={"inputs": {"x": {}}, "custom_inputs_entrypoint": "make"},
        triton_code="source",
        workloads=[{"inputs": {"x": {"type": "custom"}}}],
    )
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(data)
    OperatorRequest.model_validate(dict(data, custom_inputs_code="source"))


def test_float64_fill_is_supported():
    request = OperatorRequest.model_validate(
        body(
            workloads=[
                {
                    "inputs": {
                        "x": {"type": "random", "dtype": "float64", "shape": [2], "fill": "zero"}
                    }
                }
            ]
        )
    )
    assert normalize_workloads(request)[0]["inputs"]["x"]["dtype"] == "float64"


def test_negative_workload_axes_and_file_entrypoint_rejected():
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(
                workloads=[{"axes": {"flag": -1}, "inputs": {"x": {"type": "scalar", "value": 1}}}]
            )
        )
    with pytest.raises(ValidationError):
        OperatorRequest.model_validate(
            body(
                definition={
                    "inputs": {"x": {}},
                    "reference": "source",
                    "custom_inputs_entrypoint": "inputs.py::make",
                }
            )
        )


def test_explicit_scalar_null_survives_normalization_and_generation():
    pytest.importorskip("torch")
    from triton_evaluation.runtime.input_generation import generate_inputs

    request = OperatorRequest.model_validate(
        body(workloads=[{"inputs": {"x": {"type": "scalar", "value": None}}}])
    )
    case = normalize_workloads(request)[0]
    assert case["inputs"]["x"] == {"type": "scalar", "value": None}
    assert generate_inputs(case, device="cpu") == [None]
