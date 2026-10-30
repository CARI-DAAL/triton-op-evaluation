"""Operator input: the official definition/workloads submission and its translation.

One module owns the public request models, case materialization and the
durable engine request built from them, without importing Torch or running
caller code. ``to_evaluation_request`` is the only translation; it rejects
invalid input with a ValueError whose message is safe to return to the caller.

The data models are adapted from Ascend NPUKernelBench (Apache-2.0), commit
0607a29d231cde384c9dfdb8a39f54f91837c327. See runtime/INPUT_GENERATION_LICENSE.
"""

import ast
import json
import math
import operator
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from .api.contract import (
    MAX_SOURCE_BYTES,
    Check,
    CorrectnessParams,
    EvaluationRequest,
    PerformanceParams,
    ProfileParams,
    SourceCode,
    StrictModel,
)

Dtypes = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1fn_x2",
    "int8",
    "int16",
    "int32",
    "int64",
    "bool",
]
Dim = Annotated[int, Field(ge=0, le=2**31 - 1)]
Shape = Annotated[list[Dim], Field(max_length=32)]
Name = Annotated[str, Field(min_length=1, max_length=256)]


class Axis(StrictModel):
    type: Literal["const", "var", "expr"]
    value: Dim | None = None
    expression: str | None = Field(default=None, max_length=1024)
    description: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.type == "const" and self.value is None:
            raise ValueError("constant axis requires value")
        if self.type == "expr" and not self.expression:
            raise ValueError("expression axis requires expression")
        return self


class TensorSpec(StrictModel):
    shape: list[Name] | None = Field(default=None, max_length=32)
    dtype: Dtypes | None = None
    description: str | None = None


class Definition(StrictModel):
    name: Name = "operator"
    op_type: str | None = None
    axes: dict[Name, Axis] = Field(default_factory=dict, max_length=128)
    inputs: dict[Name, TensorSpec] = Field(max_length=128)
    outputs: dict[Name, TensorSpec] = Field(default_factory=dict)
    reference: SourceCode | None = None
    reference_hash: str = ""
    custom_inputs_entrypoint: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if set(self.inputs) & set(self.outputs):
            raise ValueError("input and output names must not overlap")
        for spec in [*self.inputs.values(), *self.outputs.values()]:
            for axis in spec.shape or []:
                if not axis.isdigit() and axis not in self.axes:
                    raise ValueError(f"undefined shape axis: {axis}")
        if self.custom_inputs_entrypoint and not re.fullmatch(
            r"[A-Za-z_]\w*", self.custom_inputs_entrypoint
        ):
            raise ValueError("custom input entrypoint must be a function identifier")
        return self


class RandomInput(StrictModel):
    type: Literal["random"] = "random"
    dtype: Dtypes | None = None
    shape: Shape | None = None
    # Upstream reads range only for integer dtypes; length is checked there.
    range: Annotated[list[int], Field(max_length=32)] | None = None
    distribution: (
        Literal["mixed", "randn", "uniform", "normal", "zeros", "ones", "uniform_signed"] | None
    ) = None
    low: float | None = None
    high: float | None = None
    mu_low: float | None = None
    mu_high: float | None = None
    sigma_low: float | None = None
    sigma_high: float | None = None
    fill: Literal["inf", "-inf", "nan", "zero"] | None = None


class ScalarInput(StrictModel):
    type: Literal["scalar"] = "scalar"
    value: int | float | bool | str | list | None

    @model_validator(mode="after")
    def finite_json(self):
        remaining = 10000

        def check(value, depth=0):
            nonlocal remaining
            remaining -= 1
            if remaining < 0 or depth > 32:
                raise ValueError("scalar JSON exceeds size or nesting limits")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("scalar values must be finite JSON")
            if isinstance(value, list):
                for item in value:
                    check(item, depth + 1)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError("scalar JSON object keys must be strings")
                    check(item, depth + 1)
            elif value is not None and type(value) not in (int, float, bool, str):
                raise ValueError("scalar values must contain only JSON values")

        check(self.value)
        return self


class TensorListInput(StrictModel):
    type: Literal["tensor_list"] = "tensor_list"
    dtype: Dtypes
    shapes: list[Shape] = Field(max_length=128)


class CustomInput(StrictModel):
    type: Literal["custom"] = "custom"


InputSpec = Annotated[
    RandomInput | ScalarInput | TensorListInput | CustomInput, Field(discriminator="type")
]


class Tolerance(StrictModel):
    """Official per-workload tolerance; a declared workload is judged by the
    pinned official check (engine/_official_tolerance.py), unset fields by
    its dtype standard. Like upstream, values are not range-checked."""

    max_atol: float | None = None
    max_rtol: float | None = None
    required_matched_ratio: float | None = None
    max_error_cap: float | None = None
    allow_negative_inf: bool = False


class Workload(StrictModel):
    uuid: Name | None = None
    axes: dict[Name, Dim] | None = Field(default=None, max_length=128)
    inputs: dict[Name, InputSpec] = Field(max_length=128)
    tolerance: Tolerance = Field(default_factory=Tolerance)
    # Official workloads carry measured upstream latencies; like upstream's
    # generator, input materialization ignores them.
    perf_baseline: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_external_resources(cls, value):
        if isinstance(value, dict):
            inputs = value.get("inputs")
            if isinstance(inputs, dict) and any(
                isinstance(spec, dict) and spec.get("type") == "safetensors"
                for spec in inputs.values()
            ):
                raise ValueError("safetensors inputs are unsupported; local paths are never read")
        return value


@lru_cache(maxsize=1024)
def _parse_axis_expression(expression: str) -> tuple[ast.expr, frozenset[str]]:
    """Parse and bound one expression once; every workload of a definition reuses it."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid axis expression syntax") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 128:
        raise ValueError("axis expression is too complex")
    return tree.body, frozenset(node.id for node in nodes if isinstance(node, ast.Name))


def resolve_axes(definition: Definition, values: dict[str, int] | None) -> dict[str, int]:
    """Upstream resolve_shape_expression semantics plus resource bounds.

    Like upstream get_resolved_axes_values, expressions are evaluated once in
    definition order with the full operator set (including any ``**``). Only
    integers wider than 1024 bits are refused, so huge powers cannot exhaust
    the API; the resolved value must still be an integral dimension.
    """
    resolved = {name: axis.value for name, axis in definition.axes.items() if axis.type == "const"}
    resolved.update(values or {})
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = node.value
        elif isinstance(node, ast.Name):
            value = resolved[node.id]
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = evaluate(node.left), evaluate(node.right)
            if (
                isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int)
                and abs(left) > 1 and (abs(left).bit_length() - 1) * right >= 1024
            ):
                raise ValueError("axis arithmetic exceeds the 1024-bit integer limit")
            value = operations[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("unsupported axis arithmetic (calls are forbidden)")
        if isinstance(value, complex):
            raise ValueError("axis arithmetic must stay real")
        if isinstance(value, int) and value.bit_length() > 1024:
            raise ValueError("axis arithmetic exceeds the 1024-bit integer limit")
        return value

    for name, axis in definition.axes.items():
        if axis.type != "expr":
            continue
        body, names = _parse_axis_expression(axis.expression)
        missing = sorted(names - resolved.keys())
        if missing:
            raise ValueError(f"axis {name} uses {missing[0]}, which is not resolved before it")
        try:
            value = evaluate(body)
        except (ZeroDivisionError, OverflowError) as error:
            raise ValueError("invalid axis arithmetic") from error
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("axis expression must resolve to an integer")
            value = int(value)
        if abs(value) > 2**31 - 1:
            raise ValueError("axis arithmetic exceeds dimension limit")
        resolved[name] = int(value)
    return resolved


def normalize_workloads(request: "OperatorRequest") -> list[dict[str, Any]]:
    cases = []
    for index, workload in enumerate(request.workloads):
        axes = resolve_axes(request.definition, workload.axes)
        # Like upstream gen_inputs, only definition inputs are materialized;
        # extra workload entries (Gather's sparse_grad) are ignored.
        custom =[name for name, spec in workload.inputs.items() if isinstance(spec, CustomInput)]
        if custom and len(custom) != len(workload.inputs):
            raise ValueError("cannot mix custom and non-custom inputs")
        if custom and not request.definition.custom_inputs_entrypoint:
            raise ValueError("custom inputs require custom_inputs_entrypoint")
        if custom and not (request.custom_inputs_code or request.definition.reference):
            raise ValueError(
                "custom inputs require reference or explicitly supplied custom_inputs_code"
            )
        inputs = {}
        for name, default in request.definition.inputs.items():
            spec = workload.inputs.get(name)
            if spec is None and default.shape is None and default.dtype is None:
                inputs[name] = {"type": "scalar", "value": None}
                continue
            if spec is None:
                spec = RandomInput()
            data = spec.model_dump(exclude_none=True)
            if isinstance(spec, ScalarInput):
                data["value"] = spec.value
            if isinstance(spec, RandomInput):
                dtype = spec.dtype or default.dtype
                if dtype is None:
                    raise ValueError(f"input {name} has no dtype")
                if dtype == "float64" and spec.fill is None:
                    raise ValueError("official random generation does not support float64")
                shape = spec.shape
                if shape is None and default.shape is not None:
                    try:
                        shape = [
                            int(axis) if axis.isdigit() else axes[axis] for axis in default.shape
                        ]
                    except KeyError as error:
                        raise ValueError(f"missing shape axis {error.args[0]}") from error
                if shape is None:
                    raise ValueError(f"input {name} has no shape")
                if any(dim < 0 or dim > 2**31 - 1 for dim in shape):
                    raise ValueError("invalid resolved shape")
                data.update(dtype=dtype, shape=shape)
                if spec.range is not None:
                    limits = {
                        "int8": (-128, 128),
                        "int16": (-32768, 32768),
                        "int32": (-(2**31), 2**31),
                        "int64": (-(2**63), 2**63 - 1),
                    }
                    if dtype not in limits:
                        # Upstream applies range only to integer dtypes.
                        data.pop("range")
                    elif spec.fill is None:
                        # torch.randint refuses these for every seed; a fill ignores range.
                        if len(spec.range) != 2:
                            raise ValueError("integer range must be [low, high]")
                        low, high = spec.range
                        minimum, maximum = limits[dtype]
                        if not minimum <= low < high <= maximum:
                            raise ValueError(
                                "integer range must satisfy dtype bounds and low < high"
                            )
                # Reject only bounds that torch refuses for every seed: an explicit
                # float distribution always reads them. The case-level family and
                # "mixed" pick bounds by RNG; fills and other dtypes ignore them.
                if dtype in ("float32", "float16", "bfloat16") and spec.fill is None:
                    bounds = {
                        "uniform": [("low", "high", -5.0, 5.0)],
                        "uniform_signed": [("low", "high", 0.3, 3.0)],
                        "normal": [
                            ("mu_low", "mu_high", -5.0, 5.0),
                            ("sigma_low", "sigma_high", 0.1, 2.0),
                        ],
                    }.get(spec.distribution, [])
                    for low_key, high_key, low_default, high_default in bounds:
                        low, high = getattr(spec, low_key), getattr(spec, high_key)
                        low = low_default if low is None else low
                        high = high_default if high is None else high
                        if low > high:
                            raise ValueError(f"{low_key} must not exceed {high_key}")
                    if spec.distribution == "normal" and (spec.sigma_high or 0.0) < 0:
                        raise ValueError("normal sigma_high must not be negative")
            if isinstance(spec, TensorListInput) and spec.dtype == "float64":
                raise ValueError("official random generation does not support float64")
            inputs[name] = data
        case = {
            "uuid": workload.uuid or f"case-{index + 1}",
            "axes": axes,
            "input_order": list(inputs),
            "inputs": inputs,
        }
        # Only a declared override changes the verdict policy; defaults keep
        # the service's own accuracy check and the existing reference identity.
        declared = {
            key: value
            for key, value in workload.tolerance.model_dump().items()
            if value is not None and value is not False
        }
        if declared:
            case["tolerance"] = declared
        cases.append(case)
    return cases


class OperatorRequest(StrictModel):
    definition: Definition
    workloads: list[Workload] = Field(min_length=1, max_length=128)
    triton_code: SourceCode | None = None
    custom_inputs_code: SourceCode | None = None
    seed: int = Field(default=200, ge=0, le=2**32 - 1)
    checks: list[Check] = Field(
        default_factory=lambda data: (
            ["correctness", "performance"]
            if data.get("definition")
            and data["definition"].reference is not None
            and data.get("triton_code") is not None
            else ["performance"]
        ),
        min_length=1,
        max_length=4,
    )
    deadline_seconds: int = Field(default=3600, ge=10, le=3600)
    reference_device_policy: Literal["cpu-first", "npu-strict"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    correctness: CorrectnessParams | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    correctness_from: str | None = Field(
        default=None, pattern=r"^ev_[0-9a-f]{32}$", exclude_if=lambda value: value is None,
    )
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    non_compute: bool = False


_REFERENCE_MODEL = """class Model(_operator_torch.nn.Module):
    def forward(self, *args):
        return run(*args)


"""
# One stderr line naming the workload the pinned generator refused; evaluators
# turn it into a non-retryable request error instead of a card fault.
INPUT_GENERATION_FAILURE_PREFIX = "TRITON_EVAL_INPUT_GENERATION_FAILED "
_INPUT_SUPPORT = """
import collections.abc as _operator_collections
import json as _operator_json
import sys as _operator_sys
import torch as _operator_torch


def get_init_inputs():
    return []


class _OperatorInputGroups(_operator_collections.Sequence):
    @property
    def _inputs_are_fresh(self):
        # Built-in generation reseeds and allocates tensors on each access.
        # A custom callback can retain storage in imported module state.
        return all(
            spec["type"] != "custom"
            for case in _operator_cases for spec in case["inputs"].values()
        )

    def __len__(self):
        return len(_operator_cases)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if type(index) is not int:
            raise TypeError("case index must be an integer")
        case = _operator_cases[index]
        try:
            return _operator_generate_inputs(
                case, seed=_operator_seed, device="npu",
                custom_inputs_code=_operator_custom_code,
                custom_inputs_entrypoint=_operator_custom_entrypoint,
            )
        except _operator_input_error as error:
            failure = {
                "case_idx": range(len(self))[index] + 1,
                "uuid": case["uuid"],
                "message": str(error),
            }
            _operator_sys.stderr.write(""" + repr(INPUT_GENERATION_FAILURE_PREFIX) + """
                                       + _operator_json.dumps(failure) + "\\n")
            _operator_sys.stderr.flush()
            raise

    def __iter__(self):
        # Sequence.__iter__ keeps the previous workload bound while generating
        # the next; yield unbound so consumers hold one device workload at a time.
        for index in range(len(self)):
            yield self[index]


def get_input_groups():
    return _OperatorInputGroups()


def get_inputs():
    return get_input_groups()[0]


def get_case_tolerances():
    return [case.get("tolerance") for case in _operator_cases]
"""
_REFERENCE_SUPPORT = _INPUT_SUPPORT + "\n" + _REFERENCE_MODEL

_CANDIDATE_SUPPORT = '''
import torch as _operator_torch

class ModelNew(_operator_torch.nn.Module):
    def forward(self, *args):
        return run(*args)
'''


def _check_operator_input(request: OperatorRequest, correctness: CorrectnessParams | None):
    """Rules only the public input can break; check coupling is the durable model's."""
    sources = (request.definition.reference, request.triton_code)
    if all(source is None for source in sources):
        raise ValueError("at least one reference or triton_code is required")
    if any(
        source is not None and not source.strip()
        for source in [*sources, request.custom_inputs_code]
    ):
        raise ValueError("provided source must not be empty")
    indices = correctness.case_indices if correctness is not None else None
    if indices is not None and any(index >= len(request.workloads) for index in indices):
        raise ValueError("correctness case index out of range")
    if request.custom_inputs_code is not None and not any(
        isinstance(spec, CustomInput)
        for workload in request.workloads
        for spec in workload.inputs.values()
    ):
        raise ValueError("custom_inputs_code requires at least one custom workload input")


def to_evaluation_request(
    request: OperatorRequest, checks: list[Check] | None = None,
) -> EvaluationRequest:
    """Translate operator input to the durable engine contract using only strings.

    ``checks`` replaces the requested checks (the /verify, /benchmark and
    /profile intents) and drops the parameters of checks it does not select.
    Every rule runs once, against that effective request.
    """
    selected = request.checks if checks is None else checks
    correctness = request.correctness if checks is None or "correctness" in selected else None
    _check_operator_input(request, correctness)
    workloads = normalize_workloads(request)
    cases = json.dumps(
        workloads, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    implementation = (
        "triton" if request.definition.reference is None
        else "torch" if request.triton_code is None else "comparison"
    )
    reference_support = _INPUT_SUPPORT if implementation == "triton" else _REFERENCE_SUPPORT
    generation_source = Path(__file__).parent.joinpath(
        "runtime/input_generation.py"
    ).read_text(encoding="utf-8")
    # Only custom workloads read the callback source; embedding the reference
    # a second time otherwise halves the usable reference size.
    uses_custom = any(
        spec["type"] == "custom" for case in workloads for spec in case["inputs"].values()
    )
    custom_code = (
        None if not uses_custom
        else request.custom_inputs_code if request.custom_inputs_code is not None
        else request.definition.reference
    )
    reference_code = (
        (request.definition.reference or "") + "\n"
        + "_operator_generator_namespace = {'__name__': '_operator_input_generation'}\n"
        + "exec(" + repr(generation_source) + ", _operator_generator_namespace)\n"
        + "_operator_generate_inputs = _operator_generator_namespace['generate_inputs']\n"
        + "_operator_input_error = _operator_generator_namespace['InputGenerationError']\n"
        + reference_support
        + "\n_operator_cases = _operator_json.loads(" + repr(cases) + ")\n"
        + "_operator_seed = " + repr(request.seed) + "\n"
        + "_operator_custom_code = " + repr(custom_code) + "\n"
        + "_operator_custom_entrypoint = "
        + repr(request.definition.custom_inputs_entrypoint) + "\n"
    )
    kernel_code = (
        request.triton_code + "\n" + _CANDIDATE_SUPPORT
        if request.triton_code is not None else ""
    )
    if any(len(source.encode("utf-8")) > MAX_SOURCE_BYTES
           for source in (reference_code, kernel_code)):
        raise ValueError("generated operator source exceeds the 2 MiB engine limit")
    try:
        return EvaluationRequest(
            schema_version="2.1", purpose="trial", op_name="operator",
            implementation=implementation,
            reference_code=reference_code, kernel_code=kernel_code,
            triton_impl_name="triton_ascend_impl", checks=selected,
            reference_device_policy=request.reference_device_policy,
            deadline_seconds=request.deadline_seconds, non_compute=request.non_compute,
            correctness_from=request.correctness_from,
            correctness=correctness,
            performance=(
                request.performance if checks is None or "performance" in selected else None
            ),
            profile=request.profile if checks is None or "profile" in selected else None,
        )
    except ValidationError as error:
        # Report the coupling rule, never pydantic's echo of the generated source.
        detail = error.errors(include_url=False)[0]
        raise ValueError(str(detail.get("ctx", {}).get("error", detail["msg"]))) from None
