"""Official named workload inputs, validated without importing Torch or running code.

Adapted from Ascend NPUKernelBench data models (Apache-2.0), commit
0607a29d231cde384c9dfdb8a39f54f91837c327. See runtime/INPUT_GENERATION_LICENSE.
"""

import ast
import math
import operator
import re
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .contract import Check, PerformanceParams, ProfileParams, SourceCode, StrictModel

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
    range: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None
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
    max_atol: float | None = None
    max_rtol: float | None = None
    required_matched_ratio: float | None = None
    max_error_cap: float | None = None
    allow_negative_inf: bool = False

    @model_validator(mode="after")
    def valid(self):
        if (
            any(
                value is not None
                for key, value in self.model_dump().items()
                if key != "allow_negative_inf"
            )
            or self.allow_negative_inf
        ):
            raise ValueError("workload tolerance overrides are not supported")
        return self


class Workload(StrictModel):
    uuid: Name | None = None
    axes: dict[Name, Dim] | None = Field(default=None, max_length=128)
    inputs: dict[Name, InputSpec] = Field(max_length=128)
    tolerance: Tolerance = Field(default_factory=Tolerance)

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


def resolve_axes(definition: Definition, values: dict[str, int] | None) -> dict[str, int]:
    resolved = {name: axis.value for name, axis in definition.axes.items() if axis.type == "const"}
    resolved.update(values or {})
    pending = {
        name: axis.expression for name, axis in definition.axes.items() if axis.type == "expr"
    }
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.Name):
            value = resolved[node.id]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = evaluate(node.left), evaluate(node.right)
            if type(exponent) is not int or not 0 <= exponent <= 31:
                raise ValueError("axis exponent must be an integer between zero and 31")
            value = base**exponent
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("unsupported axis arithmetic (calls are forbidden)")
        if abs(value) > 2**31 - 1:
            raise ValueError("axis arithmetic exceeds dimension limit")
        return value

    while pending:
        progressed = False
        for name, expression in list(pending.items()):
            try:
                tree = ast.parse(expression, mode="eval")
            except SyntaxError as error:
                raise ValueError("invalid axis expression syntax") from error
            if sum(1 for _ in ast.walk(tree)) > 128:
                raise ValueError("axis expression is too complex")
            try:
                value = evaluate(tree.body)
            except KeyError:
                continue
            except (SyntaxError, ZeroDivisionError, OverflowError) as error:
                raise ValueError("invalid axis arithmetic") from error
            if int(value) != value:
                raise ValueError("axis expression must resolve to an integer")
            resolved[name] = int(value)
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError("unresolved or cyclic axis expression")
    return resolved


def normalize_workloads(request: "OperatorRequest") -> list[dict[str, Any]]:
    cases = []
    for index, workload in enumerate(request.workloads):
        axes = resolve_axes(request.definition, workload.axes)
        unknown = set(workload.inputs) - set(request.definition.inputs)
        if unknown:
            raise ValueError(f"unknown workload inputs: {sorted(unknown)}")
        custom = [name for name, spec in workload.inputs.items() if isinstance(spec, CustomInput)]
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
                        raise ValueError("range is supported only for integer random inputs")
                    low, high = spec.range
                    minimum, maximum = limits[dtype]
                    if not minimum <= low < high <= maximum:
                        raise ValueError("integer range must satisfy dtype bounds and low < high")
                uniform_defaults = (
                    (0.3, 3.0) if spec.distribution == "uniform_signed" else (-5.0, 5.0)
                )
                for low_key, high_key, defaults in [
                    ("low", "high", uniform_defaults),
                    ("mu_low", "mu_high", (-5.0, 5.0)),
                    ("sigma_low", "sigma_high", (0.1, 2.0)),
                ]:
                    given_low, given_high = getattr(spec, low_key), getattr(spec, high_key)
                    low = given_low if given_low is not None else defaults[0]
                    high = given_high if given_high is not None else defaults[1]
                    if low > high:
                        raise ValueError(f"{low_key} must not exceed {high_key}")
                if any(
                    value is not None and value <= 0 for value in [spec.sigma_low, spec.sigma_high]
                ):
                    raise ValueError("normal sigma bounds must be positive")
            if isinstance(spec, TensorListInput) and spec.dtype == "float64":
                raise ValueError("official random generation does not support float64")
            inputs[name] = data
        cases.append(
            {
                "uuid": workload.uuid or f"case-{index + 1}",
                "axes": axes,
                "input_order": list(inputs),
                "inputs": inputs,
            }
        )
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
    deadline_seconds: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    non_compute: bool = False

    @model_validator(mode="after")
    def valid(self):
        sources = (self.definition.reference, self.triton_code)
        if all(source is None for source in sources):
            raise ValueError("at least one reference or triton_code is required")
        if any(
            source is not None and not source.strip()
            for source in [*sources, self.custom_inputs_code]
        ):
            raise ValueError("provided source must not be empty")
        if None in sources and "correctness" in self.checks:
            raise ValueError("correctness requires both implementations")
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must not contain duplicates")
        self.checks.sort(key=("compile", "correctness", "performance", "profile").index)
        if self.custom_inputs_code is not None and not any(
            isinstance(spec, CustomInput)
            for workload in self.workloads
            for spec in workload.inputs.values()
        ):
            raise ValueError("custom_inputs_code requires at least one custom workload input")
        normalize_workloads(self)
        return self
