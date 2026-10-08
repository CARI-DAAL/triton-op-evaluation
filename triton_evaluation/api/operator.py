"""Minimal operator inputs, independent of catalogs and engine module conventions."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contract import (
    Check,
    PerformanceParams,
    ProfileParams,
    SourceCode,
    StrictModel,
)

TensorDtype = Literal[
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"
]
Scalar = int | float | bool | str | None
_INTEGER_LIMITS = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 255),
}


class TensorInput(StrictModel):
    shape: list[Annotated[int, Field(ge=0)]] = Field(max_length=32)
    dtype: TensorDtype = "float32"
    range: list[int | float] | None = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _valid_range(self):
        if self.range is None:
            return self
        if self.dtype == "bool":
            raise ValueError("bool tensors do not support range")
        low, high = self.range
        if low >= high:
            raise ValueError("range must satisfy low < high")
        if self.dtype in _INTEGER_LIMITS:
            if any(value != int(value) for value in self.range):
                raise ValueError("integer tensor range endpoints must be integral")
            minimum, maximum = _INTEGER_LIMITS[self.dtype]
            if low < minimum or high > maximum + 1:
                raise ValueError("range exceeds the tensor dtype")
            self.range = [int(low), int(high)]
        return self


class Case(StrictModel):
    inputs: list[TensorInput | Scalar]
    seed: int = Field(default=0, ge=-(2**63), le=2**64 - 1)


class OperatorRequest(StrictModel):
    torch_code: SourceCode | None = None
    triton_code: SourceCode | None = None
    cases: list[Case] = Field(min_length=1, max_length=128)
    checks: list[Check] = Field(
        default_factory=lambda data: (
            ["correctness", "performance"]
            if data.get("torch_code") is not None and data.get("triton_code") is not None
            else ["performance"]
        ),
        min_length=1, max_length=4
    )
    deadline_seconds: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    non_compute: bool = False

    @model_validator(mode="after")
    def _normalize_checks(self):
        codes = (self.torch_code, self.triton_code)
        if all(code is None for code in codes):
            raise ValueError("at least one of torch_code or triton_code is required")
        if any(code is not None and not code.strip() for code in codes):
            raise ValueError("provided implementation source must not be empty or whitespace")
        if None in codes and "correctness" in self.checks:
            raise ValueError("the correctness check requires both implementations")
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must not contain duplicates")
        order = ("compile", "correctness", "performance", "profile")
        self.checks = sorted(self.checks, key=order.index)
        return self
