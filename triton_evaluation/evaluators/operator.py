"""Generate self-contained engine modules without evaluating caller source."""

import json

from ..api.contract import MAX_SOURCE_BYTES, Check, EvaluationRequest
from ..api.operator import OperatorRequest

_REFERENCE_SUPPORT = '''
import collections.abc as _operator_collections
import json as _operator_json
import torch as _operator_torch

class Model(_operator_torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def get_init_inputs():
    return []


def _operator_integer(shape, dtype, low, high, generator):
    if dtype == _operator_torch.int64 and high == 2**63:
        # randint's Python high argument cannot represent the exclusive int64
        # upper bound. Build all 64 bits, rejecting values below low.
        def draw(size):
            magnitude = _operator_torch.empty(
                size, dtype=dtype, device="cpu"
            ).random_(generator=generator)
            sign = _operator_torch.randint(
                0, 2, size, dtype=dtype, device="cpu", generator=generator
            )
            return magnitude | (sign << 63)
        if low >= 0:
            return _operator_torch.randint(
                low - 1, high - 1, shape, dtype=dtype,
                device="cpu", generator=generator,
            ) + 1
        result = draw(shape)
        invalid = result < low
        while invalid.any():
            result[invalid] = draw((int(invalid.sum()),))
            invalid = result < low
        return result
    return _operator_torch.randint(
        low, high, shape, dtype=dtype, device="cpu", generator=generator,
    )


def _operator_tensor(spec, generator):
    shape = spec["shape"]
    dtype = getattr(_operator_torch, spec["dtype"])
    limits = spec["range"]
    if dtype == _operator_torch.bool:
        return _operator_torch.randint(
            0, 2, shape, dtype=dtype, device="cpu", generator=generator,
        )
    if not dtype.is_floating_point:
        low, high = limits if limits is not None else (
            (0, 5) if dtype == _operator_torch.uint8 else (-5, 5)
        )
        return _operator_integer(shape, dtype, low, high, generator)
    if limits is None:
        return _operator_torch.randn(
            shape, dtype=dtype, device="cpu", generator=generator,
        )
    low, high = limits
    # Calculate in float64, then keep the final dtype's values inside [low, high).
    uniform = _operator_torch.rand(
        shape, dtype=_operator_torch.float64, device="cpu", generator=generator,
    )
    result = (low * (1 - uniform) + high * uniform).to(dtype)
    lower = _operator_torch.tensor(low, dtype=dtype, device="cpu")
    upper = _operator_torch.tensor(high, dtype=dtype, device="cpu")
    infinity = _operator_torch.tensor(float("inf"), dtype=dtype, device="cpu")
    if float(lower) < low:
        lower = _operator_torch.nextafter(lower, infinity)
    if float(upper) >= high:
        upper = _operator_torch.nextafter(upper, -infinity)
    if not _operator_torch.isfinite(lower) or not _operator_torch.isfinite(upper) or lower > upper:
        raise ValueError("range contains no finite values of the requested dtype")
    return result.clamp(min=lower, max=upper)


class _OperatorInputGroups(_operator_collections.Sequence):
    def __len__(self):
        return len(_operator_cases)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if type(index) is not int:
            raise TypeError("case index must be an integer")
        case = _operator_cases[index]
        generator = _operator_torch.Generator(device="cpu")
        generator.manual_seed(case["seed"])
        return [
            _operator_tensor(value, generator) if isinstance(value, dict) else value
            for value in case["inputs"]
        ]


def get_input_groups():
    return _OperatorInputGroups()


def get_inputs():
    return get_input_groups()[0]
'''

_CANDIDATE_SUPPORT = '''
import torch as _operator_torch

class ModelNew(_operator_torch.nn.Module):
    def forward(self, *args):
        return run(*args)
'''


def to_evaluation_request(
    request: OperatorRequest, checks: list[Check] | None = None,
) -> EvaluationRequest:
    """Translate operator input to the durable engine contract using only strings."""
    cases = json.dumps(
        [case.model_dump(mode="json") for case in request.cases],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    reference_code = (
        request.torch_code + "\n" + _REFERENCE_SUPPORT
        + "\n_operator_cases = _operator_json.loads(" + repr(cases) + ")\n"
    )
    kernel_code = request.triton_code + "\n" + _CANDIDATE_SUPPORT
    if any(len(source.encode("utf-8")) > MAX_SOURCE_BYTES
           for source in (reference_code, kernel_code)):
        raise ValueError("generated operator source exceeds the 2 MiB engine limit")
    selected = request.checks if checks is None else checks
    return EvaluationRequest(
        schema_version="2.1", purpose="trial", op_name="operator",
        reference_code=reference_code, kernel_code=kernel_code,
        triton_impl_name="triton_ascend_impl", checks=selected,
        deadline_seconds=request.deadline_seconds, non_compute=request.non_compute,
        performance=(
            request.performance if checks is None or "performance" in selected else None
        ),
        profile=request.profile if checks is None or "profile" in selected else None,
    )
