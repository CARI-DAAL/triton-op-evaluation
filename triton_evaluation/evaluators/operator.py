"""Generate self-contained engine modules without evaluating caller source."""

import json
from pathlib import Path

from ..api.contract import MAX_SOURCE_BYTES, Check, EvaluationRequest
from ..api.operator import OperatorRequest, normalize_workloads

_REFERENCE_MODEL = """class Model(_operator_torch.nn.Module):
    def forward(self, *args):
        return run(*args)


"""
_INPUT_SUPPORT = """
import collections.abc as _operator_collections
import json as _operator_json
import torch as _operator_torch


def get_init_inputs():
    return []


class _OperatorInputGroups(_operator_collections.Sequence):
    def __len__(self):
        return len(_operator_cases)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if type(index) is not int:
            raise TypeError("case index must be an integer")
        return _operator_generate_inputs(
            _operator_cases[index], seed=_operator_seed, device="npu",
            custom_inputs_code=_operator_custom_code,
            custom_inputs_entrypoint=_operator_custom_entrypoint,
        )

    def __iter__(self):
        # Sequence.__iter__ keeps the previous workload bound while generating
        # the next; yield unbound so consumers hold one device workload at a time.
        for index in range(len(self)):
            yield self[index]


def get_input_groups():
    return _OperatorInputGroups()


def get_inputs():
    return get_input_groups()[0]
"""
_REFERENCE_SUPPORT = _INPUT_SUPPORT + "\n" + _REFERENCE_MODEL

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
        normalize_workloads(request),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    implementation = (
        "triton" if request.definition.reference is None
        else "torch" if request.triton_code is None else "comparison"
    )
    reference_support = _INPUT_SUPPORT if implementation == "triton" else _REFERENCE_SUPPORT
    generation_source = Path(__file__).parents[1].joinpath(
        "runtime/input_generation.py"
    ).read_text(encoding="utf-8")
    custom_code = (
        request.custom_inputs_code if request.custom_inputs_code is not None
        else request.definition.reference
    )
    reference_code = (
        (request.definition.reference or "") + "\n"
        + "_operator_generator_namespace = {'__name__': '_operator_input_generation'}\n"
        + "exec(" + repr(generation_source) + ", _operator_generator_namespace)\n"
        + "_operator_generate_inputs = _operator_generator_namespace['generate_inputs']\n"
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
    selected = request.checks if checks is None else checks
    return EvaluationRequest(
        schema_version="2.1", purpose="trial", op_name="operator",
        implementation=implementation,
        reference_code=reference_code, kernel_code=kernel_code,
        triton_impl_name="triton_ascend_impl", checks=selected,
        deadline_seconds=request.deadline_seconds, non_compute=request.non_compute,
        performance=(
            request.performance if checks is None or "performance" in selected else None
        ),
        profile=request.profile if checks is None or "profile" in selected else None,
    )
