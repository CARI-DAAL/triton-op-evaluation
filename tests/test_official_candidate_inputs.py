"""Candidate calls retain official omitted/default values across every engine path."""

import json
import types

import pytest
import torch

from triton_evaluation.engine import benchmark, verify
from triton_evaluation.engine._common_utils import invoke_model, prepare_candidate_inputs
from triton_evaluation.official_task import compose_reference_source
from triton_evaluation.reference_cache import value_sha256
from triton_evaluation.reference_diagnostics import _read_trace


@pytest.fixture
def reference():
    definition = {
        "name": "Defaults",
        "inputs": {name: {} for name in ("x", "offsets", "flag", "scale", "mode")},
        "reference": (
            "import torch\n"
            "def run(x, offsets=None, flag=True, scale=7, *, mode='official'):\n"
            "    return x * scale + (1 if flag else 0) + (0 if offsets is None else offsets)\n"
        ),
    }
    workload = {"inputs": {"x": {"type": "random", "shape": [2], "dtype": "float32"}}}
    source = compose_reference_source(
        json.dumps(definition).encode(),
        json.dumps(workload).encode(),
        problem_id="official_l3_Defaults",
    )
    module = types.ModuleType("official_candidate_defaults")
    exec(compile(source, "official_candidate_defaults.py", "exec"), module.__dict__)
    return module


class Candidate(torch.nn.Module):
    def forward(self, x, offsets=None, flag=False, scale=99, *, mode="candidate"):
        # Deliberately different candidate defaults prove the reference owns
        # omission semantics, including non-None and keyword-only defaults.
        assert offsets is None and flag is True and scale == 7 and mode == "official"
        return x * scale + 1


def test_exact_reference_defaults_replace_only_its_own_sentinel(reference):
    raw = reference.get_input_groups()[0]
    before = value_sha256(raw, input_value=True)
    trace = {}
    prepared = prepare_candidate_inputs(reference.Model, raw, trace=trace)
    assert prepared[0] is raw[0]
    assert prepared[1:] == [None, True, 7, "official"]
    assert raw[1] is reference._ABSENT
    assert value_sha256(raw, input_value=True) == before
    assert trace["raw_input_sha256"] == before
    assert trace["candidate_input_sha256"] != before
    assert trace["candidate_defaulted_parameters"] == ["offsets", "flag", "scale", "mode"]
    assert torch.equal(invoke_model(Candidate(), prepared), reference.Model()(*raw))


def test_explicit_none_false_zero_and_strings_do_not_become_defaults(reference):
    raw = [torch.ones(2), None, False, 0, "explicit"]
    prepared = prepare_candidate_inputs(reference.Model(), raw)
    assert prepared[1:] == raw[1:]


def test_other_plain_object_is_not_treated_as_missing(reference):
    foreign = object()
    raw = reference.get_input_groups()[0]
    raw[1] = foreign
    assert prepare_candidate_inputs(reference.Model, raw)[1] is foreign


def test_nonofficial_references_keep_natural_inputs():
    values = [object(), None, False]
    assert prepare_candidate_inputs(None, values) == values


def test_omitted_required_parameter_is_a_reference_contract_error(reference):
    raw = reference.get_input_groups()[0]
    raw[0] = reference._ABSENT
    with pytest.raises(ValueError, match="required parameter: x"):
        prepare_candidate_inputs(reference.Model, raw)


@pytest.mark.parametrize("cache_hit", [False, True])
def test_verify_real_candidate_execution_receives_defaults_without_reference_instantiation_on_hit(
    reference, monkeypatch, tmp_path, cache_hit
):
    raw = reference.get_input_groups()[0]
    expected = reference.Model()(*raw)
    # Comparison is separately covered; this test exercises the actual candidate
    # call and both input preparation routes using torch on CPU.
    monkeypatch.setattr(verify, "compare", lambda *_args, **_kwargs: None)
    trace = [
        {
            "case_idx": 1,
            "selected_reference_device": "cpu",
            "reference_completed": False,
            "output_devices": [],
        }
    ]
    context = verify.CaseContext(
        1, 1, reference_trace=trace, cached_reference=(None, expected) if cache_hit else None
    )
    pair = verify.ModelPair(
        None if cache_hit else reference.Model(), Candidate(), "cpu", reference.Model
    )
    verify.run_single_case(pair, raw, "cpu", context)
    assert trace[0]["candidate_defaulted_parameters"] == ["offsets", "flag", "scale", "mode"]
    path = tmp_path / "verify_reference.json"
    path.write_text(json.dumps({"reference_device_policy": "npu-strict", "cases": trace}))
    assert _read_trace(path, "npu-strict")["status"] == "recorded"


@pytest.mark.parametrize("cache_hit", [False, True])
def test_benchmark_actual_candidate_function_receives_defaults(reference, monkeypatch, cache_hit):
    raw = reference.get_input_groups()[0]
    config = types.SimpleNamespace(skip_framework=cache_hit)
    captured = []

    def framework(model, inputs, *_args):
        assert inputs[1] is reference._ABSENT
        if not cache_hit:
            captured.append(model(*inputs))
        return {}, 2.0, 0.0, None

    def candidate(model, inputs, *_args):
        captured.append(invoke_model(model, inputs))
        return {}, 1.0, 0.0

    monkeypatch.setattr(benchmark, "_measure_framework", framework)
    monkeypatch.setattr(benchmark, "_measure_impl", candidate)
    pair = benchmark.ModelPair(
        None if cache_hit else reference.Model(), Candidate(), reference.Model
    )
    benchmark.run_single_benchmark(pair, raw, config, "cpu", benchmark.CaseContext(1, 1))
    assert torch.equal(captured[-1], reference.Model()(*raw))
    assert raw[1] is reference._ABSENT
