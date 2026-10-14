"""Shared phases preserve timing semantics, case identity and bounded memory."""

import gc
import sys
import weakref
from types import SimpleNamespace

import pytest

from triton_evaluation.api.contract import TraceProfileResult
from triton_evaluation.engine import benchmark
from triton_evaluation.engine import joint_measurement as joint
from triton_evaluation.evaluators import operator as operator_source


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed=lambda _: None), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    provider = (
        "import torch\n"
        "provider_calls = 0\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    global provider_calls\n"
        "    provider_calls += 1\n"
        "    return [[torch.tensor([2.])], [torch.tensor([3.])], [torch.tensor([4.])]]\n"
    )
    model = (
        "class {name}(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.register_buffer('state', torch.tensor([0.]))\n"
        "    def forward(self, x):\n"
        "        x.add_(10)\n"
        "        self.state.add_(1)\n"
        "        return x\n"
    )
    (tmp_path / "joint_torch.py").write_text(provider + model.format(name="Model"))
    (tmp_path / "joint_triton_ascend_impl.py").write_text(
        "import torch\n" + model.format(name="ModelNew")
    )
    loaded, calls = [], []
    load = joint.single_benchmark.load_module

    def tracked_load(name, path):
        loaded.append(name)
        return load(name, path)

    def measure(context, **_):
        calls.append({
            "method": context.method, "name": context.profile_name,
            "input": context.inputs[0].item(), "state": context.model.state.item(),
        })
        context.model(*context.inputs)
        context.model(*context.inputs)
        return {"kernel_a": 300.0, "kernel_b": 700.0}, 1.0, 0.0

    monkeypatch.setattr(joint.single_benchmark, "load_module", tracked_load)
    monkeypatch.setattr(benchmark, "measure_single", measure)
    config = benchmark.BenchmarkConfig(
        "joint", str(tmp_path), measurement_method="event", disable_reference_cache=True,
        case_indices=[1, 3], max_retries=1,
    )
    return config, calls, loaded


def test_event_timing_finishes_before_trace_with_one_module_load(runtime):
    config, calls, loaded = runtime
    checkpoints = []

    def checkpoint(performance):
        checkpoints.append(performance)
        assert len(calls) == 4
        assert all(call["method"] == "event" for call in calls)

    performance, trace = joint.run_measurements(
        config, trace_indices=[1, 2], performance_ready=checkpoint, device="cpu",
    )
    assert checkpoints == [performance]
    assert loaded == ["joint_torch", "joint_triton_ascend_impl"]
    assert sys.modules["joint_torch"].provider_calls == 1
    assert [call["method"] for call in calls] == ["event"] * 4 + ["profiler"] * 2
    assert performance["case_indices"] == [1, 3]
    assert [case["case_idx"] for case in trace["cases"]] == [0, 1]
    # Timing mutates both model and already-on-device inputs. Trace receives
    # fresh state and pristine input values even for an overlapping case.
    assert [call["input"] for call in calls[-2:]] == [2.0, 3.0]
    assert [call["state"] for call in calls[-2:]] == [0.0, 0.0]
    assert trace["total_cases"] == 3
    TraceProfileResult.model_validate(trace)


def test_profiler_timing_reuses_exact_candidate_capture_without_recapture(runtime):
    config, calls, _ = runtime
    config.measurement_method = "profiler"
    performance, trace = joint.run_measurements(config, trace_indices=[1, 2], device="cpu")
    # Four timing captures (two sides, two cases), one new trace-only case.
    assert len(calls) == 5
    assert [call["name"] for call in calls if call["name"].startswith("trace_")] == [
        "trace_case2"
    ]
    assert performance["measurement_method"] == "profiler_kernel_sum_v1"
    assert trace["cases"][0]["kernels"] == [
        {"op_name": "kernel_b", "time_ms": 0.7, "percentage": 70.0},
        {"op_name": "kernel_a", "time_ms": 0.3, "percentage": 30.0},
    ]
    TraceProfileResult.model_validate(trace)


@pytest.mark.parametrize("implementation", ["torch", "triton"])
@pytest.mark.parametrize("method", ["event", "profiler"])
def test_single_source_shares_preparation_and_never_measures_counterpart(
    runtime, implementation, method,
):
    config, calls, loaded = runtime
    config.measurement_method = method
    performance, trace = joint.run_measurements(
        config, implementation=implementation, trace_indices=[1, 2], device="cpu",
    )
    assert len(loaded) == (1 if implementation == "torch" else 2)
    assert len(calls) == (4 if method == "event" else 3)
    assert all("framework" not in call["name"] for call in calls)
    assert trace["correctness_verified"] is False
    assert performance["correctness_verified"] is False
    assert trace["implementation"] == implementation
    assert performance["selected_cases"] == 2
    assert [case["case_idx"] for case in performance["per_case"]] == [0, 2]
    assert [case["case_idx"] for case in trace["cases"]] == [0, 1]
    TraceProfileResult.model_validate(trace)


def test_trace_only_never_enters_timing_or_correctness_gate(runtime, monkeypatch):
    config, calls, _ = runtime
    monkeypatch.setattr(benchmark, "benchmark_implementations", lambda *_a, **_k: pytest.fail())
    monkeypatch.setattr(benchmark, "check_verify_gate", lambda *_a: pytest.fail())
    performance, trace = joint.run_measurements(
        config, timing=False, trace_indices=[2], device="cpu",
    )
    assert performance is None
    assert len(calls) == 1 and calls[0]["method"] == "profiler"
    assert [case["case_idx"] for case in trace["cases"]] == [1]
    TraceProfileResult.model_validate(trace)


def test_later_trace_failure_does_not_discard_timing(runtime, monkeypatch):
    config, calls, _ = runtime
    measure = benchmark.measure_single

    def fail_trace(context, **kwargs):
        if context.method == "profiler":
            raise RuntimeError("capture unavailable")
        return measure(context, **kwargs)

    monkeypatch.setattr(benchmark, "measure_single", fail_trace)
    saved = []
    performance, trace = joint.run_measurements(
        config, trace_indices=[1, 2], device="cpu", performance_ready=saved.append,
    )
    assert saved == [performance] and performance["passed_cases"] == 2
    assert trace["counts"] == {"selected": 2, "complete": 0, "partial": 0, "failed": 2}
    assert len(calls) == 4
    TraceProfileResult.model_validate(trace)


def test_failed_shared_profiler_capture_is_not_silently_retried_as_trace(runtime, monkeypatch):
    config, _, _ = runtime
    config.measurement_method = "profiler"
    monkeypatch.setattr(benchmark, "measure_single", lambda *_a, **_kw: ({}, None, 0.0))
    monkeypatch.setattr(joint, "_capture_case", lambda *_a: pytest.fail("duplicate capture"))
    performance, trace = joint.run_measurements(config, trace_indices=[1, 3], device="cpu")
    assert performance["failed_cases"] == 2
    assert trace["counts"]["failed"] == 2
    TraceProfileResult.model_validate(trace)


@pytest.mark.parametrize("operators", [None, {}, {"bad": float("nan")}, {"bad": -1},
                                      {"bad": True}, {"zero": 0}, {"huge": float("inf")}])
def test_invalid_capture_is_explicit_failure(operators):
    case = joint.trace_case(2, operators, 200)
    assert case["status"] == "failed" and case["kernels"] == []
    assert case["error"]["code"] == "PROFILE_CAPTURE_FAILED"


def test_kernel_limit_keeps_full_capture_denominator(runtime):
    config, _, _ = runtime
    _, trace = joint.run_measurements(
        config, trace_indices=[1], timing=False, max_kernels=1, device="cpu",
    )
    assert trace["counts"]["partial"] == 1
    assert trace["cases"][0]["kernels"][0]["percentage"] == 70
    assert trace["cases"][0]["error"]["code"] == "PROFILE_KERNEL_LIMIT"
    TraceProfileResult.model_validate(trace)


def test_streaming_provider_releases_previous_workload_before_next_allocation(monkeypatch):
    pytest.importorskip("torch")
    references = []

    class Payload:
        pass

    class Lazy:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index >= 4:
                raise IndexError
            gc.collect()
            assert all(reference() is None for reference in references)
            value = Payload()
            references.append(weakref.ref(value))
            return [value]

    monkeypatch.setattr(joint, "_rng_state", lambda: (None, None))
    groups = joint.CleanInputGroups(Lazy())
    iterator = iter(groups)
    for _ in range(4):
        values = next(iterator)
        del values
    with pytest.raises(StopIteration):
        next(iterator)
    assert all(reference() is None for reference in references)


def test_clean_inputs_preserve_aliases_views_and_adapter_sentinels():
    torch = pytest.importorskip("torch")
    base = torch.arange(10)
    sentinel = object()
    source = [base, base, base[2:5], sentinel]
    copied = joint._clone_inputs(source)
    assert copied[0] is copied[1] and copied[0] is not base
    assert copied[0].untyped_storage().data_ptr() == copied[2].untyped_storage().data_ptr()
    copied[0].zero_()
    assert copied[2].tolist() == [0, 0, 0]
    assert base.tolist() == list(range(10))
    assert copied[3] is sentinel


def test_custom_nonleaf_tensors_preserve_shared_storage_views_and_mutation_isolation():
    torch = pytest.importorskip("torch")
    base = torch.arange(12.0, requires_grad=True) * 2
    leaf = base.detach()
    view = base.as_strided((3,), (2,), 3)
    source = [leaf, base, {"view": view}, base]
    copied = joint._clone_inputs(source)
    copied_view = copied[2]["view"]
    assert copied[1] is copied[3]
    assert copied_view.stride() == view.stride()
    assert copied_view.storage_offset() == view.storage_offset()
    assert copied[1].requires_grad and copied_view.requires_grad
    assert copied[1].is_leaf and copied_view.is_leaf
    pointers = {value.untyped_storage().data_ptr() for value in [
        copied[0], copied[1], copied_view,
    ]}
    assert len(pointers) == 1 and base.untyped_storage().data_ptr() not in pointers
    with torch.no_grad():
        copied[1].add_(100)
    assert copied_view.tolist() == [106.0, 110.0, 114.0]
    assert view.tolist() == [6.0, 10.0, 14.0]
    assert leaf.tolist() == list(range(0, 24, 2))


def test_input_provider_rng_replays_for_overlapping_cases():
    torch = pytest.importorskip("torch")

    class RandomGroups:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return [torch.rand(4)]

    groups = joint.CleanInputGroups(RandomGroups())
    original = groups[1][0].tolist()
    torch.rand(30)
    groups.replay = True
    assert groups[1][0].tolist() == original


def test_durable_correctness_gate_is_explicit_and_bound_to_materialized_sources(
    tmp_path, monkeypatch,
):
    observed = []

    def validate(path, **kwargs):
        observed.append((path, kwargs))
        return 3

    monkeypatch.setitem(
        sys.modules, "triton_evaluation.runtime.correctness_evidence",
        SimpleNamespace(validate_handoff=validate),
    )
    monkeypatch.setattr(benchmark, "check_verify_gate", lambda *_: pytest.fail("old gate"))
    args = benchmark._build_argparser().parse_args([
        "--op_name", "joint", "--verify_dir", str(tmp_path),
        "--correctness-evidence", str(tmp_path / "correctness_evidence.json"),
        "--correctness-non-compute",
    ])
    assert benchmark.check_correctness_gate(args) == 3
    assert observed == [(str(tmp_path / "correctness_evidence.json"), {
        "reference_path": tmp_path / "joint_torch.py",
        "candidate_path": tmp_path / "joint_triton_ascend_impl.py",
        "triton_impl_name": "triton_ascend_impl", "non_compute": True,
    })]


def test_invalid_durable_correctness_evidence_fails_closed(tmp_path, monkeypatch):
    def validate(*args, **kwargs):
        raise ValueError("identity mismatch")

    monkeypatch.setitem(
        sys.modules, "triton_evaluation.runtime.correctness_evidence",
        SimpleNamespace(validate_handoff=validate),
    )
    args = benchmark._build_argparser().parse_args([
        "--op_name", "joint", "--correctness-evidence", str(tmp_path / "proof.json"),
    ])
    with pytest.raises(benchmark.VerifyGateError, match="reused correctness evidence"):
        benchmark.check_correctness_gate(args)


@pytest.mark.parametrize("implementation", ["comparison", "triton"])
def test_full_passes_use_linear_provider_iteration_for_timing_and_trace(
    runtime, monkeypatch, implementation,
):
    torch = pytest.importorskip("torch")
    config, calls, _ = runtime
    config.case_indices = None
    iterations = []

    class IterationOnly:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            pytest.fail("full passes must not replay prefixes through random access")

        def __iter__(self):
            iterations.append(True)
            for index in range(3):
                yield [torch.tensor([float(index)])]

    monkeypatch.setattr(benchmark, "_build_input_groups_shared", lambda _: IterationOnly())
    performance, trace = joint.run_measurements(
        config, implementation=implementation, device="cpu",
    )
    assert len(iterations) == 2
    assert len(calls) == (9 if implementation == "comparison" else 6)
    assert len(trace["cases"]) == 3 and performance is not None


def test_builtin_fresh_inputs_skip_tensor_clones_but_isolate_mutable_scalar_lists(monkeypatch):
    torch = pytest.importorskip("torch")
    tensor = torch.ones(4)
    scalar = [1, 2]

    class Fresh:
        _inputs_are_fresh = True

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return [tensor, scalar]

    monkeypatch.setattr(joint, "_rng_state", lambda: pytest.fail("redundant RNG snapshot"))
    copied = joint.CleanInputGroups(Fresh())[0]
    assert copied[0] is tensor
    assert copied[1] == scalar and copied[1] is not scalar
    copied[1].append(3)
    assert scalar == [1, 2]


@pytest.mark.parametrize("kind,expected", [("random", True), ("scalar", True), ("custom", False)])
def test_generated_provider_only_promises_fresh_storage_for_builtin_specs(kind, expected):
    pytest.importorskip("torch")
    namespace = {"_operator_cases": [{"inputs": {"x": {"type": kind}}}]}
    exec(operator_source._INPUT_SUPPORT, namespace)
    assert namespace["get_input_groups"]()._inputs_are_fresh is expected
