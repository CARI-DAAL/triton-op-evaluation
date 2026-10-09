"""Benchmarking retains evidence while releasing each device workload."""

import gc
import sys
import weakref
from types import SimpleNamespace

import pytest

from triton_evaluation.engine import benchmark
from triton_evaluation.evaluators import operator as operator_source


class Payload:
    pass


class LazyGroups:
    def __init__(self, count=4):
        self.count = count
        self.references = []
        self.generated = []
        self.iterations = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        gc.collect()
        assert all(ref() is None for ref in self.references), "previous workload retained"
        value = Payload()
        self.references.append(weakref.ref(value))
        self.generated.append(index)
        return [value]

    def __iter__(self):
        self.iterations += 1
        for index in range(self.count):
            yield self[index]


def _run_benchmark(monkeypatch, groups, indices):
    torch = SimpleNamespace(
        device=lambda _: "npu", manual_seed=lambda _: None,
        npu=SimpleNamespace(manual_seed=lambda _: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(benchmark, "_ensure_npu_ambient_shared", lambda *_a, **_k: True)
    monkeypatch.setattr(benchmark, "resolve_inputs", lambda *_: groups)
    monkeypatch.setattr(
        benchmark, "_load_benchmark_modules", lambda *_: (object, object, lambda: [])
    )
    measured = []

    def measure(config, spec, inputs, device, context):
        measured.append(context.case_idx)
        return benchmark.SingleShapeResult(case_idx=context.case_idx, input_desc=[], status="fail")

    monkeypatch.setattr(benchmark, "_run_shape_case", measure)
    config = benchmark.BenchmarkConfig(
        "task", ".", case_indices=indices, disable_reference_cache=True
    )
    return measured, benchmark.benchmark_implementations(config)


@pytest.mark.parametrize("indices", [None, [4, 2]])
def test_benchmark_releases_groups_before_next_allocation(monkeypatch, indices):
    groups = LazyGroups()
    measured, result = _run_benchmark(monkeypatch, groups, indices)
    assert measured == ([1, 2, 3, 4] if indices is None else [2, 4])
    assert groups.generated == ([0, 1, 2, 3] if indices is None else [1, 3])
    assert groups.iterations == (1 if indices is None else 0)
    assert result.case_indices == (None if indices is None else [2, 4])
    assert all(ref() is None for ref in groups.references)


def test_selected_view_cache_access_preserves_original_indices():
    groups = [["a"], ["b"], ["c"], ["d"]]
    selected = benchmark._SelectedInputGroups(groups, [2, 4])
    assert len(selected) == 2
    assert selected[0] == ["b"]
    assert selected[-1] == ["d"]
    assert list(selected) == [["b"], ["d"]]
    assert list(selected[:1]) == [["b"]]
    assert list(selected) == [["b"], ["d"]]


@pytest.mark.parametrize("indices", [None, [4, 2]])
def test_generated_operator_groups_release_each_workload(monkeypatch, indices):
    references, generated = [], []

    def generate(case, **_):
        gc.collect()
        assert all(ref() is None for ref in references), "previous workload retained"
        value = Payload()
        references.append(weakref.ref(value))
        generated.append(case["id"])
        return [value]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    namespace = {
        "_operator_cases": [{"id": index} for index in range(4)],
        "_operator_seed": 200,
        "_operator_custom_code": None,
        "_operator_custom_entrypoint": None,
        "_operator_generate_inputs": generate,
    }
    exec(operator_source._INPUT_SUPPORT, namespace)
    measured, _ = _run_benchmark(monkeypatch, namespace["get_input_groups"](), indices)
    assert measured == ([1, 2, 3, 4] if indices is None else [2, 4])
    assert generated == ([0, 1, 2, 3] if indices is None else [1, 3])
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("miss_at", [None, 2])
def test_reference_cache_lookup_releases_each_workload(monkeypatch, tmp_path, miss_at):
    from triton_evaluation.runtime import reference_cache

    loaded, fallback = [], []

    class Cache:
        deployment = SimpleNamespace(digest="d")
        device_id = 0

        def check_measurement(self, config):
            pass

        def load_case(self, source, case_idx, inputs, *, reference_device_policy):
            loaded.append(case_idx)
            if case_idx == miss_at:
                raise reference_cache.ReferenceCacheMiss("absent")
            return SimpleNamespace(
                reference_latency_median_ms=1.0, latency_samples_ms=[1.0],
                binding=SimpleNamespace(case_idx=case_idx),
                model_dump=lambda **_: {"case_idx": case_idx},
            )

        def needs_remeasurement(self, cached, measured):
            return False

    def reuse(config, spec, groups, indices, device):
        # The miss fallback may materialize a workload of its own.
        fallback.append(len(groups[0]))
        return False

    class Model:
        def __init__(self, *_):
            pass

        def to(self, _):
            return self

    torch = SimpleNamespace(
        manual_seed=lambda _: None, npu=SimpleNamespace(manual_seed=lambda _: None)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        reference_cache.ReferenceCache, "from_environment", classmethod(lambda cls: Cache())
    )
    monkeypatch.setattr(reference_cache, "official_source_metadata", lambda _: ("d", "official"))
    monkeypatch.setattr(benchmark, "_try_validated_timing_reuse", reuse)
    monkeypatch.setattr(benchmark, "_measure_framework", lambda *_: (None, 1.0, None, None))
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    (tmp_path / "task_torch.py").write_text("# official reference\n")
    config = benchmark.BenchmarkConfig("task", str(tmp_path))
    groups = LazyGroups()
    benchmark._configure_reference_cache(
        config, benchmark.BenchmarkModelSpec(Model, object, lambda: []),
        benchmark._SelectedInputGroups(groups, [1, 2, 3, 4]), "npu", case_indices=[1, 2, 3, 4],
    )
    if miss_at is None:
        # Four cache lookups, then the sentinel case.
        assert loaded == [1, 2, 3, 4] and groups.generated == [0, 1, 2, 3, 0]
        assert config.reference_evidence_source == "cached_manifest"
    else:
        assert loaded == [1, 2] and fallback == [1]
        assert groups.generated == [0, 1, 0]
        assert config.reference_measure_eight
    assert all(ref() is None for ref in groups.references)
