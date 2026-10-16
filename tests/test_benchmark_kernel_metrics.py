"""Profiler counters reach agent responses without changing timing semantics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from triton_evaluation.api.contract import (
    BenchmarkKernelMetrics,
    EvaluationResource,
    PerformanceCase,
    SinglePerformanceResult,
)
from triton_evaluation.api.routes import get_instance, require_read_caller, router
from triton_evaluation.engine import benchmark, single_benchmark
from triton_evaluation.evaluators.shaping import ShapingError, shape_performance
from triton_evaluation.service.security import Caller

FIXTURES = Path(__file__).parent / "fixtures"


def capture(tmp_path, text):
    path = tmp_path / "capture"
    path.mkdir()
    (path / "kernel_details.csv").write_text(text)
    return str(path)


def test_active_kernel_counters_and_latency_share_one_csv(tmp_path):
    pytest.importorskip("pandas")
    path = capture(tmp_path, (
        "Name,Duration(us),aic_mac_ratio,aiv_vec_ratio,cube_utilization(%)\n"
        "cube,1000,0.99,,99\nvector,1000,,0.99,\n"
        "cube,20,0.2,,20\ncube,40,0.4,,40\n"
        "vector,10,,0.6,\nvector,30,,0.8,\n"
    ))
    metrics = {}
    operators, latency = benchmark.parse_operator_latency(
        path, 1, 2, kernel_metrics=metrics,
    )
    assert operators == {"cube": 30.0, "vector": 20.0} and latency == .05
    assert not Path(path).exists()  # Original capture cleanup still runs.
    assert metrics["cube"]["latency_ms"] == .03
    assert metrics["cube"]["aic_mac_ratio"] == pytest.approx(.3)
    assert metrics["cube"]["cube_utilization"] == pytest.approx(.3)
    assert metrics["cube"]["aic_cube_ratio"] is None  # No invented MAC/Cube alias.
    assert metrics["vector"]["aiv_vec_ratio"] == pytest.approx(.7)
    assert metrics["vector"]["aiv_mte2_ratio"] is None
    for value in metrics.values():
        BenchmarkKernelMetrics.model_validate(value)
    json.dumps(metrics, allow_nan=False)


def test_invalid_missing_and_percentage_counters_preserve_latency(tmp_path):
    pytest.importorskip("pandas")
    path = capture(tmp_path, (
        "Name,Duration(us),aic_cube_ratio,aiv_vec_ratio,aiv_mte2_ratio(%),aiv_scalar_ratio\n"
        "matrix,20,bad,inf,30,True\nmatrix,20,NaN,-1,50,True\n"
        "matrix,20,1.2,0.8,inf,True\n"
    ))
    metrics = {"stale": {"latency_ms": 9}}
    assert benchmark.parse_operator_latency(path, 0, 3, kernel_metrics=metrics)[1] == .02
    assert "stale" not in metrics
    assert metrics["matrix"]["aic_cube_ratio"] is None
    assert metrics["matrix"]["aiv_scalar_ratio"] is None
    assert metrics["matrix"]["aiv_vec_ratio"] == .8
    assert metrics["matrix"]["aiv_mte2_ratio"] == pytest.approx(.4)


@pytest.mark.parametrize("method", ["event", "profiler"])
def test_measurement_keeps_three_value_unpacking_and_one_capture(tmp_path, monkeypatch, method):
    pytest.importorskip("pandas")
    path = capture(tmp_path, "Name,Duration(us),aic_cube_ratio\nmatmul,20,0.4\n")
    npu = SimpleNamespace(reset_peak_memory_stats=lambda: None,
                          max_memory_allocated=lambda: 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(benchmark, "prepare_model_fn", lambda *args: lambda: None)
    monkeypatch.setattr(benchmark, "_measure_event", lambda *args: .02)
    captures = []

    def profile(*args, **kwargs):
        captures.append(1)
        return path

    monkeypatch.setattr(benchmark, "run_profiler_with_config", profile)
    result = benchmark.measure_single(benchmark.MeasureContext(
        model=None, inputs=[], warmup=0, repeats=1, profile_name="unit", device=None,
        method=method,
    ))
    operators, latency, memory = result
    assert latency == .02 and memory == 1
    assert len(result) == 3 and result[1] == latency
    if method == "profiler":
        assert operators == {"matmul": 20}
        assert captures == [1]
        assert result.kernel_metrics["matmul"]["aic_cube_ratio"] == .4
    else:
        assert operators is None and result.kernel_metrics is None and not captures


def test_engine_pair_serialization_carries_fresh_candidate_counters(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    path = capture(tmp_path, "Name,Duration(us),aic_cube_ratio\n_bmm_kernel,20,0.4\n")
    npu = SimpleNamespace(reset_peak_memory_stats=lambda: None,
                          max_memory_allocated=lambda: 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(benchmark, "prepare_model_fn", lambda *args: lambda: None)
    monkeypatch.setattr(benchmark, "run_profiler_with_config", lambda *a, **kw: path)
    monkeypatch.setattr(benchmark, "_move_inputs_to_device", lambda values, device: values)
    config = benchmark.BenchmarkConfig(
        "unit", str(tmp_path), warmup=0, repeats=1, reference_latencies_ms={1: .04},
    )
    reference, implementation, speedup = benchmark.run_single_benchmark(
        benchmark.ModelPair(None, None), [], config, None, benchmark.CaseContext(1, 1),
    )
    assert reference.kernel_metrics is None
    assert implementation.kernel_metrics["_bmm_kernel"]["aic_cube_ratio"] == .4
    assert speedup == 2
    # Exercise the exact per-case raw layout consumed by the shaping boundary.
    row = benchmark.SingleShapeResult(
        1, [], framework=reference, implementation=implementation, speedup_vs_torch=speedup,
    )
    result = SimpleNamespace(
        op_name="unit", warmup=0, repeats=1, max_retries=1, total_cases=1,
        case_indices=[1], passed_cases=1, failed_cases=0, nan_indices=[], inf_indices=[],
        zero_indices=[], negative_indices=[], none_indices=[], measurement_method="profiler",
        framework=reference, implementation=implementation, speedup_vs_torch=speedup,
        per_shape_results=[row],
    )
    shaped = shape_performance(benchmark.result_to_dict(result))
    assert shaped.per_case[0].kernel_metrics["_bmm_kernel"].aic_cube_ratio == .4


@pytest.mark.parametrize("method", ["event", "profiler"])
def test_single_source_retains_its_own_profiler_metrics(tmp_path, monkeypatch, method):
    fake_torch = SimpleNamespace(manual_seed=lambda value: None, Tensor=type("Tensor", (), {}))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    (tmp_path / "unit_torch.py").write_text(
        "def get_init_inputs(): return []\ndef get_input_groups(): return [[]]\n"
        "class Model:\n    def to(self, device): return self\n"
    )
    metrics = {"matmul": {"latency_ms": .02, "aic_cube_ratio": .4}}
    monkeypatch.setattr(benchmark, "measure_single", lambda ctx: benchmark.Measurement(
        {}, .02, 1, metrics if method == "profiler" else None,
    ))
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    result = single_benchmark.measure_implementation(
        op_name="unit", verify_dir=tmp_path, triton_impl_name="unused", implementation="torch",
        warmup=0, repeats=1, method=method, device="cpu",
    )
    parsed = SinglePerformanceResult.model_validate(result)
    if method == "profiler":
        assert parsed.per_case[0].kernel_metrics["matmul"].aic_cube_ratio == .4
    else:
        assert "kernel_metrics" not in result["per_case"][0]


def profiler_result():
    raw = json.loads((FIXTURES / "engine/benchmark-mixed-coverage.json").read_text())
    raw["measurement_method"] = "profiler_kernel_sum_v1"
    for case in raw["per_shape_results"]:
        case["measurement_method"] = "profiler_kernel_sum_v1"
    raw["per_shape_results"][0]["framework"]["baseline_cache_hit"] = True
    raw["per_shape_results"][0]["implementation"]["kernel_metrics"] = {
        "_bmm_kernel": {"latency_ms": .5, "aic_cube_ratio": .42},
    }
    return raw


def test_shaping_preserves_candidate_metrics_with_cached_reference():
    result = shape_performance(profiler_result())
    case = result.per_case[0]
    assert case.baseline_cache_hit is True
    assert case.implementation_latency_ms == .5 and case.reference_latency_ms == 1
    assert case.speedup_vs_reference == 2
    assert case.kernel_metrics["_bmm_kernel"].aic_cube_ratio == .42
    assert case.kernel_metrics["_bmm_kernel"].aiv_vec_ratio is None
    assert "kernel_metrics" not in result.per_case[1].model_dump(mode="json")


def test_event_evidence_cannot_claim_profiler_counters():
    raw = profiler_result()
    raw["per_shape_results"][0]["measurement_method"] = "npu_event_v1"
    with pytest.raises(ShapingError):
        shape_performance(raw)
    with pytest.raises(ValueError, match="require profiler"):
        PerformanceCase(case_idx=0, input_shapes=[], implementation_latency_ms=1,
                        reference_latency_ms=1, speedup_vs_reference=1,
                        measurement_method="npu_event_v1",
                        kernel_metrics={"kernel": {"latency_ms": 1}})
    with pytest.raises(ValueError, match="require profiler"):
        SinglePerformanceResult(implementation="torch", correctness_verified=False,
                                measurement_method="npu_event_v1", selected_cases=1,
                                measured_cases=1, coverage=1,
                                per_case=[{"case_idx": 0, "input_shapes": [], "latency_ms": 1,
                                           "kernel_metrics": {"kernel": {"latency_ms": 1}}}],
                                unmeasured_cases=[])


def test_public_http_response_returns_counter_nulls_and_names():
    raw = json.loads((FIXTURES / "evaluation_api/2.1/fixtures/"
                      "resource-completed-profiler-performance.json").read_text())
    raw["result"]["performance"]["data"] = shape_performance(profiler_result()).model_dump(
        mode="json",
    )
    resource = EvaluationResource.model_validate_json(json.dumps(raw))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_instance] = lambda: SimpleNamespace(
        get=lambda eid, caller: resource,
    )
    app.dependency_overrides[require_read_caller] = lambda: Caller(
        "test", "trial", frozenset({"trial:read"}),
    )
    with TestClient(app) as client:
        response = client.get("/evaluations/" + resource.evaluation_id)
    assert response.status_code == 200
    data = response.json()["result"]["performance"]["data"]
    assert data["per_case"][0]["kernel_metrics"]["_bmm_kernel"]["aic_cube_ratio"] == .42
    assert data["per_case"][0]["kernel_metrics"]["_bmm_kernel"]["aiv_vec_ratio"] is None
    assert "kernel_metrics" in app.openapi()["components"]["schemas"]["PerformanceCase"][
        "properties"
    ]
