"""Single-side timing loads only supplied code and uses the existing measurement primitive."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_evaluation.api.contract import EvaluationRequest, SinglePerformanceResult
from triton_evaluation.engine import single_benchmark as single
from triton_evaluation.evaluators import benchmark as evaluator_module


@pytest.mark.parametrize("implementation", ["torch", "triton"])
@pytest.mark.parametrize("method", ["event", "profiler"])
def test_only_selected_model_is_loaded_and_measured(tmp_path, monkeypatch, implementation, method):
    pytest.importorskip("torch")
    factory = (
        "import torch\ndef get_init_inputs(): return []\n"
        "def get_input_groups(): return [[torch.tensor([2.])], [torch.tensor([4.])]]\n"
    )
    model = "class {name}(torch.nn.Module):\n    def forward(self, x): return x * 3\n"
    (tmp_path / "single_torch.py").write_text(
        factory + (model.format(name="Model") if implementation == "torch" else "")
    )
    (tmp_path / "single_triton_ascend_impl.py").write_text(
        "raise AssertionError('missing counterpart must never import')\n"
        if implementation == "torch"
        else "import torch\n" + model.format(name="ModelNew")
    )
    calls = []

    def measure(context):
        calls.append(context)
        assert context.model(*context.inputs).item() == 12
        return None, 0.75, 2.5

    monkeypatch.setattr(single.benchmark, "measure_single", measure)
    monkeypatch.setattr(single.benchmark, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(single.benchmark, "check_verify_gate", lambda *a: pytest.fail("no gate"))
    monkeypatch.setattr(
        single.benchmark, "_configure_reference_cache", lambda *a: pytest.fail("no reference cache")
    )
    result = single.measure_implementation(
        op_name="single",
        verify_dir=tmp_path,
        triton_impl_name="triton_ascend_impl",
        implementation=implementation,
        warmup=3,
        repeats=7,
        method=method,
        case_indices=[1],
        device="cpu",
    )
    SinglePerformanceResult.model_validate(result)
    assert result["implementation"] == implementation
    assert result["correctness_verified"] is False
    assert result["coverage"] == 1
    assert result["per_case"] == [
        {
            "case_idx": 1,
            "latency_ms": 0.75,
            "peak_memory_mb": 2.5,
            "input_shapes": [{"input_index": 0, "dimensions": [1], "dtype": "torch.float32"}],
        }
    ]
    assert calls[0].warmup == 3 and calls[0].repeats == 7 and calls[0].method == method
    assert "speedup" not in json.dumps(result) and "reference_latency" not in json.dumps(result)


def test_all_unstable_measurements_fail_instead_of_fabricating_latency(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    (tmp_path / "single_torch.py").write_text(
        "import torch\nclass Model(torch.nn.Module):\n    pass\n"
        "def get_input_groups(): return [[]]\ndef get_init_inputs(): return []\n"
    )
    monkeypatch.setattr(single.benchmark, "measure_single", lambda ctx: (None, None, 0))
    monkeypatch.setattr(single.benchmark, "cleanup_npu_memory", lambda: None)
    with pytest.raises(RuntimeError, match="no measurements"):
        single.measure_implementation(
            op_name="single",
            verify_dir=tmp_path,
            triton_impl_name="triton_ascend_impl",
            implementation="torch",
            warmup=1,
            repeats=2,
            method="event",
            device="cpu",
        )


@pytest.mark.parametrize("implementation", ["torch", "triton", "comparison"])
def test_evaluator_selects_standalone_engine_or_existing_verify_gate(
    tmp_path,
    monkeypatch,
    implementation,
):
    request = EvaluationRequest(
        schema_version="2.1",
        op_name="example",
        reference_code="pass",
        kernel_code="" if implementation == "torch" else "pass",
        implementation=implementation,
        checks=["correctness", "performance"]
        if implementation == "comparison"
        else ["performance"],
    )
    seen = []

    def run(cmd, *args, **kwargs):
        seen.extend(cmd)
        if implementation == "comparison":
            return SimpleNamespace(outcome="completed", timed_out=False, exit_code=2)
        Path(cmd[cmd.index("--output") + 1]).write_text(
            '{"implementation":"' + implementation + '"}'
        )
        assert not (tmp_path / "verify_result.json").exists()
        assert (tmp_path / (cmd[cmd.index("--op_name") + 1] + "_torch.py")).exists()
        return SimpleNamespace(outcome="completed", timed_out=False, exit_code=0)

    monkeypatch.setattr(evaluator_module, "run_script", run)
    result = evaluator_module.BenchmarkEvaluator("engine").run(
        request, tmp_path, 0, datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    if implementation == "comparison":
        assert Path(seen[1]).name == "benchmark.py"
        assert result.status == "failed" and "verify gate" in result.error.message
    else:
        assert Path(seen[1]).name == "single_benchmark.py"
        assert seen[seen.index("--implementation") + 1] == implementation
        assert result.status == "succeeded"


def test_partial_measurement_coverage_contains_no_fabricated_samples(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    (tmp_path / "single_torch.py").write_text(
        "import torch\nclass Model(torch.nn.Module):\n    pass\n"
        "def get_input_groups(): return [[], []]\ndef get_init_inputs(): return []\n"
    )
    latencies = iter([None, 0.5])
    monkeypatch.setattr(single.benchmark, "measure_single", lambda ctx: (None, next(latencies), 0))
    monkeypatch.setattr(single.benchmark, "cleanup_npu_memory", lambda: None)
    result = single.measure_implementation(
        op_name="single",
        verify_dir=tmp_path,
        triton_impl_name="triton_ascend_impl",
        implementation="torch",
        warmup=1,
        repeats=2,
        method="event",
        device="cpu",
    )
    SinglePerformanceResult.model_validate(result)
    assert result["selected_cases"] == 2 and result["measured_cases"] == 1
    assert result["coverage"] == 0.5
    assert [case["case_idx"] for case in result["per_case"]] == [1]
    assert result["unmeasured_cases"] == [
        {"case_idx": 0, "input_shapes": [], "reason_code": "NO_STABLE_MEASUREMENT"}
    ]


@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_single_implementation_never_uses_advisory_baseline_cache(
    tmp_path,
    monkeypatch,
    implementation,
):
    from triton_evaluation.api.contract import PerformanceParams

    accesses = []

    class ForbiddenCache:
        def __getattr__(self, name):
            accesses.append(name)
            raise AssertionError("single implementation cannot access baseline cache")

    request = EvaluationRequest(
        schema_version="2.1",
        op_name="example",
        reference_code="pass",
        kernel_code="" if implementation == "torch" else "pass",
        implementation=implementation,
        checks=["performance"],
        performance=PerformanceParams(use_baseline_cache=True),
    )
    runs = []
    raw = {"implementation": implementation, "correctness_verified": False}

    def run(cmd, *args, **kwargs):
        runs.append(cmd)
        assert Path(cmd[1]).name == "single_benchmark.py"
        for flag in ("--reference-only", "--disable-reference-cache", "--baseline-timings-json"):
            assert flag not in cmd
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(raw))
        return SimpleNamespace(outcome="completed", timed_out=False, exit_code=0)

    monkeypatch.setattr(
        evaluator_module, "run_script", lambda *a, **kw: pytest.fail("injected runner required"),
    )
    evaluator = evaluator_module.BenchmarkEvaluator(
        "engine", baseline_cache=ForbiddenCache(), runner=run,
    )
    result = evaluator.run(
        request,
        tmp_path,
        0,
        datetime.now(timezone.utc) + timedelta(minutes=5),
        lease_valid=lambda: True,
    )
    assert accesses == []
    assert len(runs) == 1
    assert result.status == "succeeded" and result.raw == raw
