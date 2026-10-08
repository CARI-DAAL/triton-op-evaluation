"""Host baseline timing inputs remain separate from candidate measurements."""

import sys
from types import SimpleNamespace

import pytest

from triton_evaluation.engine import benchmark


@pytest.mark.parametrize(
    "values",
    [
        None,
        [],
        {},
        {"1": True},
        {"1": 0},
        {"1": -1},
        {"1": float("nan")},
        {"1": float("inf")},
        {"1": "1"},
        {"1": 1, "2": 2},
    ],
)
def test_baseline_rejects_incomplete_or_invalid_values(values):
    with pytest.raises(ValueError):
        benchmark._validate_baseline_timings(values, [1])


def test_baseline_keeps_original_case_indices():
    assert benchmark._validate_baseline_timings({"3": 2, "1": 1.5}, [3, 1]) == {3: 2.0, 1: 1.5}


@pytest.fixture
def fake_runtime(monkeypatch):
    seeds = []
    torch = SimpleNamespace(
        device=lambda _: "npu",
        manual_seed=lambda seed: seeds.append(seed),
        npu=SimpleNamespace(manual_seed=lambda seed: seeds.append(seed)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(benchmark, "_ensure_npu_ambient_shared", lambda *_a, **_k: True)
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(benchmark, "resolve_inputs", lambda *_: [["a"], ["b"], ["c"]])
    monkeypatch.setattr(
        benchmark,
        "_configure_reference_cache",
        lambda *_a, **_k: pytest.fail("legacy cache invoked"),
    )
    return seeds


def test_reference_only_never_loads_candidate_and_measures_once(monkeypatch, fake_runtime):
    calls = []

    class Model:
        def to(self, device):
            return self

    module = SimpleNamespace(Model=Model, get_init_inputs=lambda: [])
    monkeypatch.setattr(
        benchmark.importlib,
        "import_module",
        lambda name: module if name == "task_torch" else pytest.fail("candidate imported"),
    )
    monkeypatch.setattr(
        benchmark, "_load_benchmark_modules", lambda *_: pytest.fail("candidate imported")
    )

    def measure(model, inputs, config, device, index):
        calls.append((inputs, index, config.warmup, config.repeats, config.reference_measure_eight))
        return {}, float(index), 0, None

    monkeypatch.setattr(benchmark, "_measure_framework", measure)
    result = benchmark.benchmark_reference_only(
        benchmark.BenchmarkConfig("task", ".", case_indices=[3, 1], warmup=7, repeats=11)
    )
    assert result == {
        "total_cases": 3,
        "case_indices": [1, 3],
        "reference_latencies_ms": {"3": 3.0, "1": 1.0},
        "measurement_method": "profiler",
        "warmup": 7,
        "repeats": 11,
    }
    assert calls == [(["a"], 1, 7, 11, False), (["c"], 3, 7, 11, False)]
    assert fake_runtime == [0] * 6


def test_reference_only_invalid_measurement_fails(monkeypatch, fake_runtime):
    module = SimpleNamespace(
        Model=lambda: SimpleNamespace(to=lambda _: object()), get_init_inputs=lambda: []
    )
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(benchmark, "_measure_framework", lambda *_: ({}, None, 0, None))
    with pytest.raises(ValueError):
        benchmark.benchmark_reference_only(benchmark.BenchmarkConfig("task", "."))


def test_invalid_baseline_rejected_before_candidate_import(monkeypatch, fake_runtime):
    monkeypatch.setattr(
        benchmark, "_load_benchmark_modules", lambda *_: pytest.fail("candidate imported")
    )
    with pytest.raises(ValueError):
        benchmark.benchmark_implementations(
            benchmark.BenchmarkConfig("task", ".", baseline_timings={"1": 1})
        )


@pytest.mark.parametrize("supplied", [None, {"1": 1, "2": 2, "3": 3}])
def test_internal_path_bypasses_legacy_cache(monkeypatch, fake_runtime, supplied):
    monkeypatch.setattr(
        benchmark, "_load_benchmark_modules", lambda *_: (object, object, lambda: [])
    )
    configs = []

    def run(config, spec, inputs, device, context):
        configs.append(config)
        return benchmark.SingleShapeResult(case_idx=context.case_idx, input_desc=[], status="fail")

    monkeypatch.setattr(benchmark, "_run_shape_case", run)
    config = benchmark.BenchmarkConfig(
        "task", ".", baseline_timings=supplied, disable_reference_cache=True
    )
    result = benchmark.benchmark_implementations(config)
    assert result.total_cases == 3
    assert len(configs) == 3
    assert config.reference_latencies_ms == (None if supplied is None else {1: 1.0, 2: 2.0, 3: 3.0})


def test_reference_only_cli_skips_verify_gate(monkeypatch, tmp_path):
    output = tmp_path / "baseline.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--op_name",
            "task",
            "--verify_dir",
            str(tmp_path),
            "--reference-only",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(benchmark, "_enforce_conv_fp32", lambda: None)
    monkeypatch.setattr(
        benchmark, "check_verify_gate", lambda *_: pytest.fail("candidate gate read")
    )
    monkeypatch.setattr(
        benchmark, "benchmark_reference_only", lambda _: {"reference_latencies_ms": {"1": 1.0}}
    )
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 0
    assert output.exists()


def test_supplied_denominator_skips_framework_measurement(monkeypatch):
    monkeypatch.setattr(
        benchmark, "measure_single", lambda *_a, **_k: pytest.fail("framework measured")
    )
    config = benchmark.BenchmarkConfig("task", ".", reference_latencies_ms={2: 4.5})
    assert benchmark._measure_framework(None, [], config, "npu", 2) == ({}, 4.5, 0.0, None)


def test_reference_only_cli_ignores_candidate_bytecode_and_verify_files(
    monkeypatch, tmp_path, fake_runtime
):
    import json
    import py_compile

    (tmp_path / "task_torch.py").write_text(
        "class Model:\n"
        "    def to(self, device):\n"
        "        return self\n"
        "def get_init_inputs():\n"
        "    return []\n"
    )
    candidate = tmp_path / "task_triton_ascend_impl.py"
    candidate.write_text("raise RuntimeError('candidate bytecode executed')\n")
    py_compile.compile(str(candidate), cfile=str(candidate.with_suffix(".pyc")))
    candidate.unlink()
    (tmp_path / "verify_result.json").write_text("invalid verify JSON")
    (tmp_path / "verify_reference.json").write_text("invalid reference JSON")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "task_torch", raising=False)
    monkeypatch.delitem(sys.modules, "task_triton_ascend_impl", raising=False)
    monkeypatch.setattr(benchmark, "_enforce_conv_fp32", lambda: None)
    monkeypatch.setattr(benchmark, "_measure_framework", lambda *_: ({}, 1.0, 0, None))
    monkeypatch.setattr(benchmark, "check_verify_gate", lambda *_: pytest.fail("verify file read"))
    output = tmp_path / "timings.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--op_name",
            "task",
            "--verify_dir",
            str(tmp_path),
            "--reference-only",
            "--case-indices",
            "2",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 0
    result = json.loads(output.read_text())
    assert result["total_cases"] == 3
    assert result["case_indices"] == [2]
    assert result["reference_latencies_ms"] == {"2": 1.0}
    assert "task_triton_ascend_impl" not in sys.modules
    monkeypatch.delitem(sys.modules, "task_torch", raising=False)
