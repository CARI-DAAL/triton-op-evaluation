"""One grade -> eight reference samples + median + honest provenance (CPU).

Real call paths on CPU with a fake ``measure_single`` seam (the same seam the
controlled cache tests use): the full engine ``benchmark_implementations`` ->
``result_to_dict`` -> evaluator ``shape_performance`` -> public contract JSON.
Covers uncached fresh-eight, cache-hit manifest samples labeled cached, case
selection with original identity preservation (audit one-based labels == engine
case_idx; public case_idx == audit - 1), and failure/partial-sample honesty.
"""

import importlib.metadata
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from triton_evaluation.config import Settings
from triton_evaluation.engine import benchmark
from triton_evaluation.evaluators import benchmark as benchmark_evaluator_module
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.shaping import ShapingError, shape_performance
from triton_evaluation.reference_cache import (
    CacheDeployment,
    CacheMeasurement,
    ReferenceCache,
    sha256,
)
from triton_evaluation.api.contract import (
    EvaluationRequest,
    PerformanceCase,
    PerformanceParams,
)

torch = pytest.importorskip("torch")

# Audit-selection mapping from the wiring packet: audit one-based labels
# [1, 6, 11, 18, 33, 43] == engine case_idx == public case_idx + 1.
AUDIT_SELECTED = [1, 6, 11, 18, 33, 43]
PUBLIC_SELECTED = [index - 1 for index in AUDIT_SELECTED]
TASK_CASES = 51

SOURCE_TAIL = "_ADAPTER_VERSION = \"test-v1\"\n"
SOURCE = '"""\nofficial_digest: ' + "a" * 64 + '\n"""\n' + SOURCE_TAIL

FRESH_SAMPLES = [1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.03, 0.97]
FRESH_MEDIAN = statistics.median(FRESH_SAMPLES)  # 1.0
IMPL_LATENCY = 0.5


def write_task(tmp_path: Path, cases: int = TASK_CASES) -> str:
    (tmp_path / "example_torch.py").write_text(
        SOURCE
        + "import torch\n\n\n"
        + "class Model(torch.nn.Module):\n"
        + "    def forward(self, value):\n"
        + "        return value + 1\n\n\n"
        + "def get_init_inputs():\n"
        + "    return []\n\n\n"
        + "def get_input_groups():\n"
        + f"    return [[torch.ones(2)] for _ in range({cases})]\n",
        encoding="utf-8",
    )
    (tmp_path / "example_triton_ascend_impl.py").write_text(
        "import torch\n\n\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, value):\n        return value + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "verify_reference.json").write_text("{}", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def cpu_engine(monkeypatch):
    """Run the real engine benchmark flow with CPU standing in for NPU."""
    real_device = torch.device
    monkeypatch.setattr(
        torch, "device", lambda name: real_device("cpu" if name == "npu" else name)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(manual_seed=lambda *_: None), raising=False
    )
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(benchmark, "_ensure_npu_ambient_shared", lambda _torch: True)
    # engine imports task modules by name; drop any previous test's modules
    monkeypatch.delitem(sys.modules, "example_torch", raising=False)
    monkeypatch.delitem(
        sys.modules, "example_triton_ascend_impl", raising=False
    )
    scripted_measure(monkeypatch, FRESH_SAMPLES)  # default: healthy fresh samples
    return None


def scripted_measure(monkeypatch, framework_latencies, impl_latency=IMPL_LATENCY):
    """Fake measure_single returning a scripted sequence per framework call."""
    state = {"framework": 0, "names": []}

    def measure(context, **_kwargs):
        assert (context.warmup, context.repeats) == (5, 20)
        if context.profile_name.startswith("framework"):
            latency = framework_latencies[state["framework"] % len(framework_latencies)]
            state["framework"] += 1
            state["names"].append(context.profile_name)
            return {}, latency, 2.0
        return {}, impl_latency, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    return state


def make_cache(tmp_path):
    runtime_file = tmp_path / "runtime-version"
    runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(benchmark.__file__).parent
    deployment = CacheDeployment(
        generation="cpu-controlled-test",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(runtime_file): sha256(runtime_file.read_bytes())},
        algorithm_files={
            name: sha256((engine / name).read_bytes())
            for name in ["verify.py", "benchmark.py", "_common_utils.py"]
        },
        device_conditions={"3": {"hardware_model": "controlled-cpu"}},
        measurement=CacheMeasurement(method="event", warmup=5, repeats=20, cv_threshold=0.3),
    )
    return ReferenceCache(
        tmp_path / "cache", deployment, "3", entries={}, state_dir=tmp_path / "state"
    )


def store_case(cache, case_idx, samples, source):
    inputs = [torch.ones(2)]
    key, digest = cache.write_case(
        source,
        case_idx,
        inputs,
        inputs[0],
        repeated_outputs=inputs[0],
        reference_device_policy="cpu-first",
        selected_reference_device="cpu",
        latency_samples_ms=samples,
    )
    cache.entries[key] = digest
    return statistics.median(samples)


def public_json(result_dict):
    """Engine dict -> real evaluator shaping -> public model JSON."""
    performance = shape_performance(result_dict)
    payload = json.dumps(performance.model_dump(mode="json"), allow_nan=False)
    return performance, json.loads(payload)


def test_uncached_grade_measures_eight_fresh_reference_samples_per_case(
    tmp_path, monkeypatch, cpu_engine
):
    for name in (
        "TRITON_EVAL_REFERENCE_CACHE_DIR",
        "TRITON_EVAL_REFERENCE_CACHE_BINDING",
        "TRITON_EVAL_REFERENCE_CACHE_SHA256",
        "TRITON_EVAL_REFERENCE_CACHE_STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    scripted = scripted_measure(monkeypatch, FRESH_SAMPLES)
    config = benchmark.BenchmarkConfig(
        "example", write_task(tmp_path, cases=3), measurement_method="event"
    )
    assert config.reference_measure_eight is False  # before configuration
    result = benchmark.benchmark_implementations(config)
    assert config.reference_measure_eight is True  # uncached grade forces eight
    assert config.reference_cache_trace is None

    case = result.per_shape_results[0]
    assert case.framework.latency_samples_ms == FRESH_SAMPLES
    assert case.framework.latency_samples_source == "fresh_measurements"
    assert case.framework.avg_latency_ms == round(FRESH_MEDIAN, 4)
    assert case.speedup_vs_torch == round(FRESH_MEDIAN / IMPL_LATENCY, 4)
    # exactly eight reference measurements per case, one candidate measurement
    assert scripted["framework"] == 8 * 3
    assert result.case_indices is None and result.total_cases == 3

    performance, payload = public_json(benchmark.result_to_dict(result))
    first = performance.per_case[0]
    assert first.reference_latency_samples_ms == FRESH_SAMPLES
    assert first.reference_latency_samples_source == "fresh_measurements"
    assert first.reference_latency_ms == round(FRESH_MEDIAN, 4)
    assert first.speedup_vs_reference == pytest.approx(FRESH_MEDIAN / IMPL_LATENCY)
    assert payload["per_case"][0]["reference_latency_samples_ms"] == FRESH_SAMPLES


def test_selected_cases_preserve_original_identity_through_the_real_path(
    tmp_path, monkeypatch, cpu_engine
):
    scripted = scripted_measure(monkeypatch, FRESH_SAMPLES)
    config = benchmark.BenchmarkConfig(
        "example",
        write_task(tmp_path),
        measurement_method="event",
        case_indices=list(AUDIT_SELECTED),  # one-based engine/audit labels
    )
    result = benchmark.benchmark_implementations(config)

    assert result.case_indices == AUDIT_SELECTED  # original order preserved
    assert result.total_cases == len(AUDIT_SELECTED)
    assert [case.case_idx for case in result.per_shape_results] == AUDIT_SELECTED
    assert [case.case_idx for case in result.per_shape_results] != list(
        range(1, len(AUDIT_SELECTED) + 1)
    )  # never renumbered
    # only the selected cases are measured: 8 reference + 1 candidate each
    assert scripted["framework"] == 8 * len(AUDIT_SELECTED)
    assert sorted(set(scripted["names"])) == sorted(
        f"framework_profile_case{index}" for index in AUDIT_SELECTED
    )

    performance, payload = public_json(benchmark.result_to_dict(result))
    assert [case.case_idx for case in performance.per_case] == PUBLIC_SELECTED
    assert performance.selected_cases == len(AUDIT_SELECTED)
    assert performance.measured_cases == len(AUDIT_SELECTED)
    assert payload["per_case"][4]["case_idx"] == 32  # audit 33 -> public 32


def test_cache_hit_returns_real_manifest_samples_as_cached_evidence(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir = write_task(tmp_path)
    active = make_cache(tmp_path)
    task_source = Path(verify_dir, "example_torch.py").read_text(encoding="utf-8")
    cached_medians = {}
    for index in AUDIT_SELECTED:
        base = 1.0 + index * 0.1
        cached_medians[index] = store_case(
            active, index, [round(base + k * 0.01, 4) for k in range(8)], task_source
        )
    monkeypatch.setattr(ReferenceCache, "from_environment", lambda: active)

    state = scripted_measure(
        monkeypatch,
        [cached_medians[1] * 1.01] * 8,  # sentinel within +/-3% -> cache hit
    )
    config = benchmark.BenchmarkConfig(
        "example",
        verify_dir,
        measurement_method="event",
        case_indices=list(AUDIT_SELECTED),
    )
    result = benchmark.benchmark_implementations(config)

    # sentinel measured the first selected case eight times, then reuse
    assert state["framework"] == 8
    for case, index in zip(result.per_shape_results, AUDIT_SELECTED):
        assert case.framework.latency_samples_source == "cached_manifest"
        assert case.framework.latency_samples_ms == [
            round(1.0 + index * 0.1 + k * 0.01, 4) for k in range(8)
        ]
        assert case.framework.avg_latency_ms == round(cached_medians[index], 4)

    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    performance_cache = trace["performance_cache"]
    assert performance_cache["status"] == "hit"
    assert performance_cache["sentinel_case_idx"] == 1
    assert len(performance_cache["manifests"]) == len(AUDIT_SELECTED)

    performance, payload = public_json(benchmark.result_to_dict(result))
    assert [case.case_idx for case in performance.per_case] == PUBLIC_SELECTED
    for case, index in zip(performance.per_case, AUDIT_SELECTED):
        assert case.reference_latency_samples_source == "cached_manifest"
        assert case.reference_latency_samples_ms == [
            round(1.0 + index * 0.1 + k * 0.01, 4) for k in range(8)
        ]
        assert case.reference_latency_ms == round(cached_medians[index], 4)
    assert payload["per_case"][0]["reference_latency_samples_source"] == "cached_manifest"


def test_invalid_or_empty_selection_is_rejected_honestly(tmp_path, cpu_engine, monkeypatch):
    with pytest.raises(RuntimeError, match="selection is empty"):
        benchmark._resolve_selected_indices([], 5)
    with pytest.raises(RuntimeError, match="out of range"):
        benchmark._resolve_selected_indices([0], 5)
    with pytest.raises(RuntimeError, match="out of range"):
        benchmark._resolve_selected_indices([1, 52], TASK_CASES)

    with pytest.raises(ValueError):
        benchmark._parse_case_indices("")
    with pytest.raises(ValueError):
        benchmark._parse_case_indices("1,,6")
    with pytest.raises(ValueError):
        benchmark._parse_case_indices("0")
    assert benchmark._parse_case_indices("1,6,11") == [1, 6, 11]
    assert benchmark._parse_case_indices(None) is None

    # out-of-range engine selection fails the whole performance run
    scripted_measure(monkeypatch, FRESH_SAMPLES)
    config = benchmark.BenchmarkConfig(
        "example", write_task(tmp_path, cases=3), measurement_method="event",
        case_indices=[4],
    )
    with pytest.raises(RuntimeError, match="out of range"):
        benchmark.benchmark_implementations(config)

    # the public contract mirrors ProfileParams validation
    with pytest.raises(ValidationError):
        PerformanceParams(case_indices=[3, 3])
    with pytest.raises(ValidationError):
        PerformanceParams(case_indices=[-1])
    assert PerformanceParams(case_indices=[42, 0, 5]).case_indices == [0, 5, 42]
    assert PerformanceParams().case_indices is None


def test_evaluator_rejects_empty_selection_and_wires_real_argv(tmp_path, monkeypatch):
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)

    def request(case_indices):
        return EvaluationRequest(
            schema_version="2.1",
            purpose="trial",
            op_name="example",
            reference_code="pass",
            kernel_code="pass",
            checks=["correctness", "performance"],
            performance=PerformanceParams(case_indices=case_indices),
        )

    evaluator = BenchmarkEvaluator("unused")
    outcome = evaluator.run(request([]), tmp_path, 0, deadline)
    assert outcome.status == "failed"
    assert "selection is empty" in outcome.error.message

    captured = {}

    def fake_run_script(cmd, cwd, device_id, deadline_at, lease_valid=None,
                        budget_seconds=None):
        captured["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_text("{}", encoding="utf-8")
        return SimpleNamespace(outcome="completed", timed_out=False, exit_code=0)

    monkeypatch.setattr(benchmark_evaluator_module, "run_script", fake_run_script)
    assert evaluator.run(request(PUBLIC_SELECTED), tmp_path, 0, deadline).status == "succeeded"
    assert captured["cmd"][
        captured["cmd"].index("--case-indices") + 1
    ] == ",".join(str(i) for i in AUDIT_SELECTED)

    assert evaluator.run(request(None), tmp_path, 0, deadline).status == "succeeded"
    assert "--case-indices" not in captured["cmd"]


@pytest.mark.parametrize(
    "bad_samples",
    [
        [1.0] * 7,  # partial: seven samples is not evidence of eight
        [1.0] * 8 + [1.0],  # nine
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],  # non-positive
        [float("inf")] + [1.0] * 7,  # non-finite
        [0.5, 0.5, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0],  # median inconsistent with scalar
    ],
)
def test_public_case_rejects_fabricated_or_partial_sample_evidence(bad_samples):
    with pytest.raises(ValidationError):
        PerformanceCase(
            case_idx=0,
            input_shapes=[],
            implementation_latency_ms=0.5,
            reference_latency_ms=1.0,
            speedup_vs_reference=2.0,
            reference_latency_samples_ms=bad_samples,
            reference_latency_samples_source="fresh_measurements",
        )


def test_public_case_requires_source_label_with_samples_and_allows_legacy_scalar():
    with pytest.raises(ValidationError):
        PerformanceCase(
            case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
            reference_latency_ms=1.0, speedup_vs_reference=2.0,
            reference_latency_samples_ms=[1.0] * 8,
        )
    with pytest.raises(ValidationError):
        PerformanceCase(
            case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
            reference_latency_ms=1.0, speedup_vs_reference=2.0,
            reference_latency_samples_source="fresh_measurements",
        )
    legacy = PerformanceCase(
        case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
        reference_latency_ms=1.0, speedup_vs_reference=2.0,
    )
    assert legacy.reference_latency_samples_ms is None


def test_partial_or_invalid_reference_measurement_stays_a_failed_case(
    tmp_path, monkeypatch, cpu_engine
):
    # seven good samples then a missing latency -> not a successful eight-sample
    # result; the case fails after retries and shaping reports no measurement.
    scripted_measure(
        monkeypatch,
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None] * 6,
    )
    config = benchmark.BenchmarkConfig(
        "example", write_task(tmp_path, cases=2), measurement_method="event"
    )
    result = benchmark.benchmark_implementations(config)
    assert result.failed_cases == result.total_cases == 2
    assert all(case.framework is None for case in result.per_shape_results)
    with pytest.raises(ShapingError):
        public_json(benchmark.result_to_dict(result))

    # a non-finite sample is equally refused
    scripted_measure(
        monkeypatch, [1.0, 1.0, 1.0, 1.0, float("inf"), 1.0, 1.0, 1.0] * 6
    )
    result = benchmark.benchmark_implementations(config)
    assert result.failed_cases == 2


def test_settings_declare_the_reference8_benchmark_protocol():
    assert Settings().benchmark_algorithm_version == "geometric-mean-v1:reference8-v1"
