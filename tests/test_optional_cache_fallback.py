"""Reference caching is a pure optional accelerator (ruling 2026-10-06).

CPU tests through the REAL engine path (fake ``measure_single`` seam) for the
five rule-5 cases: eligible hit, ineligible task (machine2 gate analog),
eligible miss, drift-invalidated card and no cache configured. Every non-hit
case must take the service's ORIGINAL e4a36f7 reference measurement path for
its configuration — never a newly invented one:

* no cache configured / task not cache-eligible -> the original single
  measurement per case (``reference_path == "original_measurements"``);
* cache configured + official task + no entry / drift -> e4a36f7's own
  eight-sample remeasure path (``reference_path == "fresh_measurements"``);
* valid hit (sentinel within 3%) -> cached evidence, unchanged.
"""

import importlib.metadata
import json
from pathlib import Path

import pytest

from triton_evaluation.engine import benchmark
from triton_evaluation.reference_cache import (
    CacheDeployment,
    CacheMeasurement,
    ReferenceCache,
    sha256,
)

torch = pytest.importorskip("torch")

SOURCE = '"""\nofficial_digest: ' + "a" * 64 + '\n"""\n_ADAPTER_VERSION = "test-v1"\n'
PLAIN_SOURCE = "# machine2-local reference, no official pin\n"
IMPL_LATENCY = 0.5


def write_task(tmp_path, source=SOURCE, cases=2):
    (tmp_path / "example_torch.py").write_text(
        source + "import torch\n\n\nclass Model(torch.nn.Module):\n"
        "    def forward(self, value):\n        return value + 1\n\n\n"
        "def get_init_inputs():\n    return []\n\n\ndef get_input_groups():\n"
        f"    return [[torch.ones(2)] for _ in range({cases})]\n",
        encoding="utf-8",
    )
    (tmp_path / "example_triton_ascend_impl.py").write_text(
        "import torch\n\nclass ModelNew(torch.nn.Module):\n"
        "    def forward(self, value):\n        return value + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "verify_reference.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def cpu_engine(monkeypatch):
    real_device = torch.device
    monkeypatch.setattr(
        torch, "device", lambda name: real_device("cpu" if name == "npu" else name)
    )
    monkeypatch.setitem(sys_modules(), "torch_npu", _stub())
    monkeypatch.setattr(
        torch, "npu", _stub(manual_seed=lambda *_: None), raising=False
    )
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(benchmark, "_ensure_npu_ambient_shared", lambda _t: True)
    for name in (
        "TRITON_EVAL_REFERENCE_CACHE_DIR",
        "TRITON_EVAL_REFERENCE_CACHE_BINDING",
        "TRITON_EVAL_REFERENCE_CACHE_SHA256",
        "TRITON_EVAL_REFERENCE_CACHE_STATE_DIR",
        "TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID",
        "TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY",
        "TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def sys_modules():
    import sys

    return sys.modules


def _stub(**attrs):
    import types

    return types.SimpleNamespace(**attrs)


def make_cache(tmp_path, monkeypatch, *, seal, sentinel_factor):
    """Configured per-card cache env; ``seal`` stores entries for all cases."""
    runtime_file = tmp_path / "runtime-version"
    runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(benchmark.__file__).parent
    deployment = CacheDeployment(
        generation="cpu-controlled-optional-cache",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(runtime_file): sha256(runtime_file.read_bytes())},
        algorithm_files={
            name: sha256((engine / name).read_bytes())
            for name in ["verify.py", "benchmark.py", "_common_utils.py"]
        },
        device_conditions={"4": {"hardware_model": "controlled-cpu"}},
        measurement=CacheMeasurement(
            method="profiler", warmup=5, repeats=20, cv_threshold=0.3
        ),
    )
    cache = ReferenceCache(
        tmp_path / "cache", deployment, "4", entries={}, state_dir=tmp_path / "state"
    )
    source = (tmp_path / "example_torch.py").read_text(encoding="utf-8")
    inputs = [torch.ones(2)]
    samples_by_case = {}
    if seal:
        for case_idx in (1, 2):
            base = 1.0 + case_idx * 0.1
            samples = [round(base + k * 0.01, 4) for k in range(8)]
            samples_by_case[case_idx] = samples
            key, digest = cache.write_case(
                source, case_idx, inputs, inputs[0], repeated_outputs=inputs[0],
                reference_device_policy="npu-strict", selected_reference_device="npu",
                latency_samples_ms=samples,
            )
            cache.entries[key] = digest
    index_path = tmp_path / "index.json"
    index_sha = cache.write_index(index_path, cache.entries)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DIR", str(cache.root))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_BINDING", str(index_path))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_SHA256", index_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_STATE_DIR", str(cache.state_dir))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID", "4")
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")

    import statistics

    first_median = statistics.median(samples_by_case[1]) if seal else 1.0
    calls = {"framework": 0}

    def measure(context, **_kwargs):
        assert (context.warmup, context.repeats) == (5, 20)  # e4a36f7 parameters
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, round(first_median * sentinel_factor, 4), 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    return cache, samples_by_case, calls


def run(verify_dir):
    config = benchmark.BenchmarkConfig(
        "example", str(verify_dir), measurement_method="profiler"
    )
    result = benchmark.benchmark_implementations(config)
    return config, result


def test_eligible_hit_serves_cached_evidence_unchanged(tmp_path, monkeypatch, cpu_engine):
    verify_dir = write_task(tmp_path)
    _, samples_by_case, calls = make_cache(
        tmp_path, monkeypatch, seal=True, sentinel_factor=1.01
    )
    config, result = run(verify_dir)
    assert config.reference_measure_eight is False
    assert calls["framework"] == 8  # exactly the sentinel, nothing else
    for case in result.per_shape_results:
        assert case.framework.reference_path == "cached_manifest"
        assert case.framework.latency_samples_ms == samples_by_case[case.case_idx]
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "hit"


def test_ineligible_task_takes_the_original_e4a36f7_path(
    tmp_path, monkeypatch, cpu_engine
):
    # machine2 gate analog: cache configured, but the task reference is not a
    # pinned official one -> ORIGINAL path, one measurement per case.
    verify_dir = write_task(tmp_path, source=PLAIN_SOURCE)
    _, _, calls = make_cache(tmp_path, monkeypatch, seal=False, sentinel_factor=1.01)
    config, result = run(verify_dir)
    assert config.reference_measure_eight is False  # NOT the fresh-eight path
    assert config.reference_latencies_ms is None
    assert config.reference_cache_trace is None
    assert calls["framework"] == 2  # one original-path measurement per case
    for case in result.per_shape_results:
        assert case.framework.reference_path == "original_measurements"
        assert case.framework.latency_samples_ms is None
        assert case.framework.avg_latency_ms == 1.01  # the single measurement
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert "performance_cache" not in trace  # e4a36f7 writes no trace


def test_eligible_miss_keeps_e4a36f7_remeasure_semantics(tmp_path, monkeypatch, cpu_engine):
    verify_dir = write_task(tmp_path)
    _, _, calls = make_cache(tmp_path, monkeypatch, seal=False, sentinel_factor=1.01)
    config, result = run(verify_dir)
    assert config.reference_measure_eight is True  # e4a36f7 miss behavior
    assert calls["framework"] == 8 * 2
    for case in result.per_shape_results:
        assert case.framework.reference_path == "fresh_measurements"
        assert case.framework.latency_samples_source == "fresh_measurements"
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "miss"


def test_drift_invalidated_card_remeasures_like_e4a36f7(tmp_path, monkeypatch, cpu_engine):
    verify_dir = write_task(tmp_path)
    cache, _, calls = make_cache(tmp_path, monkeypatch, seal=True, sentinel_factor=1.05)
    config, result = run(verify_dir)
    assert config.reference_measure_eight is True
    assert calls["framework"] == 8 + 8 * 2  # sentinel, then eight per case
    for case in result.per_shape_results:
        assert case.framework.reference_path == "fresh_measurements"
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "remeasured"
    assert trace["performance_cache"]["sentinel_relative_drift"] > 0.03
    markers = list(Path(tmp_path, "state").rglob("card-*.invalid.json"))
    assert [marker.name for marker in markers] == ["card-4.invalid.json"]


def test_no_cache_configured_is_exactly_the_original_path(tmp_path, monkeypatch, cpu_engine):
    verify_dir = write_task(tmp_path)
    calls = {"framework": 0}

    def measure(context, **_kwargs):
        assert (context.warmup, context.repeats) == (5, 20)
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, 1.0, 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    config, result = run(verify_dir)
    assert config.reference_measure_eight is False
    assert config.reference_latencies_ms is None
    assert config.reference_cache_trace is None
    assert calls["framework"] == 2  # one original-path measurement per case
    for case in result.per_shape_results:
        assert case.framework.reference_path == "original_measurements"
        assert case.framework.latency_samples_ms is None
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert "performance_cache" not in trace
