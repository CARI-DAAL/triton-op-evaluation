"""Controlled CPU cache checks; native NPU A/B remains a release gate."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from triton_evaluation.config import Settings
from triton_evaluation.engine import benchmark, verify
from triton_evaluation.reference_cache import (
    BINDING_ENV,
    CACHE_ENV,
    DIGEST_ENV,
    STATE_ENV,
    CacheDeployment,
    CacheMeasurement,
    ReferenceCache,
    ReferenceCacheError,
    ReferenceCacheMiss,
    sha256,
)

torch = pytest.importorskip("torch")

SOURCE = '"""\nofficial_digest: ' + "a" * 64 + '\n"""\n_ADAPTER_VERSION = "test-v1"\n'


@pytest.fixture
def cache(tmp_path):
    runtime_file = tmp_path / "runtime-version"
    runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(verify.__file__).parent
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


def store(cache, inputs, output, *, case_idx=1, repeated=None, samples=None):
    key, digest = cache.write_case(
        SOURCE,
        case_idx,
        inputs,
        output,
        repeated_outputs=output if repeated is None else repeated,
        reference_device_policy="npu-strict",
        selected_reference_device="npu:0",
        latency_samples_ms=samples or [1.0] * 8,
    )
    cache.entries[key] = digest
    return cache.load_case(SOURCE, case_idx, inputs, reference_device_policy="npu-strict")


def test_roundtrip_multitensor_dtype_shape_structure_and_clone(cache):
    inputs = [torch.arange(6, dtype=torch.int32).reshape(2, 3), 0.25, object(), True]
    outputs = (
        torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        [torch.tensor(True), torch.empty(0, 2)],
        {"count": torch.tensor(7, dtype=torch.int64)},
    )
    manifest = store(cache, inputs, outputs)
    loaded = cache.outputs(manifest)
    assert isinstance(loaded, tuple) and isinstance(loaded[1], list)
    assert loaded[0].dtype == torch.bfloat16 and loaded[0].shape == (2, 3)
    assert loaded[1][0].dtype == torch.bool and loaded[1][0].shape == ()
    assert loaded[1][1].shape == (0, 2)
    assert torch.equal(loaded[2]["count"], outputs[2]["count"])
    loaded[0].fill_(100)
    assert torch.equal(cache.outputs(manifest)[0], outputs[0])
    fresh = [inputs[0].clone(), 0.25, object(), True]
    assert cache.load_case(SOURCE, 1, fresh, reference_device_policy="npu-strict") == manifest


@pytest.mark.parametrize(
    "change", ["source", "input", "dtype", "shape", "stride", "case", "card", "policy"]
)
def test_different_immutable_binding_is_a_real_miss(cache, change):
    inputs = [torch.arange(4, dtype=torch.float32).reshape(2, 2)]
    store(cache, inputs, inputs[0])
    source, case, policy = SOURCE, 1, "npu-strict"
    if change == "source":
        source += "# changed reference\n"
    elif change == "input":
        inputs[0] = inputs[0] + 1
    elif change == "dtype":
        inputs[0] = inputs[0].half()
    elif change == "shape":
        inputs[0] = inputs[0].reshape(4)
    elif change == "stride":
        inputs[0] = inputs[0].t()
    elif change == "case":
        case = 2
    elif change == "card":
        cache.device_id = "4"
    elif change == "policy":
        policy = "cpu-first"
    with pytest.raises(ReferenceCacheMiss):
        cache.load_case(source, case, inputs, reference_device_policy=policy)


@pytest.mark.parametrize("target", ["manifest", "blob", "missing_manifest", "missing_blob"])
def test_pinned_corruption_is_never_a_miss_or_an_answer(cache, target):
    inputs = [torch.ones(3)]
    manifest = store(cache, inputs, torch.ones(3))
    if "manifest" in target:
        path = cache.root / "cases" / f"{manifest.binding.key}.json"
    else:
        path = cache.root / "blobs" / f"{manifest.outputs['sha256']}.bin"
    if target.startswith("missing"):
        path.unlink()
    else:
        path.write_bytes(b"tampered")
    with pytest.raises((ReferenceCacheError, FileNotFoundError)):
        loaded = cache.load_case(SOURCE, 1, inputs, reference_device_policy="npu-strict")
        cache.outputs(loaded)


def test_nondeterministic_reference_and_invalid_measurements_refused(cache):
    with pytest.raises(ReferenceCacheError, match="not bitwise deterministic"):
        store(cache, [torch.ones(2)], torch.ones(2), repeated=torch.zeros(2))
    with pytest.raises(ValidationError):
        store(cache, [], torch.ones(1), samples=[float("nan")] * 8)
    with pytest.raises(ValidationError):
        store(cache, [], torch.ones(1), samples=[1.0] * 7)


def test_index_pin_runtime_drift_and_incomplete_deployment(cache, tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    digest = cache.write_index(index, {})
    monkeypatch.setenv(CACHE_ENV, str(cache.root))
    with pytest.raises(ReferenceCacheError, match="all reference cache"):
        ReferenceCache.from_environment()
    monkeypatch.setenv(BINDING_ENV, str(index))
    monkeypatch.setenv(DIGEST_ENV, digest)
    monkeypatch.setenv(STATE_ENV, str(cache.state_dir))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    assert ReferenceCache.from_environment().deployment.digest == cache.deployment.digest
    index.write_bytes(index.read_bytes() + b" ")
    with pytest.raises(ReferenceCacheError, match="pinned SHA256"):
        ReferenceCache.from_environment()
    index.write_bytes(index.read_bytes().rstrip())
    Path(next(iter(cache.deployment.runtime_files))).write_text("runtime changed")
    with pytest.raises(ReferenceCacheError, match="runtime file changed"):
        ReferenceCache.from_environment()


def test_immutable_index_cannot_be_overwritten(cache, tmp_path):
    index = tmp_path / "index.json"
    digest = cache.write_index(index, {})
    assert cache.write_index(index, {}) == digest
    with pytest.raises(ReferenceCacheError, match="immutable"):
        cache.write_index(index, {"a" * 64: "b" * 64})


def test_drift_invalidates_whole_card_generation_monotonically(cache):
    assert not cache.needs_remeasurement(1.0, 1.02)
    assert not cache.needs_remeasurement(1.0, 1.03)
    assert cache.needs_remeasurement(1.0, 1.031)
    assert cache.needs_remeasurement(1.0, 1.0)
    other = ReferenceCache(
        cache.root, cache.deployment, "4", entries=cache.entries, state_dir=cache.state_dir
    )
    assert not other.needs_remeasurement(1.0, 1.0)
    with pytest.raises(ReferenceCacheError, match="sentinel"):
        cache.needs_remeasurement(1.0, float("inf"))


@pytest.mark.parametrize(
    "non_compute,dtype",
    [(False, torch.float32), (False, torch.int32), (False, torch.bool), (True, torch.float32)],
)
def test_hit_executes_candidate_with_existing_comparison(cache, monkeypatch, non_compute, dtype):
    output = torch.ones(3, dtype=dtype)
    inputs = [output.clone()]
    manifest = store(cache, inputs, output)
    forwards = []

    class Candidate(torch.nn.Module):
        def forward(self, value):
            forwards.append(1)
            return value.clone()

    def forbidden_reference(*_args):
        raise RuntimeError("reference constructor ran on cache hit")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed=lambda _: None), raising=False)
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    context = verify.CaseContext(
        1,
        1,
        reference_trace=[],
        cached_reference=(manifest, cache.outputs(manifest)),
        reference_cache_status="hit",
    )
    modules = {
        "framework_cls": forbidden_reference,
        "impl_cls": Candidate,
        "get_init_inputs": lambda: [],
    }
    assert verify._try_run_case(
        modules,
        context,
        inputs,
        torch.device("cpu"),
        non_compute=non_compute,
        reference_device_policy="npu-strict",
    ) == (True, None)
    assert forwards == [1]
    assert context.reference_trace[0]["reference_completed"] is False
    assert context.reference_trace[0]["reference_output_sha256"] == manifest.output_sha256


def test_wrong_candidate_is_wrong_with_cache(cache, monkeypatch):
    output = torch.ones(3)
    inputs = [output.clone()]
    manifest = store(cache, inputs, output)

    class Wrong(torch.nn.Module):
        def forward(self, value):
            return value * 0

    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed=lambda _: None), raising=False)
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    context = verify.CaseContext(
        1, 1, reference_trace=[], cached_reference=(manifest, cache.outputs(manifest))
    )
    passed, failure = verify._try_run_case(
        {"impl_cls": Wrong, "get_init_inputs": lambda: []},
        context,
        inputs,
        torch.device("cpu"),
        reference_device_policy="npu-strict",
    )
    assert not passed and failure["error_type"] == "AccuracyError"


@pytest.mark.parametrize("sentinel", [1.01, 1.05])
def test_sentinel_selects_hit_or_full_remeasurement(cache, tmp_path, monkeypatch, sentinel):
    inputs = [[torch.ones(3)], [torch.ones(2)]]
    for index, case in enumerate(inputs, start=1):
        store(cache, case, case[0], case_idx=index)
    (tmp_path / "example_torch.py").write_text(SOURCE)
    monkeypatch.setattr(ReferenceCache, "from_environment", lambda: cache)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed=lambda _: None), raising=False)
    monkeypatch.setattr(benchmark, "cleanup_npu_memory", lambda: None)
    measurements = []

    def measure(context, **_kwargs):
        measurements.append(context.profile_name)
        return {}, sentinel if context.profile_name.startswith("framework") else 0.5, 4.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    config = benchmark.BenchmarkConfig("example", str(tmp_path), measurement_method="event")
    spec = benchmark.BenchmarkModelSpec(torch.nn.Identity, torch.nn.Identity, lambda: [])
    benchmark._configure_reference_cache(config, spec, inputs, torch.device("cpu"))
    assert measurements == ["framework_profile_case1"] * 8
    assert config.reference_cache_trace["status"] == ("hit" if sentinel == 1.01 else "remeasured")
    result = benchmark.run_single_benchmark(
        benchmark.ModelPair(torch.nn.Identity(), torch.nn.Identity()),
        inputs[1],
        config,
        torch.device("cpu"),
        benchmark.CaseContext(2, 2),
    )
    assert result[0].avg_latency_ms == (1.0 if sentinel == 1.01 else 1.05)
    assert measurements.count("impl_profile_case2") == 1
    assert measurements.count("framework_profile_case2") == (0 if sentinel == 1.01 else 8)


def test_missing_card_runs_full_reference(cache, tmp_path, monkeypatch):
    (tmp_path / "example_torch.py").write_text(SOURCE)
    monkeypatch.setattr(ReferenceCache, "from_environment", lambda: cache)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    config = benchmark.BenchmarkConfig("example", str(tmp_path), measurement_method="event")
    benchmark._configure_reference_cache(config, SimpleNamespace(), [[torch.ones(1)]], "cpu")
    assert config.reference_cache_trace["status"] == "miss"
    assert config.reference_measure_eight and config.reference_latencies_ms is None


def test_config_cache_index_attestation_preserves_existing_anchors(tmp_path):
    with pytest.raises(ValidationError, match="four reference cache"):
        Settings(reference_cache_dir=tmp_path)
    settings = Settings(
        reference_cache_dir=tmp_path,
        reference_cache_binding=tmp_path / "index",
        reference_cache_sha256="a" * 64,
        reference_cache_state_dir=tmp_path / "state",
    )
    assert settings.effective_correctness_algorithm_version.endswith(":cache-v1:" + "a" * 64)
    assert settings.correctness_algorithm_version == "correctness-v1"


def test_retained_private_cache_trace_has_no_tensor_values(tmp_path):
    from triton_evaluation.reference_diagnostics import _read_trace

    trace = {
        "reference_device_policy": "npu-strict",
        "cases": [
            {
                "case_idx": 1,
                "selected_reference_device": "npu:0",
                "reference_completed": False,
                "output_devices": [{"output_index": 0, "device": "cpu"}],
                "reference_cache_status": "hit",
                "reference_manifest_sha256": "a" * 64,
                "reference_binding_sha256": "b" * 64,
                "reference_output_sha256": "c" * 64,
            }
        ],
        "performance_cache": {
            "deployment_sha256": "d" * 64,
            "device_id": "3",
            "status": "hit",
            "manifests": ["a" * 64],
            "reference_memory_measured": False,
            "sentinel_case_idx": 1,
            "sentinel_cached_ms": 1.0,
            "sentinel_measured_ms": 1.01,
            "sentinel_relative_drift": 0.01,
        },
    }
    path = tmp_path / "verify_reference.json"
    path.write_text(json.dumps(trace))
    assert _read_trace(path, "npu-strict") == {"status": "recorded", "observations": trace}


def test_real_engines_share_lazy_seedzero_official_input_route(monkeypatch):
    requests = []

    def single(index):
        requests.append(index)
        # Simulate all preceding draws of the official adapter, releasing old tensors.
        values = None
        for _ in range(index + 1):
            values = [torch.randn(2)]
        return values

    module = SimpleNamespace(
        _ADAPTER_VERSION="official-test",
        get_case_count=lambda: 3,
        get_input_group=single,
        get_input_groups=lambda: pytest.fail("eager input provider used"),
    )
    groups, total = verify.resolve_input_provider(module)
    assert total == 3 and requests == []
    torch.manual_seed(0)
    expected = [torch.randn(2) for _ in range(3)]
    assert torch.equal(groups[2][0], expected[2])
    torch.manual_seed(999)
    assert torch.equal(groups[2][0], expected[2])
    assert requests == [2, 2]


def test_seal_index_validates_blobs_and_preserves_unknown_coverage(cache, tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "reference_cache_index.py"
    spec = importlib.util.spec_from_file_location("cache_index_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seal_index = module.seal_index

    manifest = store(cache, [torch.ones(3)], torch.ones(3))
    coverage = seal_index(cache.root, cache.deployment, tmp_path / "sealed.json")
    assert coverage["entry_count"] == 1
    assert coverage["coverage_complete"] is None
    assert coverage["observed_case_indices_by_source_and_card"] == {
        manifest.binding.reference_source_sha256: {"3": [1]}
    }
    blob = cache.root / "blobs" / f"{manifest.outputs['sha256']}.bin"
    blob.write_bytes(b"tampered")
    with pytest.raises(ReferenceCacheError, match="blob hash"):
        seal_index(cache.root, cache.deployment, tmp_path / "second.json")


def test_official_cache_miss_really_executes_reference_and_candidate(cache, tmp_path, monkeypatch):
    import sys

    reference_calls = []
    candidate_calls = []

    class Reference(torch.nn.Module):
        def forward(self, value):
            reference_calls.append(1)
            return value + 1

    class Candidate(torch.nn.Module):
        def forward(self, value):
            candidate_calls.append(1)
            return value + 1

    module = SimpleNamespace(get_input_groups=lambda: [[torch.ones(2)], [torch.ones(3)]])
    modules = {
        "torch_module": module,
        "framework_cls": Reference,
        "impl_cls": Candidate,
        "get_init_inputs": lambda: [],
    }
    (tmp_path / "example_torch.py").write_text(SOURCE)
    monkeypatch.setattr(ReferenceCache, "from_environment", lambda: cache)
    monkeypatch.setattr(verify, "_load_verify_modules", lambda *_: modules)
    monkeypatch.setattr(verify, "_ensure_npu_ambient_shared", lambda _t, **_k: None)
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed=lambda _: None), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda name: real_device("cpu" if name == "npu" else name))
    assert verify.verify_implementations(
        "example", str(tmp_path), reference_device_policy="npu-strict"
    ) == (2, 2)
    assert reference_calls == [1, 1] and candidate_calls == [1, 1]
    trace = json.loads((tmp_path / "verify_reference.json").read_text())
    assert all(case["reference_cache_status"] == "miss" for case in trace["cases"])
    assert all(case["reference_completed"] for case in trace["cases"])
