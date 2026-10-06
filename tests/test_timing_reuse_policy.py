"""Validated historical timing reuse: owner-pinned policy at the real lookup.

CPU tests through the REAL engine/evaluator/API path (fake ``measure_single``
seam), the REAL sandbox delivery path (``isolate_command`` argv after
``--clearenv``), plus offline acceptance against the ACTUAL retained artifacts:
sealed 2026-10-04 old cache index (1046 entries), real old manifests, staged
patched runtime files, and the owner policy artifact bound to this source.
Fail-closed behavior asserted everywhere.
"""

import hashlib
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from triton_evaluation import reference_timing_reuse as reuse_module
from triton_evaluation.api.contract import PerformanceCase
from triton_evaluation.engine import benchmark, verify
from triton_evaluation.evaluators import sandbox
from triton_evaluation.evaluators.shaping import shape_performance
from triton_evaluation.reference_cache import (
    CacheDeployment,
    CacheMeasurement,
    CaseBinding,
    CaseManifest,
    ReferenceCache,
    sha256,
)
from triton_evaluation.reference_timing_reuse import (
    POLICY_ENV,
    POLICY_SHA_ENV,
    TimingReuseError,
    TimingReusePolicy,
    load_validated_timing_reuse,
)

torch = pytest.importorskip("torch")

SOURCE = '"""\nofficial_digest: ' + "a" * 64 + '\n"""\n' + '_ADAPTER_VERSION = "test-v1"\n'
FRESH_SAMPLES = [1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.03, 0.97]
IMPL_LATENCY = 0.5

# Actual retained artifacts (read-only; retained for this task).
REAL_INDEX = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/state/sealed-cache-index.json")
REAL_INDEX_SHA = "4c165dfca2675951c7dabf3caeabbef22f54d69a44e1540e95110b470db4fa87"
REAL_CACHE_DIR = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/cache")
REAL_DEPLOYMENT_SHA = "84154dfb56d4a06a818cee7e110589d27817fce784881edaa864210412932f5d"
STAGED_TRITON = Path(
    "/root/.paseo/worktrees/08x5542i/v3-cache-cutover-prep-20261003/.scratch/"
    "v3-cache-cutover-prep/formal-staging-v1-patched-triton/staged-triton"
)
REAL_DRIVER_SHA = "3942188b279b331b82abece2a59b4837cf8fb5d5ab894840876533c7478e297b"
REAL_NPU_UTILS_SHA = "6bd908f6e53c8ae243a44def9af8a3706034fcbab7cc627658b6742a3f609140"
DELIVERY_DIR = Path(__file__).parents[1] / (
    ".scratch/v3-service-reference8/timing-reuse-delivery"
)
REAL_POLICY = DELIVERY_DIR / "validated-timing-reuse-policy-v2.json"

requires_real_artifacts = pytest.mark.skipif(
    not REAL_INDEX.is_file() or not REAL_CACHE_DIR.is_dir(),
    reason="retained old-cache artifacts are not present on this host",
)


# ---------------------------------------------------------------------------
# Engine helpers (mirrors phase-1 tests; fake measurement seam)
# ---------------------------------------------------------------------------

def write_task(tmp_path: Path, cases: int = 6, size: int = 2) -> str:
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # engine imports task modules by name; drop any earlier test's modules
    sys.modules.pop("example_torch", None)
    sys.modules.pop("example_triton_ascend_impl", None)
    (tmp_path / "example_torch.py").write_text(
        SOURCE
        + "import torch\n\n\n"
        + "class Model(torch.nn.Module):\n"
        + "    def forward(self, value):\n"
        + "        return value + 1\n\n\n"
        + "def get_init_inputs():\n"
        + "    return []\n\n\n"
        + "def get_input_groups():\n"
        + f"    return [[torch.ones({size})] for _ in range({cases})]\n",
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
    monkeypatch.delitem(sys.modules, "example_torch", raising=False)
    monkeypatch.delitem(sys.modules, "example_triton_ascend_impl", raising=False)
    for name in (POLICY_ENV, POLICY_SHA_ENV):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "TRITON_EVAL_REFERENCE_CACHE_DIR",
        "TRITON_EVAL_REFERENCE_CACHE_BINDING",
        "TRITON_EVAL_REFERENCE_CACHE_SHA256",
        "TRITON_EVAL_REFERENCE_CACHE_STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = {"framework": 0}

    def measure(context, **_kwargs):
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, FRESH_SAMPLES[calls["framework"] % 8], 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    return calls


def make_deployment(tmp_path: Path) -> CacheDeployment:
    runtime_file = tmp_path / "runtime-version"
    runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(benchmark.__file__).parent
    return CacheDeployment(
        generation="cpu-controlled-reuse",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(runtime_file): sha256(runtime_file.read_bytes())},
        algorithm_files={
            name: sha256((engine / name).read_bytes())
            for name in ["verify.py", "benchmark.py", "_common_utils.py"]
        },
        device_conditions={"6": {"hardware_model": "controlled-cpu"}},
        measurement=CacheMeasurement(
            method="profiler", warmup=5, repeats=20, cv_threshold=0.3
        ),
    )


def store_old_case(cache, source, case_idx, samples):
    inputs = [torch.ones(2)]
    key, digest = cache.write_case(
        source,
        case_idx,
        inputs,
        inputs[0],
        repeated_outputs=inputs[0],
        reference_device_policy="npu-strict",
        selected_reference_device="npu",
        latency_samples_ms=samples,
    )
    cache.entries[key] = digest


def build_policy(tmp_path: Path, *, cache_dir, index_path, index_sha, state_dir, devices,
                 patched, service_files=None, deployment_sha=None):
    if service_files is None:
        service_files = {
            name: reuse_module.current_service_source_sha256(name)
            for name in ("engine/benchmark.py", "evaluators/sandbox.py")
        }
    policy = {
        "schema_version": "validated-timing-reuse-policy-v2",
        "scope_statement": "six sampled BMM cases within +/-3%; owner-adopted policy",
        "source_deployment_sha256": deployment_sha or make_deployment(tmp_path).digest,
        "source_cache_dir": str(cache_dir),
        "source_index_path": str(index_path),
        "source_index_sha256": index_sha,
        "source_state_dir": str(state_dir),
        "source_devices": list(devices),
        "timing_contract": {"method": "profiler", "warmup": 5, "repeats": 20, "samples": 8},
        "contract_reconciliation": (
            "old audit label and reference8 use the same underlying "
            "8x(warmup5/repeats20) profiler observations; reconciled explicitly"
        ),
        "target_patched_runtime_files": patched,
        "target_service_source_sha256": service_files,
        "evidence": {"sampled_problem": "official_l3_BatchMatmul"},
    }
    path = tmp_path / "policy.json"
    payload = json.dumps(policy, indent=2, sort_keys=True).encode()
    path.write_bytes(payload)
    return path, sha256(payload)


def configure_reuse(tmp_path, monkeypatch, *, cases=6, devices=("6",), patched=None,
                    size=2, sentinel_factor=1.01):
    """Old cache + sealed index + v2 policy + sentinel-scripted measurements."""
    tmp_path = Path(tmp_path)
    verify_dir = Path(write_task(tmp_path, cases=cases, size=size))
    source = (verify_dir / "example_torch.py").read_text(encoding="utf-8")
    deployment = make_deployment(tmp_path)
    old_cache = ReferenceCache(
        tmp_path / "old-cache", deployment, "6", entries={}, state_dir=tmp_path / "state"
    )
    samples_by_case = {}
    for case_idx in range(1, cases + 1):
        base = 1.0 + case_idx * 0.1
        samples = [round(base + k * 0.01, 4) for k in range(8)]
        store_old_case(old_cache, source, case_idx, samples)
        samples_by_case[case_idx] = samples
    index_path = tmp_path / "sealed-index.json"
    index_sha = old_cache.write_index(index_path, old_cache.entries)
    state_dir = tmp_path / "reuse-state"
    state_dir.mkdir()
    patched = patched or {
        "backends/ascend/driver.py": "a" * 64,
        "backends/ascend/npu_utils.cpp": "b" * 64,
    }
    policy_path, policy_sha = build_policy(
        tmp_path,
        cache_dir=old_cache.root,
        index_path=index_path,
        index_sha=index_sha,
        state_dir=state_dir,
        devices=devices,
        patched=patched,
        deployment_sha=deployment.digest,
    )
    monkeypatch.setenv(POLICY_ENV, str(policy_path))
    monkeypatch.setenv(POLICY_SHA_ENV, policy_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID", devices[0])
    monkeypatch.setattr(
        reuse_module,
        "current_patched_runtime_sha256",
        lambda name: patched.get(name),
    )
    # The live sentinel measures the first selected case; keep it within (or
    # beyond) the existing +/-3% drift threshold on demand.
    sentinel_value = statistics.median(samples_by_case[1]) * sentinel_factor
    calls = {"framework": 0}

    def measure(context, **_kwargs):
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, sentinel_value, 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    return verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment


def public_per_case(result):
    performance = shape_performance(benchmark.result_to_dict(result))
    return performance, performance.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Real retained artifacts (offline acceptance)
# ---------------------------------------------------------------------------

@requires_real_artifacts
def test_real_sealed_index_is_pinned_and_carries_1046_real_bindings():
    payload = REAL_INDEX.read_bytes()
    assert sha256(payload) == REAL_INDEX_SHA
    index = reuse_module.CacheIndex.model_validate_json(payload)
    assert index.deployment.digest == REAL_DEPLOYMENT_SHA
    assert index.deployment.measurement.method == "profiler"
    assert (index.deployment.measurement.warmup, index.deployment.measurement.repeats) == (5, 20)
    assert len(index.entries) == 1046
    # A real manifest round-trips through its sealed binding pin.
    manifest_path = sorted((REAL_CACHE_DIR / "cases").glob("*.json"))[0]
    manifest_payload = manifest_path.read_bytes()
    manifest = CaseManifest.model_validate_json(manifest_payload)
    key = CaseBinding.model_validate(manifest.binding.model_dump()).key
    assert index.entries[key] == sha256(manifest_payload)
    assert len(manifest.latency_samples_ms) == 8
    assert all(0.0 < value < float("inf") for value in manifest.latency_samples_ms)
    assert manifest.reference_latency_median_ms == statistics.median(
        manifest.latency_samples_ms
    )


@requires_real_artifacts
def test_real_policy_binds_real_identities_and_real_patched_files():
    assert REAL_POLICY.is_file()
    payload = REAL_POLICY.read_bytes()
    policy = TimingReusePolicy.model_validate_json(payload)
    assert policy.source_index_sha256 == REAL_INDEX_SHA
    assert policy.source_deployment_sha256 == REAL_DEPLOYMENT_SHA
    assert Path(policy.source_state_dir).is_absolute()
    assert policy.timing_contract.method == "profiler"
    assert (policy.timing_contract.warmup, policy.timing_contract.repeats) == (5, 20)
    assert policy.timing_contract.samples == 8
    assert policy.evidence["sampled_cases_one_based"] == [1, 6, 11, 18, 33, 43]
    # The declared patched runtime files are the real staged bytes.
    staged_driver = sha256((STAGED_TRITON / "backends/ascend/driver.py").read_bytes())
    assert staged_driver == REAL_DRIVER_SHA
    assert policy.target_patched_runtime_files["backends/ascend/driver.py"] == staged_driver
    staged_utils = sha256((STAGED_TRITON / "backends/ascend/npu_utils.cpp").read_bytes())
    assert staged_utils == REAL_NPU_UTILS_SHA
    assert policy.target_patched_runtime_files["backends/ascend/npu_utils.cpp"] == staged_utils
    # The policy binds this actual service source (engine lookup + delivery).
    for name, expected in policy.target_service_source_sha256.items():
        assert reuse_module.current_service_source_sha256(name) == expected


@requires_real_artifacts
def test_real_policy_denies_reuse_on_this_unpatched_host(monkeypatch):
    # This host has no patched triton runtime: the real policy must authorize
    # nothing here (fail-closed), never fall back to relabeled reuse.
    monkeypatch.setenv(POLICY_ENV, str(REAL_POLICY))
    monkeypatch.setenv(POLICY_SHA_ENV, sha256(REAL_POLICY.read_bytes()))
    assert load_validated_timing_reuse() is None


# ---------------------------------------------------------------------------
# Real engine path: reuse, sentinel and fail-closed behavior
# ---------------------------------------------------------------------------

def test_valid_policy_reuses_old_eight_samples_after_a_fresh_sentinel(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = configure_reuse(
        tmp_path, monkeypatch, cases=6
    )
    config = benchmark.BenchmarkConfig(
        "example", str(verify_dir), measurement_method="profiler",
        case_indices=[1, 6, 3],
    )
    result = benchmark.benchmark_implementations(config)

    # exactly one fresh eight-sample sentinel; zero per-case reference timing
    assert calls["framework"] == 8
    assert [case.case_idx for case in result.per_shape_results] == [1, 3, 6]  # task order
    for case in result.per_shape_results:
        assert case.framework.latency_samples_ms == samples_by_case[case.case_idx]
        assert case.framework.latency_samples_source == "historical_validated"
        assert case.framework.latency_samples_proof_sha256 == policy_sha
        assert case.framework.avg_latency_ms == round(
            statistics.median(samples_by_case[case.case_idx]), 4
        )
        assert case.speedup_vs_torch == round(
            statistics.median(samples_by_case[case.case_idx]) / IMPL_LATENCY, 4
        )

    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "timing_reuse"
    assert trace["performance_cache"]["timing_reuse_policy_sha256"] == policy_sha
    assert trace["performance_cache"]["sentinel_case_idx"] == 1
    assert trace["performance_cache"]["sentinel_cached_ms"] == statistics.median(
        samples_by_case[1]
    )
    assert len(trace["performance_cache"]["manifests"]) == 3
    # No drift marker: the sentinel stayed within the existing threshold.
    assert not list(state_dir.rglob("card-*.invalid.json"))

    performance, payload = public_per_case(result)
    assert [case.case_idx for case in performance.per_case] == [0, 2, 5]
    for case in performance.per_case:
        assert case.reference_latency_samples_source == "historical_validated"
        assert case.reference_timing_policy_sha256 == policy_sha
        assert case.reference_latency_samples_ms == samples_by_case[case.case_idx + 1]
    assert payload["per_case"][0]["reference_timing_policy_sha256"] == policy_sha


def test_historical_sentinel_drift_forces_fresh_measurement_monotonically(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = configure_reuse(
        tmp_path, monkeypatch, cases=2, sentinel_factor=1.05  # > 3% drift
    )
    config = benchmark.BenchmarkConfig(
        "example", str(verify_dir), measurement_method="profiler"
    )
    result = benchmark.benchmark_implementations(config)
    # sentinel (8) + the ORIGINAL e4a36f7 path (one measurement per case):
    # nothing was reused, and with no cache configured the service runs its
    # original reference path exactly as production.
    assert calls["framework"] == 8 + 2
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.latency_samples_ms is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "timing_reuse_drift"
    assert trace["performance_cache"]["sentinel_relative_drift"] > 0.03
    markers = list(state_dir.rglob("card-*.invalid.json"))
    assert len(markers) == 1  # monotone marker under the source generation digest
    assert markers[0].parent.name == deployment.digest

    # A later in-range sentinel cannot restore reuse within this generation.
    calls["framework"] = 0
    (verify_dir / "verify_reference.json").write_text("{}", encoding="utf-8")
    in_range = statistics.median(samples_by_case[1]) * 1.01

    def measure(context, **_kwargs):
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, in_range, 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 8 + 2  # sentinel still runs, then original path
    assert all(
        case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "timing_reuse_drift"


def test_absent_policy_preserves_exact_original_e4a36f7_behavior(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir = write_task(tmp_path, cases=2)
    config = benchmark.BenchmarkConfig(
        "example", str(verify_dir), measurement_method="profiler"
    )
    result = benchmark.benchmark_implementations(config)
    # No cache and no policy: the ORIGINAL e4a36f7 path measures each case
    # exactly once; no sample evidence, no trace, per-case path recorded.
    assert cpu_engine["framework"] == 2
    assert result.per_shape_results[0].framework.latency_samples_source is None
    assert result.per_shape_results[0].framework.latency_samples_ms is None
    assert result.per_shape_results[0].framework.reference_path == (
        "original_measurements"
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert "performance_cache" not in trace


def test_tampered_policy_is_an_infrastructure_error_never_reuse(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir, *_ = configure_reuse(tmp_path, monkeypatch, cases=2)
    import os

    policy_path = Path(os.environ[POLICY_ENV])
    policy_path.write_bytes(policy_path.read_bytes() + b" \n")
    config = benchmark.BenchmarkConfig(
        "example", str(verify_dir), measurement_method="profiler"
    )
    with pytest.raises(TimingReuseError, match="pinned SHA256"):
        benchmark.benchmark_implementations(config)


def test_wrong_transition_input_device_or_protocol_cannot_yield_reuse(
    tmp_path, monkeypatch, cpu_engine
):
    # wrong target runtime identity
    verify_dir, _, _, calls, _, _ = configure_reuse(tmp_path, monkeypatch, cases=2)
    monkeypatch.setattr(
        reuse_module, "current_patched_runtime_sha256", lambda name: "0" * 64
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2  # original path: one measurement per case
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )

    # wrong physical device: card 9 was never audited, never borrowed
    verify_dir, _, _, calls, _, _ = configure_reuse(
        tmp_path / "d9", monkeypatch, cases=2, devices=("9",)
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2

    # wrong inputs: different tensor bytes cannot match the sealed binding
    verify_dir, _, _, calls, _, _ = configure_reuse(
        tmp_path / "in", monkeypatch, cases=2, size=3
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2

    # wrong measurement protocol: a repeats=50 request takes the original path
    verify_dir, _, _, calls, _, _ = configure_reuse(tmp_path / "pr", monkeypatch, cases=2)
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig(
            "example", str(verify_dir), measurement_method="profiler", repeats=50
        )
    )
    assert calls["framework"] == 2  # one original-path measurement per case
    assert all(
        case.framework.latency_samples_source is None
        for case in result.per_shape_results
    )


def test_partial_coverage_follows_original_execution_for_all_cases(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir = Path(write_task(tmp_path, cases=6))
    source = (verify_dir / "example_torch.py").read_text(encoding="utf-8")
    deployment = make_deployment(tmp_path)
    old_cache = ReferenceCache(
        tmp_path / "old-cache", deployment, "6", entries={}, state_dir=tmp_path / "state"
    )
    store_old_case(old_cache, source, 1, [1.0] * 8)  # only case 1 is represented
    index_path = tmp_path / "sealed-index.json"
    index_sha = old_cache.write_index(index_path, old_cache.entries)
    state_dir = tmp_path / "reuse-state"
    state_dir.mkdir()
    patched = {"backends/ascend/driver.py": "a" * 64}
    policy_path, policy_sha = build_policy(
        tmp_path,
        cache_dir=old_cache.root,
        index_path=index_path,
        index_sha=index_sha,
        state_dir=state_dir,
        devices=("6",),
        patched=patched,
        deployment_sha=deployment.digest,
    )
    monkeypatch.setenv(POLICY_ENV, str(policy_path))
    monkeypatch.setenv(POLICY_SHA_ENV, policy_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID", "6")
    monkeypatch.setattr(
        reuse_module, "current_patched_runtime_sha256", lambda name: patched.get(name)
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    # case 2 has no old binding: the whole run follows the original execution
    # path (no cache configured -> one measurement per case); nothing is
    # partially relabeled.
    assert cpu_engine["framework"] == 6
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )


def test_current_cache_miss_can_fall_back_to_validated_reuse(
    tmp_path, monkeypatch, cpu_engine
):
    # A same-generation cache is configured but has no entry: the validated
    # historical policy may serve the timing after its own fresh sentinel.
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = configure_reuse(
        tmp_path, monkeypatch, cases=2
    )
    current = ReferenceCache(
        tmp_path / "current-cache", deployment, "6", entries={}, state_dir=tmp_path / "cstate"
    )
    index_path = tmp_path / "current-index.json"
    index_sha = current.write_index(index_path, current.entries)  # empty: always miss
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DIR", str(current.root))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_BINDING", str(index_path))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_SHA256", index_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_STATE_DIR", str(current.state_dir))
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 8  # historical sentinel only
    assert all(
        case.framework.latency_samples_source == "historical_validated"
        and case.framework.latency_samples_proof_sha256 == policy_sha
        for case in result.per_shape_results
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())
    assert trace["performance_cache"]["status"] == "timing_reuse"


def test_correctness_engine_never_consults_the_reuse_policy():
    source = Path(verify.__file__).read_text(encoding="utf-8")
    assert "reference_timing_reuse" not in source
    assert "TIMING_REUSE" not in source


def test_policy_settings_pair_and_public_contract_validation(tmp_path):
    from triton_evaluation.config import Settings

    with pytest.raises(ValidationError):
        Settings(reference_timing_reuse_policy=tmp_path / "p.json")
    with pytest.raises(ValidationError):
        Settings(
            reference_timing_reuse_policy=Path("../escape.json"),
            reference_timing_reuse_sha256="a" * 64,
        )
    settings = Settings(
        reference_timing_reuse_policy=tmp_path / "p.json",
        reference_timing_reuse_sha256="a" * 64,
    )
    assert settings.reference_timing_reuse_sha256 == "a" * 64

    # public honesty: historical samples require the policy proof digest
    with pytest.raises(ValidationError):
        PerformanceCase(
            case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
            reference_latency_ms=1.0, speedup_vs_reference=2.0,
            reference_latency_samples_ms=[1.0] * 8,
            reference_latency_samples_source="historical_validated",
        )
    with pytest.raises(ValidationError):
        PerformanceCase(
            case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
            reference_latency_ms=1.0, speedup_vs_reference=2.0,
            reference_latency_samples_ms=[1.0] * 8,
            reference_latency_samples_source="fresh_measurements",
            reference_timing_policy_sha256="a" * 64,
        )
    case = PerformanceCase(
        case_idx=0, input_shapes=[], implementation_latency_ms=0.5,
        reference_latency_ms=1.0, speedup_vs_reference=2.0,
        reference_latency_samples_ms=[1.0] * 8,
        reference_latency_samples_source="historical_validated",
        reference_timing_policy_sha256="a" * 64,
    )
    assert case.reference_timing_policy_sha256 == "a" * 64


# ---------------------------------------------------------------------------
# Real sandbox delivery: the explicit pair survives --clearenv, nothing else
# ---------------------------------------------------------------------------

@pytest.fixture
def delivery(tmp_path, monkeypatch):
    """Base sandbox manifest + overlay carrying ONLY the policy pair."""
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = configure_reuse(
        tmp_path, monkeypatch, cases=2
    )
    policy_path = Path(__import__("os").environ[POLICY_ENV])
    index_path = Path(json.loads(policy_path.read_text())["source_index_path"])
    cache_dir = Path(json.loads(policy_path.read_text())["source_cache_dir"])
    mounts = [(path, path) for path in dict.fromkeys(("/usr", sys.base_prefix, sys.prefix))]
    original = {
        "readonly_mounts": mounts,
        "environment": {"PATH": "/usr/bin:/bin"},
    }
    base = tmp_path / "base-sandbox.json"
    base.write_text(json.dumps(original))
    monkeypatch.setenv(sandbox.CONFIG_ENV, str(base))
    monkeypatch.setenv(sandbox.DIGEST_ENV, hashlib.sha256(base.read_bytes()).hexdigest())
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "npu-strict")
    overlay = sandbox.EngineRuntimeOverlay(
        environment={
            POLICY_ENV: str(policy_path),
            POLICY_SHA_ENV: policy_sha,
        }
    )
    overlay_path = tmp_path / "runtime-overlay.json"
    overlay_path.write_text(overlay.model_dump_json())
    monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(overlay_path))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, sha256(overlay_path.read_bytes()))
    work = tmp_path / "attempt"
    work.mkdir()
    return {
        "tmp_path": tmp_path, "base": base, "overlay_path": overlay_path,
        "policy_path": policy_path, "policy_sha": policy_sha, "index_path": index_path,
        "cache_dir": cache_dir, "state_dir": state_dir, "work": work,
        "deployment": deployment,
    }


def test_policy_pair_survives_clearenv_through_the_real_overlay(delivery, monkeypatch):
    # a parent env value the overlay never declared must not leak to the child
    monkeypatch.setenv("TRITON_EVAL_API_PORT", "12345")
    wrapped, _outer = sandbox.isolate_command(
        [sys.executable, "-c", "pass"], delivery["work"], None, {}
    )
    assert "--clearenv" in wrapped
    for key in (POLICY_ENV, POLICY_SHA_ENV):
        position = wrapped.index(key)
        assert wrapped[position + 1] == (
            str(delivery["policy_path"]) if key == POLICY_ENV else delivery["policy_sha"]
        )
        assert position > wrapped.index("--clearenv")  # set AFTER clearenv
    assert "TRITON_EVAL_API_PORT" not in wrapped
    # exactly the declared artifacts are mounted read-only, state writable
    ro_pairs = {
        (wrapped[i + 1], wrapped[i + 2])
        for i, arg in enumerate(wrapped) if arg == "--ro-bind"
    }
    for required in (
        delivery["policy_path"],
        delivery["index_path"],
        delivery["cache_dir"] / "cases",
        delivery["cache_dir"] / "blobs",
    ):
        assert (str(required), str(required)) in ro_pairs
    bind_pairs = {
        (wrapped[i + 1], wrapped[i + 2])
        for i, arg in enumerate(wrapped) if arg == "--bind"
    }
    assert (str(delivery["state_dir"]), str(delivery["state_dir"])) in bind_pairs
    boundary = json.loads(
        next(delivery["work"].glob("engine-runtime-boundary-*.json")).read_bytes()
    )
    assert boundary["timing_reuse_policy_sha256"] == delivery["policy_sha"]
    assert boundary["timing_reuse_state_dir"] == str(delivery["state_dir"])


def test_half_pair_or_tampered_policy_is_refused(delivery, monkeypatch, tmp_path):
    # half a pair in the overlay is structurally invalid
    with pytest.raises(ValidationError):
        sandbox.EngineRuntimeOverlay(environment={POLICY_ENV: str(delivery["policy_path"])})
    # half a pair in the sandbox manifest is structurally invalid
    with pytest.raises(ValidationError, match="both validated timing-reuse"):
        sandbox.EngineSandbox(
            readonly_mounts=[(Path("/usr"), Path("/usr"))],
            environment={"PATH": "/usr/bin:/bin", POLICY_ENV: "/x"},
        )
    # a state mount without the pair is invalid too
    with pytest.raises(ValidationError, match="requires the policy pair"):
        sandbox.EngineSandbox(
            readonly_mounts=[(Path("/usr"), Path("/usr"))],
            environment={"PATH": "/usr/bin:/bin"},
            timing_reuse_state_dir=delivery["state_dir"],
        )
    # tampered policy bytes (same declared path+SHA) never reach the child
    original_bytes = delivery["policy_path"].read_bytes()
    try:
        delivery["policy_path"].write_bytes(original_bytes + b" ")
        monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(delivery["overlay_path"]))
        monkeypatch.setenv(
            sandbox.OVERLAY_DIGEST_ENV, sha256(delivery["overlay_path"].read_bytes())
        )
        with pytest.raises(ValueError, match="SHA pin"):
            sandbox.isolate_command([sys.executable, "-c", "pass"], delivery["work"], None, {})
    finally:
        delivery["policy_path"].write_bytes(original_bytes)
    # an overlay pointing at a different policy than the worker's declared
    # runtime is refused before any child exists
    tampered = tmp_path / "tampered-policy.json"
    tampered.write_bytes(original_bytes)
    overlay = sandbox.EngineRuntimeOverlay(
        environment={POLICY_ENV: str(tampered), POLICY_SHA_ENV: delivery["policy_sha"]}
    )
    overlay_path = tmp_path / "bad-overlay.json"
    overlay_path.write_text(overlay.model_dump_json())
    monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(overlay_path))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, sha256(overlay_path.read_bytes()))
    with pytest.raises(ValueError, match="worker's declared runtime"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], delivery["work"], None, {})
    # a non-dedicated state directory is refused
    (delivery["state_dir"] / "stray.txt").write_text("not a dedicated state namespace")
    monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(delivery["overlay_path"]))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, sha256(delivery["overlay_path"].read_bytes()))
    with pytest.raises(ValueError, match="dedicated timing-reuse state"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], delivery["work"], None, {})


def test_parent_policy_without_overlay_never_reaches_a_clearenv_child(
    delivery, monkeypatch
):
    # The worker declares the pair but provides no pinned overlay: refusing is
    # the only safe outcome; --clearenv must never silently drop it.
    for key in sandbox.OVERLAY_ENV_KEYS:
        monkeypatch.delenv(key)
    monkeypatch.setenv(POLICY_ENV, str(delivery["policy_path"]))
    monkeypatch.setenv(POLICY_SHA_ENV, delivery["policy_sha"])
    with pytest.raises(ValueError, match="overlay before --clearenv"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], delivery["work"], None, {})
