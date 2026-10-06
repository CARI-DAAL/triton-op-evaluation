"""Cross-card (policy v4) timing reuse: sealed entry from ANY card serves the
current card, validated per evaluation by the fresh sentinel on the CURRENT
card.

CPU tests through the REAL engine path (fake ``measure_single`` seam) plus
offline acceptance against the ACTUAL retained artifacts: the sealed
2026-10-04 index (1046 entries), the real card-8 BMM51 input identities from
the live reuse run, and the delivered policy v4. Fail-closed behavior
(fleet gate, ambiguity, unknown workload, v2/v3 per-card semantics) asserted
everywhere.
"""

import hashlib
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path

import pytest

from triton_evaluation import reference_timing_reuse as reuse_module
from triton_evaluation.engine import benchmark
from triton_evaluation.reference_cache import (
    CacheDeployment,
    CacheMeasurement,
    ReferenceCache,
    ReferenceCacheMiss,
    sha256,
)
from triton_evaluation.reference_timing_reuse import (
    POLICY_ENV,
    POLICY_SHA_ENV,
    TimingReusePolicy,
    load_validated_timing_reuse,
)
from test_timing_reuse_policy import IMPL_LATENCY, write_task

torch = pytest.importorskip("torch")

FLEET = [str(card) for card in range(16)]
REAL_INDEX = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/state/sealed-cache-index.json")
REAL_INDEX_SHA = "4c165dfca2675951c7dabf3caeabbef22f54d69a44e1540e95110b470db4fa87"
REAL_CACHE_DIR = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/cache")
REAL_DEPLOYMENT_SHA = "84154dfb56d4a06a818cee7e110589d27817fce784881edaa864210412932f5d"
REAL_CARD8_INPUTS = Path(
    "/root/.paseo/worktrees/0u9809pj/nb7-live-reuse-run2-20261005/.scratch/"
    "nb7-live-reuse-run2-20261005/pairs-root2/official_l3_BatchMatmul/"
    "uncached/reference.json"
)
DELIVERY_DIR = Path(__file__).parents[1] / ".scratch/nb7-cross-card-reuse-20261006"
REAL_POLICY_V4 = DELIVERY_DIR / "validated-timing-reuse-policy-v4.json"
EXPECTED_CARD_COUNTS = {"1": 150, "2": 131, "3": 175, "4": 110, "5": 100, "6": 124, "7": 107, "8": 149}

requires_real_artifacts = pytest.mark.skipif(
    not REAL_INDEX.is_file() or not REAL_CACHE_DIR.is_dir(),
    reason="retained old-cache artifacts are not present on this host",
)
requires_delivery = pytest.mark.skipif(
    not REAL_POLICY_V4.is_file(), reason="policy v4 delivery is not present"
)


@pytest.fixture
def cpu_engine(monkeypatch):
    real_device = torch.device
    monkeypatch.setattr(
        torch, "device", lambda name: real_device("cpu" if name == "npu" else name)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", __import__("types").SimpleNamespace())
    monkeypatch.setattr(
        torch, "npu", __import__("types").SimpleNamespace(manual_seed=lambda *_: None),
        raising=False,
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
        "TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def make_deployment(tmp_path: Path, cards: tuple[str, ...]) -> CacheDeployment:
    runtime_file = tmp_path / "runtime-version"
    runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(benchmark.__file__).parent
    return CacheDeployment(
        generation="cpu-controlled-cross-card",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(runtime_file): sha256(runtime_file.read_bytes())},
        algorithm_files={
            name: sha256((engine / name).read_bytes())
            for name in ["verify.py", "benchmark.py", "_common_utils.py"]
        },
        device_conditions={card: {"hardware_model": "controlled-cpu"} for card in cards},
        measurement=CacheMeasurement(
            method="profiler", warmup=5, repeats=20, cv_threshold=0.3
        ),
    )


def build_v4_policy(tmp_path: Path, *, cache_dir, index_path, index_sha, state_dir,
                    devices, patched, deployment_sha, cross_card=True):
    policy = {
        "schema_version": "validated-timing-reuse-policy-v2",
        "cross_card_scope": cross_card,
        "scope_statement": (
            "cross-card reuse of sealed references validated per evaluation "
            "by the current-card sentinel"
        ),
        "source_deployment_sha256": deployment_sha,
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
        "target_service_source_sha256": {
            name: reuse_module.current_service_source_sha256(name)
            for name in (
                "engine/benchmark.py",
                "reference_timing_reuse.py",
                "reference_cache.py",
            )
        },
        "evidence": {"sampled_problem": "official_l3_BatchMatmul"},
    }
    path = tmp_path / "policy-v4.json"
    payload = json.dumps(policy, indent=2, sort_keys=True).encode()
    path.write_bytes(payload)
    return path, sha256(payload)


def configure_cross_card(tmp_path, monkeypatch, *, cases=6, size=2, seal_size=None,
                         entry_cards=("6",), current_card="9", devices=FLEET,
                         sentinel_factor=1.01, cross_card=True, seal_twice=False):
    """Sealed old cache on ``entry_cards`` + v4 policy + scripted sentinel."""
    tmp_path = Path(tmp_path)
    verify_dir = Path(write_task(tmp_path, cases=cases, size=size))
    source = (verify_dir / "example_torch.py").read_text(encoding="utf-8")
    deployment = make_deployment(tmp_path, entry_cards)
    old_cache = ReferenceCache(
        tmp_path / "old-cache", deployment, entry_cards[0], entries={},
        state_dir=tmp_path / "state",
    )
    samples_by_case = {}
    sealed_inputs = [torch.ones(size if seal_size is None else seal_size)]
    for case_idx in range(1, cases + 1):
        base = 1.0 + case_idx * 0.1
        samples = [round(base + k * 0.01, 4) for k in range(8)]
        for card in (entry_cards if seal_twice else entry_cards[:1]):
            writer = ReferenceCache(
                tmp_path / "old-cache", deployment, card, entries=old_cache.entries,
                state_dir=tmp_path / "state",
            )
            key, digest = writer.write_case(
                source, case_idx, sealed_inputs, sealed_inputs[0],
                repeated_outputs=sealed_inputs[0],
                reference_device_policy="npu-strict",
                selected_reference_device="npu",
                latency_samples_ms=samples,
            )
            old_cache.entries[key] = digest
        samples_by_case[case_idx] = samples
    index_path = tmp_path / "sealed-index.json"
    index_sha = old_cache.write_index(index_path, old_cache.entries)
    state_dir = tmp_path / "reuse-state"
    state_dir.mkdir()
    patched = {"backends/ascend/driver.py": "a" * 64}
    policy_path, policy_sha = build_v4_policy(
        tmp_path, cache_dir=old_cache.root, index_path=index_path, index_sha=index_sha,
        state_dir=state_dir, devices=devices, patched=patched,
        deployment_sha=deployment.digest, cross_card=cross_card,
    )
    monkeypatch.setenv(POLICY_ENV, str(policy_path))
    monkeypatch.setenv(POLICY_SHA_ENV, policy_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID", current_card)
    monkeypatch.setattr(
        reuse_module, "current_patched_runtime_sha256", lambda name: patched.get(name)
    )
    sentinel_value = statistics.median(samples_by_case[1]) * sentinel_factor
    calls = {"framework": 0}

    def measure(context, **_kwargs):
        if context.profile_name.startswith("framework"):
            calls["framework"] += 1
            return {}, sentinel_value, 2.0
        return {}, IMPL_LATENCY, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    return verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment


def test_cross_card_policy_serves_entry_card_on_current_card(
    tmp_path, monkeypatch, cpu_engine
):
    # Sealed on card 6, evaluated on card 9, sentinel in range: full reuse.
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = (
        configure_cross_card(tmp_path, monkeypatch, cases=3, entry_cards=("6",))
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 8  # exactly the fresh sentinel, nothing else
    for case in result.per_shape_results:
        assert case.framework.latency_samples_ms == samples_by_case[case.case_idx]
        assert case.framework.latency_samples_source == "historical_validated"
        assert case.framework.latency_samples_proof_sha256 == policy_sha
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())["performance_cache"]
    assert trace["status"] == "timing_reuse"
    assert trace["device_id"] == "9"  # current card ran the sentinel
    assert trace["sentinel_case_idx"] == 1
    assert trace["sentinel_cached_ms"] == statistics.median(samples_by_case[1])
    assert not list(state_dir.rglob("card-*.invalid.json"))


def test_cross_card_drift_marks_remeasure_for_the_current_card_monotonically(
    tmp_path, monkeypatch, cpu_engine
):
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = (
        configure_cross_card(
            tmp_path, monkeypatch, cases=2, entry_cards=("8",), sentinel_factor=1.05
        )
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    # Drift: sentinel runs, then the ORIGINAL e4a36f7 path serves every case
    # (no cache configured here; the accelerator is optional).
    assert calls["framework"] == 8 + 2
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        and case.framework.latency_samples_proof_sha256 is None
        for case in result.per_shape_results
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())["performance_cache"]
    assert trace["status"] == "timing_reuse_drift"
    assert trace["sentinel_relative_drift"] > 0.03
    markers = list(state_dir.rglob("card-*.invalid.json"))
    assert [marker.name for marker in markers] == ["card-9.invalid.json"]  # CURRENT card
    assert markers[0].parent.name == deployment.digest

    # Monotone: a later in-range sentinel on card 9 cannot restore reuse.
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
    assert calls["framework"] == 8 + 2  # sentinel, then the original path
    assert all(
        case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )
    # card 8 (the sealed card) stays clean: the marker is per current card
    assert not list(state_dir.rglob("card-8.invalid.json"))


def test_cross_card_unknown_workload_takes_the_original_path(
    tmp_path, monkeypatch, cpu_engine
):
    # No sealed entry for these inputs anywhere: no reuse on any card, and
    # with no cache configured the service runs its ORIGINAL e4a36f7 path.
    verify_dir, _, _, calls, state_dir, _ = configure_cross_card(
        tmp_path, monkeypatch, cases=2, size=7, seal_size=2, entry_cards=("6",)
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2  # one original-path measurement per case
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )
    assert not list(state_dir.rglob("card-*.invalid.json"))


def test_cross_card_fleet_gate_denies_a_card_outside_the_policy(
    tmp_path, monkeypatch, cpu_engine
):
    # Policy scope is the fleet; card 16 (or any non-declared card) reuses
    # nothing even though a sealed entry exists on card 6.
    verify_dir, _, _, calls, _, _ = configure_cross_card(
        tmp_path, monkeypatch, cases=2, entry_cards=("6",), current_card="16",
        devices=[str(card) for card in range(16)],
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    # Card 16 borrows nothing; no cache configured -> original path.
    assert calls["framework"] == 2
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )


def test_policy_without_cross_card_scope_keeps_exact_per_card_semantics(
    tmp_path, monkeypatch, cpu_engine
):
    # A v2/v3-style policy (no cross_card_scope) never borrows card 6's entry
    # for card 9, even when card 9 is inside source_devices; with no cache
    # configured the ORIGINAL e4a36f7 path serves the run.
    verify_dir, _, _, calls, _, _ = configure_cross_card(
        tmp_path, monkeypatch, cases=2, entry_cards=("6",), current_card="9",
        devices=["9"], cross_card=False,
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )


def test_ambiguous_cross_card_seal_fails_closed_to_the_original_path(
    tmp_path, monkeypatch, cpu_engine
):
    # The same workload sealed on two cards has no unique cross-card entry;
    # nothing is borrowed and the original path measures (no cache here).
    verify_dir, _, _, calls, _, _ = configure_cross_card(
        tmp_path, monkeypatch, cases=2, entry_cards=("6", "7"), seal_twice=True
    )
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 2
    assert all(
        case.framework.latency_samples_source is None
        and case.framework.reference_path == "original_measurements"
        for case in result.per_shape_results
    )


def test_current_cache_miss_falls_back_to_cross_card_reuse(
    tmp_path, monkeypatch, cpu_engine
):
    # The live same-generation cache (from_environment) misses on card 9;
    # the v4 policy serves the card-6 sealed entry after its own sentinel.
    verify_dir, samples_by_case, policy_sha, calls, state_dir, deployment = (
        configure_cross_card(tmp_path, monkeypatch, cases=2, entry_cards=("6",))
    )
    current = ReferenceCache(
        tmp_path / "current-cache", deployment, "9", entries={},
        state_dir=tmp_path / "cstate",
    )
    index_path = tmp_path / "current-index.json"
    index_sha = current.write_index(index_path, current.entries)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_DIR", str(current.root))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_BINDING", str(index_path))
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_SHA256", index_sha)
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_CACHE_STATE_DIR", str(current.state_dir))
    result = benchmark.benchmark_implementations(
        benchmark.BenchmarkConfig("example", str(verify_dir), measurement_method="profiler")
    )
    assert calls["framework"] == 8
    assert all(
        case.framework.latency_samples_source == "historical_validated"
        and case.framework.latency_samples_proof_sha256 == policy_sha
        for case in result.per_shape_results
    )
    trace = json.loads(Path(verify_dir, "verify_reference.json").read_text())["performance_cache"]
    assert trace["status"] == "timing_reuse"


def test_cross_card_lookup_unit_semantics(tmp_path):
    # Unit level: candidates are the declared cards plus the anchor card; a
    # unique match loads the sealed manifest; ambiguity and absence miss.
    tmp_path = Path(tmp_path)
    deployment = make_deployment(tmp_path, ("6", "8"))
    source = Path(write_task(tmp_path, cases=1)).joinpath("example_torch.py").read_text("utf-8")
    inputs = [torch.ones(2)]
    cache = ReferenceCache(tmp_path / "old", deployment, "9", entries={}, state_dir=None)
    writer = ReferenceCache(tmp_path / "old", deployment, "8", entries={}, state_dir=None)
    key, digest = writer.write_case(
        source, 1, inputs, inputs[0], repeated_outputs=inputs[0],
        reference_device_policy="npu-strict", selected_reference_device="npu",
        latency_samples_ms=[1.0] * 8,
    )
    entries = {key: digest}
    reader = ReferenceCache(tmp_path / "old", deployment, "9", entries=entries, state_dir=None)
    manifest = reader.load_case_cross_card(
        source, 1, inputs, reference_device_policy="npu-strict"
    )
    assert manifest.binding.device_id == "8"  # resolved the sealed card, not "9"
    assert manifest.binding.key == key
    with pytest.raises(ReferenceCacheMiss):
        reader.load_case_cross_card(
            source, 2, inputs, reference_device_policy="npu-strict"
        )
    # a second card's seal for the same workload makes the entry ambiguous
    other = ReferenceCache(tmp_path / "old", deployment, "6", entries=dict(entries), state_dir=None)
    key6, digest6 = other.write_case(
        source, 1, inputs, inputs[0], repeated_outputs=inputs[0],
        reference_device_policy="npu-strict", selected_reference_device="npu",
        latency_samples_ms=[1.0] * 8,
    )
    ambiguous = ReferenceCache(
        tmp_path / "old", deployment, "9", entries={key: digest, key6: digest6}, state_dir=None
    )
    with pytest.raises(ReferenceCacheMiss):
        ambiguous.load_case_cross_card(
            source, 1, inputs, reference_device_policy="npu-strict"
        )


# ---------------------------------------------------------------------------
# Real retained artifacts (offline acceptance)
# ---------------------------------------------------------------------------

@requires_real_artifacts
def test_real_sealed_entries_are_unique_per_workload_across_cards():
    payload = REAL_INDEX.read_bytes()
    assert sha256(payload) == REAL_INDEX_SHA
    index = reuse_module.CacheIndex.model_validate_json(payload)
    assert index.deployment.digest == REAL_DEPLOYMENT_SHA
    workload = lambda binding: json.dumps(
        {k: v for k, v in binding.model_dump().items() if k != "device_id"}, sort_keys=True
    )
    groups: dict[str, int] = {}
    cards: dict[str, int] = {}
    for key, manifest_sha in index.entries.items():
        manifest = reuse_module.CaseManifest.model_validate_json(
            (REAL_CACHE_DIR / "cases" / f"{key}.json").read_bytes()
        )
        assert manifest.binding.key == key
        identity = workload(manifest.binding)
        groups[identity] = groups.get(identity, 0) + 1
        cards[manifest.binding.device_id] = cards.get(manifest.binding.device_id, 0) + 1
    assert len(groups) == len(index.entries) == 1046  # every workload unique
    assert cards == EXPECTED_CARD_COUNTS


@requires_real_artifacts
def test_real_bmm51_card8_entry_resolves_from_device9_with_retained_input_sha():
    # Input identity: the live card-8 run's candidate_input_sha256 values all
    # equal sealed card-8 binding input_sha256 values (device TYPE only), so a
    # device-9 anchor differs ONLY in device_id and must reach the card-8 entry.
    live = json.loads(REAL_CARD8_INPUTS.read_bytes())
    assert live["physical_card"] == 8 and len(live["observations"]["cases"]) == 51
    index = reuse_module.CacheIndex.model_validate_json(REAL_INDEX.read_bytes())
    cache = ReferenceCache(
        REAL_CACHE_DIR, index.deployment, "9", entries=index.entries, state_dir=None
    )
    from triton_evaluation.reference_cache import CaseBinding

    sealed_by_input = {}
    for key in index.entries:
        binding = json.loads(
            (REAL_CACHE_DIR / "cases" / f"{key}.json").read_bytes()
        )["binding"]
        sealed_by_input[binding["input_sha256"]] = binding
    resolved = 0
    for case in live["observations"]["cases"]:
        sealed = sealed_by_input[case["candidate_input_sha256"]]
        assert sealed["device_id"] == "8"  # BMM51 was audited entirely on card 8
        anchor = CaseBinding(
            deployment_sha256=sealed["deployment_sha256"],
            reference_source_sha256=sealed["reference_source_sha256"],
            official_digest=sealed["official_digest"],
            adapter_version=sealed["adapter_version"],
            case_idx=case["case_idx"],
            input_sha256=case["candidate_input_sha256"],
            device_id="9",
            reference_device_policy=sealed["reference_device_policy"],
        )
        manifest = cache.cross_card_entry(anchor)
        assert manifest.binding.device_id == "8"
        assert manifest.binding.case_idx == case["case_idx"]
        assert manifest.binding.input_sha256 == case["candidate_input_sha256"]
        resolved += 1
    assert resolved == 51


@requires_real_artifacts
@requires_delivery
def test_real_policy_v4_pins_and_fail_closed_on_unpatched_host(monkeypatch):
    payload = REAL_POLICY_V4.read_bytes()
    policy = TimingReusePolicy.model_validate_json(payload)
    assert policy.cross_card_scope is True
    assert policy.source_index_sha256 == REAL_INDEX_SHA
    assert policy.source_deployment_sha256 == REAL_DEPLOYMENT_SHA
    assert policy.source_devices == FLEET
    assert policy.timing_contract.samples == 8
    # v4 pins the branch bytes of the lookup implementation itself
    for name, expected in policy.target_service_source_sha256.items():
        assert reuse_module.current_service_source_sha256(name) == expected
    # unpatched host: the configured policy authorizes nothing (fail-closed)
    monkeypatch.setenv(POLICY_ENV, str(REAL_POLICY_V4))
    monkeypatch.setenv(POLICY_SHA_ENV, sha256(payload))
    assert load_validated_timing_reuse() is None
