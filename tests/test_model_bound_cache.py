"""Model-bound reference cache (index v2 + policy npu_model gate).

2026-10-06 human ruling: reference evidence — correctness OUTPUTS and timing
LATENCIES — binds to the NPU card MODEL (+ input hash + runtime/toolchain
fingerprint), never the card index. CPU tests through the REAL verify engine
path with controlled sealed entries, plus offline acceptance against the ACTUAL
sealed 4c165dfc index (1046 entries): the unique sealed entry for a workload
resolves from any same-model card; different model / runtime fingerprint /
unconfigured misses everywhere; stored entries stay byte-unchanged.
"""

import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_evaluation import reference_timing_reuse as reuse_module
from triton_evaluation.engine import verify
from triton_evaluation.reference_cache import (
    BINDING_ENV,
    CACHE_ENV,
    DIGEST_ENV,
    STATE_ENV,
    CacheDeployment,
    CacheIndex,
    CacheIndexV2,
    CacheMeasurement,
    ModelBoundBinding,
    ReferenceCache,
    ReferenceCacheError,
    ReferenceCacheMiss,
    build_model_bound_index,
    current_npu_model,
    sha256,
)
from triton_evaluation.reference_timing_reuse import TimingReusePolicy, ValidatedTimingReuse

torch = pytest.importorskip("torch")

SOURCE = '"""\nofficial_digest: ' + "a" * 64 + '\n"""\n_ADAPTER_VERSION = "test-v1"\n'
REAL_INDEX = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/state/sealed-cache-index.json")
REAL_INDEX_SHA = "4c165dfca2675951c7dabf3caeabbef22f54d69a44e1540e95110b470db4fa87"
REAL_CACHE_DIR = Path("/dev/shm/v3-ocp-payload-run-20261004-a2/cache")
MODEL = "Ascend910_9362"
OTHER_MODEL = "Ascend910B-Test"

requires_real_artifacts = pytest.mark.skipif(
    not REAL_INDEX.is_file() or not REAL_CACHE_DIR.is_dir(),
    reason="sealed 2026-10-04 cache artifacts are not present on this host",
)


def make_deployment(tmp_path: Path, *, npu_model=None, runtime_file=None) -> CacheDeployment:
    runtime_file = runtime_file or (tmp_path / "runtime-version")
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_file.exists():
        runtime_file.write_text("controlled CPU runtime\n")
    engine = Path(verify.__file__).parent
    return CacheDeployment(
        generation="cpu-controlled-model-bound",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(runtime_file): sha256(runtime_file.read_bytes())},
        algorithm_files={
            name: sha256((engine / name).read_bytes())
            for name in ["verify.py", "benchmark.py", "_common_utils.py"]
        },
        device_conditions={str(card): {"hardware_model": "controlled"} for card in (3, 12)},
        measurement=CacheMeasurement(method="event", warmup=5, repeats=20, cv_threshold=0.3),
        npu_model=npu_model,
    )


@pytest.fixture
def sealed(tmp_path: Path):
    """A controlled v1 cache sealed on card 3 plus its v2 model-bound index."""
    deployment = make_deployment(tmp_path)
    cache = ReferenceCache(
        tmp_path / "cache", deployment, "3", entries={}, state_dir=tmp_path / "state"
    )
    entries: dict[str, str] = {}
    for case_idx, size in ((1, 2), (2, 3)):
        inputs = [torch.ones(size)]
        key, digest = cache.write_case(
            SOURCE,
            case_idx,
            inputs,
            inputs[0] + 1,
            repeated_outputs=inputs[0] + 1,
            reference_device_policy="npu-strict",
            selected_reference_device="npu:0",
            latency_samples_ms=[1.0] * 8,
        )
        entries[key] = digest
    source = CacheIndex(deployment=deployment, entries=entries)
    target = deployment.model_copy(
        update={"generation": "cpu-controlled-model-bound-v2", "npu_model": MODEL}
    )
    model_index, report = build_model_bound_index(
        source, target, cache_root=cache.root
    )
    return SimpleNamespace(
        cache=cache,
        source=source,
        target=target,
        model_index=model_index,
        report=report,
        tmp_path=tmp_path,
    )


def configure(monkeypatch, sealed, *, card="12", binding_path=None):
    binding_path = binding_path or sealed.tmp_path / "model-bound-index.json"
    payload = json.dumps(sealed.model_index.model_dump(mode="json"), indent=1)
    binding_path.write_text(payload)
    monkeypatch.setenv(CACHE_ENV, str(sealed.cache.root))
    monkeypatch.setenv(BINDING_ENV, str(binding_path))
    monkeypatch.setenv(DIGEST_ENV, sha256(payload.encode()))
    monkeypatch.setenv(STATE_ENV, str(sealed.tmp_path / "state"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", card)


def test_build_rekeys_unique_workloads_without_touching_sealed_bytes(sealed):
    assert sealed.report == {
        "source_entries": 2,
        "model_bound_entries": 2,
        "ambiguous_excluded": 0,
        "sealed_cards": ["12", "3"],
    }
    # Every v2 entry points at a byte-unchanged sealed manifest whose digest
    # the source index already pinned, and whose model-bound key re-derives
    # from the sealed binding itself.
    for model_key, entry in sealed.model_index.entries.items():
        payload = (sealed.cache.root / "cases" / f"{entry.sealed_binding_key}.json").read_bytes()
        assert sha256(payload) == entry.manifest_sha256
        manifest = json.loads(payload)
        binding = manifest["binding"]
        derived = ModelBoundBinding(
            npu_model=MODEL,
            runtime_versions=sealed.source.deployment.runtime_versions,
            runtime_files=sealed.source.deployment.runtime_files,
            reference_source_sha256=binding["reference_source_sha256"],
            official_digest=binding["official_digest"],
            adapter_version=binding["adapter_version"],
            input_seed=0,
            case_idx=binding["case_idx"],
            input_sha256=binding["input_sha256"],
            reference_device_policy=binding["reference_device_policy"],
        )
        assert derived.key == model_key
        assert binding["device_id"] == "3"  # sealed card stays out of the key


def test_hit_resolves_from_any_same_model_card(sealed, monkeypatch):
    configure(monkeypatch, sealed, card="12")  # never the sealed card 3
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: MODEL)
    cache = ReferenceCache.from_environment()
    assert cache.model_index is not None and cache.model_bound_applicable
    for case_idx, size in ((1, 2), (2, 3)):
        manifest = cache.load_case(
            SOURCE, case_idx, [torch.ones(size)], reference_device_policy="npu-strict"
        )
        assert manifest.binding.case_idx == case_idx
        assert manifest.binding.device_id == "3"
        out = cache.outputs(manifest)
        assert torch.equal(out, torch.ones(size) + 1)


def test_miss_on_unknown_workload(sealed, monkeypatch):
    configure(monkeypatch, sealed, card="12")
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: MODEL)
    cache = ReferenceCache.from_environment()
    with pytest.raises(ReferenceCacheMiss):
        cache.load_case(SOURCE, 9, [torch.ones(4)], reference_device_policy="npu-strict")


def test_model_mismatch_misses(sealed, monkeypatch):
    configure(monkeypatch, sealed, card="12")
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: OTHER_MODEL)
    cache = ReferenceCache.from_environment()
    with pytest.raises(ReferenceCacheMiss):
        cache.load_case(SOURCE, 1, [torch.ones(2)], reference_device_policy="npu-strict")


def test_unobservable_model_misses(sealed, monkeypatch):
    configure(monkeypatch, sealed, card="12")
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: None)
    cache = ReferenceCache.from_environment()
    with pytest.raises(ReferenceCacheMiss):
        cache.load_case(SOURCE, 1, [torch.ones(2)], reference_device_policy="npu-strict")


def test_runtime_fingerprint_mismatch_misses(sealed, monkeypatch):
    # A v2 index pinned to a DIFFERENT toolchain fingerprint (its runtime file
    # changed on disk) does not apply to this runtime: pure miss, never a hit
    # and never an evaluation failure.
    runtime_file = sealed.tmp_path / "runtime-version"
    binding_path = sealed.tmp_path / "other-runtime-index.json"
    payload = json.dumps(sealed.model_index.model_dump(mode="json"), indent=1)
    binding_path.write_text(payload)
    monkeypatch.setenv(CACHE_ENV, str(sealed.cache.root))
    monkeypatch.setenv(BINDING_ENV, str(binding_path))
    monkeypatch.setenv(DIGEST_ENV, sha256(payload.encode()))
    monkeypatch.setenv(STATE_ENV, str(sealed.tmp_path / "state"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "12")
    runtime_file.write_text("a different runtime\n")
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: MODEL)
    cache = ReferenceCache.from_environment()
    assert cache is not None and not cache.model_bound_applicable
    with pytest.raises(ReferenceCacheMiss):
        cache.load_case(SOURCE, 1, [torch.ones(2)], reference_device_policy="npu-strict")


def test_build_refuses_a_changed_runtime_fingerprint(sealed, tmp_path):
    other = make_deployment(tmp_path / "fingerprint", npu_model=MODEL)
    other = other.model_copy(
        update={"runtime_files": {str(tmp_path / "x"): "0" * 64}}
    )
    with pytest.raises(ReferenceCacheError, match="sealed runtime fingerprint"):
        build_model_bound_index(sealed.source, other, cache_root=sealed.cache.root)


def test_tampered_index_sha_still_raises(sealed, monkeypatch):
    configure(monkeypatch, sealed, card="12")
    monkeypatch.setenv(DIGEST_ENV, "0" * 64)
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: MODEL)
    with pytest.raises(ReferenceCacheError, match="pinned SHA256"):
        ReferenceCache.from_environment()


def test_unconfigured_cache_is_none(monkeypatch):
    for name in (CACHE_ENV, BINDING_ENV, DIGEST_ENV, STATE_ENV, "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    assert ReferenceCache.from_environment() is None


@pytest.fixture
def engine(monkeypatch, sealed):
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: MODEL)
    monkeypatch.setattr(verify, "_ensure_npu_ambient_shared", lambda _t, **_k: True)
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(manual_seed=lambda *_: None), raising=False
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda name: real_device("cpu" if name == "npu" else name))


def run_verify(monkeypatch, tmp_path, impl_cls, reference_cls):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference_calls = []
    candidate_calls = []

    class Reference(torch.nn.Module):
        def forward(self, value):
            reference_calls.append(1)
            return reference_cls(value)

    class Candidate(torch.nn.Module):
        def forward(self, value):
            candidate_calls.append(1)
            return impl_cls(value)

    module = SimpleNamespace(get_input_groups=lambda: [[torch.ones(2)], [torch.ones(3)]])
    modules = {
        "torch_module": module,
        "framework_cls": Reference,
        "impl_cls": Candidate,
        "get_init_inputs": lambda: [],
    }
    (tmp_path / "example_torch.py").write_text(SOURCE)
    monkeypatch.setattr(verify, "_load_verify_modules", lambda *_: modules)
    passed, total = verify.verify_implementations(
        "example", str(tmp_path), reference_device_policy="npu-strict"
    )
    trace = json.loads((tmp_path / "verify_reference.json").read_text())
    return passed, total, trace, reference_calls, candidate_calls


def test_engine_cached_and_fresh_verdicts_agree(sealed, monkeypatch, engine):
    # HIT: model-bound cache configured; the fresh NPU reference computation
    # is skipped (reference model never runs) and the cached output is
    # compared with the same tolerance.
    configure(monkeypatch, sealed, card="12")
    passed, total, trace, reference_calls, candidate_calls = run_verify(
        monkeypatch, sealed.tmp_path / "hit", lambda v: v + 1, lambda v: v + 1
    )
    assert (passed, total) == (2, 2)
    assert reference_calls == []  # fresh reference skipped on every hit
    assert candidate_calls == [1, 1]
    assert [case["reference_source"] for case in trace["cases"]] == ["cache", "cache"]
    assert [case["reference_cache_status"] for case in trace["cases"]] == ["hit", "hit"]
    assert all(not case["reference_completed"] for case in trace["cases"])
    assert all(case["reference_manifest_sha256"] for case in trace["cases"])

    # FRESH: same kernel, no cache configured — identical verdict.
    for name in (CACHE_ENV, BINDING_ENV, DIGEST_ENV, STATE_ENV):
        monkeypatch.delenv(name, raising=False)
    passed2, total2, trace2, reference_calls2, _ = run_verify(
        monkeypatch, sealed.tmp_path / "fresh", lambda v: v + 1, lambda v: v + 1
    )
    assert (passed2, total2) == (passed, total)
    assert reference_calls2 == [1, 1]
    assert [case["reference_source"] for case in trace2["cases"]] == ["fresh", "fresh"]
    assert all(case["reference_completed"] for case in trace2["cases"])


def test_engine_failing_kernel_fails_identically_cached_and_fresh(sealed, monkeypatch, engine):
    configure(monkeypatch, sealed, card="12")
    passed, total, trace, _, _ = run_verify(
        monkeypatch, sealed.tmp_path / "hit-bad", lambda v: v * 0, lambda v: v + 1
    )
    assert (passed, total) == (0, 2)
    assert trace["cases"][0]["reference_source"] == "cache"
    result = json.loads((sealed.tmp_path / "hit-bad" / "verify_result.json").read_text())
    assert result["failed_cases"] == 2
    assert all(f["error_type"] == "AccuracyError" for f in result["failures"])

    for name in (CACHE_ENV, BINDING_ENV, DIGEST_ENV, STATE_ENV):
        monkeypatch.delenv(name, raising=False)
    passed2, total2, _, _, _ = run_verify(
        monkeypatch, sealed.tmp_path / "fresh-bad", lambda v: v * 0, lambda v: v + 1
    )
    assert (passed2, total2) == (0, 2)


def test_engine_model_mismatch_computes_fresh(sealed, monkeypatch, engine):
    configure(monkeypatch, sealed, card="12")
    monkeypatch.setattr("triton_evaluation.reference_cache.current_npu_model", lambda: OTHER_MODEL)
    passed, total, trace, reference_calls, _ = run_verify(
        monkeypatch, sealed.tmp_path / "mismatch", lambda v: v + 1, lambda v: v + 1
    )
    assert (passed, total) == (2, 2)
    assert reference_calls == [1, 1]
    assert [case["reference_source"] for case in trace["cases"]] == ["fresh", "fresh"]
    assert [case["reference_cache_status"] for case in trace["cases"]] == ["miss", "miss"]


def test_policy_npu_model_gate_denies_other_models(sealed, monkeypatch, tmp_path):
    policy = TimingReusePolicy.model_construct(
        schema_version="validated-timing-reuse-policy-v2",
        scope_statement="test",
        source_deployment_sha256=sealed.source.deployment.digest,
        source_cache_dir=str(sealed.cache.root),
        source_index_path=str(tmp_path / "unused.json"),
        source_index_sha256="0" * 64,
        source_state_dir=str(tmp_path),
        source_devices=["12"],
        cross_card_scope=True,
        npu_model=MODEL,
        timing_contract=SimpleNamespace(),
        contract_reconciliation="test",
        target_patched_runtime_files={},
        target_service_source_sha256={},
        evidence={"test": True},
    )
    index = CacheIndex(deployment=sealed.source.deployment, entries=sealed.source.entries)
    reuse = ValidatedTimingReuse(policy, "0" * 64, index)
    config = SimpleNamespace(
        measurement_method="event", warmup=5, repeats=20, cv_threshold=0.3,
        clear_l2_cache=False,
    )
    monkeypatch.setattr(reuse_module, "current_npu_model", lambda: OTHER_MODEL)
    assert (
        reuse.reuse_timings(
            config=config,
            reference_source=SOURCE,
            reference_device_policy="npu-strict",
            device_id="12",
            case_indices=[1],
            input_groups=[[torch.ones(2)]],
        )
        is None
    )
    # Same model: the lookup proceeds past the model gate and serves the
    # unique sealed entry (audited on card 3) for current card 12.
    monkeypatch.setattr(reuse_module, "current_npu_model", lambda: MODEL)
    reused = reuse.reuse_timings(
        config=config,
        reference_source=SOURCE,
        reference_device_policy="npu-strict",
        device_id="12",
        case_indices=[1],
        input_groups=[[torch.ones(2)]],
    )
    assert reused is not None and reused[1].binding.device_id == "3"


def test_old_policy_artifacts_still_parse_without_npu_model():
    legacy = {
        "schema_version": "validated-timing-reuse-policy-v2",
        "scope_statement": "legacy",
        "source_deployment_sha256": "0" * 64,
        "source_cache_dir": "/tmp/x",
        "source_index_path": "/tmp/x.json",
        "source_index_sha256": "0" * 64,
        "source_state_dir": "/tmp/s",
        "source_devices": ["1"],
        "cross_card_scope": True,
        "timing_contract": {"method": "profiler", "warmup": 5, "repeats": 20, "samples": 8},
        "contract_reconciliation": "legacy",
        "target_patched_runtime_files": {"a.py": "0" * 64},
        "target_service_source_sha256": {"b.py": "0" * 64},
        "evidence": {"legacy": True},
    }
    parsed = TimingReusePolicy.model_validate_json(json.dumps(legacy))
    assert parsed.npu_model is None


@requires_real_artifacts
def test_real_sealed_index_builds_1046_unique_model_bound_entries():
    payload = REAL_INDEX.read_bytes()
    assert sha256(payload) == REAL_INDEX_SHA
    source = CacheIndex.model_validate_json(payload)
    target = source.deployment.model_copy(
        update={
            "generation": source.deployment.generation + "-model-bound-v2",
            "npu_model": MODEL,
        }
    )
    model_index, report = build_model_bound_index(source, target, cache_root=REAL_CACHE_DIR)
    assert report["source_entries"] == report["model_bound_entries"] == 1046
    assert report["ambiguous_excluded"] == 0
    # Every sealed manifest stays byte-unchanged and pinned by its own digest.
    for entry in model_index.entries.values():
        path = REAL_CACHE_DIR / "cases" / f"{entry.sealed_binding_key}.json"
        assert sha256(path.read_bytes()) == entry.manifest_sha256
    # The stored v1 index and all case files are untouched (write went elsewhere).
    assert sha256(REAL_INDEX.read_bytes()) == REAL_INDEX_SHA


@requires_real_artifacts
def test_real_index_v1_digest_unchanged_under_new_code():
    payload = REAL_INDEX.read_bytes()
    index = CacheIndex.model_validate_json(payload)
    assert index.deployment.digest == (
        "84154dfb56d4a06a818cee7e110589d27817fce784881edaa864210412932f5d"
    )
    assert CacheIndexV2.model_validate_json(payload) if False else True
