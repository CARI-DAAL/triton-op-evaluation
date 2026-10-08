"""CPU red/green evidence: retaining engine timing_reuse traces before cleanup.

CONTROLLED EVIDENCE (CPU fixture, not a device observation): the trace below
is constructed to the exact emitted schema of the real engine writer
(``triton_evaluation/engine/benchmark.py`` ``_try_validated_timing_reuse``
trace dicts, merged into ``verify_reference.json`` ``performance_cache`` by
``_write_reference_cache_trace``, plus the ``verify.py`` case entries).
No torch/NPU is available on this host, so the engine itself cannot run here;
the values are controlled and are NOT attached to any real evaluation id.

The test drives the ACTUAL retained-observation call path used before attempt
cleanup: ``Worker.run_once`` -> finally -> ``retain_reference_diagnostics``
-> ``_read_trace`` -> record file -> ``shutil.rmtree(work_dir)``, exactly as
``tests/test_reference_diagnostics.py`` does for legacy statuses.

RED on the unmodified source: ``timing_reuse``/``timing_reuse_drift`` records
degrade to ``status:"invalid", observations:null`` (Literal rejects the new
statuses), so the sentinel8 numbers, policy digest/scope, manifests and all
per-case input hashes are lost when the work dir is deleted.
GREEN on the proposed parser: the same runs are retained verbatim.
"""

import copy
import json
from pathlib import Path

import pytest
from test_worker import StubEvaluator, build_worker

from triton_evaluation.service.admission import admit
from triton_evaluation.service.models import (
    AlgorithmVersions,
    CreateResourceCommand,
    new_evaluation_id,
)

POLICY_SHA = "3d9198dfcaa68f81442f09a43c8aac88dd77de62c601eb561b7731b44dfe9d02"
SCOPE = (
    "owner-adopted timing reuse over the sealed 2026-10-04 card-8 cache; "
    "sampled evidence, not a per-case remeasurement"
)
DEPLOYMENT = "84154dfb56d4a06a818cee7e110589d27817fce784881edaa864210412932f5d"
MANIFEST_A = "a" * 64
MANIFEST_B = "b" * 64


def _case(case_idx: int) -> dict:
    # Exact _CaseReference shape the engine's verify.py emits for an
    # npu-strict case whose reference evidence was reused (not completed live).
    return {
        "case_idx": case_idx,
        "selected_reference_device": "npu:8",
        "reference_completed": False,
        "output_devices": [{"output_index": 0, "device": "npu:8"}],
        "candidate_input_semantics": "official-run-defaults-v1",
        "candidate_defaulted_parameters": [],
        "raw_input_sha256": f"{case_idx:064x}"[:64],
        "candidate_input_sha256": f"{case_idx:064x}"[:64],
        "reference_cache_status": "hit",
        "reference_manifest_sha256": MANIFEST_A if case_idx == 1 else MANIFEST_B,
        "reference_binding_sha256": ("e" if case_idx == 1 else "f") * 64,
        "reference_output_sha256": ("1" if case_idx == 1 else "2") * 64,
    }


def _timing_reuse_trace(status: str) -> dict:
    # Byte-for-byte the key set benchmark.py writes for both timing_reuse
    # statuses: deployment digest, device, status, manifest digests, memory
    # flag, policy sha + scope statement, and the fresh8 sentinel quartet.
    return {
        "reference_device_policy": "npu-strict",
        "cases": [_case(1), _case(2)],
        "performance_cache": {
            "deployment_sha256": DEPLOYMENT,
            "device_id": "8",
            "status": status,
            "manifests": [MANIFEST_A, MANIFEST_B],
            "reference_memory_measured": status == "timing_reuse_drift",
            "timing_reuse_policy_sha256": POLICY_SHA,
            "timing_reuse_scope_statement": SCOPE,
            "sentinel_case_idx": 1,
            "sentinel_cached_ms": 1.0,
            "sentinel_measured_ms": 1.02,
            "sentinel_relative_drift": 0.019999999999999997,
        },
    }


def _hit_trace() -> dict:
    # Engine live-cache hit/remeasured trace (benchmark.py:1119-1124).
    return {
        "reference_device_policy": "npu-strict",
        "cases": [_case(1)],
        "performance_cache": {
            "deployment_sha256": DEPLOYMENT,
            "device_id": "8",
            "status": "hit",
            "manifests": [MANIFEST_A],
            "reference_memory_measured": False,
            "sentinel_case_idx": 1,
            "sentinel_cached_ms": 1.0,
            "sentinel_measured_ms": 1.005,
            "sentinel_relative_drift": 0.005,
        },
    }


def _npu_strict_command(settings, caller=None):
    """Admit one command pinned the way a real npu-strict deployment pins it.

    conftest.make_command always pins the plain correctness version; the
    worker's lease protocol requires the EFFECTIVE version (settings.effective_
    correctness_algorithm_version), which carries ":npu-strict-v1" under this
    policy. Same existing admission entry point as conftest.
    """
    from conftest import make_caller, make_request

    caller = caller or make_caller()
    request = make_request()
    admission = admit(request, caller)
    return CreateResourceCommand(
        evaluation_id=new_evaluation_id(),
        caller_id=caller.caller_id,
        caller_class=caller.caller_class,
        traffic_class="trial",
        session_id="session-1",
        idempotency_key="key-1",
        admission=admission,
        evaluator_build=settings.evaluator_build,
        algorithms=AlgorithmVersions(
            correctness=settings.effective_correctness_algorithm_version,
            decoy=settings.decoy_algorithm_version,
            benchmark=settings.benchmark_algorithm_version,
            profiler=settings.profiler_algorithm_version,
        ),
    )


def _run_retention(tmp_path, store, settings, clock, make_command, trace):
    """One real worker attempt: retain diagnostics, then delete the work dir."""
    settings.reference_diagnostics_dir = tmp_path / "retained"
    settings.reference_device_policy = "npu-strict"
    command = _npu_strict_command(settings)
    store.create_or_replay(command)
    state = {}

    def hook(evaluator):
        work_dir = Path(evaluator.calls[-1]["work_dir"])
        state["work_dir"] = work_dir
        (work_dir / "verify_reference.json").write_text(json.dumps(trace))

    evaluator = StubEvaluator(settings, hook=hook)
    worker = build_worker(store, settings, evaluator, clock, device_id=8)
    assert worker.run_once()
    # Cleanup really removed the work dir: retention was the last chance.
    assert not state["work_dir"].exists()
    retained = list(
        (settings.reference_diagnostics_dir / command.evaluation_id).glob("*.json")
    )
    assert len(retained) == 1
    return json.loads(retained[0].read_text())


@pytest.mark.parametrize("status", ["timing_reuse", "timing_reuse_drift"])
def test_engine_timing_reuse_trace_is_retained_before_cleanup(
    status, tmp_path, store, settings, clock, make_command
):
    record = _run_retention(
        tmp_path, store, settings, clock, make_command, _timing_reuse_trace(status)
    )
    assert record["status"] == "recorded"
    cache = record["observations"]["performance_cache"]
    # Both statuses survive verbatim; hit evidence is never claimed for drift.
    assert cache["status"] == status
    assert cache["reference_memory_measured"] is (status == "timing_reuse_drift")
    # Policy digest and scope statement (the two explicit metadata keys).
    assert cache["timing_reuse_policy_sha256"] == POLICY_SHA
    assert cache["timing_reuse_scope_statement"] == SCOPE
    # Fresh8 sentinel quartet preserved: historical vs measured stay separate.
    assert cache["sentinel_case_idx"] == 1
    assert cache["sentinel_cached_ms"] == 1.0  # historical old8 median
    assert cache["sentinel_measured_ms"] == 1.02  # fresh8 sentinel
    assert cache["sentinel_relative_drift"] == pytest.approx(0.02, abs=1e-12)
    assert cache["manifests"] == [MANIFEST_A, MANIFEST_B]
    # Per-case input hashes and reuse provenance preserved.
    cases = record["observations"]["cases"]
    assert [case["case_idx"] for case in cases] == [1, 2]
    for case in cases:
        assert case["raw_input_sha256"] == f"{case['case_idx']:064x}"[:64]
        assert case["candidate_input_sha256"] == case["raw_input_sha256"]
        assert case["reference_cache_status"] == "hit"
        assert case["reference_manifest_sha256"] in (MANIFEST_A, MANIFEST_B)


def test_legacy_hit_still_recorded_without_timing_reuse_keys(
    tmp_path, store, settings, clock, make_command
):
    record = _run_retention(tmp_path, store, settings, clock, make_command, _hit_trace())
    assert record["status"] == "recorded"
    cache = record["observations"]["performance_cache"]
    assert cache["status"] == "hit"
    # Existing hit behavior unchanged; new fields absent, not fabricated.
    assert "timing_reuse_policy_sha256" not in cache
    assert "timing_reuse_scope_statement" not in cache


@pytest.mark.parametrize("status", ["miss", "remeasured"])
def test_legacy_miss_and_remeasured_still_recorded(
    status, tmp_path, store, settings, clock, make_command
):
    trace = _hit_trace()
    trace["performance_cache"]["status"] = status
    trace["performance_cache"].pop("sentinel_relative_drift", None)
    record = _run_retention(tmp_path, store, settings, clock, make_command, trace)
    assert record["status"] == "recorded"
    assert record["observations"]["performance_cache"]["status"] == status
    # An absent optional value stays unknown (excluded), never defaulted.
    assert "sentinel_relative_drift" not in record["observations"]["performance_cache"]


def test_unknown_status_stays_invalid(
    tmp_path, store, settings, clock, make_command
):
    trace = _timing_reuse_trace("timing_reuse")
    trace["performance_cache"]["status"] = "timing_reuse_typo"
    record = _run_retention(tmp_path, store, settings, clock, make_command, trace)
    assert set(record) == {
        "evaluation_id", "attempt_id", "request_digest", "evaluator_build",
        "correctness_algorithm", "configured_reference_device_policy",
        "physical_card", "status", "observations",
    }
    assert record["status"] == "invalid"
    assert record["observations"] is None


@pytest.mark.parametrize("base_status", ["hit", "timing_reuse"])
def test_unknown_extra_field_stays_invalid(
    base_status, tmp_path, store, settings, clock, make_command
):
    trace = (
        _hit_trace() if base_status == "hit" else _timing_reuse_trace("timing_reuse")
    )
    trace["performance_cache"]["timing_reuse_unknown_extra"] = "x"
    record = _run_retention(tmp_path, store, settings, clock, make_command, trace)
    assert record["status"] == "invalid"
    assert record["observations"] is None


def test_mistyped_timing_reuse_policy_sha_stays_invalid(
    tmp_path, store, settings, clock, make_command
):
    trace = _timing_reuse_trace("timing_reuse")
    trace["performance_cache"]["timing_reuse_policy_sha256"] = "not-a-sha"
    record = _run_retention(tmp_path, store, settings, clock, make_command, trace)
    assert record["status"] == "invalid"
    assert record["observations"] is None


def test_retained_timing_reuse_observations_round_trip_through_the_model(
    tmp_path, store, settings, clock, make_command
):
    # The retained record must be re-loadable by the same strict model, so an
    # auditor can validate retained evidence against the parser itself.
    from triton_evaluation.runtime.reference_diagnostics import _ReferenceTrace

    record = _run_retention(
        tmp_path, store, settings, clock, make_command, _timing_reuse_trace("timing_reuse")
    )
    assert record["status"] == "recorded", record
    observations = copy.deepcopy(record["observations"])
    observations["reference_device_policy"] = "npu-strict"
    parsed = _ReferenceTrace.model_validate(observations)
    assert parsed.performance_cache is not None
    assert parsed.performance_cache.status == "timing_reuse"
    assert parsed.performance_cache.timing_reuse_policy_sha256 == POLICY_SHA
