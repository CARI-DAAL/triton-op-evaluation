import json
from types import SimpleNamespace

import fakeredis
import pytest

from triton_evaluation.api.contract import (
    CheckSucceeded,
    CorrectnessResult,
    EvaluationRequest,
    EvaluationResult,
    Provenance,
)
from triton_evaluation.config import Settings
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex
from triton_evaluation.service.admission import CanonicalAdmission, admit
from triton_evaluation.service.broker import EvaluationStore
from triton_evaluation.service.models import (
    AlgorithmVersions,
    CreateResourceCommand,
    WorkerIdentity,
    new_evaluation_id,
    new_worker_id,
)
from triton_evaluation.service.security import Caller


@pytest.fixture
def settings(tmp_path):
    # Startup canary off: protocol tests predate card health and keep their
    # exact pre-feature code path; card-health tests opt in explicitly.
    return Settings(
        engine_dir=str(tmp_path),
        cann_fingerprint="cann-test",
        card_canary_on_startup=False,
    )


class FakeClock:
    """Deterministic epoch-millisecond clock injected into the store."""

    def __init__(self, now_ms: int = 1_786_000_000_000) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self.now_ms += delta_ms


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


@pytest.fixture
def redis_client(redis_server):
    # Binary connection: canonical request bytes are never decoded/re-encoded.
    return fakeredis.FakeStrictRedis(server=redis_server)


@pytest.fixture
def store(redis_client, settings, clock):
    return EvaluationStore(redis_client, settings, clock=clock)


CATALOG_REFERENCE_CODE = "import torch\n\n\nclass Model(torch.nn.Module):\n    pass\n"


def make_caller(caller_id="caller-1", caller_class="trial", scopes=("trial:submit", "trial:read")):
    return Caller(
        caller_id=caller_id, caller_class=caller_class, scopes=frozenset(scopes)
    )


def make_request(**overrides):
    payload = {
        "schema_version": "2.1",
        "op_name": "addmm",
        "reference_code": "REF",
        "kernel_code": "IMPL",
        "checks": ["correctness"],
    }
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


def make_grade_caller(caller_id="grader-1", scopes=("grade:submit", "grade:read")):
    return make_caller(caller_id, "trusted", scopes)


def make_grade_request(**overrides):
    payload = {
        "schema_version": "2.1",
        "purpose": "grade",
        "op_name": "addmm",
        "reference_code": CATALOG_REFERENCE_CODE,
        "kernel_code": "IMPL",
        "checks": ["correctness"],
        "grade_context": {
            "manifest_id": "release-1",
            "problem_id": "addmm-1",
            "reference_digest": sha256_hex(CATALOG_REFERENCE_CODE.encode("utf-8")),
        },
    }
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


@pytest.fixture
def make_command(settings):
    """Build trial admission or seed a previously admitted legacy grade record."""

    def _make(
        *,
        caller=None,
        request=None,
        session_id="session-1",
        idempotency_key="key-1",
        evaluation_id=None,
        traffic_class=None,
    ):
        caller = caller or make_caller()
        request = request or make_request()
        if request.purpose == "grade":
            # Recovery/fencing tests seed historical work directly. This is
            # intentionally not a route through the retired grade admission.
            canonical = canonical_json_bytes(request.model_dump(mode="json", exclude_none=False))
            admission = CanonicalAdmission(
                canonical_request=canonical,
                request_digest=sha256_hex(canonical),
                reference_digest=sha256_hex(request.reference_code.encode("utf-8")),
                requested_checks=tuple(request.checks),
                catalog_binding=request.grade_context,
            )
        else:
            admission = admit(request, caller)
        return CreateResourceCommand(
            evaluation_id=evaluation_id or new_evaluation_id(),
            caller_id=caller.caller_id,
            caller_class=caller.caller_class,
            traffic_class=traffic_class
            or ("trial" if request.purpose == "trial" else "trusted"),
            session_id=session_id,
            idempotency_key=idempotency_key,
            admission=admission,
            evaluator_build=settings.evaluator_build,
            algorithms=AlgorithmVersions(
                correctness=settings.correctness_algorithm_version,
                decoy=settings.decoy_algorithm_version,
                benchmark=settings.benchmark_algorithm_version,
                profiler=settings.profiler_algorithm_version,
            ),
        )

    return _make


@pytest.fixture
def release_catalog_files(tmp_path):
    """Legacy fixture data retained for historical records and HTTP test helpers."""
    root = tmp_path / "catalog"
    references = root / "references"
    references.mkdir(parents=True)
    (references / "addmm-1.py").write_text(CATALOG_REFERENCE_CODE, encoding="utf-8")
    reference_digest = sha256_hex(CATALOG_REFERENCE_CODE.encode("utf-8"))
    manifest_path = root / "catalog.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": [
                    {
                        "manifest_id": "release-1",
                        "problem_id": "addmm-1",
                        "reference_digest": reference_digest,
                        "reference_path": "references/addmm-1.py",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=root,
        manifest_path=manifest_path,
        reference_code=CATALOG_REFERENCE_CODE,
        reference_digest=reference_digest,
    )


@pytest.fixture
def make_worker(settings):
    """Build a WorkerIdentity pinned to this deployment's build and algorithms."""

    def _make(worker_id=None, *, evaluator_build=None, algorithms=None):
        return WorkerIdentity(
            worker_id=worker_id or new_worker_id(),
            evaluator_build=evaluator_build or settings.evaluator_build,
            algorithms=algorithms
            or AlgorithmVersions(
                correctness=settings.correctness_algorithm_version,
                decoy=settings.decoy_algorithm_version,
                benchmark=settings.benchmark_algorithm_version,
                profiler=settings.profiler_algorithm_version,
            ),
        )

    return _make


def make_provenance(settings, **overrides):
    payload = {
        "schema_version": "2.1",
        "evaluator_build": settings.evaluator_build,
        "correctness_algorithm": settings.correctness_algorithm_version,
        "decoy_algorithm": settings.decoy_algorithm_version,
        "benchmark_algorithm": settings.benchmark_algorithm_version,
        "profiler_algorithm": settings.profiler_algorithm_version,
        "grade_eligible": False,
        "manifest_id": None,
        "problem_id": None,
        "execution_environment": None,
    }
    payload.update(overrides)
    return Provenance(**payload)


def make_correctness_success(passed_cases=1, total_cases=1):
    return CheckSucceeded[CorrectnessResult](
        status="succeeded",
        data=CorrectnessResult(
            outcome="correct",
            passed=True,
            passed_cases=passed_cases,
            total_cases=total_cases,
            failures=[],
        ),
    )


def make_completed_result(settings):
    """A complete result for a correctness-only request."""
    return EvaluationResult(
        correctness=make_correctness_success(),
        performance=None,
        profile=None,
        provenance=make_provenance(settings),
    )
