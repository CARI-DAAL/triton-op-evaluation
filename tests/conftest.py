import json
from types import SimpleNamespace

import fakeredis
import pytest

from triton_evaluation.admission import admit
from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.broker import EvaluationStore
from triton_evaluation.catalog import ReleaseCatalog
from triton_evaluation.config import Settings
from triton_evaluation.models import (
    AlgorithmVersions,
    CreateResourceCommand,
    new_evaluation_id,
)
from triton_evaluation.security import Caller


@pytest.fixture
def settings(tmp_path):
    return Settings(engine_dir=str(tmp_path), dequeue_block_seconds=1)


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
        "schema_version": "2.0",
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
        "schema_version": "2.0",
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
def make_command(settings, release_catalog):
    """Build a fully admitted CreateResourceCommand with a fresh evaluation ID."""

    def _make(
        *,
        caller=None,
        request=None,
        session_id="session-1",
        idempotency_key="key-1",
        evaluation_id=None,
    ):
        caller = caller or make_caller()
        request = request or make_request()
        catalog = release_catalog if request.purpose == "grade" else ReleaseCatalog({})
        admission = admit(request, caller, catalog)
        return CreateResourceCommand(
            evaluation_id=evaluation_id or new_evaluation_id(),
            caller_id=caller.caller_id,
            caller_class=caller.caller_class,
            traffic_class="trial" if request.purpose == "trial" else "trusted",
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
    """One valid release catalog root: the manifest plus the addmm-1 reference artifact."""
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
def release_catalog(release_catalog_files):
    return ReleaseCatalog.load(
        release_catalog_files.root, release_catalog_files.manifest_path
    )
