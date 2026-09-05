"""Contract artifact generation and pinning tests (schema 2.0).

These tests pin the Phase 0 publication: RFC 8785 canonical bytes, the exact
four publication path groups, recomputable digests, the sorted fixture
inventory, strict-model round trips, and jsonschema consumer entrypoints.
"""
import json
import math
from pathlib import PurePosixPath

import pytest

from triton_evaluation.api import _frozen_v20
from triton_evaluation.api.contract_artifacts import (
    build_artifacts,
    canonical_json_bytes,
    sha256_hex,
    verify_artifacts,
    write_artifacts,
)

EXPECTED_FIXTURES = [
    "error-deadline-exceeded.json",
    "error-evaluator-execution-failed.json",
    "error-forbidden.json",
    "error-idempotency-conflict.json",
    "error-internal.json",
    "error-invalid-engine-output.json",
    "error-invalid-request.json",
    "error-limit-exceeded.json",
    "error-manifest-binding-failed.json",
    "error-not-found.json",
    "error-not-ready.json",
    "error-queue-unavailable.json",
    "error-rate-limited.json",
    "error-unauthenticated.json",
    "error-worker-lost.json",
    "request-grade-full.json",
    "request-trial-minimal.json",
    "resource-completed-compile-error.json",
    "resource-completed-correct.json",
    "resource-completed-decoy.json",
    "resource-completed-execution-timeout.json",
    "resource-completed-performance-profile-op.json",
    "resource-completed-profile-deep-partial.json",
    "resource-completed-runtime-error.json",
    "resource-completed-wrong.json",
    "resource-failed.json",
    "resource-queued.json",
    "resource-running.json",
    "resource-timed-out.json",
]

# The published 2.0 bytes correspond to the frozen snapshot models; the live
# models have since gained v2.1 fields that must not appear in the tree.
ENTRYPOINT_MODELS = {
    "request": _frozen_v20.EvaluationRequest,
    "resource": _frozen_v20.EvaluationResource,
    "error": _frozen_v20.ErrorEnvelope,
}


# ---------------------------------------------------------------------------
# RFC 8785 canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (333333333.33333329, b"333333333.3333333"),
        (1e+30, b"1e+30"),
        (4.50, b"4.5"),
        (2e-3, b"0.002"),
        (1e-27, b"1e-27"),
    ],
)
def test_rfc8785_appendix_b_number_representations(value, expected):
    assert canonical_json_bytes(value) == expected


def test_rfc8785_keys_sort_in_utf16_order():
    shuffled = {"דּ": 0, "😀": 0, "€": 0, "ö": 0, "\u0080": 0, "1": 0, "\r": 0}
    encoded = canonical_json_bytes(shuffled)
    assert list(json.loads(encoded).keys()) == ["\r", "1", "\u0080", "ö", "€", "😀", "דּ"]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_rfc8785_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value)


def test_rfc8785_rejects_lone_surrogates():
    with pytest.raises(ValueError):
        canonical_json_bytes("\ud800")


def test_json_dumps_sort_keys_is_not_the_canonicalizer():
    # codepoint order puts דּ (U+FB33) before 😀 (U+1F600); UTF-16 code unit
    # order reverses them, and json.dumps also ASCII-escapes by default
    pair = {"דּ": 0, "😀": 0}
    assert canonical_json_bytes(pair) != json.dumps(pair, sort_keys=True).encode()


# ---------------------------------------------------------------------------
# Publication path set and digest pinning
# ---------------------------------------------------------------------------


def test_artifact_paths_are_exactly_the_publication_set():
    paths = {path.as_posix() for path in build_artifacts()}
    expected = {"schema.json", "contract.json", "fixtures/index.json"}
    expected |= {f"fixtures/{name}" for name in EXPECTED_FIXTURES}
    assert paths == expected


def test_contract_json_has_exact_keys_and_recomputable_digests():
    artifacts = build_artifacts()
    contract = json.loads(artifacts[PurePosixPath("contract.json")])
    assert set(contract) == {
        "evaluation_api_schema_version",
        "evaluation_api_schema_digest",
        "evaluation_api_golden_fixture_digest",
        "json_schema_path",
        "golden_fixture_index_path",
    }
    assert contract["evaluation_api_schema_version"] == "2.0"
    assert contract["json_schema_path"] == "schema.json"
    assert contract["golden_fixture_index_path"] == "fixtures/index.json"
    assert contract["evaluation_api_schema_digest"] == sha256_hex(
        artifacts[PurePosixPath("schema.json")]
    )
    assert contract["evaluation_api_golden_fixture_digest"] == sha256_hex(
        artifacts[PurePosixPath("fixtures/index.json")]
    )


def test_fixture_index_is_sorted_unique_and_pins_every_fixture():
    artifacts = build_artifacts()
    index = json.loads(artifacts[PurePosixPath("fixtures/index.json")])
    assert index["schema_version"] == "2.0"
    entries = index["fixtures"]
    names = [entry["name"] for entry in entries]
    assert names == sorted(EXPECTED_FIXTURES)
    assert len(set(names)) == len(names)
    for entry in entries:
        assert set(entry) == {"name", "path", "sha256"}
        path = PurePosixPath(entry["path"])
        assert path.as_posix() == f"fixtures/{entry['name']}"
        assert not path.is_absolute()
        assert entry["sha256"] == sha256_hex(artifacts[path])


def test_artifacts_have_no_trailing_newline():
    for path, data in build_artifacts().items():
        assert not data.endswith(b"\n"), path


def test_every_fixture_round_trips_through_its_strict_model():
    artifacts = build_artifacts()
    for name in EXPECTED_FIXTURES:
        raw = artifacts[PurePosixPath(f"fixtures/{name}")]
        model = ENTRYPOINT_MODELS[name.split("-", maxsplit=1)[0]]
        assert model.model_validate_json(raw).model_dump(mode="json") == json.loads(raw)


# ---------------------------------------------------------------------------
# Independent jsonschema consumer entrypoints
# ---------------------------------------------------------------------------


def _entrypoint_validator(def_name):
    import jsonschema
    from referencing import Registry, Resource

    schema = json.loads(build_artifacts()[PurePosixPath("schema.json")])
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    return jsonschema.Draft202012Validator(
        {"$ref": f"{schema['$id']}#/$defs/{def_name}"}, registry=registry
    )


def test_consumer_entrypoints_resolve_and_validate_independently():
    import jsonschema

    artifacts = build_artifacts()
    cases = {
        "EvaluationRequest": "fixtures/request-trial-minimal.json",
        "EvaluationResource": "fixtures/resource-completed-correct.json",
        "ErrorEnvelope": "fixtures/error-internal.json",
    }
    for def_name, fixture in cases.items():
        _entrypoint_validator(def_name).validate(json.loads(artifacts[PurePosixPath(fixture)]))

    request = json.loads(artifacts[PurePosixPath("fixtures/request-trial-minimal.json")])
    request["checks"] = []
    with pytest.raises(jsonschema.ValidationError):
        _entrypoint_validator("EvaluationRequest").validate(request)


# ---------------------------------------------------------------------------
# Write and verify behavior
# ---------------------------------------------------------------------------


def test_write_then_verify_is_clean(tmp_path):
    write_artifacts(tmp_path)
    assert verify_artifacts(tmp_path) == []


def test_verify_artifacts_detects_drift(tmp_path):
    write_artifacts(tmp_path)
    target = tmp_path / "fixtures" / "error-internal.json"
    data = target.read_bytes()
    target.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))
    (tmp_path / "fixtures" / "unexpected.json").write_bytes(b"{}")
    (tmp_path / "schema.json").unlink()
    assert verify_artifacts(tmp_path) == [
        "fixtures/error-internal.json",
        "fixtures/unexpected.json",
        "schema.json",
    ]
