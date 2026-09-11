"""Contract artifact generation and pinning tests (schemas 2.0 and 2.1).

These tests pin the Phase 0 publication: RFC 8785 canonical bytes, the exact
publication path groups, recomputable digests, the sorted fixture
inventory, strict-model round trips, and jsonschema consumer entrypoints.
"""
import json
import math
from pathlib import PurePosixPath

import pytest

from triton_evaluation.api import _frozen_v20
from triton_evaluation.api import contract as live_contract
from triton_evaluation.api.contract_artifacts import (
    build_v20_artifacts,
    build_v21_artifacts,
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

EXPECTED_FIXTURES_V21 = [
    "request-trial-compile-only.json",
    "request-trial-rollout.json",
    "resource-completed-compile-check.json",
    "resource-failed-poison.json",
]

# The 2.1 tree generates from the live models.
ENTRYPOINT_MODELS_V21 = {
    "request": live_contract.EvaluationRequest,
    "resource": live_contract.EvaluationResource,
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
    paths = {path.as_posix() for path in build_v20_artifacts()}
    expected = {"schema.json", "contract.json", "fixtures/index.json"}
    expected |= {f"fixtures/{name}" for name in EXPECTED_FIXTURES}
    assert paths == expected


def test_contract_json_has_exact_keys_and_recomputable_digests():
    artifacts = build_v20_artifacts()
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
    artifacts = build_v20_artifacts()
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
    for path, data in build_v20_artifacts().items():
        assert not data.endswith(b"\n"), path


def test_every_fixture_round_trips_through_its_strict_model():
    artifacts = build_v20_artifacts()
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

    schema = json.loads(build_v20_artifacts()[PurePosixPath("schema.json")])
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    return jsonschema.Draft202012Validator(
        {"$ref": f"{schema['$id']}#/$defs/{def_name}"}, registry=registry
    )


def test_consumer_entrypoints_resolve_and_validate_independently():
    import jsonschema

    artifacts = build_v20_artifacts()
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
    write_artifacts(tmp_path, build_v20_artifacts())
    assert verify_artifacts(tmp_path, build_v20_artifacts()) == []


def test_write_then_verify_is_clean_v21(tmp_path):
    write_artifacts(tmp_path, build_v21_artifacts())
    assert verify_artifacts(tmp_path, build_v21_artifacts()) == []


def test_verify_artifacts_detects_drift(tmp_path):
    write_artifacts(tmp_path, build_v20_artifacts())
    target = tmp_path / "fixtures" / "error-internal.json"
    data = target.read_bytes()
    target.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))
    (tmp_path / "fixtures" / "unexpected.json").write_bytes(b"{}")
    (tmp_path / "schema.json").unlink()
    assert verify_artifacts(tmp_path, build_v20_artifacts()) == [
        "fixtures/error-internal.json",
        "fixtures/unexpected.json",
        "schema.json",
    ]


# ---------------------------------------------------------------------------
# v2.1 publication path set, digest self-consistency, consumer entrypoints
# ---------------------------------------------------------------------------


def test_v21_artifact_paths_are_exactly_the_publication_set():
    paths = {path.as_posix() for path in build_v21_artifacts()}
    expected = {"schema.json", "contract.json", "fixtures/index.json"}
    expected |= {f"fixtures/{name}" for name in EXPECTED_FIXTURES_V21}
    assert paths == expected


def test_v21_contract_json_has_exact_keys_and_recomputable_digests():
    artifacts = build_v21_artifacts()
    contract = json.loads(artifacts[PurePosixPath("contract.json")])
    assert set(contract) == {
        "evaluation_api_schema_version",
        "evaluation_api_schema_digest",
        "evaluation_api_golden_fixture_digest",
        "json_schema_path",
        "golden_fixture_index_path",
    }
    assert contract["evaluation_api_schema_version"] == "2.1"
    assert contract["json_schema_path"] == "schema.json"
    assert contract["golden_fixture_index_path"] == "fixtures/index.json"
    assert contract["evaluation_api_schema_digest"] == sha256_hex(
        artifacts[PurePosixPath("schema.json")]
    )
    assert contract["evaluation_api_golden_fixture_digest"] == sha256_hex(
        artifacts[PurePosixPath("fixtures/index.json")]
    )


def test_v21_fixture_index_is_sorted_unique_and_pins_every_fixture():
    artifacts = build_v21_artifacts()
    index = json.loads(artifacts[PurePosixPath("fixtures/index.json")])
    assert index["schema_version"] == "2.1"
    entries = index["fixtures"]
    names = [entry["name"] for entry in entries]
    assert names == sorted(EXPECTED_FIXTURES_V21)
    assert len(set(names)) == len(names)
    for entry in entries:
        assert set(entry) == {"name", "path", "sha256"}
        path = PurePosixPath(entry["path"])
        assert path.as_posix() == f"fixtures/{entry['name']}"
        assert not path.is_absolute()
        assert entry["sha256"] == sha256_hex(artifacts[path])


def test_v21_every_fixture_round_trips_through_its_strict_model():
    artifacts = build_v21_artifacts()
    for name in EXPECTED_FIXTURES_V21:
        raw = artifacts[PurePosixPath(f"fixtures/{name}")]
        model = ENTRYPOINT_MODELS_V21[name.split("-", maxsplit=1)[0]]
        assert model.model_validate_json(raw).model_dump(mode="json") == json.loads(raw)


def test_v21_schema_carries_the_compile_slot_and_measurement_method():
    schema = json.loads(build_v21_artifacts()[PurePosixPath("schema.json")])
    assert schema["$id"] == "urn:triton-evaluation:evaluation-api:2.1:schema"
    result = schema["$defs"]["EvaluationResult"]["properties"]
    assert "compile" in result
    performance = schema["$defs"]["PerformanceResult"]["properties"]
    assert "measurement_method" in performance
    case = schema["$defs"]["PerformanceCase"]["properties"]
    assert "measurement_method" in case
    # The frozen 2.0 schema must NOT carry the v2.1-only fields.
    frozen = json.loads(build_v20_artifacts()[PurePosixPath("schema.json")])
    assert "compile" not in frozen["$defs"]["EvaluationResult"]["properties"]
    assert "measurement_method" not in frozen["$defs"]["PerformanceCase"]["properties"]


def _v21_entrypoint_validator(def_name):
    import jsonschema
    from referencing import Registry, Resource

    schema = json.loads(build_v21_artifacts()[PurePosixPath("schema.json")])
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    return jsonschema.Draft202012Validator(
        {"$ref": f"{schema['$id']}#/$defs/{def_name}"}, registry=registry
    )


def test_v21_consumer_entrypoints_resolve_and_validate_independently():
    import jsonschema

    artifacts = build_v21_artifacts()
    cases = {
        "EvaluationRequest": "fixtures/request-trial-rollout.json",
        "EvaluationResource": "fixtures/resource-completed-compile-check.json",
        "EvaluationResource (poison)": "fixtures/resource-failed-poison.json",
    }
    for def_name, fixture in cases.items():
        _v21_entrypoint_validator(def_name.split(" ")[0]).validate(
            json.loads(artifacts[PurePosixPath(fixture)])
        )

    request = json.loads(artifacts[PurePosixPath("fixtures/request-trial-rollout.json")])
    request["checks"] = []
    with pytest.raises(jsonschema.ValidationError):
        _v21_entrypoint_validator("EvaluationRequest").validate(request)
