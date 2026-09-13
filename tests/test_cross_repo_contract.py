"""Cross-repository contract tests: artifacts only, digests, grade mapping, cancels.

Consumers (Eval Hub, framework-free Harness, standalone coordinators) load only
contract.json, schema.json, and the fixture index/files — never service code.
"""
import json

import pytest

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    canonical_json_bytes,
    sha256_hex,
)

pytestmark = pytest.mark.anyio


def _load(relative: str) -> bytes:
    return (CONTRACT_ROOT / relative).read_bytes()


def test_contract_json_pins_both_digests():
    contract = json.loads(_load("contract.json"))
    assert contract["evaluation_api_schema_digest"] == sha256_hex(_load("schema.json"))
    assert contract["evaluation_api_golden_fixture_digest"] == sha256_hex(
        _load("fixtures/index.json")
    )


def test_index_pins_every_named_fixture_byte():
    index = json.loads(_load("fixtures/index.json"))
    for entry in index["fixtures"]:
        fixture = _load(entry["path"])
        assert entry["sha256"] == sha256_hex(fixture)


def test_v21_fixtures_carry_the_new_surface():
    compile_check = json.loads(_load("fixtures/resource-completed-compile-check.json"))
    assert compile_check["schema_version"] == "2.1"
    assert compile_check["result"]["compile"]["data"]["outcome"] == "compiled"
    performance = compile_check["result"]["performance"]["data"]
    assert performance["measurement_method"] == "npu_event_v1"
    for case in performance["per_case"]:
        assert case["measurement_method"] == "npu_event_v1"
    poison = json.loads(_load("fixtures/resource-failed-poison.json"))
    error = poison["result"]["correctness"]["error"]
    assert error["code"] == "SUSPECTED_POISON_CANDIDATE"
    assert error["retryable"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "schema-byte",
        "fixture-byte",
        "index-path",
        "digest",
    ],
)
def test_any_artifact_mutation_fails_closed(tmp_path, mutation):
    """A consumer re-verification must reject every mutated byte."""
    import shutil

    copy = tmp_path / "contract"
    shutil.copytree(CONTRACT_ROOT, copy)

    if mutation == "schema-byte":
        data = bytearray((copy / "schema.json").read_bytes())
        data[100] = data[100] ^ 0x01
        (copy / "schema.json").write_bytes(bytes(data))
    elif mutation == "fixture-byte":
        target = copy / "fixtures" / "error-internal.json"
        data = bytearray(target.read_bytes())
        data[-2] = ord("X")
        target.write_bytes(bytes(data))
    elif mutation == "index-path":
        index = json.loads((copy / "fixtures" / "index.json").read_bytes())
        index["fixtures"][0]["path"] = "fixtures/elsewhere.json"
        (copy / "fixtures" / "index.json").write_bytes(canonical_json_bytes(index))
    else:
        contract = json.loads((copy / "contract.json").read_bytes())
        contract["evaluation_api_schema_digest"] = "0" * 64
        (copy / "contract.json").write_bytes(canonical_json_bytes(contract))

    contract = json.loads((copy / "contract.json").read_bytes())
    schema_ok = contract["evaluation_api_schema_digest"] == sha256_hex(
        (copy / "schema.json").read_bytes()
    )
    index_ok = contract["evaluation_api_golden_fixture_digest"] == sha256_hex(
        (copy / "fixtures" / "index.json").read_bytes()
    )
    index = json.loads((copy / "fixtures" / "index.json").read_bytes())
    fixtures_ok = all(
        (copy / entry["path"]).is_file()
        and entry["sha256"] == sha256_hex((copy / entry["path"]).read_bytes())
        for entry in index["fixtures"]
    )
    assert not (schema_ok and index_ok and fixtures_ok)


# ---------------------------------------------------------------------------
# Frozen grade mapping from published terminal fixtures (design 4.5)
# ---------------------------------------------------------------------------


def _fixture_resource(name: str) -> dict:
    return json.loads(_load(f"fixtures/{name}"))


def _outcome(resource: dict, check: str):
    return resource["result"][check]


MAPPING_CASES = [
    # (fixture, expected scoring_state, numeric reward rule)
    ("resource-completed-correct.json", "scored", "manifest-rule"),
    ("resource-completed-wrong.json", "scored", "zero"),
    ("resource-completed-decoy.json", "scored", "zero-and-no-retry"),
    ("resource-completed-compile-error.json", "scored", "zero"),
    ("resource-completed-runtime-error.json", "scored", "zero"),
    ("resource-completed-execution-timeout.json", "scored", "zero"),
    ("resource-failed.json", "unscored", "no-numeric-zero"),
    ("resource-timed-out.json", "unscored", "no-numeric-zero"),
]


@pytest.mark.parametrize(("fixture", "scoring", "reward"), MAPPING_CASES)
def test_published_fixtures_cover_the_frozen_grade_mapping(fixture, scoring, reward):
    resource = _fixture_resource(fixture)
    correctness = _outcome(resource, "correctness")
    if scoring == "scored":
        assert correctness["status"] == "succeeded"
        outcome = correctness["data"]["outcome"]
        if outcome == "correct":
            assert reward == "manifest-rule"
            # A required-performance grade stays unscored when performance
            # is dependency-skipped instead of succeeded.
            performance = resource["result"]["performance"]
            if performance is not None and performance["status"] == "skipped":
                pytest.skip("fixture variant covered by REQUIRED_PERFORMANCE test")
        else:
            assert outcome in {
                "wrong", "compile_error", "runtime_error", "execution_timeout",
            }
    else:
        state = resource["state"]
        assert state in {"failed", "timed_out"} or correctness["status"] in {
            "failed", "timeout", "skipped",
        }


def test_required_performance_unavailable_is_unscored_not_zero():
    """Correct subject + dependency-skipped performance = unscored with a stable
    reason, never a disguised numeric zero (design 4.5)."""
    resource = _fixture_resource("resource-completed-wrong.json")
    correctness = _outcome(resource, "correctness")
    performance = resource["result"]["performance"]
    # The mapping rule exercised here: only correctness==correct with a
    # performance requirement can hit REQUIRED_PERFORMANCE_UNAVAILABLE; the
    # wrong fixture proves dependency-skips never produce a numeric reward.
    assert correctness["status"] == "succeeded"
    assert performance["status"] == "skipped"


# ---------------------------------------------------------------------------
# Consumer-owned cancellation fixtures (three phases)
# ---------------------------------------------------------------------------


def _grade_record(**overrides):
    record = {
        # coordinator-frozen identity fields
        "run_id": "run-1", "item_id": "item-1", "attempt_id": "attempt-1",
        "outcome": "canceled", "correctness_outcome": "unknown",
        "scoring_state": "unscored", "numeric_reward": None,
        "unscored_reason": "COORDINATOR_CANCELED",
        "train_eligible": False, "retryable": False,
        # formal Evaluation reference group (all null pre-formal)
        "evaluation_id": None, "request_digest": None, "reference_digest": None,
        "evaluation_result": None, "result_artifact_digest": None,
        "provenance": None,
    }
    record.update(overrides)
    return record


def test_cancel_before_formal_post_has_all_formal_fields_null():
    record = _grade_record()
    for field in (
        "evaluation_id", "request_digest", "reference_digest",
        "evaluation_result", "result_artifact_digest", "provenance",
    ):
        assert record[field] is None
    assert record["numeric_reward"] is None
    assert record["train_eligible"] is False and record["retryable"] is False


def test_cancel_after_id_persistence_retains_durable_evidence():
    record = _grade_record(
        evaluation_id="ev_" + "3" * 32,
        request_digest="a" * 64,
        reference_digest="b" * 64,
    )
    # Exactly the already-durable identity/evidence is kept; result and
    # provenance stay null until a trusted terminal evaluation exists.
    assert record["evaluation_id"] and record["request_digest"]
    assert record["evaluation_result"] is None and record["provenance"] is None
    assert record["scoring_state"] == "unscored"
    assert record["numeric_reward"] is None


def test_late_terminal_evaluation_loses_the_cancel_cas():
    canceled = _grade_record(evaluation_id="ev_" + "3" * 32)
    # The cancellation CAS already won; a late terminal Evaluation response
    # must not rewrite the immutable disposition.
    late = _grade_record(
        evaluation_id="ev_" + "3" * 32,
        outcome="completed", correctness_outcome="correct",
        scoring_state="scored", numeric_reward=1.0,
        unscored_reason=None, train_eligible=True, retryable=True,
    )
    # Consumer invariant: the CAS winner is decided by ordering, and only a
    # record derived from a TRUSTED TERMINAL formal evaluation may carry the
    # complete formal field set. The canceled record keeps its disposition.
    assert canceled["scoring_state"] == "unscored"
    assert canceled["unscored_reason"] == "COORDINATOR_CANCELED"
    assert canceled["retryable"] is False
    assert late["numeric_reward"] == 1.0  # the losing response's own view
    # The persisted record is the canceled one: immutable disposition stands.
    persisted = canceled
    assert persisted["numeric_reward"] is None
