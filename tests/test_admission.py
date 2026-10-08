"""Canonical trial/rollout admission and passive legacy-grade decoding."""
import json
import re

import pytest
from pydantic import ValidationError

from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.serialization import canonical_json_bytes, sha256_hex
from triton_evaluation.service.admission import ForbiddenError, admit
from triton_evaluation.service.security import Caller

KERNEL_CODE = "import torch\n\n\nclass ModelNew(torch.nn.Module):\n    pass\n"

TRIAL = Caller(
    caller_id="trial-1", caller_class="trial",
    scopes=frozenset({"trial:submit", "trial:read"}),
)
TRIAL_READONLY = Caller(
    caller_id="trial-2", caller_class="trial", scopes=frozenset({"trial:read"}),
)
GRADE = Caller(
    caller_id="hub-1", caller_class="trusted",
    scopes=frozenset({"grade:submit", "grade:read"}),
)
OBSERVER = Caller(
    caller_id="obs-1", caller_class="observer", scopes=frozenset({"observer:read"}),
)
OBSERVER_WITH_GRADE_SUBMIT = Caller(
    caller_id="obs-2", caller_class="observer",
    scopes=frozenset({"observer:read", "grade:submit"}),
)


def make_request(reference_code, checks=("correctness",), **overrides):
    payload = {
        "schema_version": "2.1",
        "op_name": "addmm",
        "reference_code": reference_code,
        "kernel_code": KERNEL_CODE,
        "checks": list(checks),
    }
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


def make_grade_request(files, reference_code=None, reference_digest=None,
                       manifest_id="release-1", problem_id="addmm-1"):
    code = files.reference_code if reference_code is None else reference_code
    digest = files.reference_digest if reference_digest is None else reference_digest
    return make_request(
        code,
        purpose="grade",
        grade_context={
            "manifest_id": manifest_id,
            "problem_id": problem_id,
            "reference_digest": digest,
        },
    )


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonicalization_collapses_explicit_defaults(release_catalog_files):
    minimal = make_request(release_catalog_files.reference_code)
    explicit = make_request(
        release_catalog_files.reference_code,
        purpose="trial",
        triton_impl_name="triton_ascend_impl",
        non_compute=False,
        deadline_seconds=900,
        performance=None,
        profile=None,
        grade_context=None,
    )
    first = admit(minimal, TRIAL)
    second = admit(explicit, TRIAL)
    assert first.canonical_request == second.canonical_request
    assert first.request_digest == second.request_digest


def test_canonicalization_collapses_check_order(release_catalog_files):
    forward = make_request(
        release_catalog_files.reference_code, checks=("correctness", "performance")
    )
    reverse = make_request(
        release_catalog_files.reference_code, checks=("performance", "correctness")
    )
    assert admit(forward, TRIAL).canonical_request == admit(
        reverse, TRIAL
    ).canonical_request


def test_canonicalization_collapses_profile_case_order(release_catalog_files):
    ascending = make_request(
        release_catalog_files.reference_code,
        checks=("correctness", "profile"),
        profile={"case_indices": [0, 2]},
    )
    descending = make_request(
        release_catalog_files.reference_code,
        checks=("correctness", "profile"),
        profile={"case_indices": [2, 0]},
    )
    assert admit(ascending, TRIAL).request_digest == admit(
        descending, TRIAL
    ).request_digest


def test_requested_checks_follow_the_fixed_execution_order(
    release_catalog_files
):
    request = make_request(
        release_catalog_files.reference_code, checks=("profile", "correctness")
    )
    admission = admit(request, TRIAL)
    assert tuple(admission.requested_checks) == ("correctness", "profile")


def test_source_bytes_are_preserved_verbatim():
    source = "import torch  \n\n# trailing whitespace and blank lines\n\n\n"
    admission = admit(make_request(source), TRIAL)
    payload = json.loads(admission.canonical_request.decode("utf-8"))
    assert payload["reference_code"] == source
    assert payload["kernel_code"] == KERNEL_CODE
    assert admission.reference_digest == sha256_hex(source.encode("utf-8"))
    other = admit(make_request(source + " "), TRIAL)
    assert other.request_digest != admission.request_digest
    assert other.reference_digest != admission.reference_digest


def test_canonical_bytes_match_the_rfc8785_contract(release_catalog_files):
    request = make_request(release_catalog_files.reference_code)
    admission = admit(request, TRIAL)
    expected = canonical_json_bytes(request.model_dump(mode="json", exclude_none=False))
    assert admission.canonical_request == expected
    assert admission.request_digest == sha256_hex(expected)
    assert re.fullmatch(r"[0-9a-f]{64}", admission.request_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", admission.reference_digest)


def test_wait_session_and_headers_never_enter_the_canonical_form(
    release_catalog_files
):
    request = make_request(release_catalog_files.reference_code)
    first = admit(request, TRIAL)
    other_caller = Caller(
        caller_id="trial-9", caller_class="trial", scopes=frozenset({"trial:submit"}),
    )
    second = admit(request, other_caller)
    # Caller identity, session correlation, and wait budgets are transport
    # metadata: the canonical request is identical for any admittable caller.
    assert first.canonical_request == second.canonical_request
    assert first.request_digest == second.request_digest
    for marker in (b"wait_seconds", b"session", b"Authorization", b"Bearer", b"trial-1"):
        assert marker not in first.canonical_request


# ---------------------------------------------------------------------------
# Purpose and scope isolation
# ---------------------------------------------------------------------------


def test_trial_admission_has_no_legacy_grade_binding(release_catalog_files):
    admission = admit(make_request(release_catalog_files.reference_code), TRIAL)
    assert admission.catalog_binding is None


@pytest.mark.parametrize("who", [GRADE, OBSERVER, OBSERVER_WITH_GRADE_SUBMIT])
def test_trial_requests_require_trial_submit(release_catalog_files, who):
    request = make_request(release_catalog_files.reference_code)
    with pytest.raises(ForbiddenError) as excinfo:
        admit(request, who)
    assert excinfo.value.code == "FORBIDDEN"


def test_trial_read_alone_cannot_submit(release_catalog_files):
    request = make_request(release_catalog_files.reference_code)
    with pytest.raises(ForbiddenError):
        admit(request, TRIAL_READONLY)


@pytest.mark.parametrize("who", [TRIAL, GRADE, OBSERVER, OBSERVER_WITH_GRADE_SUBMIT])
def test_new_grade_requests_are_forbidden_even_with_grade_scope(release_catalog_files, who):
    request = make_grade_request(release_catalog_files)
    with pytest.raises(ForbiddenError):
        admit(request, who)


# ---------------------------------------------------------------------------
# Contract coupling (rejected before admission ever runs)
# ---------------------------------------------------------------------------


def test_trial_requests_reject_grade_context(release_catalog_files):
    with pytest.raises(ValidationError):
        make_request(
            release_catalog_files.reference_code,
            grade_context={
                "manifest_id": "release-1",
                "problem_id": "addmm-1",
                "reference_digest": release_catalog_files.reference_digest,
            },
        )


def test_grade_requests_require_grade_context(release_catalog_files):
    with pytest.raises(ValidationError):
        make_request(release_catalog_files.reference_code, purpose="grade")


def test_old_grade_records_remain_decodable_with_pinned_context(
    release_catalog_files, make_command,
):
    request = make_grade_request(release_catalog_files)
    command = make_command(request=request, caller=GRADE)
    restored = EvaluationRequest.model_validate_json(command.admission.canonical_request)
    assert restored == request
    assert restored.purpose == "grade"
    assert restored.grade_context == command.admission.catalog_binding
    assert restored.grade_context.reference_digest == command.admission.reference_digest
    with pytest.raises(ForbiddenError):
        admit(restored, GRADE)


def test_new_grade_is_forbidden_even_with_both_submit_scopes(release_catalog_files):
    caller = Caller(
        caller_id="combined", caller_class="trusted",
        scopes=frozenset({"grade:submit", "trial:submit"}),
    )
    with pytest.raises(ForbiddenError):
        admit(make_grade_request(release_catalog_files), caller)
