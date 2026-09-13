"""RED tests for canonical grade admission (design sections 3.1-3.2)."""
import json
import re

import pytest
from pydantic import ValidationError

from triton_evaluation.admission import ForbiddenError, admit
from triton_evaluation.api.contract import EvaluationRequest
from triton_evaluation.api.contract_artifacts import canonical_json_bytes, sha256_hex
from triton_evaluation.catalog import ManifestBindingError, ReleaseCatalog
from triton_evaluation.security import Caller

KERNEL_CODE = "import torch\n\n\nclass ModelNew(torch.nn.Module):\n    pass\n"
OTHER_REFERENCE_CODE = "import torch\n\n\nclass Other(torch.nn.Module):\n    pass\n"

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


@pytest.fixture
def catalog(release_catalog_files):
    return ReleaseCatalog.load(
        release_catalog_files.root, release_catalog_files.manifest_path
    )


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonicalization_collapses_explicit_defaults(catalog, release_catalog_files):
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
    first = admit(minimal, TRIAL, catalog)
    second = admit(explicit, TRIAL, catalog)
    assert first.canonical_request == second.canonical_request
    assert first.request_digest == second.request_digest


def test_canonicalization_collapses_check_order(catalog, release_catalog_files):
    forward = make_request(
        release_catalog_files.reference_code, checks=("correctness", "performance")
    )
    reverse = make_request(
        release_catalog_files.reference_code, checks=("performance", "correctness")
    )
    assert admit(forward, TRIAL, catalog).canonical_request == admit(
        reverse, TRIAL, catalog
    ).canonical_request


def test_canonicalization_collapses_profile_case_order(catalog, release_catalog_files):
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
    assert admit(ascending, TRIAL, catalog).request_digest == admit(
        descending, TRIAL, catalog
    ).request_digest


def test_requested_checks_follow_the_fixed_execution_order(
    catalog, release_catalog_files
):
    request = make_request(
        release_catalog_files.reference_code, checks=("profile", "correctness")
    )
    admission = admit(request, TRIAL, catalog)
    assert tuple(admission.requested_checks) == ("correctness", "profile")


def test_source_bytes_are_preserved_verbatim(catalog):
    source = "import torch  \n\n# trailing whitespace and blank lines\n\n\n"
    admission = admit(make_request(source), TRIAL, catalog)
    payload = json.loads(admission.canonical_request.decode("utf-8"))
    assert payload["reference_code"] == source
    assert payload["kernel_code"] == KERNEL_CODE
    assert admission.reference_digest == sha256_hex(source.encode("utf-8"))
    other = admit(make_request(source + " "), TRIAL, catalog)
    assert other.request_digest != admission.request_digest
    assert other.reference_digest != admission.reference_digest


def test_canonical_bytes_match_the_rfc8785_contract(catalog, release_catalog_files):
    request = make_request(release_catalog_files.reference_code)
    admission = admit(request, TRIAL, catalog)
    expected = canonical_json_bytes(request.model_dump(mode="json", exclude_none=False))
    assert admission.canonical_request == expected
    assert admission.request_digest == sha256_hex(expected)
    assert re.fullmatch(r"[0-9a-f]{64}", admission.request_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", admission.reference_digest)


def test_wait_session_and_headers_never_enter_the_canonical_form(
    catalog, release_catalog_files
):
    request = make_request(release_catalog_files.reference_code)
    first = admit(request, TRIAL, catalog)
    other_caller = Caller(
        caller_id="trial-9", caller_class="trial", scopes=frozenset({"trial:submit"}),
    )
    second = admit(request, other_caller, catalog)
    # Caller identity, session correlation, and wait budgets are transport
    # metadata: the canonical request is identical for any admittable caller.
    assert first.canonical_request == second.canonical_request
    assert first.request_digest == second.request_digest
    for marker in (b"wait_seconds", b"session", b"Authorization", b"Bearer", b"trial-1"):
        assert marker not in first.canonical_request


# ---------------------------------------------------------------------------
# Purpose and scope isolation
# ---------------------------------------------------------------------------


def test_trial_admission_has_no_catalog_binding(catalog, release_catalog_files):
    admission = admit(make_request(release_catalog_files.reference_code), TRIAL, catalog)
    assert admission.catalog_binding is None


@pytest.mark.parametrize("who", [GRADE, OBSERVER, OBSERVER_WITH_GRADE_SUBMIT])
def test_trial_requests_require_trial_submit(catalog, release_catalog_files, who):
    request = make_request(release_catalog_files.reference_code)
    with pytest.raises(ForbiddenError) as excinfo:
        admit(request, who, catalog)
    assert excinfo.value.code == "FORBIDDEN"


def test_trial_read_alone_cannot_submit(catalog, release_catalog_files):
    request = make_request(release_catalog_files.reference_code)
    with pytest.raises(ForbiddenError):
        admit(request, TRIAL_READONLY, catalog)


@pytest.mark.parametrize("who", [TRIAL, OBSERVER])
def test_grade_requests_require_grade_submit(catalog, release_catalog_files, who):
    request = make_grade_request(release_catalog_files)
    with pytest.raises(ForbiddenError):
        admit(request, who, catalog)


def test_grade_admission_binds_the_catalog(catalog, release_catalog_files):
    admission = admit(make_grade_request(release_catalog_files), GRADE, catalog)
    binding = admission.catalog_binding
    assert binding is not None
    assert binding.manifest_id == "release-1"
    assert binding.problem_id == "addmm-1"
    assert binding.reference_digest == release_catalog_files.reference_digest


def test_observer_with_explicit_grade_submit_uses_grade_path(
    catalog, release_catalog_files
):
    admission = admit(
        make_grade_request(release_catalog_files), OBSERVER_WITH_GRADE_SUBMIT, catalog
    )
    assert admission.catalog_binding is not None


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


# ---------------------------------------------------------------------------
# Catalog triple and body digest are checked independently
# ---------------------------------------------------------------------------


def test_unknown_manifest_is_rejected(catalog, release_catalog_files):
    request = make_grade_request(release_catalog_files, manifest_id="release-9")
    with pytest.raises(ManifestBindingError) as excinfo:
        admit(request, GRADE, catalog)
    message = str(excinfo.value)
    assert "release-9" not in message
    assert release_catalog_files.reference_code not in message


def test_unknown_problem_is_rejected(catalog, release_catalog_files):
    request = make_grade_request(release_catalog_files, problem_id="relu-9")
    with pytest.raises(ManifestBindingError):
        admit(request, GRADE, catalog)


def test_context_digest_must_match_the_catalog_entry(catalog, release_catalog_files):
    request = make_grade_request(
        release_catalog_files,
        reference_digest=sha256_hex(OTHER_REFERENCE_CODE.encode("utf-8")),
    )
    with pytest.raises(ManifestBindingError):
        admit(request, GRADE, catalog)


def test_body_reference_must_match_the_bound_digest(catalog, release_catalog_files):
    request = make_grade_request(
        release_catalog_files, reference_code=OTHER_REFERENCE_CODE
    )
    with pytest.raises(ManifestBindingError) as excinfo:
        admit(request, GRADE, catalog)
    assert OTHER_REFERENCE_CODE not in str(excinfo.value)
