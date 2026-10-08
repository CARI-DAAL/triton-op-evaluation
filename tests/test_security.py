"""RED tests for static bearer-capability authentication (design section 3.1)."""
import dataclasses
import hashlib
import json
import os
import secrets

import pytest

import triton_evaluation.service.security as security
from triton_evaluation.service.security import (
    Caller,
    CapabilityRegistry,
    UnauthenticatedError,
)

TOKEN_TRIAL = "tok-trial-aaaa"
TOKEN_TRIAL_ROTATION = "tok-trial-bbbb"
TOKEN_GRADE = "tok-grade-cccc"
TOKEN_OBSERVER = "tok-observer-dddd"


def capability(token=TOKEN_TRIAL, caller_id="trial-1", caller_class="trial",
               scopes=("trial:submit", "trial:read"), **overrides):
    entry = {
        "token": token,
        "caller_id": caller_id,
        "caller_class": caller_class,
        "scopes": list(scopes),
    }
    entry.update(overrides)
    return entry


def write_secret(tmp_path, entries, mode=0o600, name="capabilities.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"capabilities": entries}), encoding="utf-8")
    os.chmod(path, mode)
    return path


def load_registry(tmp_path, entries=None, mode=0o600):
    if entries is None:
        entries = [capability()]
    return CapabilityRegistry.from_secret_file(write_secret(tmp_path, entries, mode))


def three_caller_entries():
    return [
        capability(),
        capability(token=TOKEN_GRADE, caller_id="hub-1", caller_class="trusted",
                   scopes=("grade:submit", "grade:read")),
        capability(token=TOKEN_OBSERVER, caller_id="obs-1", caller_class="observer",
                   scopes=("observer:read",)),
    ]


# ---------------------------------------------------------------------------
# Authentication outcomes
# ---------------------------------------------------------------------------


def test_authenticates_valid_bearer_token(tmp_path):
    registry = load_registry(tmp_path)
    caller = registry.authenticate(f"Bearer {TOKEN_TRIAL}")
    assert caller == Caller(
        caller_id="trial-1",
        caller_class="trial",
        scopes=frozenset({"trial:submit", "trial:read"}),
    )


def test_missing_authorization_is_unauthenticated(tmp_path):
    registry = load_registry(tmp_path)
    with pytest.raises(UnauthenticatedError) as excinfo:
        registry.authenticate(None)
    assert excinfo.value.code == "UNAUTHENTICATED"
    assert str(excinfo.value) == "missing or invalid bearer credential"


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        "bearer " + TOKEN_TRIAL,
        "Basic " + TOKEN_TRIAL,
        "Bearer one two",
        "Bearer  " + TOKEN_TRIAL,
        "Bearer " + TOKEN_TRIAL + " ",
        "Bearer\t" + TOKEN_TRIAL,
        "Bearer " + TOKEN_TRIAL + " Bearer " + TOKEN_GRADE,
    ],
)
def test_malformed_authorization_is_unauthenticated(tmp_path, header):
    registry = load_registry(tmp_path, three_caller_entries())
    with pytest.raises(UnauthenticatedError) as excinfo:
        registry.authenticate(header)
    assert str(excinfo.value) == "missing or invalid bearer credential"


def test_wrong_token_gets_the_same_generic_error(tmp_path):
    registry = load_registry(tmp_path)
    with pytest.raises(UnauthenticatedError) as missing:
        registry.authenticate(None)
    with pytest.raises(UnauthenticatedError) as wrong:
        registry.authenticate("Bearer tok-not-configured")
    assert str(wrong.value) == str(missing.value)
    assert "tok-not-configured" not in str(wrong.value)
    assert "tok-not-configured" not in repr(wrong.value)


def test_token_text_never_appears_in_authentication_errors(tmp_path):
    registry = load_registry(tmp_path)
    presented = "Bearer " + TOKEN_GRADE  # a real token presented to the wrong registry
    with pytest.raises(UnauthenticatedError) as excinfo:
        registry.authenticate(presented)
    assert TOKEN_GRADE not in str(excinfo.value)
    assert TOKEN_GRADE not in repr(excinfo.value)


# ---------------------------------------------------------------------------
# Constant-time digest lookup
# ---------------------------------------------------------------------------


def test_digest_lookup_compares_every_configured_digest(tmp_path, monkeypatch):
    registry = load_registry(tmp_path, three_caller_entries())
    calls = []
    real_compare = secrets.compare_digest

    def spy(first, second):
        calls.append((first, second))
        return real_compare(first, second)

    monkeypatch.setattr(secrets, "compare_digest", spy)

    caller = registry.authenticate(f"Bearer {TOKEN_GRADE}")
    assert caller.caller_id == "hub-1"
    # No early exit: every configured digest is compared even after a match.
    assert len(calls) == 3
    expected = hashlib.sha256(TOKEN_GRADE.encode("utf-8")).digest()
    for presented, configured in calls:
        assert presented == expected  # the presented token is hashed, never compared raw
        assert isinstance(configured, bytes) and len(configured) == 32

    calls.clear()
    with pytest.raises(UnauthenticatedError):
        registry.authenticate("Bearer tok-unknown")
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Secret-file validation
# ---------------------------------------------------------------------------


def test_duplicate_token_is_rejected(tmp_path):
    entries = [capability(), capability(caller_id="trial-2")]
    with pytest.raises(ValueError, match="unique") as excinfo:
        load_registry(tmp_path, entries)
    assert TOKEN_TRIAL not in str(excinfo.value)


def test_duplicate_scopes_within_one_entry_are_rejected(tmp_path):
    entries = [capability(scopes=["trial:read", "trial:read"])]
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(tmp_path, entries)


def test_conflicting_caller_class_is_rejected(tmp_path):
    entries = [
        capability(),
        capability(token=TOKEN_TRIAL_ROTATION, caller_class="trusted",
                   scopes=("grade:read",)),
    ]
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        load_registry(tmp_path, entries)


def test_conflicting_caller_scopes_are_rejected(tmp_path):
    entries = [
        capability(),
        capability(token=TOKEN_TRIAL_ROTATION, scopes=("trial:read",)),
    ]
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        load_registry(tmp_path, entries)


def test_rotation_tokens_share_one_identical_caller(tmp_path):
    entries = [capability(), capability(token=TOKEN_TRIAL_ROTATION)]
    registry = load_registry(tmp_path, entries)
    first = registry.authenticate(f"Bearer {TOKEN_TRIAL}")
    second = registry.authenticate(f"Bearer {TOKEN_TRIAL_ROTATION}")
    assert first == second


@pytest.mark.parametrize(
    ("caller_class", "scopes"),
    [
        ("trial", ["grade:read"]),
        ("trial", ["observer:read"]),
        ("trusted", ["trial:submit"]),
        ("trusted", ["trial:read"]),
        ("observer", ["trial:read"]),
        ("observer", ["grade:read", "grade:submit"]),  # observer:read is mandatory
    ],
)
def test_scope_grants_stay_within_caller_class(tmp_path, caller_class, scopes):
    entries = [capability(caller_class=caller_class, scopes=scopes)]
    with pytest.raises(ValueError):
        load_registry(tmp_path, entries)


@pytest.mark.parametrize(
    ("caller_class", "scopes"),
    [
        ("trial", ["trial:read"]),
        ("trusted", ["observer:read"]),
        ("observer", ["observer:read", "grade:submit", "grade:read"]),
    ],
)
def test_allowed_scope_grants_load(tmp_path, caller_class, scopes):
    registry = load_registry(
        tmp_path, [capability(caller_class=caller_class, scopes=scopes)]
    )
    caller = registry.authenticate(f"Bearer {TOKEN_TRIAL}")
    assert caller.scopes == frozenset(scopes)


def test_at_least_one_scope_is_required(tmp_path):
    with pytest.raises(ValueError):
        load_registry(tmp_path, [capability(scopes=[])])


@pytest.mark.parametrize("caller_id", ["bad id", "bad/id", "x" * 129, ""])
def test_caller_id_must_match_the_stable_pattern(tmp_path, caller_id):
    with pytest.raises(ValueError):
        load_registry(tmp_path, [capability(caller_id=caller_id)])


def test_caller_id_accepts_the_documented_alphabet(tmp_path):
    registry = load_registry(tmp_path, [capability(caller_id="A-z_09._:-")])
    assert registry.authenticate(f"Bearer {TOKEN_TRIAL}").caller_id == "A-z_09._:-"


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o777])
def test_secret_file_mode_broader_than_0600_is_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="0600"):
        load_registry(tmp_path, mode=mode)


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_secret_file_mode_at_or_below_0600_loads(tmp_path, mode):
    load_registry(tmp_path, mode=mode)


def test_missing_secret_file_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        CapabilityRegistry.from_secret_file(tmp_path / "absent.json")


def test_malformed_secret_json_is_rejected_without_echoing_tokens(tmp_path):
    path = tmp_path / "capabilities.json"
    path.write_text('{"capabilities": [{"token": "s3cr3t"', encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(ValueError) as excinfo:
        CapabilityRegistry.from_secret_file(path)
    assert "s3cr3t" not in str(excinfo.value)


def test_schema_failures_do_not_echo_token_text(tmp_path):
    entries = [capability(token="s3cr3t-value", caller_id="bad id")]
    with pytest.raises(ValueError) as excinfo:
        load_registry(tmp_path, entries)
    assert "s3cr3t-value" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Surface discipline: static capabilities only
# ---------------------------------------------------------------------------


def test_no_login_or_token_minting_surface():
    registry_api = {name for name in vars(CapabilityRegistry) if not name.startswith("_")}
    assert registry_api == {"from_secret_file", "authenticate"}
    assert {f.name for f in dataclasses.fields(Caller)} == {
        "caller_id",
        "caller_class",
        "scopes",
    }
    module_api = {name for name in vars(security) if not name.startswith("_")}
    forbidden = [
        name
        for name in module_api
        if any(term in name.lower() for term in ("login", "mint", "issue", "signin"))
    ]
    assert forbidden == []
