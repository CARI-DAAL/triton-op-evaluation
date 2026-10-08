"""Every non-2xx path returns a schema 2.1 ErrorEnvelope with stable mappings."""

import pytest
from test_api import build_app, post, request_payload, trial_headers

from triton_evaluation.api.contract import ErrorEnvelope

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


async def exercise(app, case):
    async with app.http as client:
        if case == "unauthenticated":
            headers = trial_headers()
            del headers["Authorization"]
            return await post(client, headers=headers)
        if case == "forbidden-no-submit-scope":
            headers = trial_headers() | {"Authorization": "Bearer observer-token-1"}
            return await post(client, headers=headers)
        if case == "invalid-session":
            return await post(client, headers=trial_headers(session="bad id"))
        if case == "invalid-wait":
            return await post(client, params={"wait_seconds": "99"})
        if case == "malformed-json":
            return await client.post(
                "/evaluations", content=b"{oops", headers=trial_headers()
            )
        if case == "invalid-body-encoding":
            return await client.post(
                "/evaluations", content=b"\xff", headers=trial_headers()
            )
        if case == "field-error":
            return await post(client, payload=request_payload(op_name=[]))
        if case == "combination-error":
            return await post(client, payload=request_payload(checks=["performance"]))
        if case == "unknown-route":
            return await client.get("/nope")
        if case == "disallowed-method":
            return await client.delete("/evaluations", headers=trial_headers())
        if case == "unknown-id":
            return await client.get(
                f"/evaluations/ev_{'0' * 32}", headers=trial_headers()
            )
        raise AssertionError(f"unhandled case {case}")


CASES = {
    "unauthenticated": (401, "UNAUTHENTICATED", "auth", False, True, False),
    "forbidden-no-submit-scope": (403, "FORBIDDEN", "auth", False, False, False),
    "invalid-session": (422, "INVALID_REQUEST", "request", False, False, False),
    "invalid-wait": (422, "INVALID_REQUEST", "request", False, False, False),
    "malformed-json": (422, "INVALID_REQUEST", "request", False, False, False),
    "invalid-body-encoding": (400, "INVALID_REQUEST", "request", False, False, False),
    "field-error": (422, "INVALID_REQUEST", "request", False, False, False),
    "combination-error": (422, "INVALID_REQUEST", "request", False, False, False),
    "unknown-route": (404, "NOT_FOUND", "request", False, False, False),
    # The router answers an unregistered method with a 404 envelope.
    "disallowed-method": (404, "NOT_FOUND", "request", False, False, False),
    "unknown-id": (404, "NOT_FOUND", "request", False, False, False),
}


@pytest.mark.parametrize("case", list(CASES))
async def test_non_2xx_paths_return_stable_error_envelopes(app, case):
    status, code, phase, retryable, www_auth, retry_after = CASES[case]
    response = await exercise(app, case)
    assert response.status_code == status, response.text
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.schema_version == "2.1"
    assert envelope.error.code == code
    assert envelope.error.phase == phase
    assert envelope.error.retryable is retryable
    assert envelope.request_id
    assert ("WWW-Authenticate" in response.headers) is www_auth
    assert ("Retry-After" in response.headers) is retry_after


async def test_no_submitted_source_or_token_appears_in_any_error(app):
    secret_source = "import torch\nclass Secret:\n    pass\n"
    async with app.http as client:
        response = await post(
            client, payload=request_payload(torch_code=secret_source, checks=[])
        )
    body = response.text
    assert "Secret" not in body
    assert "trial-token-1" not in body
    assert "import torch" not in body
