"""Header, body, and admission validation at the durable HTTP boundary."""
import json

import pytest
from test_api import (
    OBSERVER_TOKEN,
    build_app,
    grade_headers,
    post,
    request_payload,
    seed_worker,
    trial_headers,
)

from triton_evaluation.api.contract import ErrorEnvelope

pytestmark = pytest.mark.anyio

GRADE_BODY = {
    "schema_version": "2.1",
    "purpose": "grade",
    "op_name": "addmm",
    "reference_code": "import torch\n\n\nclass Model(torch.nn.Module):\n    pass\n",
    "kernel_code": "IMPL",
    "checks": ["correctness"],
}


def grade_payload():
    from conftest import CATALOG_REFERENCE_CODE

    from triton_evaluation.serialization import sha256_hex

    return GRADE_BODY | {
        "reference_code": CATALOG_REFERENCE_CODE,
        "grade_context": {
            "manifest_id": "release-1",
            "problem_id": "addmm-1",
            "reference_digest": sha256_hex(CATALOG_REFERENCE_CODE.encode("utf-8")),
        },
    }


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_missing_authorization_is_401(app):
    async with app.http as client:
        headers = trial_headers()
        del headers["Authorization"]
        response = await post(client, headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "UNAUTHENTICATED"


async def test_bearer_scheme_uses_native_case_insensitive_parsing(app):
    seed_worker(app)
    async with app.http as client:
        response = await post(
            client, headers=trial_headers() | {"Authorization": "bearer trial-token-1"},
        )
    assert response.status_code == 202


async def test_wrong_token_is_indistinguishable_401(app):
    async with app.http as client:
        response = await post(
            client, headers=trial_headers() | {"Authorization": "Bearer nope"}
        )
        response2 = await post(
            client, headers=trial_headers() | {"Authorization": "Basic abc"}
        )
    assert response.status_code == response2.status_code == 401
    assert response.json()["error"]["message"] == response2.json()["error"]["message"]
    assert "nope" not in response.text


async def test_read_only_caller_cannot_submit(app):
    async with app.http as client:
        response = await post(
            client,
            headers=trial_headers() | {"Authorization": f"Bearer {OBSERVER_TOKEN}"},
        )
    assert response.status_code == 403
    assert ErrorEnvelope.model_validate_json(response.content).error.code == "FORBIDDEN"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


async def test_missing_session_id_is_rejected(app):
    async with app.http as client:
        headers = trial_headers()
        del headers["X-Session-ID"]
        response = await post(client, headers=headers)
    assert response.status_code == 422
    assert ErrorEnvelope.model_validate_json(response.content).error.code == (
        "INVALID_REQUEST"
    )


async def test_invalid_session_id_is_rejected(app):
    async with app.http as client:
        response = await post(client, headers=trial_headers(session="bad session id"))
    assert response.status_code == 422


async def test_missing_idempotency_key_is_rejected(app):
    async with app.http as client:
        headers = trial_headers()
        del headers["Idempotency-Key"]
        response = await post(client, headers=headers)
    assert response.status_code == 422


async def test_idempotency_key_rejects_non_visible_ascii(app):
    async with app.http as client:
        response = await post(client, headers=trial_headers(key="bad\tkey"))
        response2 = await post(client, headers=trial_headers(key="x" * 129))
    assert response.status_code == response2.status_code == 422


async def test_session_header_is_not_stripped(app):
    async with app.http as client:
        response = await post(client, headers=trial_headers(session="session-1 "))
    assert response.status_code == 422  # trailing space is not trimmed


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------


async def test_missing_content_type_is_rejected(app):
    async with app.http as client:
        headers = trial_headers()
        del headers["Content-Type"]
        response = await post(client, headers=headers)
    assert response.status_code == 422


async def test_malformed_json_is_422(app):
    async with app.http as client:
        response = await client.post(
            "/evaluations", content=b"{not json", headers=trial_headers()
        )
    assert response.status_code == 422
    assert ErrorEnvelope.model_validate_json(response.content).error.code == (
        "INVALID_REQUEST"
    )


async def test_nonstandard_json_constants_are_rejected(app):
    async with app.http as client:
        body = json.dumps(request_payload()).replace(
            '"checks":', '"padding": NaN, "checks":', 1
        ).encode()
        response = await client.post("/evaluations", content=body,
                                     headers=trial_headers())
    assert response.status_code == 422


async def test_field_errors_are_422(app):
    async with app.http as client:
        response = await post(client, payload=request_payload(op_name=123))
    assert response.status_code == 422
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "INVALID_REQUEST"


async def test_cross_field_combination_errors_are_422(app):
    async with app.http as client:
        response = await post(
            client,
            payload=request_payload(
                purpose="grade", grade_context=None
            ),
        )
    assert response.status_code == 422


async def test_source_over_two_mib_is_422(app):
    async with app.http as client:
        payload = request_payload(kernel_code="k" * (2 * 1024 * 1024 + 1))
        response = await post(client, payload=payload)
    assert response.status_code == 422
    assert ErrorEnvelope.model_validate_json(response.content).error.code == (
        "INVALID_REQUEST"
    )


async def test_body_over_four_mib_is_413_with_streamed_count(app):
    async with app.http as client:
        payload = request_payload(kernel_code="k" * (5 * 1024 * 1024))
        response = await post(client, payload=payload)
    assert response.status_code == 413


async def test_body_over_four_mib_without_content_length_is_413(app):
    async with app.http as client:
        payload = request_payload(kernel_code="k" * (5 * 1024 * 1024))
        response = await client.post(
            "/evaluations",
            content=_chunked(raw_chunk(payload)),
            headers=trial_headers(),
        )
    assert response.status_code == 413


def raw_chunk(payload):
    return json.dumps(payload).encode()


async def _chunked(data, size=1024 * 1024):
    for index in range(0, len(data), size):
        yield data[index : index + size]


async def test_invalid_wait_range_is_422(app):
    async with app.http as client:
        response = await post(client, params={"wait_seconds": "61"})
        response2 = await post(client, params={"wait_seconds": "-1"})
    assert response.status_code == response2.status_code == 422


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


async def test_trial_caller_cannot_submit_grade(app):
    async with app.http as client:
        response = await post(
            client, payload=grade_payload(), headers=trial_headers()
        )
    assert response.status_code == 403
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "FORBIDDEN"


async def test_grade_binding_failure_is_403_manifest_code(app):
    async with app.http as client:
        payload = grade_payload() | {
            "grade_context": grade_payload()["grade_context"]
            | {"reference_digest": "0" * 64}
        }
        response = await post(client, payload=payload, headers=grade_headers())
    assert response.status_code == 403
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "MANIFEST_BINDING_FAILED"
    assert envelope.error.phase == "admission"
    assert envelope.error.retryable is False


async def test_idempotency_conflict_is_409(app):
    seed_worker(app)
    async with app.http as client:
        await post(client)  # same key, canonical request pinned
        response = await post(client, payload=request_payload(reference_code="OTHER"))
    assert response.status_code == 409
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "IDEMPOTENCY_CONFLICT"


# ---------------------------------------------------------------------------
# Cross-caller isolation
# ---------------------------------------------------------------------------


async def test_cross_caller_get_is_indistinguishable_404(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
        # trial-caller-2 holds no observer scope: another caller's resource
        # is indistinguishable from an unknown ID.
        response = await client.get(
            f"/evaluations/{evaluation_id}",
            headers=trial_headers() | {"Authorization": "Bearer trial-token-2"},
        )
    assert response.status_code == 404


async def test_forged_shared_session_does_not_authorize(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client, headers=trial_headers(session="shared"))
        evaluation_id = created.json()["evaluation_id"]
        # Same session header, different caller: still invisible.
        response = await client.get(
            f"/evaluations/{evaluation_id}",
            headers=trial_headers(session="shared")
            | {"Authorization": "Bearer trial-token-2"},
        )
    assert response.status_code == 404


async def test_observer_reads_any_resource(app):
    seed_worker(app)
    async with app.http as client:
        created = await post(client)
        evaluation_id = created.json()["evaluation_id"]
        response = await client.get(
            f"/evaluations/{evaluation_id}",
            headers=trial_headers() | {"Authorization": f"Bearer {OBSERVER_TOKEN}"},
        )
    assert response.status_code == 200
    assert response.json()["evaluation_id"] == evaluation_id
