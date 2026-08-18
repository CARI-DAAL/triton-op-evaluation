"""HTTP security: request IDs, redaction, and header hygiene."""
import re

import pytest
from test_api import build_app, post, trial_headers

from triton_evaluation.api.contract import ErrorEnvelope

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


async def test_valid_inbound_request_id_is_propagated(app):
    async with app.http as client:
        response = await post(
            client,
            params={"wait_seconds": "999"},  # invalid wait -> error path
            headers=trial_headers(request_id="client-generated-id-1"),
        )
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.request_id == "client-generated-id-1"


async def test_invalid_inbound_request_id_is_replaced(app):
    async with app.http as client:
        response = await post(
            client,
            params={"wait_seconds": "999"},
            headers=trial_headers(request_id="bad request id!!"),
        )
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert re.fullmatch(r"rq_[0-9a-f]{32}", envelope.request_id)


async def test_generated_request_id_shape(app):
    async with app.http as client:
        response = await post(
            client, params={"wait_seconds": "999"}, headers=trial_headers()
        )
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert re.fullmatch(r"rq_[0-9a-f]{32}", envelope.request_id)


async def test_error_bodies_are_single_line_and_byte_bounded(app):
    async with app.http as client:
        response = await post(
            client, params={"wait_seconds": "999"}, headers=trial_headers()
        )
    text = response.text
    assert "\n" not in text and "\r" not in text
    assert len(text.encode()) <= 2048


async def test_no_worker_or_queue_details_in_errors(app):
    async with app.http as client:
        response = await post(client, headers=trial_headers())
        # No worker heartbeats seeded: readiness fails closed without detail.
    assert response.status_code == 503
    envelope = ErrorEnvelope.model_validate_json(response.content)
    assert envelope.error.code == "NOT_READY"
    assert response.headers.get("Retry-After") == "1"
    for leak in ("worker", "queue", "heartbeat", "redis", "6379"):
        assert leak not in response.text.lower()
