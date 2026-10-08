"""The live HTTP API is the discoverable, model-derived wire contract."""

import pytest
from jsonschema import Draft202012Validator
from test_api import build_app, post, request_payload, seed_worker, trial_headers
from test_durable_http import claim_and_complete

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(tmp_path, release_catalog_files):
    return build_app(tmp_path, release_catalog_files)


def resolve(document, pointer):
    assert pointer.startswith("#/")
    target = document
    for token in pointer[2:].split("/"):
        target = target[token.replace("~1", "/").replace("~0", "~")]
    return target


def check_references(document, node):
    if isinstance(node, dict):
        if "$ref" in node:
            assert resolve(document, node["$ref"])
        for value in node.values():
            check_references(document, value)
    elif isinstance(node, list):
        for value in node:
            check_references(document, value)


def validate(document, schema, payload):
    # Local components remain visible when validating one operation's schema.
    Draft202012Validator({**schema, "components": document["components"]}).validate(payload)


def body_schema(operation):
    return operation["requestBody"]["content"]["application/json"]["schema"]


def response_schema(operation, status):
    return operation["responses"][str(status)]["content"]["application/json"]["schema"]


async def test_live_spec_documents_requests_responses_and_auth(app):
    async with app.http as client:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        assert (await client.get("/docs")).status_code == 200
    document = response.json()
    check_references(document, document)
    submit = document["paths"]["/evaluations"]["post"]
    assert body_schema(submit)
    validate(document, body_schema(submit), request_payload())
    assert response_schema(submit, 200)
    assert response_schema(submit, 202)
    assert "Location" in submit["responses"]["202"]["headers"]
    assert submit["security"]
    schemes = document["components"]["securitySchemes"]
    assert schemes[next(iter(submit["security"][0]))]["scheme"] == "bearer"
    parameters = {parameter["name"]: parameter for parameter in submit["parameters"]}
    assert not parameters["X-Session-ID"]["required"]
    assert not parameters["Idempotency-Key"]["required"]
    assert parameters["wait_seconds"]["schema"]["maximum"] == 60
    for status in (400, 401, 403, 409, 413, 422, 429, 500, 503):
        assert response_schema(submit, status)


async def test_real_async_replay_and_error_responses_match_live_spec(app):
    seed_worker(app)
    async with app.http as client:
        document = (await client.get("/openapi.json")).json()
        accepted = await post(client)
        assert accepted.status_code == 202
        fetched = await client.get(accepted.headers["Location"], headers=trial_headers())
        assert fetched.status_code == 200
        assert fetched.headers["Retry-After"] == "1"
        claim_and_complete(app)
        completed = await post(client)
        assert completed.status_code == 200
        unauthorized = await client.post("/evaluations", json=request_payload())
        assert unauthorized.status_code == 401
    submit = document["paths"]["/evaluations"]["post"]
    for response in (accepted, completed, unauthorized):
        schema = response_schema(submit, response.status_code)
        assert schema
        validate(document, schema, response.json())
    recover = document["paths"]["/evaluations/{evaluation_id}"]["get"]
    assert "Retry-After" in recover["responses"]["200"]["headers"]
    validate(document, response_schema(recover, 200), fetched.json())


@pytest.mark.parametrize("action", ["verify", "benchmark", "profile"])
async def test_tool_endpoints_share_operator_and_resource_models(app, action):
    seed_worker(app)
    async with app.http as client:
        document = (await client.get("/openapi.json")).json()
        operation = document["paths"][f"/{action}"]["post"]
        assert body_schema(operation) == body_schema(document["paths"]["/evaluations"]["post"])
        validate(document, body_schema(operation), request_payload())
        accepted = await client.post(f"/{action}", json=request_payload(), headers=trial_headers())
        assert accepted.status_code == 202
        refused = await client.post(f"/{action}", json=request_payload())
        assert refused.status_code == 401
    for response in (accepted, refused):
        validate(document, response_schema(operation, response.status_code), response.json())


async def test_health_metrics_and_error_content_are_typed(app):
    async with app.http as client:
        document = (await client.get("/openapi.json")).json()
    for path in document["paths"].values():
        for operation in path.values():
            for response in operation["responses"].values():
                for media in response.get("content", {}).values():
                    assert media["schema"], operation["operationId"]
    metrics = document["paths"]["/internal/metrics"]["get"]
    assert metrics["security"] == [{"CapabilityBearer": []}]
    content = metrics["responses"]["200"]["content"]
    assert set(content) == {"application/openmetrics-text"}
    assert content["application/openmetrics-text"]["schema"] == {"type": "string"}
    for name in ("/livez", "/readyz", "/health"):
        assert response_schema(document["paths"][name]["get"], 200)
    verify = document["paths"]["/verify"]["post"]
    assert verify["security"] == [{"CapabilityBearer": []}]
    assert all(
        not item["required"] for item in verify["parameters"] if item["in"] == "header"
    )
