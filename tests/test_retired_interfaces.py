"""Removed public interfaces stay absent even if stale deployment flags remain."""
import pytest

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("method,path", [
    ("GET", "/tasks/ev_removed"),
])
async def test_retired_routes_are_absent(
    tmp_path, release_catalog_files, monkeypatch, method, path,
):
    from test_api import build_app, trial_headers

    monkeypatch.setenv("TRITON_EVAL_LEGACY_ROUTES_ENABLED", "true")
    app = build_app(tmp_path, release_catalog_files)
    async with app.http as client:
        response = await client.request(method, path, headers=trial_headers())
    assert response.status_code == 404
    assert response.json()["schema_version"] == "2.1"
    assert response.json()["error"]["code"] == "NOT_FOUND"
