"""Stage snapshots use real HTTP authorization and durable attempt evidence."""
import pytest
from conftest import make_correctness_success
from test_api import (
    TRIAL_TOKEN_2,
    algorithms,
    build_app,
    request_payload,
    seed_worker,
    trial_headers,
)

from triton_evaluation.api.contract import EvaluationProgress
from triton_evaluation.service.models import WorkerIdentity, new_worker_id

pytestmark = pytest.mark.anyio


async def test_progress_http_exposes_check_before_resource_finishes(
    tmp_path, release_catalog_files,
):
    app = build_app(tmp_path, release_catalog_files)
    seed_worker(app)
    async with app.http as client:
        submitted = await client.post(
            "/evaluations", headers=trial_headers(), json=request_payload(),
        )
        assert submitted.status_code == 202
        evaluation_id = submitted.json()["evaluation_id"]
        lease = app.store.claim_next(WorkerIdentity(
            new_worker_id(), app.settings.evaluator_build, algorithms(app.settings),
        ))
        assert app.store.set_active_check(lease, "correctness")
        assert app.store.save_check_outcome(lease, "correctness", make_correctness_success())
        response = await client.get(
            f"/evaluations/{evaluation_id}/progress", headers=trial_headers(),
        )
        assert response.status_code == 200 and response.headers["Retry-After"] == "1"
        progress = EvaluationProgress.model_validate_json(response.content)
        assert progress.state == "running" and progress.attempt_id == lease.attempt_id
        assert progress.completed_checks.correctness.data.outcome == "correct"
        resource = await client.get(f"/evaluations/{evaluation_id}", headers=trial_headers())
        assert resource.json()["result"] is None
        missing = await client.get(f"/evaluations/{evaluation_id}/progress")
        assert missing.status_code == 401
        invisible = await client.get(
            f"/evaluations/{evaluation_id}/progress",
            headers={"Authorization": "Bearer " + TRIAL_TOKEN_2},
        )
        assert invisible.status_code == 404
        unknown = await client.get(
            "/evaluations/ev_" + "0" * 32 + "/progress", headers=trial_headers(),
        )
        assert unknown.status_code == 404
