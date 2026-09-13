"""Caller and purpose policy select quota and scheduling through submission."""
from conftest import make_caller, make_grade_caller, make_grade_request, make_request

from triton_evaluation.instance import EvaluationInstance


def test_declared_rollout_caller_uses_bulk_queue(settings, store, release_catalog, make_worker):
    instance = EvaluationInstance(
        settings=settings.model_copy(update={"rollout_caller_ids": ["rl-1"]}),
        store=store, catalog=release_catalog, artifacts_verified=True,
    )
    worker = make_worker()
    store.heartbeat(worker)
    rollout = instance.submit(
        make_request(), make_caller("rl-1"), session_id="s1", idempotency_key="k1",
    )
    plain = instance.submit(
        make_request(), make_caller("plain-1"), session_id="s2", idempotency_key="k2",
    )
    # Interactive work is claimed before an earlier rollout submission.
    assert store.claim_next(worker).evaluation_id == plain.resource.evaluation_id
    assert store.claim_next(worker).evaluation_id == rollout.resource.evaluation_id


def test_grade_caller_is_never_reclassified_as_rollout(
    settings, store, release_catalog, make_worker,
):
    instance = EvaluationInstance(
        settings=settings.model_copy(update={"rollout_caller_ids": ["grader-1"]}),
        store=store, catalog=release_catalog, artifacts_verified=True,
    )
    worker = make_worker()
    store.heartbeat(worker)
    rollout = instance.submit(
        make_request(purpose="rollout"), make_caller(), session_id="s1", idempotency_key="k1",
    )
    grade = instance.submit(
        make_grade_request(), make_grade_caller(), session_id="s2", idempotency_key="k2",
    )
    assert store.claim_next(worker).evaluation_id == grade.resource.evaluation_id
    assert store.claim_next(worker).evaluation_id == rollout.resource.evaluation_id
