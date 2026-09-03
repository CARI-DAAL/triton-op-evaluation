"""Rollout traffic-class wiring: declared callers map to the rollout bucket."""
from types import SimpleNamespace

from conftest import make_caller, make_grade_caller, make_grade_request, make_request

from triton_evaluation.admission import admit
from triton_evaluation.api.routes import _build_command
from triton_evaluation.catalog import ReleaseCatalog


def test_build_command_maps_declared_rollout_callers(settings, release_catalog):
    rollout_settings = settings.model_copy(update={"rollout_caller_ids": ["rl-1"]})
    context = SimpleNamespace(settings=rollout_settings)

    rollout_caller = make_caller(caller_id="rl-1")
    admission = admit(make_request(), rollout_caller, ReleaseCatalog({}))
    cmd = _build_command(context, rollout_caller, "s1", "k1", admission, "trial")
    assert cmd.traffic_class == "rollout"

    plain_caller = make_caller(caller_id="someone-else")
    admission2 = admit(make_request(), plain_caller, ReleaseCatalog({}))
    cmd2 = _build_command(context, plain_caller, "s2", "k2", admission2, "trial")
    assert cmd2.traffic_class == "trial"


def test_build_command_never_maps_grade_to_rollout(settings, release_catalog):
    rollout_settings = settings.model_copy(update={"rollout_caller_ids": ["grader-1"]})
    context = SimpleNamespace(settings=rollout_settings)
    grade_caller = make_grade_caller()
    admission = admit(make_grade_request(), grade_caller, release_catalog)
    cmd = _build_command(context, grade_caller, "s1", "k1", admission, "grade")
    assert cmd.traffic_class == "trusted"
