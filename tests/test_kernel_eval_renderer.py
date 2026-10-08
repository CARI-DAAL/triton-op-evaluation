"""Tool observations replay from frozen bytes without importing a bridge."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from triton_evaluation import kernel_eval_renderer as renderer

FIXTURES = Path(__file__).parent / "fixtures" / "kernel_eval"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())


def render_fixture(fixture):
    params = fixture["params"]
    common = {
        "action": params["action"],
        "kernel_code": params["kernel_code"],
        "eval_budget": params["eval_budget"],
    }
    resource = fixture.get("resource")
    if resource is None:
        if "message" in params:
            return renderer.render_infra_error(
                **common, eval_turn=params.get("eval_turn"), message=params["message"]
            )
        return renderer.render_budget_exhausted(**common)
    return renderer.render_envelope(
        resource,
        **common,
        eval_turn=params.get("eval_turn"),
        diagnostics=renderer.compose_diagnostics(params.get("diagnostics_profile")),
    )


@pytest.mark.parametrize("entry", MANIFEST["fixtures"], ids=lambda entry: entry["name"])
def test_frozen_envelope_and_observation_bytes(entry):
    name = entry["name"]
    input_bytes = (FIXTURES / f"{name}.input.json").read_bytes()
    baseline_bytes = (FIXTURES / f"{name}.baseline.envelope.json").read_bytes()
    observation_bytes = (FIXTURES / f"{name}.baseline.observation.txt").read_bytes()
    assert hashlib.sha256(input_bytes).hexdigest() == entry["sha256"]["input"]
    expected = json.loads(baseline_bytes)
    # Historical MANIFEST hashes canonical envelope/observation bytes; disk
    # files were pretty printed and given a trailing newline by the miner.
    assert (
        hashlib.sha256(renderer.envelope_bytes(expected)).hexdigest()
        == entry["sha256"]["baseline_envelope"]
    )
    assert (
        hashlib.sha256(observation_bytes[:-1]).hexdigest()
        == entry["sha256"]["baseline_observation"]
    )
    actual = render_fixture(json.loads(input_bytes))
    assert renderer.envelope_bytes(actual) == renderer.envelope_bytes(expected)
    assert actual["observation"].encode() + b"\n" == observation_bytes


def fixture(name):
    return copy.deepcopy(json.loads((FIXTURES / f"{name}.input.json").read_bytes()))


def test_benchmark_unavailable():
    data = fixture("benchmark_measured")
    data["resource"]["result"]["performance"] = None
    result = render_fixture(data)
    assert result["request_status"] == "completed"
    assert result["correct"] is True
    assert result["benchmark_available"] is False
    assert result["observation"].endswith(
        "[kernel_eval][benchmark][unavailable] "
        f"verify_cases={result['passed_cases']}/{result['total_cases']}"
    )


def test_correctness_skipped():
    data = fixture("verify_correct")
    data["params"].pop("diagnostics_profile", None)
    data["resource"]["result"]["correctness"] = {
        "status": "skipped",
        "reason": {"code": "DEPENDENCY_FAILED", "message": "compile unavailable"},
    }
    result = render_fixture(data)
    assert result["request_status"] == "infra_error"
    assert result["observation"].endswith(
        "error=verify skipped: DEPENDENCY_FAILED: compile unavailable"
    )


def test_profile_rejected():
    data = fixture("profile_complete")
    data["resource"]["result"]["profile"] = {
        "status": "skipped",
        "reason": {"code": "DEPENDENCY_FAILED", "message": "compile unavailable"},
    }
    result = render_fixture(data)
    assert result["task_status"] == "REJECTED"
    assert result["profile_state"] == "infra_error"
    assert result["request_status"] == "infra_error"
    assert result["observation"].endswith("error=DEPENDENCY_FAILED: compile unavailable")


def test_evaluation_failed_uses_first_error():
    data = fixture("verify_correct")
    data["params"].pop("diagnostics_profile", None)
    data["resource"]["state"] = "failed"
    data["resource"]["result"]["correctness"] = {
        "status": "failed",
        "error": {"code": "INTERNAL", "message": "worker unavailable"},
    }
    result = render_fixture(data)
    evaluation_id = data["resource"]["evaluation_id"]
    assert result["request_status"] == "infra_error"
    assert result["observation"].endswith(
        f"error=evaluation {evaluation_id} failed: INTERNAL: worker unavailable"
    )
