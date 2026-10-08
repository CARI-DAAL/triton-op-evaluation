import json
from pathlib import Path

import pytest
from test_worker import StubEvaluator, build_worker

from triton_evaluation.runtime.reference_diagnostics import _read_trace


def _trace():
    return {
        "reference_device_policy": "cpu-first",
        "cases": [
            {
                "case_idx": 1,
                "selected_reference_device": "cpu",
                "reference_completed": True,
                "output_devices": [{"output_index": 0, "device": "cpu"}],
            }
        ],
    }


@pytest.mark.parametrize("source", ["valid", "missing", "invalid", "link", "fifo", "large"])
def test_worker_retains_diagnostics_before_cleaning_attempt(
    source,
    tmp_path,
    store,
    settings,
    clock,
    make_command,
):
    import os

    settings.reference_diagnostics_dir = tmp_path / "retained"
    command = make_command()
    store.create_or_replay(command)
    secret = tmp_path / "host-private"
    secret.write_text("MUST NOT COPY THIS")

    def hook(evaluator):
        source_path = Path(evaluator.calls[-1]["work_dir"]) / "verify_reference.json"
        if source == "valid":
            source_path.write_text(json.dumps(_trace()))
        elif source == "invalid":
            source_path.write_text('{"secret":"MUST NOT COPY THIS"}')
        elif source == "link":
            source_path.symlink_to(secret)
        elif source == "fifo":
            os.mkfifo(source_path)
        elif source == "large":
            source_path.write_bytes(b"x" * (1024 * 1024 + 1))

    evaluator = StubEvaluator(settings, hook=hook)
    worker = build_worker(store, settings, evaluator, clock, device_id=3)
    assert worker.run_once()
    assert not Path(evaluator.calls[-1]["work_dir"]).exists()
    retained = list((settings.reference_diagnostics_dir / command.evaluation_id).glob("*.json"))
    assert len(retained) == 1
    text = retained[0].read_text()
    assert "MUST NOT COPY THIS" not in text
    record = json.loads(text)
    assert record["evaluation_id"] == command.evaluation_id
    assert record["attempt_id"].startswith("at_")
    assert record["physical_card"] == 3
    assert record["request_digest"]
    assert record["evaluator_build"] == settings.evaluator_build
    assert record["status"] == (
        "recorded" if source == "valid" else "missing" if source == "missing" else "invalid"
    )
    assert record["observations"] == (_trace() if source == "valid" else None)


def test_trace_policy_mismatch_is_unknown(tmp_path):
    sidecar = tmp_path / "verify_reference.json"
    sidecar.write_text(json.dumps(_trace()))
    assert _read_trace(sidecar, "npu-strict") == {"status": "invalid", "observations": None}
