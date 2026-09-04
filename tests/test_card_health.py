"""Card health module unit tests: recovery ladder routing and metric allowlists.
The canary body itself is NPU-only (tests/test_card_health_npu.py)."""
from types import SimpleNamespace
from unittest import mock

import pytest

import triton_evaluation.health as health
from triton_evaluation import metrics
from triton_evaluation.config import Settings


def test_ladder_cleanup_step_runs_without_device():
    with mock.patch.object(health, "_cleanup_in_process", return_value=None) as cleanup:
        name = health.run_recovery_step(1, card=7, reset_command="")
    assert name == "cleanup"
    cleanup.assert_called_once()


def test_ladder_reset_uses_configured_command():
    with mock.patch.object(
        health.subprocess, "run", return_value=SimpleNamespace(returncode=0)
    ) as run:
        name = health.run_recovery_step(2, card=7, reset_command="npu-smi reset -c {card}")
    assert name == "reset"
    run.assert_called_once()
    assert (
        run.call_args.kwargs.get("args") == "npu-smi reset -c 7"
        or run.call_args[0][0] == "npu-smi reset -c 7"
    )


def test_ladder_reset_without_command_is_noop_but_named():
    with mock.patch.object(health.subprocess, "run") as run:
        name = health.run_recovery_step(2, card=7, reset_command="")
    assert name == "reset"
    run.assert_not_called()


def test_ladder_restart_execs_process():
    with mock.patch.object(health.os, "execv") as execv:
        # execv is mocked so it returns; the function then hits its
        # unreachable-after-execv guard.
        with pytest.raises(AssertionError, match="unreachable"):
            health.run_recovery_step(3, card=7, reset_command="")
    execv.assert_called_once()


def test_health_metric_allowlists_registered():
    assert metrics.ALLOWED_METRICS["card_canary_total"] == ("outcome",)
    assert metrics.ALLOWED_METRICS["card_recovery_total"] == ("step",)
    assert metrics.ALLOWED_METRICS["card_health_state"] == ("card",)
    assert "card_health_state" in metrics._GAUGE_NAMES
    assert "pass" in metrics.LABEL_VALUES["outcome"]
    assert "fail" in metrics.LABEL_VALUES["outcome"]
    assert "cleanup" in metrics.LABEL_VALUES["step"]
    assert metrics.LABEL_VALUES["card"] == tuple(str(i) for i in range(16))


def test_health_settings_defaults_and_env(monkeypatch):
    s = Settings()
    assert s.canary_budget_seconds == 120
    assert s.canary_interval_seconds == 60.0
    assert s.card_reset_command == ""
    assert s.card_recovery_max_step == 2
    assert s.card_recovery_retry_seconds == 300.0
    assert s.card_canary_on_startup is True
    monkeypatch.setenv("TRITON_EVAL_CARD_RESET_COMMAND", "npu-smi reset -c {card}")
    monkeypatch.setenv("TRITON_EVAL_CARD_RECOVERY_MAX_STEP", "3")
    s2 = Settings.from_env()
    assert s2.card_reset_command == "npu-smi reset -c {card}"
    assert s2.card_recovery_max_step == 3
