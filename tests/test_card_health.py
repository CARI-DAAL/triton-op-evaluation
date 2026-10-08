"""Card health module unit tests: recovery ladder routing and metric allowlists.
The canary body itself is NPU-only (tests/test_card_health_npu.py)."""
from types import SimpleNamespace
from unittest import mock

import pytest
import redis

import triton_evaluation.runtime.health as health
from triton_evaluation.config import Settings
from triton_evaluation.service import metrics


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


@pytest.mark.parametrize("recovery_probes", [[True], [False, True], [False, False, True]])
def test_card_health_resumes_after_direct_or_ladder_recovery(
    store, settings, clock, recovery_probes,
):
    probes = iter([False, False, False, *recovery_probes])
    controller = health.CardHealth(
        store, settings.model_copy(update={"card_canary_on_startup": True}),
        device_id=0, clock=clock,
        canary=lambda **kwargs: next(probes),
        recovery_step=lambda step, **kwargs: "cleanup" if step == 1 else "reset",
    )
    assert controller.prepare_claim() is False
    assert store.is_card_unhealthy(0)
    clock.advance_ms(300_001)
    assert controller.prepare_claim() is False
    assert not store.is_card_unhealthy(0)
    assert controller.prepare_claim() is True


def test_health_gauge_is_current_state_across_failed_probes_and_worker_restart(
    store, settings, clock, redis_client,
):
    settings = settings.model_copy(update={"card_canary_on_startup": True})
    recorder = metrics.RedisMetrics(redis_client, settings.redis_prefix)
    probes = iter([True, False, False, False])
    controller = health.CardHealth(
        store, settings, device_id=0, clock=clock, metrics=recorder,
        canary=lambda **kwargs: next(probes),
        recovery_step=lambda step, **kwargs: "cleanup" if step == 1 else "reset",
    )
    assert controller.prepare_claim() is True
    assert b'card_health_state{card="0"} 0\n' in recorder.render_openmetrics()
    controller.mark_suspicious()
    assert controller.prepare_claim() is False
    assert b'card_health_state{card="0"} 1\n' in recorder.render_openmetrics()

    restarted = health.CardHealth(
        store, settings, device_id=0, clock=clock, metrics=recorder,
        canary=lambda **kwargs: True,
    )
    assert restarted.prepare_claim() is True
    assert b'card_health_state{card="0"} 0\n' in recorder.render_openmetrics()


def test_drained_card_respects_probe_and_recovery_intervals_even_at_clock_zero(
    store, settings, clock,
):
    clock.now_ms = 0
    probes = []
    steps = []
    controller = health.CardHealth(
        store, settings.model_copy(update={"card_canary_on_startup": True}),
        device_id=0, clock=clock,
        canary=lambda **kwargs: probes.append(clock()) or False,
        recovery_step=lambda step, **kwargs: steps.append(step) or "cleanup",
    )
    assert controller.prepare_claim() is False
    assert steps == [1, 2]
    assert len(probes) == 3

    controller.mark_suspicious()
    clock.advance_ms(59_999)
    assert controller.prepare_claim() is False
    assert len(probes) == 3
    clock.advance_ms(1)
    assert controller.prepare_claim() is False
    assert len(probes) == 4
    assert steps == [1, 2]  # A probe is due, but disruptive recovery is not.

    clock.advance_ms(240_000)
    assert controller.prepare_claim() is False
    assert steps == [1, 2, 1, 2]
    assert len(probes) == 7


def test_probe_and_recovery_exceptions_drain_without_exposing_error_details(
    store, settings, clock, caplog,
):
    def broken_probe(**kwargs):
        raise ImportError("private /path/to/device stack")

    def broken_recovery(step, **kwargs):
        raise RuntimeError("private reset command")

    controller = health.CardHealth(
        store, settings.model_copy(update={"card_canary_on_startup": True}),
        device_id=0, clock=clock, canary=broken_probe, recovery_step=broken_recovery,
    )
    assert controller.prepare_claim() is False
    assert store.is_card_unhealthy(0)
    assert "ImportError" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "private" not in caplog.text


def test_metrics_outage_does_not_prevent_recovery(store, settings, clock):
    def unavailable(*args, **kwargs):
        raise RuntimeError("metrics unavailable")

    probes = iter([False, True])
    controller = health.CardHealth(
        store, settings.model_copy(update={"card_canary_on_startup": True}),
        device_id=0, clock=clock, canary=lambda **kwargs: next(probes),
        recovery_step=lambda step, **kwargs: "cleanup",
        metrics=SimpleNamespace(increment=unavailable, set_gauge=unavailable),
    )
    assert controller.prepare_claim() is False
    assert not store.is_card_unhealthy(0)
    assert controller.prepare_claim() is True


@pytest.mark.parametrize("first_probe", [True, False])
def test_failed_state_publication_never_grants_claim_eligibility(
    store, settings, clock, redis_server, first_probe,
):
    probes = iter([first_probe, False, False, False])
    controller = health.CardHealth(
        store, settings.model_copy(update={"card_canary_on_startup": True}),
        device_id=0, clock=clock, canary=lambda **kwargs: next(probes),
        recovery_step=lambda step, **kwargs: "cleanup",
    )
    redis_server.connected = False
    with pytest.raises(redis.ConnectionError):
        controller.prepare_claim()
    redis_server.connected = True
    clock.advance_ms(60_000)
    assert controller.prepare_claim() is False
    assert store.is_card_unhealthy(0)
