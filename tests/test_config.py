import os

from triton_evaluation.config import Settings


def test_defaults():
    s = Settings()
    assert s.redis_prefix == "triton-eval:v2.1"
    assert s.api_poll_interval_seconds == 0.25
    # engine is vendored inside the package; default points at the bundled dir
    assert s.engine_dir.endswith(os.path.join("triton_evaluation", "engine"))
    assert s.capability_secret_path is None


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://x:1/2")
    monkeypatch.setenv("TRITON_EVAL_ENGINE_DIR", "/opt/engine")
    s = Settings.from_env()
    assert s.redis_url == "redis://x:1/2"
    assert s.engine_dir == "/opt/engine"


def test_rollout_quota_defaults():
    s = Settings()
    assert s.rollout_rate_limit_per_minute == 600
    assert s.rollout_max_inflight == 256
    assert s.rollout_caller_ids == []


def test_rollout_settings_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_ROLLOUT_CALLER_IDS", '["rl-1", "rl-2"]')
    monkeypatch.setenv("TRITON_EVAL_ROLLOUT_MAX_INFLIGHT", "512")
    s = Settings.from_env()
    assert s.rollout_caller_ids == ["rl-1", "rl-2"]
    assert s.rollout_max_inflight == 512


def test_plan_c_health_defaults():
    s = Settings()
    assert s.poison_death_threshold == 2
    assert s.card_affinity_avoidance is True


def test_plan_c_health_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_POISON_DEATH_THRESHOLD", "3")
    monkeypatch.setenv("TRITON_EVAL_CARD_AFFINITY_AVOIDANCE", "false")
    s = Settings.from_env()
    assert s.poison_death_threshold == 3
    assert s.card_affinity_avoidance is False


def test_command_environment_uses_settings_and_preserves_toolchain(monkeypatch):
    from triton_evaluation.config import command_environment

    settings = Settings(
        engine_dir="/validated/engine", reference_device_policy="npu-strict",
        reference_diagnostics_dir=None, engine_sandbox_config=None,
        engine_sandbox_sha256=None,
    )
    inherited = {
        "PATH": "/toolchain/bin", "LD_LIBRARY_PATH": "/toolchain/lib",
        "TRITON_EVAL_ENGINE_DIR": "/stale/engine",
        "TRITON_EVAL_REFERENCE_DEVICE_POLICY": "cpu-first",
        "TRITON_EVAL_REFERENCE_DIAGNOSTICS_DIR": "/stale/diagnostics",
        "TRITON_EVAL_ENGINE_SANDBOX_CONFIG": "/stale/sandbox",
        "TRITON_EVAL_ENGINE_SANDBOX_SHA256": "0" * 64,
    }
    before = inherited.copy()
    environment = command_environment(settings, inherited)
    assert inherited == before
    assert environment["PATH"] == "/toolchain/bin"
    assert environment["LD_LIBRARY_PATH"] == "/toolchain/lib"
    assert environment["TRITON_EVAL_ENGINE_DIR"] == "/validated/engine"
    assert environment["TRITON_EVAL_REFERENCE_DEVICE_POLICY"] == "npu-strict"
    assert "TRITON_EVAL_REFERENCE_DIAGNOSTICS_DIR" not in environment
    assert "TRITON_EVAL_ENGINE_SANDBOX_CONFIG" not in environment
    assert "TRITON_EVAL_ENGINE_SANDBOX_SHA256" not in environment


def test_sandbox_settings_are_loaded_and_validated(monkeypatch):
    import pytest
    from pydantic import ValidationError

    from triton_evaluation.config import command_environment

    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_CONFIG", "/runtime/sandbox.json")
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_SHA256", "a" * 64)
    settings = Settings.from_env()
    environment = command_environment(settings, {})
    assert environment["TRITON_EVAL_ENGINE_SANDBOX_CONFIG"] == "/runtime/sandbox.json"
    assert environment["TRITON_EVAL_ENGINE_SANDBOX_SHA256"] == "a" * 64
    monkeypatch.delenv("TRITON_EVAL_ENGINE_SANDBOX_SHA256")
    with pytest.raises(ValidationError, match="configured together"):
        Settings.from_env()
