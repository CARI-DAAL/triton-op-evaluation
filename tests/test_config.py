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
