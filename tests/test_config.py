import os
from triton_evaluation.config import Settings


def test_defaults():
    s = Settings()
    assert s.queue_key == "eval:queue"
    assert s.task_key_prefix == "eval:task:"
    assert s.task_ttl_seconds == 24 * 3600


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://x:1/2")
    monkeypatch.setenv("TRITON_EVAL_SKILL_SCRIPTS_DIR", "/skills/scripts")
    s = Settings.from_env()
    assert s.redis_url == "redis://x:1/2"
    assert s.skill_scripts_dir == "/skills/scripts"
