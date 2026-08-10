import os

from triton_evaluation.config import Settings


def test_defaults():
    s = Settings()
    assert s.queue_key == "eval:queue"
    assert s.task_key_prefix == "eval:task:"
    assert s.session_key_prefix == "eval:session:"
    assert s.api_poll_interval_seconds == 0.25
    assert s.task_ttl_seconds == 24 * 3600
    # engine is vendored inside the package; default points at the bundled dir
    assert s.engine_dir.endswith(os.path.join("triton_evaluation", "engine"))


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://x:1/2")
    monkeypatch.setenv("TRITON_EVAL_ENGINE_DIR", "/opt/engine")
    s = Settings.from_env()
    assert s.redis_url == "redis://x:1/2"
    assert s.engine_dir == "/opt/engine"
