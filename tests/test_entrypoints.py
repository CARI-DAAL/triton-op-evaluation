import importlib
import os


def test_build_worker_sets_visible_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_EVAL_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TRITON_EVAL_ENGINE_DIR", str(tmp_path))
    run_worker = importlib.import_module("run_worker")
    w = run_worker.build_worker(card=3)
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert w.device_id == 3
    assert set(w.registry) == {"verify", "profile", "evaluation"}
    assert "benchmark" not in w.registry


def test_run_api_defaults_to_loopback(monkeypatch):
    run_api = importlib.import_module("run_api")
    seen = {}
    monkeypatch.delenv("TRITON_EVAL_API_HOST", raising=False)
    monkeypatch.setattr(run_api.uvicorn, "run", lambda app, **kwargs: seen.update(kwargs))

    run_api.main()

    assert seen["host"] == "127.0.0.1"
