"""Admission pre-check: syntax/import classification, cache integration."""
from unittest import mock

from triton_evaluation import precheck


def test_syntax_error_classified(tmp_path):
    module = tmp_path / "bad_mod.py"
    module.write_text("def broken(:\n    return 1\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] == "syntax"


def test_import_error_classified(tmp_path):
    module = tmp_path / "bad_mod.py"
    module.write_text("import definitely_not_a_real_pkg_xyz\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] == "import"


def test_good_module_passes(tmp_path):
    module = tmp_path / "good_mod.py"
    module.write_text("class ModelNew:\n    def __init__(self, *a, **k):\n        pass\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] is None


def test_torch_import_is_inconclusive_on_torchless_host(tmp_path, store):
    # This host has no torch: a candidate that imports it must classify as
    # inconclusive (fail-open), never as compile_error.
    from triton_evaluation.config import Settings

    module = tmp_path / "uses_torch.py"
    module.write_text("import torch\n")
    verdict = precheck._classify_module(str(module))
    assert verdict.get("inconclusive") is True
    assert verdict["stage"] is None
    settings = Settings(
        engine_dir=str(tmp_path), compile_precheck_enabled=True,
        verdict_cache_enabled=True,
    )
    assert precheck.run_precheck(store, settings, "import torch\n",
                                 "triton_ascend_impl") is None


def test_run_precheck_caches_pass_verdict(store, settings, redis_client):
    settings = settings.model_copy(update={
        "verdict_cache_enabled": True, "compile_precheck_enabled": True,
    })
    good_source = "class ModelNew:\n    def __init__(self, *a, **k):\n        pass\n"
    digest = precheck._digest_for(good_source, "triton_ascend_impl", settings.cann_fingerprint)
    assert precheck.run_precheck(store, settings, good_source, "triton_ascend_impl") is None
    # second call must be a cache hit without a subprocess
    with mock.patch.object(precheck.subprocess, "run") as run_mock:
        verdict = precheck.run_precheck(store, settings, good_source, "triton_ascend_impl")
    assert verdict is None  # pass verdict cached; not compile_error
    run_mock.assert_not_called()
    from triton_evaluation import verdict_cache

    assert verdict_cache.get(store, digest) is not None


def test_run_precheck_negative_verdict_shortcircuits(store, settings):
    settings = settings.model_copy(update={
        "verdict_cache_enabled": True, "compile_precheck_enabled": True,
    })
    bad_source = "def broken(:\n    return 1\n"
    assert (
        precheck.run_precheck(store, settings, bad_source, "triton_ascend_impl")
        == "compile_error"
    )
    # cached: repeat hits cache, still compile_error, no subprocess
    with mock.patch.object(precheck.subprocess, "run") as run_mock:
        assert (
            precheck.run_precheck(store, settings, bad_source, "triton_ascend_impl")
            == "compile_error"
        )
    run_mock.assert_not_called()
