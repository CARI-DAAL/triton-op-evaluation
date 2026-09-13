"""Admission pre-check behavior through the same interface used by admission."""
import importlib.util
import subprocess
from unittest import mock

import pytest
import redis

from triton_evaluation import precheck, verdict_cache


@pytest.fixture
def precheck_settings(settings):
    return settings.model_copy(update={
        "verdict_cache_enabled": True,
        "compile_precheck_enabled": True,
        "cann_fingerprint": "cann-test",
    })


@pytest.mark.parametrize("source", [
    "def broken(:\n    return 1\n",
    "import definitely_not_a_real_pkg_xyz\n",
    "from math import definitely_not_a_real_symbol_xyz\n",
])
def test_definitive_failure_is_cached_without_repeating_subprocess(
    store, precheck_settings, source,
):
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "compile_error"
    with mock.patch.object(precheck.subprocess, "run") as run:
        assert precheck.run_precheck(store, precheck_settings, source, "impl") == "compile_error"
    run.assert_not_called()


def test_cached_pass_only_skips_precheck_and_allows_evaluation(store, precheck_settings):
    source = "class ModelNew:\n    pass\n"
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) == "precheck_passed"
    with mock.patch.object(precheck.subprocess, "run") as run:
        assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"
    run.assert_not_called()


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None,
                    reason="torch-equipped host: import torch succeeds")
def test_torch_import_is_inconclusive_on_torchless_host(store, precheck_settings):
    source = "import torch\n"
    assert precheck.run_precheck(store, precheck_settings, source, "impl") is None
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


@pytest.mark.parametrize("source", [
    "raise OSError('temporary filesystem failure')\n",
    "raise RuntimeError('temporary runtime failure')\n",
    "raise TimeoutError('temporary dependency timeout')\n",
    "raise ImportError('native runtime library unavailable')\n",
])
def test_runtime_import_failure_is_inconclusive_and_not_cached(
    store, precheck_settings, source,
):
    with mock.patch.object(precheck.subprocess, "run", wraps=subprocess.run) as run:
        for _ in range(2):
            assert precheck.run_precheck(store, precheck_settings, source, "impl") is None
    assert run.call_count == 2
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


def test_missing_cann_identity_still_runs_precheck_without_reuse(store, precheck_settings):
    settings = precheck_settings.model_copy(update={"cann_fingerprint": ""})
    with mock.patch.object(precheck.subprocess, "run", wraps=subprocess.run) as run:
        for _ in range(2):
            assert precheck.run_precheck(
                store, settings, "class ModelNew: pass", "impl"
            ) == "precheck_passed"
    assert run.call_count == 2


def test_concurrent_failure_during_successful_precheck_remains_authoritative(
    store, precheck_settings,
):
    source = "class ModelNew: pass"
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)

    def concurrent_compile_failure(*args, **kwargs):
        verdict_cache.put(store, digest, "compile_error", 60)
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout='VERDICT {"stage": null}\n',
        )

    with mock.patch.object(precheck.subprocess, "run", side_effect=concurrent_compile_failure):
        assert precheck.run_precheck(store, precheck_settings, source, "impl") == "compile_error"
    assert verdict_cache.get(store, digest) == "compile_error"


@pytest.mark.parametrize("payload", [
    "not-json", "[]", "null", "1", "{}",
    '{"stage": "unknown"}', '{"stage": null, "inconclusive": true}',
])
def test_invalid_or_inconclusive_output_is_not_cached(store, precheck_settings, payload):
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"VERDICT {payload}\n")
    with mock.patch.object(precheck.subprocess, "run", return_value=result):
        assert precheck.run_precheck(store, precheck_settings, "code", "impl") is None
    digest = verdict_cache.digest_for("code", "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


def test_candidate_stdout_does_not_replace_runner_verdict(store, precheck_settings):
    source = "print('VERDICT {\"stage\": \"syntax\"}')\nclass ModelNew: pass\n"
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("runner", 1)])
def test_subprocess_failure_is_advisory(store, precheck_settings, error):
    with mock.patch.object(precheck.subprocess, "run", side_effect=error):
        assert precheck.run_precheck(store, precheck_settings, "code", "impl") is None
    digest = verdict_cache.digest_for("code", "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


def test_temporary_file_failure_is_advisory(store, precheck_settings):
    with mock.patch.object(precheck.Path, "write_text", side_effect=OSError("disk full")):
        assert precheck.run_precheck(store, precheck_settings, "code", "impl") is None


@pytest.mark.parametrize("source, expected", [
    ("class ModelNew: pass", "precheck_passed"),
    ("def broken(: pass", "compile_error"),
])
def test_redis_failure_preserves_direct_precheck_verdict(
    store, precheck_settings, source, expected,
):
    with mock.patch.object(store._r, "hgetall", side_effect=redis.ConnectionError("down")):
        with mock.patch.object(store._r, "pipeline", side_effect=redis.ConnectionError("down")):
            assert precheck.run_precheck(store, precheck_settings, source, "impl") == expected
