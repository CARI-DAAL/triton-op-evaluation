"""Attempt precheck behavior through its public execution interface."""
import importlib.util
from unittest import mock

import pytest
import redis

from triton_evaluation import precheck, verdict_cache
from triton_evaluation.evaluators.base import ScriptOutcome


@pytest.fixture
def precheck_settings(settings):
    return settings.model_copy(update={
        "verdict_cache_enabled": True,
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
    with mock.patch.object(precheck, "run_script") as run:
        assert precheck.run_precheck(store, precheck_settings, source, "impl") == "compile_error"
    run.assert_not_called()


def test_cached_pass_only_skips_precheck_and_allows_evaluation(store, precheck_settings):
    source = "class ModelNew:\n    pass\n"
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) == "precheck_passed"
    with mock.patch.object(precheck, "run_script") as run:
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
    with mock.patch.object(precheck, "run_script", wraps=precheck.run_script) as run:
        for _ in range(2):
            assert precheck.run_precheck(store, precheck_settings, source, "impl") is None
    assert run.call_count == 2
    digest = verdict_cache.digest_for(source, "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


def test_missing_cann_identity_still_runs_precheck_without_reuse(store, precheck_settings):
    settings = precheck_settings.model_copy(update={"cann_fingerprint": ""})
    with mock.patch.object(precheck, "run_script", wraps=precheck.run_script) as run:
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
        return ScriptOutcome(0, 'VERDICT {"stage": null}\n', "", False)

    with mock.patch.object(precheck, "run_script", side_effect=concurrent_compile_failure):
        assert precheck.run_precheck(store, precheck_settings, source, "impl") == "compile_error"
    assert verdict_cache.get(store, digest) == "compile_error"


@pytest.mark.parametrize("payload", [
    "not-json", "[]", "null", "1", "{}",
    '{"stage": "unknown"}', '{"stage": null, "inconclusive": true}',
])
def test_invalid_or_inconclusive_output_is_not_cached(store, precheck_settings, payload):
    result = ScriptOutcome(0, f"VERDICT {payload}\n", "", False)
    with mock.patch.object(precheck, "run_script", return_value=result):
        assert precheck.run_precheck(store, precheck_settings, "code", "impl") is None
    digest = verdict_cache.digest_for("code", "impl", precheck_settings.cann_fingerprint)
    assert verdict_cache.get(store, digest) is None


def test_candidate_stdout_does_not_replace_runner_verdict(store, precheck_settings):
    source = "print('VERDICT {\"stage\": \"syntax\"}')\nclass ModelNew: pass\n"
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"


@pytest.mark.parametrize("error", [OSError("unavailable")])
def test_subprocess_failure_is_advisory(store, precheck_settings, error):
    with mock.patch.object(precheck, "run_script", side_effect=error):
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


def test_precheck_never_calls_candidate_input_factories(store, precheck_settings, tmp_path):
    marker = tmp_path / "candidate-inputs-ran"
    source = (
        f"def get_input_groups():\n    open({str(marker)!r}, 'w').write('called')\n"
        "    return [[]]\n"
    )
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"
    assert not marker.exists()


def test_precheck_lost_lease_kills_candidate_process_group(store, precheck_settings, tmp_path):
    import time

    from triton_evaluation.evaluators.base import LeaseLostError

    child_marker = tmp_path / "child-survived"
    source = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(0.5); "
        f"open({str(child_marker)!r}, 'w').write('survived')\"])\n"
        "time.sleep(20)\n"
    )
    start = time.monotonic()
    with pytest.raises(LeaseLostError):
        precheck.run_precheck(
            store, precheck_settings, source, "impl",
            lease_valid=lambda: time.monotonic() - start < 0.15,
        )
    assert time.monotonic() - start < 2
    time.sleep(0.6)
    assert not child_marker.exists()


def test_expired_resource_never_executes_candidate(store, precheck_settings, tmp_path):
    from datetime import datetime, timedelta, timezone

    marker = tmp_path / "candidate-ran"
    source = f"open({str(marker)!r}, 'w').write('called')\n"
    assert precheck.run_precheck(
        store, precheck_settings, source, "impl",
        deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    ) is None
    assert not marker.exists()


def test_successful_import_does_not_leave_candidate_children_running(
    store, precheck_settings, tmp_path,
):
    import time

    child_marker = tmp_path / "child-survived"
    source = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(0.4); "
        f"open({str(child_marker)!r}, 'w').write('survived')\"])\n"
    )
    assert precheck.run_precheck(store, precheck_settings, source, "impl") == "precheck_passed"
    time.sleep(0.5)
    assert not child_marker.exists()


def _parse_validator_output(stdout):
    """--json 输出是缩进多行 JSON；从首个 { 起整体解析。"""
    import json as _j
    text = stdout[stdout.find("{"):]
    try:
        return _j.loads(text)
    except ValueError:
        return None


def test_bare_builtin_min_max_are_not_tensor_methods(tmp_path):
    """裸 min()/max() 是 Python 内建（标准 tiling 写法），不是 tensor 方法。

    沙箱版校验器有 qual-is-None 守卫并明确注释了原因；vendoring 时丢失，
    导致 conv 任务里 ``grid = min(...)`` 之类写法被误判 Type-3 decoy。
    属性形式 ``x.min()`` 仍然必须拒绝。
    """
    import subprocess, sys, json as _json
    from pathlib import Path as _P
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def _k(o, n, B: tl.constexpr):\n"
        "    p = tl.program_id(0)\n"
        "    off = p * B + tl.arange(0, B)\n"
        "    tl.store(o + off, tl.zeros((B,), dtype=tl.float32), mask=off < n)\n"
        "\n"
        "class ModelNew(nn.Module):\n"
        "    def forward(self, a, b):\n"
        "        out = torch.empty_like(a)\n"
        "        grid = min(128, triton.cdiv(a.numel(), 1024))\n"
        "        blocks = max(1, grid // 2)\n"
        "        _k[(blocks,)](out, a.numel(), B=1024)\n"
        "        return out\n",
        encoding="utf-8",
    )
    engine = _P(__file__).resolve().parents[1] / "triton_evaluation" / "engine"
    result = subprocess.run(
        [sys.executable, str(engine / "validate_triton_impl.py"), str(kernel), "--json"],
        capture_output=True, text=True, timeout=120,
    )
    payload = _parse_validator_output(result.stdout)
    assert payload is not None, result.stdout[-300:]
    assert payload["valid"] is True, payload.get("suggestion")

    # 属性调用 x.min() 仍必须拒绝
    bad = tmp_path / "bad.py"
    bad.write_text(
        kernel.read_text(encoding="utf-8").replace(
            "blocks = max(1, grid // 2)", "blocks = a.min()"
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(engine / "validate_triton_impl.py"), str(bad), "--json"],
        capture_output=True, text=True, timeout=120,
    )
    payload = _parse_validator_output(result.stdout)
    assert payload is not None
    assert payload["valid"] is False
    assert payload["regression_type"] == 3
