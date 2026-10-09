"""Both bwrap environment interfaces preserve the explicit child allowlist."""

import json
import subprocess
import sys

import pytest

from triton_evaluation.runtime import sandbox


@pytest.fixture
def fake_bwrap(tmp_path):
    def make(modern):
        path = tmp_path / 'bwrap'
        path.write_text(
            f'#!{sys.executable}\n'
            'import os, sys\n'
            f'modern = {modern!r}\n'
            'if sys.argv[1:] == ["--help"]:\n'
            '    print("--setenv --unsetenv" + (" --clearenv" if modern else ""))\n'
            '    raise SystemExit(0)\n'
            # Python's C-locale startup adds this; real bwrap is a C binary.
            'os.environ.pop("LC_CTYPE", None)\n'
            'args = sys.argv[1:]\n'
            'while args:\n'
            '    arg = args.pop(0)\n'
            '    if arg == "--":\n'
            '        os.execvpe(args[0], args, os.environ)\n'
            '    if arg == "--clearenv":\n'
            '        if not modern: raise SystemExit("unsupported --clearenv")\n'
            '        os.environ.clear()\n'
            '    elif arg == "--unsetenv":\n'
            '        os.environ.pop(args.pop(0), None)\n'
            '    elif arg == "--setenv":\n'
            '        key, value = args[:2]\n'
            '        del args[:2]\n'
            '        os.environ[key] = value\n'
        )
        path.chmod(0o755)
        return path
    return make


@pytest.mark.parametrize('modern', [False, True])
def test_exact_environment_with_both_interfaces(tmp_path, monkeypatch, fake_bwrap, modern):
    monkeypatch.setenv('HQ_SENTINEL_SECRET', 'credential-canary')
    monkeypatch.setenv('TRITON_EVAL_API_PORT', '12345')
    model = sandbox.EngineSandbox(
        executable=fake_bwrap(modern), readonly_mounts=[], environment={'LANG': 'C.UTF-8'},
    )
    wrapped, outer = model.wrap(
        [sys.executable, '-c', 'import json,os; print(json.dumps(dict(os.environ)))'],
        tmp_path, None,
    )
    assert outer == {'PATH': '/usr/bin:/bin'}
    reset = '--clearenv' if modern else '--unsetenv'
    assert reset in wrapped
    assert ('--clearenv' in wrapped) == modern
    assert wrapped.index(reset) < wrapped.index('--setenv')
    if not modern:
        assert wrapped[wrapped.index(reset) + 1] == 'PATH'
    expected = {
        wrapped[i + 1]: wrapped[i + 2]
        for i, arg in enumerate(wrapped) if arg == '--setenv'
    }
    result = subprocess.run(wrapped, env=outer, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected
    assert 'HQ_SENTINEL_SECRET' not in expected
    assert 'credential-canary' not in result.stdout
    assert 'TRITON_EVAL_API_PORT' not in expected


def test_probe_cache_tracks_executable_replacement(tmp_path, fake_bwrap, monkeypatch):
    original_run = subprocess.run
    calls = []
    def count(*args, **kwargs):
        calls.append((args, kwargs))
        return original_run(*args, **kwargs)
    monkeypatch.setattr(sandbox.subprocess, 'run', count)
    model = sandbox.EngineSandbox(executable=fake_bwrap(False), readonly_mounts=[])
    first, _ = model.wrap(['true'], tmp_path, None)
    model.wrap(['true'], tmp_path, None)
    assert len(calls) == 1
    fake_bwrap(True)
    second, _ = model.wrap(['true'], tmp_path, None)
    assert len(calls) == 2
    assert '--clearenv' not in first
    assert '--clearenv' in second
    assert calls[0][1]['timeout'] == 5
    assert calls[0][1]['env'] == {'PATH': '/usr/bin:/bin'}


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'unsupported'])
def test_probe_failure_is_closed(tmp_path, fake_bwrap, monkeypatch, failure):
    path = fake_bwrap(False)
    def fail(*args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(str(path), 5)
        if failure == 'exit':
            raise subprocess.CalledProcessError(1, str(path))
        return subprocess.CompletedProcess(args[0], 0, stdout='--setenv', stderr='')
    monkeypatch.setattr(sandbox.subprocess, 'run', fail)
    marker = tmp_path / 'candidate-ran'
    model = sandbox.EngineSandbox(executable=path, readonly_mounts=[])
    with pytest.raises(RuntimeError, match='environment-reset'):
        model.wrap(['/usr/bin/touch', str(marker)], tmp_path, None)
    assert not marker.exists()
