"""Exercise candidate execution through the real invocation boundary."""

import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import pytest

from triton_evaluation.evaluators.base import run_script
from triton_evaluation.evaluators.sandbox import EngineSandbox


def test_runtime_manifest_cannot_overlay_attempt_state(tmp_path):
    work = tmp_path / "attempt"
    runtime = tmp_path / "runtime"
    work.mkdir()
    runtime.mkdir()
    with pytest.raises(ValueError, match="parent traversal"):
        EngineSandbox(readonly_mounts=[(runtime, work / "nested" / "..")])
    for destination in (work, work.parent, work / "nested"):
        sandbox = EngineSandbox(readonly_mounts=[(runtime, destination)])
        with pytest.raises(ValueError, match="overlay attempt state"):
            sandbox.wrap([sys.executable, "-c", "pass"], work, None)


def test_configured_boundary_never_falls_back_to_host(tmp_path, monkeypatch):
    """Missing sandbox configuration must fail before the candidate executes."""
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_SHA256", "0" * 64)
    marker = tmp_path / "executed"
    try:
        run_script(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"],
            tmp_path,
            None,
        )
    except OSError:
        pass
    assert not marker.exists()


@pytest.mark.parametrize("expected_digest", [None, "0" * 64])
def test_unbound_or_changed_manifest_never_executes(tmp_path, monkeypatch, expected_digest):
    work = tmp_path / "attempt"
    work.mkdir()
    config = tmp_path / "sandbox.json"
    config.write_text(json.dumps({"readonly_mounts": [["/usr", "/usr"]]}))
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_CONFIG", str(config))
    if expected_digest is None:
        monkeypatch.delenv("TRITON_EVAL_ENGINE_SANDBOX_SHA256", raising=False)
    else:
        monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_SHA256", expected_digest)
    marker = work / "executed"
    with pytest.raises(ValueError, match="sandbox manifest"):
        run_script(["/usr/bin/touch", str(marker)], work, None)
    assert not marker.exists()


def test_candidate_sees_only_own_files_and_runtime(tmp_path, monkeypatch):
    import shutil
    import subprocess

    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is required for the real boundary test")
    probe = subprocess.run(
        ["bwrap", "--unshare-all", "--ro-bind", "/", "/", "--", "true"],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("this host does not allow sandbox namespace creation")
    work = tmp_path / "attempt"
    work.mkdir()
    secret = tmp_path / "private-host-canary"
    secret.write_text("canary")
    config = tmp_path / "sandbox.json"
    roots = ["/usr", sys.base_prefix, sys.prefix]
    executable = Path(sys.executable)
    # uv's venv Python may pass through a version alias before its real prefix.
    mounts = [[p, p] for p in dict.fromkeys(roots)]
    if executable.is_symlink() and executable.readlink().is_absolute():
        mounts.append([sys.base_prefix, str(executable.readlink().parent.parent)])
    config.write_text(json.dumps({"readonly_mounts": mounts, "environment": {}}))
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_CONFIG", str(config))
    monkeypatch.setenv(
        "TRITON_EVAL_ENGINE_SANDBOX_SHA256", hashlib.sha256(config.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("HQ_SENTINEL_SECRET", "must-not-cross")
    code = (
        "import os,json; from pathlib import Path; "
        "Path('own').write_text('ok'); "
        f"print(json.dumps({{'host_visible':Path({str(secret)!r}).exists(), "
        "'secret_present':'HQ_SENTINEL_SECRET' in os.environ, "
        "'own_write':Path('own').read_text()=='ok', 'pid':os.getpid()}))"
    )
    result = run_script([sys.executable, "-c", code], work, None, budget_seconds=10)
    assert result.exit_code == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["host_visible"] is False
    assert actual["secret_present"] is False
    assert actual["own_write"] is True
    assert actual["pid"] <= 3
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        network_code = (
            "import socket; s=socket.socket(); s.settimeout(0.2); "
            f"print(s.connect_ex(('127.0.0.1', {port})) != 0)"
        )
        isolated = run_script([sys.executable, "-c", network_code], work, None)
        assert isolated.exit_code == 0, isolated.stderr
        assert isolated.stdout.strip() == "True"
    # A new session must not let candidate descendants outlive the invocation.
    child = "import time; from pathlib import Path; time.sleep(0.7); Path('leaked').touch()"
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}], start_new_session=True); "
        "time.sleep(30)"
    )
    ended = run_script([sys.executable, "-c", parent], work, None, budget_seconds=0.2)
    assert ended.outcome == "budget"
    time.sleep(1)
    assert not (work / "leaked").exists()
