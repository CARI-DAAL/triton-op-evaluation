"""Host result collection cannot enlarge the candidate's filesystem boundary."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from triton_evaluation.evaluators.base import read_json, run_script


@pytest.mark.parametrize("entry_kind", ["symbolic", "hard"])
def test_restaging_replaces_entries_without_mutating_previous_targets(tmp_path, entry_kind):
    from conftest import make_request

    from triton_evaluation.evaluators.base import materialize, safe_module_stem

    work = tmp_path / "attempt"
    work.mkdir()
    previous = tmp_path / "previous.py"
    previous.write_text("previous contents\n")
    request = make_request()
    destination = work / f"{safe_module_stem(request)}_torch.py"
    if entry_kind == "symbolic":
        destination.symlink_to(previous)
    else:
        os.link(previous, destination)
    materialize(request, work)
    assert previous.read_text() == "previous contents\n"
    assert destination.read_text() == request.reference_code
    assert not destination.is_symlink()


def test_collects_regular_json_owned_by_the_attempt(tmp_path):
    work = tmp_path / "attempt"
    (work / "nested").mkdir(parents=True)
    result = work / "nested" / "result.json"
    result.write_text('{"passed_cases":1}')
    assert read_json(result, root=work) == {"passed_cases": 1}


@pytest.mark.parametrize("link_kind", ["leaf", "parent", "root", "outside", "traversal"])
def test_collection_refuses_any_link_or_path_escape(tmp_path, link_kind):
    work = tmp_path / "attempt"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "result.json"
    target.write_text('{"host_marker":"unreachable"}')
    root = work
    if link_kind == "leaf":
        result = work / "result.json"
        result.symlink_to(target)
    elif link_kind == "parent":
        (work / "nested").symlink_to(outside, target_is_directory=True)
        result = work / "nested/result.json"
    elif link_kind == "root":
        root = tmp_path / "alias"
        root.symlink_to(outside, target_is_directory=True)
        result = root / "result.json"
    elif link_kind == "outside":
        result = target
    else:
        result = work / "../outside/result.json"
    assert read_json(result, root=root) is None


@pytest.mark.parametrize("kind", ["fifo", "directory", "missing", "invalid-json"])
def test_nonregular_or_unusable_output_is_not_a_result(tmp_path, kind):
    result = tmp_path / "result.json"
    if kind == "fifo":
        os.mkfifo(result)
    elif kind == "directory":
        result.mkdir()
    elif kind == "invalid-json":
        result.write_text("{")
    assert read_json(result, root=tmp_path) is None


def test_parent_replacement_race_cannot_redirect_an_open_directory(tmp_path, monkeypatch):
    work = tmp_path / "attempt"
    (work / "nested").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "nested/result.json").write_text('{"source":"attempt"}')
    (outside / "result.json").write_text('{"source":"outside"}')
    original_open = os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        if path == "result.json" and kwargs.get("dir_fd") is not None:
            raced = True
            (work / "nested").rename(work / "original")
            (work / "nested").symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    assert read_json(work / "nested/result.json", root=work) == {"source": "attempt"}
    assert raced


def test_actual_sandbox_cannot_redirect_host_result_collection(tmp_path, monkeypatch):
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is required")
    probe = subprocess.run(
        ["bwrap", "--unshare-all", "--ro-bind", "/", "/", "--", "true"],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("host does not allow namespaces")
    work = tmp_path / "attempt"
    work.mkdir()
    host = tmp_path / "private-host.json"
    host.write_text('{"host_marker":"unreachable"}')
    roots = ["/usr", sys.base_prefix, sys.prefix]
    mounts = [[p, p] for p in dict.fromkeys(roots)]
    executable = Path(sys.executable)
    if executable.is_symlink() and executable.readlink().is_absolute():
        mounts.append([sys.base_prefix, str(executable.readlink().parent.parent)])
    config = tmp_path / "sandbox.json"
    config.write_text(json.dumps({"readonly_mounts": mounts, "environment": {}}))
    monkeypatch.setenv("TRITON_EVAL_ENGINE_SANDBOX_CONFIG", str(config))
    monkeypatch.setenv(
        "TRITON_EVAL_ENGINE_SANDBOX_SHA256", hashlib.sha256(config.read_bytes()).hexdigest()
    )
    code = (
        "from pathlib import Path; import json; "
        f"host=Path({str(host)!r}); "
        "Path('verify_result.json').symlink_to(host); "
        "print(json.dumps({'host_visible':host.exists()}))"
    )
    execution = run_script([sys.executable, "-c", code], work, None, budget_seconds=10)
    assert execution.exit_code == 0, execution.stderr
    assert json.loads(execution.stdout)["host_visible"] is False
    assert stat.S_ISLNK((work / "verify_result.json").lstat().st_mode)
    assert read_json(work / "verify_result.json", root=work) is None
