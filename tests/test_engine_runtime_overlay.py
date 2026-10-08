"""Original anchor bytes plus a real CPU bwrap child; never NPU admission."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_profiler_export_replay import elf_binding  # noqa: F401
from test_profiler_runtime import deployed_runtime  # noqa: F401

from triton_evaluation.evaluators.base import run_script
from triton_evaluation.runtime import profiler_runtime, sandbox


@pytest.fixture
def anchored_overlay(tmp_path, monkeypatch, deployed_runtime):  # noqa: F811
    runtime_manifest, model = deployed_runtime
    mounts = [(path, path) for path in dict.fromkeys(("/usr", sys.base_prefix, sys.prefix))]
    executable = Path(sys.executable)
    if executable.is_symlink() and executable.readlink().is_absolute():
        mounts.append((sys.base_prefix, str(executable.readlink().parent.parent)))
    # Model the original manifest: no runtime pair and no new service source.
    original = {
        "readonly_mounts": mounts,
        "environment": {"PATH": "/usr/bin:/bin"},
    }
    base = tmp_path / "original-sandbox.json"
    base.write_text(json.dumps(original))
    original_bytes = base.read_bytes()
    base_sha = hashlib.sha256(original_bytes).hexdigest()
    monkeypatch.setenv(sandbox.CONFIG_ENV, str(base))
    monkeypatch.setenv(sandbox.DIGEST_ENV, base_sha)
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "npu-strict")
    overlay = tmp_path / "runtime-overlay.json"
    environment = {
        profiler_runtime.MANIFEST_ENV: str(runtime_manifest),
        profiler_runtime.DIGEST_ENV: profiler_runtime.sha(runtime_manifest),
    }
    overlay.write_text(sandbox.EngineRuntimeOverlay(environment=environment).model_dump_json())
    overlay.chmod(0o444)
    monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(overlay))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, profiler_runtime.sha(overlay))
    work = tmp_path / "attempt"
    work.mkdir()
    return base, original_bytes, overlay, work, model


def test_unchanged_anchor_uses_overlay_in_actual_clearenv_command(anchored_overlay):
    base, original, _overlay, work, model = anchored_overlay
    wrapped, _env = sandbox.isolate_command(
        [sys.executable, "-c", "pass"], work, None, dict(os.environ)
    )
    assert "--clearenv" in wrapped
    position = wrapped.index(sandbox.REFERENCE_POLICY_ENV)
    assert wrapped[position + 1] == "npu-strict"
    for key in profiler_runtime.RUNTIME_ENV_KEYS:
        assert key in wrapped
    assert str(model.alias_path.parent.parent) in wrapped
    assert str(Path(profiler_runtime.__file__).parents[1]) in wrapped
    assert base.read_bytes() == original
    boundary = json.loads(next(work.glob("engine-runtime-boundary-*.json")).read_bytes())
    assert boundary["base_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert boundary["state"] == "prepared" and boundary["executed"] is False
    assert boundary["command_sha256"] == hashlib.sha256(json.dumps(wrapped).encode()).hexdigest()


def test_parent_runtime_is_never_silently_dropped_without_overlay(anchored_overlay, monkeypatch):
    base, original, _overlay, work, _model = anchored_overlay
    for key in sandbox.OVERLAY_ENV_KEYS:
        monkeypatch.delenv(key)
    with pytest.raises(ValueError, match="overlay before --clearenv"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], work, None, dict(os.environ))
    assert base.read_bytes() == original


@pytest.mark.parametrize("bad", ["extra-env", "partial-pair", "caller-mount"])
def test_overlay_cannot_request_arbitrary_environment_or_mounts(bad):
    if bad == "extra-env":
        raw = {"environment": {"HQ_SENTINEL_SECRET": "forbidden"}}
    elif bad == "partial-pair":
        raw = {"environment": {profiler_runtime.MANIFEST_ENV: "/missing"}}
    else:
        raw = {
            "environment": {profiler_runtime.MANIFEST_ENV: "/missing"},
            "readonly_mounts": [["/", "/"]],
        }
    with pytest.raises(ValueError):
        sandbox.EngineRuntimeOverlay.model_validate(raw)


def test_changed_overlay_or_missing_pin_never_prepares_child(anchored_overlay, monkeypatch):
    _base, _original, overlay, work, _model = anchored_overlay
    monkeypatch.delenv(sandbox.OVERLAY_DIGEST_ENV)
    with pytest.raises(ValueError, match="together"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], work, None, dict(os.environ))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, "0" * 64)
    with pytest.raises(ValueError, match="SHA pin"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], work, None, dict(os.environ))
    assert overlay.exists() and not list(work.glob("engine-runtime-boundary-*.json"))


def test_old_engine_override_and_host_scripts_are_refused(anchored_overlay, monkeypatch, tmp_path):
    _base, _original, _overlay, work, _model = anchored_overlay
    monkeypatch.setenv("TRITON_EVAL_ENGINE_DIR", str(tmp_path / "old-live/engine"))
    with pytest.raises(ValueError, match="another service source"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], work, None, dict(os.environ))
    monkeypatch.delenv("TRITON_EVAL_ENGINE_DIR")
    host_script = tmp_path / "outside.py"
    host_script.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="another service source"):
        sandbox.isolate_command([sys.executable, str(host_script)], work, None, dict(os.environ))
    attempt_script = work / "probe.py"
    attempt_script.write_text("print('trusted prepared probe')")
    sandbox.isolate_command([sys.executable, str(attempt_script)], work, None, dict(os.environ))
    boundary = json.loads(next(work.glob("engine-runtime-boundary-*.json")).read_bytes())
    assert boundary["engine_script_sha256"] == profiler_runtime.sha(attempt_script)


@pytest.mark.parametrize("original_path", ["missing-msprof", "wrong-msprof"])
def test_real_bwrap_child_loads_pinned_alias_and_new_helper_without_host_canary(
    tmp_path, monkeypatch, request, original_path
):
    # The bwrap guard must run before anchored_overlay/deployed_runtime setup:
    # that fixture globally monkeypatches shutil.which (to pin the controlled
    # msprof), which would hide a missing real bwrap from the check below.
    if not shutil.which("bwrap"):
        pytest.skip("real bwrap binary is absent")
    bootstrap = [
        "bwrap",
        "--unshare-all",
        "--ro-bind",
        "/usr",
        "/usr",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for alias in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(alias)
        if path.is_symlink():
            bootstrap += ["--symlink", str(path.readlink()), alias]
    available = subprocess.run([*bootstrap, "--", "/usr/bin/true"], capture_output=True)
    if available.returncode and any(
        message in available.stderr
        for message in (
            b"Operation not permitted",
            b"No permissions",
            b"Creating new namespace failed",
        )
    ):
        pytest.skip("this host denies real namespace creation")
    assert available.returncode == 0, available.stderr.decode(errors="replace")
    base, original, _overlay, work, model = request.getfixturevalue("anchored_overlay")
    if original_path == "wrong-msprof":
        wrong_dir = tmp_path / "old-tool-bin"
        wrong_dir.mkdir()
        wrong = wrong_dir / "msprof"
        wrong.write_text("#!/bin/sh\nexit 79\n")
        wrong.chmod(0o755)
        raw = json.loads(original)
        raw["readonly_mounts"].append([str(wrong), str(wrong)])
        raw["environment"]["PATH"] = str(wrong_dir) + ":/usr/bin:/bin"
        base.write_text(json.dumps(raw))
        original = base.read_bytes()
        monkeypatch.setenv(sandbox.DIGEST_ENV, hashlib.sha256(original).hexdigest())
    canary = tmp_path / "host-canary"
    canary.write_text("must remain outside the boundary")
    monkeypatch.setenv("HQ_SENTINEL_SECRET", "must-not-cross")
    probe = work / "probe.py"
    probe.write_text(
        "import json,os,stat,subprocess,sys\nfrom pathlib import Path\n"
        "from triton_evaluation.runtime import profiler_runtime as r\n"
        "m=r.load_runtime()\n"
        "observed=r.describe_runtime()\n"
        "with r.profiler_export_runtime() as trace:\n"
        # CANN starts a new exporter process. ELF loaders do not re-read
        # LD_LIBRARY_PATH for in-process dlopen after Python starts.
        "    subprocess.run([sys.executable,'-c',"
        "'import ctypes; ctypes.CDLL('+repr(str(m.consumer_path))+')'],check=True)\n"
        "print(json.dumps({\n"
        f"'host_visible':Path({str(canary)!r}).exists(),\n"
        "'secret_present':'HQ_SENTINEL_SECRET' in os.environ,\n"
        "'alias_mode':stat.S_IMODE(m.alias_path.parent.stat().st_mode),\n"
        "'manifest_mode':stat.S_IMODE(Path(os.environ[r.MANIFEST_ENV]).parent.stat().st_mode),\n"
        "'policy':os.environ.get('TRITON_EVAL_REFERENCE_DEVICE_POLICY'),\n"
        "'resolved_msprof_path':observed['resolved_msprof_path'],\n"
        "'resolved_msprof_sha256':observed['resolved_msprof_sha256'],\n"
        "'helper_path':r.__file__,'trace':trace}))\n"
    )
    result = run_script([sys.executable, str(probe)], work, None, budget_seconds=10)
    assert result.exit_code == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["host_visible"] is actual["secret_present"] is False
    assert actual["alias_mode"] == actual["manifest_mode"] == 0o555
    assert actual["helper_path"] == profiler_runtime.__file__
    assert actual["policy"] == "npu-strict"
    assert actual["resolved_msprof_path"] == str(model.msprof_path)
    assert actual["resolved_msprof_sha256"] == model.msprof_sha256
    assert actual["trace"]["helper_path"] == profiler_runtime.__file__
    assert actual["trace"]["reference_device_policy"] == "npu-strict"
    assert actual["trace"]["helper_sha256"] == profiler_runtime.sha(Path(profiler_runtime.__file__))
    assert actual["trace"]["alias_sha256"] == model.library_sha256
    assert (
        actual["trace"]["callback_scope_completed"] is actual["trace"]["base_ld_restored"] is True
    )
    assert base.read_bytes() == original


def test_overlay_cannot_supply_or_change_existing_policy_anchor(anchored_overlay, monkeypatch):
    _base, original, overlay_path, work, _model = anchored_overlay
    overlay = sandbox.read_runtime_overlay(str(overlay_path), profiler_runtime.sha(overlay_path))
    with pytest.raises(ValueError, match="only profiler and reference-cache"):
        sandbox.EngineRuntimeOverlay(
            environment={**overlay.environment, sandbox.REFERENCE_POLICY_ENV: "cpu-first"}
        )
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "unknown-policy")
    with pytest.raises(ValueError, match="valid reference policy anchor"):
        sandbox.isolate_command([sys.executable, "-c", "pass"], work, None, dict(os.environ))
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "cpu-first")
    base = sandbox.EngineSandbox.model_validate_json(original)
    conflict = base.model_copy(
        update={"environment": {**base.environment, sandbox.REFERENCE_POLICY_ENV: "npu-strict"}}
    )
    with pytest.raises(ValueError, match="existing reference policy anchor"):
        sandbox.apply_runtime_overlay(conflict, overlay)


def test_overlay_cache_uses_two_artifact_namespaces_and_exact_state(tmp_path, anchored_overlay):
    import importlib.metadata

    from triton_evaluation.runtime.reference_cache import (
        BINDING_ENV,
        CACHE_ENV,
        DIGEST_ENV,
        STATE_ENV,
        CacheDeployment,
        CacheIndex,
        CacheMeasurement,
    )

    base, original, overlay_path, _work, profiler = anchored_overlay
    cache = tmp_path / "audited-cache"
    cache.mkdir()
    for name in ("cases", "blobs"):
        (cache / name).mkdir()
    state = tmp_path / "cache-invalidations"
    state.mkdir()
    marker = tmp_path / "runtime-version"
    marker.write_text("CPU-controlled closure")
    package = Path(profiler_runtime.__file__).parents[1]
    index = CacheIndex(
        deployment=CacheDeployment(
            generation="controlled-overlay-not-NPU",
            runtime_versions={"pytest": importlib.metadata.version("pytest")},
            runtime_files={str(marker): profiler_runtime.sha(marker)},
            algorithm_files={
                name: profiler_runtime.sha(package / "engine" / name)
                for name in ("verify.py", "benchmark.py", "_common_utils.py")
            },
            device_conditions={"8": {"hardware": "CPU-control only"}},
            measurement=CacheMeasurement(method="profiler", warmup=5, repeats=20, cv_threshold=0.3),
        ),
        entries={},
    )
    index_path = tmp_path / "cache-index.json"
    index_path.write_text(index.model_dump_json())
    original_overlay = sandbox.read_runtime_overlay(
        str(overlay_path), profiler_runtime.sha(overlay_path)
    )
    overlay = sandbox.EngineRuntimeOverlay(
        environment={
            **original_overlay.environment,
            CACHE_ENV: str(cache),
            BINDING_ENV: str(index_path),
            DIGEST_ENV: profiler_runtime.sha(index_path),
            STATE_ENV: str(state),
        }
    )
    effective, evidence = sandbox.apply_runtime_overlay(
        sandbox.EngineSandbox.model_validate_json(original), overlay
    )
    assert (cache, cache) not in effective.readonly_mounts
    assert (cache / "cases", cache / "cases") in effective.readonly_mounts
    assert (cache / "blobs", cache / "blobs") in effective.readonly_mounts
    assert effective.reference_cache_state_dir == state
    assert evidence["reference_cache_state_dir"] == str(state)
    assert base.read_bytes() == original
    (state / "host-secret").write_text("not a dedicated state namespace")
    with pytest.raises(ValueError, match="dedicated reference-cache state"):
        sandbox.apply_runtime_overlay(sandbox.EngineSandbox.model_validate_json(original), overlay)


def test_overlay_prepare_cli_seals_without_modifying_anchor(anchored_overlay, tmp_path):
    from scripts.prepare_engine_runtime_overlay import prepare

    base, original, overlay, _work, _model = anchored_overlay
    configured = sandbox.read_runtime_overlay(str(overlay), profiler_runtime.sha(overlay))
    output = tmp_path / "prepared-overlay.json"
    observed = prepare(output, base, hashlib.sha256(original).hexdigest(), configured.environment)
    assert observed["overlay_sha256"] == profiler_runtime.sha(output)
    assert observed["executed"] is False
    assert base.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare(output, base, hashlib.sha256(original).hexdigest(), configured.environment)


def test_real_bwrap_child_reads_dev_subtree_ro_bind(tmp_path, monkeypatch):
    """A /dev-subtree ro-bind must stay readable under the sandbox --dev mount.

    Reference-cache artifacts are bound under /dev/shm; wrap() emits
    ``--dev /dev`` as a namespace foundation, and bwrap applies argv in order,
    so emitting it after those binds shadows them inside the child (the NB7
    sealed-index FileNotFoundError). This is the in-child read the previous
    argv-membership assertions could not see.
    """
    import os
    import uuid

    if not shutil.which("bwrap"):
        pytest.skip("real bwrap binary is absent")
    bootstrap = [
        "bwrap",
        "--unshare-all",
        "--ro-bind",
        "/usr",
        "/usr",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for alias in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(alias)
        if path.is_symlink():
            bootstrap += ["--symlink", str(path.readlink()), alias]
    available = subprocess.run([*bootstrap, "--", "/usr/bin/true"], capture_output=True)
    if available.returncode and any(
        message in available.stderr
        for message in (
            b"Operation not permitted",
            b"No permissions",
            b"Creating new namespace failed",
        )
    ):
        pytest.skip("this host denies real namespace creation")
    assert available.returncode == 0, available.stderr.decode(errors="replace")
    # Disposable per-run sentinel state only; never the live cache namespaces.
    sentinel_root = Path("/dev/shm") / (
        f"nb7-mount-order-regression-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    sentinel_root.mkdir(parents=True, exist_ok=False)
    sentinel = sentinel_root / "sealed.json"
    payload = b"nb7-mount-order regression sentinel"
    sentinel.write_bytes(payload)
    try:
        mounts = [(path, path) for path in dict.fromkeys(("/usr", sys.base_prefix, sys.prefix))]
        executable = Path(sys.executable)
        if executable.is_symlink() and executable.readlink().is_absolute():
            mounts.append((sys.base_prefix, str(executable.readlink().parent.parent)))
        mounts.append((str(sentinel), str(sentinel)))
        original = {
            "readonly_mounts": mounts,
            "environment": {"PATH": "/usr/bin:/bin"},
        }
        base = tmp_path / "dev-subtree-sandbox.json"
        base.write_text(json.dumps(original))
        monkeypatch.setenv(sandbox.CONFIG_ENV, str(base))
        monkeypatch.setenv(sandbox.DIGEST_ENV, hashlib.sha256(base.read_bytes()).hexdigest())
        monkeypatch.delenv(sandbox.OVERLAY_CONFIG_ENV, raising=False)
        monkeypatch.delenv(sandbox.OVERLAY_DIGEST_ENV, raising=False)
        work = tmp_path / "attempt"
        work.mkdir()
        probe = work / "probe.py"
        probe.write_text(
            "import hashlib,sys\n"
            "from pathlib import Path\n"
            f"p=Path({str(sentinel)!r})\n"
            "try:\n"
            "    raw=p.read_bytes()\n"
            "except FileNotFoundError:\n"
            "    print('SENTINEL_MISSING',file=sys.stderr);sys.exit(3)\n"
            "print('SENTINEL_OK',len(raw),hashlib.sha256(raw).hexdigest())\n"
        )
        result = run_script([sys.executable, str(probe)], work, None, budget_seconds=10)
        assert result.exit_code == 0, result.stderr
        digest = hashlib.sha256(payload).hexdigest()
        assert result.stdout.strip() == f"SENTINEL_OK {len(payload)} {digest}"
    finally:
        shutil.rmtree(sentinel_root, ignore_errors=True)


def test_supplied_environment_controls_overlay_despite_ambient_changes(
    anchored_overlay, monkeypatch
):
    _base, _original, _overlay, work, _model = anchored_overlay
    environment = dict(os.environ)
    monkeypatch.setenv(sandbox.CONFIG_ENV, "/missing/ambient-sandbox.json")
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, "0" * 64)
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "unknown-policy")
    wrapped, _outer = sandbox.isolate_command(
        [sys.executable, "-c", "pass"], work, None, environment
    )
    index = wrapped.index(sandbox.REFERENCE_POLICY_ENV)
    assert wrapped[index + 1] == environment[sandbox.REFERENCE_POLICY_ENV]


def test_script_uses_snapshot_without_mutating_environment(tmp_path, monkeypatch):
    from triton_evaluation.config import Settings, command_environment

    settings = Settings(reference_device_policy="npu-strict")
    environment = command_environment(settings, {"PATH": os.environ["PATH"]})
    before = environment.copy()
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    outcome = run_script(
        [sys.executable, "-c",
         "import os; print(os.environ['TRITON_EVAL_REFERENCE_DEVICE_POLICY']); "
         "print(os.environ['ASCEND_RT_VISIBLE_DEVICES'])"],
        tmp_path, 3, environment=environment, budget_seconds=10,
    )
    assert outcome.exit_code == 0
    assert outcome.stdout.splitlines() == ["npu-strict", "3"]
    assert environment == before
    assert os.environ["TRITON_EVAL_REFERENCE_DEVICE_POLICY"] == "cpu-first"
