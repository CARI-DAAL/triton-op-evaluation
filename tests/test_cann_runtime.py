"""Existing CANN bytes, clean preparation and real unchanged-anchor bwrap children."""

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_profiler_export_replay import elf_binding  # noqa: F401

from triton_evaluation import profiler_runtime as runtime
from triton_evaluation.evaluators import sandbox
from triton_evaluation.evaluators.base import run_script


@pytest.fixture
def canonical_cann(tmp_path, elf_binding):  # noqa: F811
    library, library_sha, consumer, consumer_sha = elf_binding
    root = tmp_path / "install" / "cann-9.1.1"
    required = (
        "lib64",
        "opp",
        "toolkit",
        "tools/ccec_compiler/bin",
        "tools/profiler/bin",
        "python/site-packages",
        "opp/built-in/op_impl/ai_core/tbe",
        "tools/profiler/profiler_tool/analysis/lib64",
    )
    for relative in required:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("lib64", "lib64/common", "lib64/driver"):
        (root.parent / "driver" / relative).mkdir(parents=True, exist_ok=True)
    actual_consumer = root / "tools/profiler/profiler_tool/analysis/lib64/msprof_analysis.so"
    shutil.copyfile(consumer, actual_consumer)
    msprof = root / "tools/profiler/bin/msprof"
    msprof.write_text("#!/bin/sh\nexit 0\n")
    msprof.chmod(0o755)
    (root / "opp/version.info").write_text("controlled existing stack 9.1.1\n")
    for relative in runtime.CANN_REQUIRED_LIBRARIES:
        shutil.copyfile(library, root / relative)
    source = root / "set_env.sh"
    lines = ['test -z "${HQ_SENTINEL_SECRET:-}" || return 91']
    for key in runtime.CANN_ENV_KEYS - runtime.CANN_PATH_KEYS:
        relative = {"ASCEND_OPP_PATH": "opp", "TOOLCHAIN_HOME": "toolkit"}.get(key, "")
        lines.append(f"export {key}={shlex.quote(str(root / relative))}")
    lines += [
        f'export PATH={shlex.quote(str(root / "tools/profiler/bin"))}:"$PATH"',
        f'export LD_LIBRARY_PATH={shlex.quote(str(root / "lib64"))}:"$LD_LIBRARY_PATH"',
        "export PYTHONPATH="
        + shlex.quote(str(root / "python/site-packages"))
        + ":"
        + shlex.quote(str(root / "opp/built-in/op_impl/ai_core/tbe"))
        + ':"$PYTHONPATH"',
    ]
    source.write_text("\n".join(lines) + "\n")
    return root, source, library, library_sha, actual_consumer, consumer_sha, msprof


def prepare(canonical_cann, output):
    _root, source, library, library_sha, consumer, consumer_sha, msprof = canonical_cann
    return runtime.prepare_runtime(
        output,
        library,
        library_sha,
        consumer,
        consumer_sha,
        msprof,
        runtime.sha(msprof),
        source,
        runtime.sha(source),
    )


def test_prepare_uses_clean_vendor_capture_and_pins_metadata_existing_libraries(
    canonical_cann, tmp_path, monkeypatch
):
    monkeypatch.setenv("HQ_SENTINEL_SECRET", "not inherited by trusted source")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/old/deleted")
    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    cann = runtime.read_cann_environment(model)
    root = canonical_cann[0]
    assert cann.environment["ASCEND_HOME_PATH"] == str(root)
    assert cann.preparation["returncode"] == 0
    assert cann.preparation["imports_npu"] is False
    assert "HQ_SENTINEL_SECRET" not in cann.preparation["clean_environment"]
    assert cann.runtime_files[str(root / "opp/version.info")] == runtime.sha(
        root / "opp/version.info"
    )
    assert cann.runtime_files[str(root / "lib64/libascendcl.so")] == runtime.sha(
        root / "lib64/libascendcl.so"
    )
    # Vendor-advertised optional directories can be absent; required ones cannot.
    optional = str(root / "tools/show_kernel_debug_data")
    assert cann.directories[optional] == {"required": False, "exists": False, "resolved_path": None}
    assert model.cann_environment_path.stat().st_mode & 0o222 == 0
    before = canonical_cann[1].read_bytes()
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *a, **kw: pytest.fail("worker sourced shell")
    )
    runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    assert canonical_cann[1].read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "source-pin",
        "different-root",
        "missing-required",
        "extra-source-path",
        "missing-library",
        "directory-escape",
    ],
)
def test_preparation_rejects_noncanonical_or_incomplete_runtime(canonical_cann, tmp_path, fault):
    root, source, _library, _library_sha, consumer, _consumer_sha, msprof = canonical_cann
    if fault == "source-pin":
        digest = "0" * 64
    else:
        if fault == "different-root":
            msprof = tmp_path / "other-msprof"
        elif fault == "missing-required":
            shutil.rmtree(root / "toolkit")
        elif fault == "missing-library":
            (root / "lib64/libruntime.so").unlink()
        elif fault == "directory-escape":
            shutil.rmtree(root / "toolkit")
            (root / "toolkit").symlink_to(tmp_path, target_is_directory=True)
        else:
            source.write_text(source.read_text() + "export LD_LIBRARY_PATH=/untrusted/lib\n")
        digest = runtime.sha(source)
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.capture_cann_environment(source, digest, msprof, consumer)


def test_cann_quartet_layout_hash_unknown_keys_and_required_directory_fail_closed(
    canonical_cann, tmp_path
):
    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    for field in (
        "cann_set_env_path",
        "cann_set_env_sha256",
        "cann_environment_path",
        "cann_environment_sha256",
    ):
        raw = model.model_dump(mode="json")
        raw[field] = None
        with pytest.raises(ValueError, match="all four pins"):
            runtime.ProfilerRuntimeManifest.model_validate(raw)
    raw_cann = json.loads(model.cann_environment_path.read_bytes())
    raw_cann["environment"]["HQ_SENTINEL_SECRET"] = "invalid"
    with pytest.raises(ValueError, match="fixed eight"):
        runtime.CannRuntimeEnvironment.model_validate(raw_cann)
    canonical_cann[1].write_text("changed source bytes\n")
    with pytest.raises(ValueError, match="cann_set_env differs"):
        runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)


def test_derive_removes_only_version_boundary_keeps_non_cann_driver_and_languages(
    canonical_cann, tmp_path
):
    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    root = canonical_cann[0]
    old = root.parent / "cann-8.5.2"
    outside = tmp_path / "outside/cann-8.5.2/keep"
    similar = root.parent / "cannulated/keep"
    original = {
        "PATH": f"{old}/bin:{outside}:{similar}:/usr/bin",
        "LD_LIBRARY_PATH": f"{old}/lib64:{root.parent}/driver/lib64",
        "PYTHONPATH": f"{old}/python/site-packages:/trusted/noncann",
        "LANG": "zh_CN",
    }
    actual = runtime.derive_cann_environment(model, original)
    assert "8.5.2/bin" not in actual["PATH"]
    assert str(outside) in actual["PATH"] and str(similar) in actual["PATH"]
    assert "/trusted/noncann" in actual["PYTHONPATH"]
    assert actual["LD_LIBRARY_PATH"].split(":").count(str(root.parent / "driver/lib64")) == 1
    assert "LANG" not in actual and original["LANG"] == "zh_CN"


def require_bwrap():
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed on this controlled CPU host")
    args = [
        shutil.which("bwrap"),
        "--unshare-all",
        "--ro-bind",
        "/usr",
        "/usr",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for name in ("/bin", "/sbin", "/lib", "/lib64"):
        if Path(name).is_symlink():
            args += ["--symlink", str(Path(name).readlink()), name]
    observed = subprocess.run([*args, "--", "/usr/bin/true"], capture_output=True)
    if observed.returncode and any(
        message in observed.stderr
        for message in (
            b"Operation not permitted",
            b"No permissions",
            b"Creating new namespace failed",
        )
    ):
        pytest.skip("actual namespace permission denial")
    assert observed.returncode == 0, observed.stderr


def test_real_bwrap_old_cann_red_then_derived_child_loader_and_conditions_green(
    canonical_cann, tmp_path, monkeypatch
):
    require_bwrap()
    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    root = canonical_cann[0]
    mounts = [(path, path) for path in dict.fromkeys(("/usr", sys.base_prefix, sys.prefix))]
    executable = Path(sys.executable)
    if executable.is_symlink() and executable.readlink().is_absolute():
        mounts.append((sys.base_prefix, str(executable.readlink().parent.parent)))
    old = root.parent / "cann-8.5.2"
    original_env = {
        "PATH": f"{old}/bin:{old}/tools/profiler/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{old}/lib64:{root.parent}/driver/lib64",
        "PYTHONPATH": f"{old}/python/site-packages:/trusted/noncann",
        "LANG": "C",
        "LC_ALL": "C",
        **{
            key: str(old / {"ASCEND_OPP_PATH": "opp", "TOOLCHAIN_HOME": "toolkit"}.get(key, ""))
            for key in runtime.CANN_ENV_KEYS - runtime.CANN_PATH_KEYS
        },
    }
    base = tmp_path / "original-sandbox.json"
    base.write_text(json.dumps({"readonly_mounts": mounts, "environment": original_env}))
    before = base.read_bytes()
    base_sha = hashlib.sha256(before).hexdigest()
    monkeypatch.setenv(sandbox.CONFIG_ENV, str(base))
    monkeypatch.setenv(sandbox.DIGEST_ENV, base_sha)
    monkeypatch.setenv(sandbox.REFERENCE_POLICY_ENV, "npu-strict")
    work = tmp_path / "attempt"
    work.mkdir()
    # Actual old --clearenv child fails the scalar contract before any Torch import.
    old_args, old_env = sandbox.EngineSandbox.model_validate_json(before).wrap(
        [sys.executable, "-c", f"import os; assert os.environ['ASCEND_HOME_PATH']=={str(root)!r}"],
        work,
        None,
    )
    red = subprocess.run(old_args, env=old_env, capture_output=True)
    assert red.returncode != 0 and b"AssertionError" in red.stderr
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        sandbox.EngineRuntimeOverlay(
            environment={runtime.MANIFEST_ENV: str(target), runtime.DIGEST_ENV: runtime.sha(target)}
        ).model_dump_json()
    )
    monkeypatch.setenv(sandbox.OVERLAY_CONFIG_ENV, str(overlay))
    monkeypatch.setenv(sandbox.OVERLAY_DIGEST_ENV, runtime.sha(overlay))
    monkeypatch.setenv("HQ_SENTINEL_SECRET", "must-not-cross")
    canary = tmp_path / "outside-canary"
    canary.write_text("host-only")
    probe = work / "probe.py"
    probe.write_text(
        "import json,os,subprocess,sys\nfrom pathlib import Path\n"
        "from triton_evaluation import profiler_runtime as r\n"
        "description=r.describe_runtime()\n"
        "with r.profiler_export_runtime() as trace:\n"
        " code='import ctypes; ctypes.CDLL('+repr(description['consumer_path'])+')'\n"
        " subprocess.run([sys.executable,'-c',code],check=True)\n"
        "print(json.dumps({'actual':description,'trace':trace,\n"
        f"'canary':Path({str(canary)!r}).exists(),\n"
        "'secret':'HQ_SENTINEL_SECRET' in os.environ,'lang':os.environ['LANG']}))\n"
    )
    outcome = run_script([sys.executable, str(probe)], work, None, budget_seconds=10)
    assert outcome.exit_code == 0, outcome.stderr
    observed = json.loads(outcome.stdout)
    assert observed["canary"] is observed["secret"] is False
    assert observed["lang"] == "C"
    actual = observed["actual"]["cann_runtime"]
    assert actual["cann_set_env_sha256"] == runtime.sha(canonical_cann[1])
    assert actual["cann_environment_sha256"] == runtime.sha(model.cann_environment_path)
    assert actual["effective_environment"]["ASCEND_HOME_PATH"] == str(root)
    assert "8.5.2" not in json.dumps(actual["effective_environment"])
    assert actual["effective_environment"]["PYTHONPATH"].split(":")[0] == str(
        Path(runtime.__file__).parent.parent
    )
    assert (
        actual["effective_environment_sha256"]
        == hashlib.sha256(
            json.dumps(actual["effective_environment"], sort_keys=True).encode()
        ).hexdigest()
    )
    assert observed["trace"]["cann_runtime"] == actual
    boundary = json.loads(next(work.glob("engine-runtime-boundary-*.json")).read_bytes())
    assert (
        boundary["cann_runtime"]["effective_environment_sha256"]
        == actual["effective_environment_sha256"]
    )
    assert boundary["executed"] is False
    assert base.read_bytes() == before and runtime.sha(base) == base_sha


def test_actual_cache_requires_cann_source_artifact_metadata_and_library_pins(
    canonical_cann, tmp_path, monkeypatch
):
    import importlib.metadata

    from triton_evaluation.reference_cache import (
        CacheDeployment,
        CacheMeasurement,
        ReferenceCacheError,
    )

    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    cann = runtime.read_cann_environment(model)
    for key, value in runtime.derive_cann_environment(model, {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(runtime.MANIFEST_ENV, str(target))
    monkeypatch.setenv(runtime.DIGEST_ENV, runtime.sha(target))
    files = {str(target): runtime.sha(target), **cann.runtime_files}
    for name in (
        "library",
        "consumer",
        "msprof",
        "alias",
        "elf_observations",
        "cann_set_env",
        "cann_environment",
    ):
        files[str(getattr(model, name + "_path"))] = getattr(model, name + "_sha256")
    engine = Path(runtime.__file__).parent / "engine"
    algorithms = {
        name: runtime.sha(engine / name)
        for name in ("verify.py", "benchmark.py", "_common_utils.py")
    }
    algorithms["../profiler_runtime.py"] = runtime.sha(Path(runtime.__file__))
    deployment = CacheDeployment(
        generation="controlled-cpu-cann",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files=files,
        algorithm_files=algorithms,
        device_conditions={"8": {"hardware": "CPU control; no NPU acceptance"}},
        measurement=CacheMeasurement(method="profiler", warmup=5, repeats=20, cv_threshold=0.3),
    )
    deployment.check_runtime(engine)
    for missing in [
        str(model.cann_set_env_path),
        str(model.cann_environment_path),
        str(cann.cann_root / "opp/version.info"),
        str(cann.cann_root / "lib64/libruntime.so"),
    ]:
        partial = deployment.model_copy(
            update={
                "runtime_files": {path: digest for path, digest in files.items() if path != missing}
            }
        )
        with pytest.raises(ReferenceCacheError, match="every configured profiler runtime file"):
            partial.check_runtime(engine)


@pytest.mark.parametrize(
    "fault", ["environment-bytes", "library-bytes", "required-directory", "stale-actual-child"]
)
def test_worker_rejects_changed_cann_conditions_before_measurement(
    canonical_cann, tmp_path, monkeypatch, fault
):
    target = prepare(canonical_cann, tmp_path / "runtime")
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    for key, value in runtime.derive_cann_environment(model, {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(runtime.MANIFEST_ENV, str(target))
    monkeypatch.setenv(runtime.DIGEST_ENV, runtime.sha(target))
    if fault == "environment-bytes":
        model.cann_environment_path.chmod(0o644)
        model.cann_environment_path.write_bytes(b"{}")
    elif fault == "library-bytes":
        (canonical_cann[0] / "lib64/libruntime.so").write_bytes(b"changed existing library")
    elif fault == "required-directory":
        shutil.rmtree(canonical_cann[0] / "toolkit")
    else:
        monkeypatch.setenv("ASCEND_HOME_PATH", str(canonical_cann[0].parent / "cann-8.5.2"))
    with pytest.raises(ValueError):
        runtime.load_runtime()


def test_actual_prepare_cli_import_ban_keeps_existing_cann_and_system_bytes(
    canonical_cann, tmp_path
):
    root, source, library, library_sha, consumer, consumer_sha, msprof = canonical_cann
    before = {path: runtime.sha(path) for path in (source, library, consumer, msprof)}
    output = tmp_path / "cli-runtime"
    script = Path(runtime.__file__).parent.parent / "scripts/prepare_profiler_runtime.py"
    argv = [
        str(script),
        "--output",
        str(output),
        "--sqlite-library",
        str(library),
        "--sqlite-library-sha256",
        library_sha,
        "--sqlite-consumer",
        str(consumer),
        "--sqlite-consumer-sha256",
        consumer_sha,
        "--msprof",
        str(msprof),
        "--msprof-sha256",
        runtime.sha(msprof),
        "--cann-set-env",
        str(source),
        "--cann-set-env-sha256",
        runtime.sha(source),
    ]
    code = (
        "import importlib.abc,runpy,sys\n"
        "class Ban(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        "  if fullname.split('.')[0] in {'torch','torch_npu','triton'}:\n"
        "   raise RuntimeError('forbidden NPU import: '+fullname)\n"
        "sys.meta_path.insert(0,Ban())\n"
        f"sys.argv={argv!r}\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    target = Path(observed["manifest_path"])
    assert observed["manifest_sha256"] == runtime.sha(target)
    model = runtime.read_runtime(str(target), runtime.sha(target), check_resolution=False)
    assert runtime.read_cann_environment(model).cann_root == root
    assert {path: runtime.sha(path) for path in before} == before


def test_repeated_clean_capture_artifact_is_byte_identical(canonical_cann, tmp_path):
    first = prepare(canonical_cann, tmp_path / "runtime-one")
    second = prepare(canonical_cann, tmp_path / "runtime-two")
    first_model = runtime.read_runtime(str(first), runtime.sha(first), check_resolution=False)
    second_model = runtime.read_runtime(str(second), runtime.sha(second), check_resolution=False)
    assert (
        first_model.cann_environment_path.read_bytes()
        == second_model.cann_environment_path.read_bytes()
    )
    assert first_model.cann_environment_sha256 == second_model.cann_environment_sha256
