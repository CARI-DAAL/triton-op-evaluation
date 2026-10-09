"""Actual loader/parser boundaries under CPU control; not NPU acceptance."""

import importlib.metadata
import json
import os
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_profiler_export_replay import elf_binding  # noqa: F401

from triton_evaluation.runtime import profiler_runtime as runtime


@pytest.fixture
def deployed_runtime(tmp_path, monkeypatch, elf_binding):  # noqa: F811
    library, library_sha, consumer, consumer_sha = elf_binding
    msprof = tmp_path / "msprof"
    msprof.write_text(f"#!{sys.executable}\nprint('controlled tool')\n")
    msprof.chmod(0o755)
    target = runtime.prepare_runtime(
        tmp_path / "immutable-runtime",
        library,
        library_sha,
        consumer,
        consumer_sha,
        msprof,
        runtime.sha(msprof),
    )
    monkeypatch.setenv(runtime.MANIFEST_ENV, str(target))
    monkeypatch.setenv(runtime.DIGEST_ENV, runtime.sha(target))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/controlled/base")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: str(msprof))
    model = runtime.load_runtime()
    try:
        yield target, model
    finally:
        # Controlled temporary artifacts are removed by pytest after the checks.
        target.parent.chmod(0o755)
        model.alias_path.parent.chmod(0o755)


def test_runtime_is_immutable_exact_bytes_and_cwd_independent(
    tmp_path, monkeypatch, deployed_runtime
):
    target, model = deployed_runtime
    monkeypatch.chdir(tmp_path / "link-only")
    assert runtime.load_runtime() == model
    assert runtime.sha(model.alias_path) == model.library_sha256
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert stat.S_IMODE(model.alias_path.stat().st_mode) == 0o444
    description = runtime.describe_runtime()
    assert description["manifest_sha256"] == runtime.sha(target)
    assert description["required_sqlite_symbols"] == ["sqlite3_compatibility_fixture"]


def test_writable_alias_directory_cannot_be_an_immutable_runtime(deployed_runtime):
    _target, model = deployed_runtime
    model.alias_path.parent.chmod(0o755)
    with pytest.raises(ValueError, match="directories must be read-only"):
        runtime.load_runtime()


def test_missing_pair_tamper_and_writable_alias_refuse_before_capture(
    monkeypatch, deployed_runtime
):
    _target, model = deployed_runtime
    with monkeypatch.context() as temporary:
        temporary.delenv(runtime.DIGEST_ENV)
        with pytest.raises(ValueError, match="together"):
            runtime.load_runtime()
    model.alias_path.chmod(0o644)
    with pytest.raises(ValueError, match="read-only"):
        runtime.load_runtime()
    model.alias_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="alias differs"):
        runtime.load_runtime()


def test_context_restores_absent_or_present_loader_after_parser_exception(
    monkeypatch, deployed_runtime
):
    _target, model = deployed_runtime
    for previous in (None, "/controlled/base"):
        if previous is None:
            monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        else:
            monkeypatch.setenv("LD_LIBRARY_PATH", previous)
        with pytest.raises(RuntimeError, match="callback failed"):
            with runtime.profiler_export_runtime():
                assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(
                    model.alias_path.parent
                )
                raise RuntimeError("callback failed")
        assert os.environ.get("LD_LIBRARY_PATH") == previous


def test_context_refuses_nested_or_non_main_thread_profiles(deployed_runtime):
    with runtime.profiler_export_runtime():
        with pytest.raises(RuntimeError, match="main-thread"):
            with runtime.profiler_export_runtime():
                pytest.fail("nested capture cannot execute")
    errors = []

    def worker():
        try:
            with runtime.profiler_export_runtime():
                errors.append("unexpected execution")
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(errors) == 1 and "main-thread" in errors[0]


def test_configured_runtime_refuses_async_or_unknown_callback_contract():
    def asynchronous(path, async_mode=True):
        pass

    def unknown(path):
        pass

    for handler in (asynchronous, unknown):
        with pytest.raises(ValueError, match="synchronous"):
            runtime.check_synchronous_handler(handler)


def test_actual_profile_entry_holds_alias_until_sync_callback_then_original_csv_parser(
    tmp_path, monkeypatch, deployed_runtime
):
    pytest.importorskip("pandas")
    from triton_evaluation.engine import benchmark

    _target, model = deployed_runtime
    calls = []
    configurations = []

    class Profile:
        def __init__(self, **kwargs):
            configurations.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            # Actual dynamic loader in a fresh subprocess at the sync callback
            # seam. It depends on the private alias remaining in scope.
            import subprocess

            subprocess.run(
                [sys.executable, "-c", f"import ctypes; ctypes.CDLL({str(model.consumer_path)!r})"],
                check=True,
                capture_output=True,
            )
            directory = Path(configurations[-1]["on_trace_ready"])
            directory.mkdir()
            (directory / "kernel_details.csv").write_text(
                "Name,Duration(us)\n" + "matrix,1000\n" * 5 + "matrix,20\n" * 20
            )

    def trace_handler(path, async_mode=False):
        return path

    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    fake_npu = SimpleNamespace(
        profiler=SimpleNamespace(
            _ExperimentalConfig=lambda **kwargs: kwargs,
            AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
            ProfilerLevel=SimpleNamespace(Level1="level1"),
            ProfilerActivity=SimpleNamespace(NPU="npu"),
            profile=Profile,
            tensorboard_trace_handler=trace_handler,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(runtime=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "triton.runtime", SimpleNamespace())
    monkeypatch.chdir(tmp_path)
    path = benchmark.run_profiler_with_config(lambda: calls.append(1), 5, 20, "controlled")
    operators, latency = benchmark.parse_operator_latency(path, 5, 20, keep_res=True)
    assert len(calls) == 26  # Original one manual warmup plus unchanged 5+20.
    assert operators == {"matrix": 20.0} and latency == 0.02
    assert configurations[0]["experimental_config"] == {
        "aic_metrics": "pipe",
        "profiler_level": "level1",
        "l2_cache": False,
        "data_simplification": False,
    }
    assert os.environ["LD_LIBRARY_PATH"] == "/controlled/base"
    trace = json.loads((Path(path) / "profiler-runtime.json").read_bytes())
    assert trace["manifest_sha256"] == runtime.sha(_target)
    assert trace["alias_sha256"] == model.library_sha256
    assert trace["effective_ld_prefix"] == str(model.alias_path.parent)
    assert trace["process_id"] == os.getpid()
    assert trace["callback_scope_completed"] is trace["base_ld_restored"] is True
    assert trace["base_preload_restored"] is True
    assert trace["alias_binding"] == {
        "consumer_path": str(model.consumer_path),
        "probe_symbol": model.required_sqlite_symbols[0],
        "provider_path": str(model.alias_path),
        "alias_path": str(model.alias_path),
        "verified": True,
    }
    assert trace["effective_preload"].split()[0] == str(model.alias_path)


def test_sandbox_requires_profile_pair_and_every_pinned_path_readonly_visible(deployed_runtime):
    from triton_evaluation.runtime.sandbox import EngineSandbox

    target, model = deployed_runtime
    env = {runtime.MANIFEST_ENV: str(target), runtime.DIGEST_ENV: runtime.sha(target)}
    with pytest.raises(ValueError, match="both pinned"):
        EngineSandbox(readonly_mounts=[], environment={runtime.MANIFEST_ENV: str(target)})
    with pytest.raises(ValueError, match="explicit read-only mount"):
        EngineSandbox(readonly_mounts=[], environment=env)
    required = [
        target.parent,
        model.library_path,
        model.consumer_path,
        model.msprof_path,
    ]
    sandbox = EngineSandbox(readonly_mounts=[(path, path) for path in required], environment=env)
    assert sandbox.environment == env
    for missing in required:
        with pytest.raises(ValueError, match="explicit read-only mount"):
            EngineSandbox(
                readonly_mounts=[(path, path) for path in required if path != missing],
                environment=env,
            )


def test_sandbox_actual_command_preserves_runtime_pair_and_readonly_alias(
    tmp_path, deployed_runtime, monkeypatch
):
    from triton_evaluation.runtime import sandbox as sandbox_module
    from triton_evaluation.runtime.sandbox import EngineSandbox

    monkeypatch.setattr(sandbox_module, "_environment_reset_option", lambda *_: "--clearenv")
    target, model = deployed_runtime
    env = {runtime.MANIFEST_ENV: str(target), runtime.DIGEST_ENV: runtime.sha(target)}
    required = [
        target.parent,
        model.library_path,
        model.consumer_path,
        model.msprof_path,
    ]
    sandbox = EngineSandbox(readonly_mounts=[(path, path) for path in required], environment=env)
    work = tmp_path / "engine-attempt"
    work.mkdir()
    command, outer_env = sandbox.wrap([sys.executable, "benchmark.py"], work, None)
    assert ["--ro-bind", str(target.parent), str(target.parent)] == command[
        command.index(str(target.parent)) - 1 : command.index(str(target.parent)) + 2
    ]
    assert runtime.MANIFEST_ENV in command and runtime.DIGEST_ENV in command
    assert command[-4:] == ["/", "--", sys.executable, "benchmark.py"]
    assert set(outer_env) == {"PATH"}


def test_settings_profiler_runtime_pair_is_separate_from_existing_anchors(monkeypatch, tmp_path):
    from triton_evaluation.config import Settings

    monkeypatch.setenv(runtime.MANIFEST_ENV, str(tmp_path / "manifest.json"))
    monkeypatch.delenv(runtime.DIGEST_ENV, raising=False)
    with pytest.raises(ValueError, match="configured together"):
        Settings.from_env()
    monkeypatch.setenv(runtime.DIGEST_ENV, "a" * 64)
    settings = Settings.from_env()
    assert settings.profiler_runtime_manifest == tmp_path / "manifest.json"
    assert settings.profiler_runtime_sha256 == "a" * 64
    monkeypatch.setenv(runtime.MANIFEST_ENV, "relative.json")
    with pytest.raises(ValueError, match="absolute"):
        Settings.from_env()


def test_cache_runtime_files_bind_actual_prepared_manifest_and_elf_bytes(
    tmp_path, deployed_runtime
):
    from triton_evaluation.runtime.reference_cache import (
        CacheDeployment,
        CacheMeasurement,
        ReferenceCacheError,
    )

    target, profiler = deployed_runtime
    engine = tmp_path / "engine"
    engine.mkdir()
    for name in ("verify.py", "benchmark.py", "_common_utils.py"):
        (engine / name).write_text("controlled engine implementation")
    helper = tmp_path / "runtime" / "profiler_runtime.py"
    helper.parent.mkdir()
    helper.write_bytes(Path(runtime.__file__).read_bytes())
    files = {str(target): runtime.sha(target)}
    for field in ("library", "consumer", "msprof", "alias", "elf_observations"):
        path = getattr(profiler, field + "_path")
        files[str(path)] = runtime.sha(path)
    model = CacheDeployment(
        generation="controlled-runtime-closure",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files=files,
        algorithm_files={
            **{path.name: runtime.sha(path) for path in engine.iterdir()},
            "../runtime/profiler_runtime.py": runtime.sha(helper),
        },
        device_conditions={"8": {"hardware": "CPU-controlled; not NPU acceptance"}},
        measurement=CacheMeasurement(method="profiler", warmup=5, repeats=20, cv_threshold=0.3),
    )
    model.check_runtime(engine)
    for missing in files:
        incomplete = model.model_copy(
            update={
                "runtime_files": {path: digest for path, digest in files.items() if path != missing}
            }
        )
        with pytest.raises(ReferenceCacheError, match="every configured profiler runtime file"):
            incomplete.check_runtime(engine)


@pytest.mark.parametrize("previous_preload", [None, "existing-preload.so"])
@pytest.mark.parametrize("failure", ["callback", "probe"])
def test_preload_restored_and_lock_released_on_failure(
    monkeypatch, deployed_runtime, previous_preload, failure
):
    _, model = deployed_runtime
    if previous_preload is None:
        monkeypatch.delenv("LD_PRELOAD", raising=False)
    else:
        monkeypatch.setenv("LD_PRELOAD", previous_preload)

    def probe(*args):
        assert os.environ["LD_PRELOAD"].split()[0] == str(model.alias_path)
        if failure == "probe":
            raise RuntimeError("probe failed")
        return str(model.alias_path)

    monkeypatch.setattr(runtime, "probe_alias_binding", probe)
    with pytest.raises(RuntimeError, match="failed"):
        with runtime.profiler_export_runtime():
            raise RuntimeError("callback failed")
    assert os.environ.get("LD_PRELOAD") == previous_preload
    assert os.environ["LD_LIBRARY_PATH"] == "/controlled/base"
    assert runtime._ENV_LOCK.acquire(False)
    runtime._ENV_LOCK.release()


def test_real_legacy_rpath_decoy_requires_preload(tmp_path, elf_binding):  # noqa: F811
    import shutil
    import subprocess

    library, _, consumer, _ = elf_binding
    decoy = tmp_path / "legacy"
    decoy.mkdir()
    source = tmp_path / "decoy.c"
    source.write_text("int sqlite3_compatibility_fixture(void) { return 7; }\n")
    subprocess.run(
        [shutil.which("cc"), "-shared", "-fPIC", "-o", str(decoy / "libsqlite3.so"), str(source)],
        check=True,
    )
    subprocess.run(
        [
            shutil.which("cc"),
            "-shared",
            "-fPIC",
            "-o",
            str(consumer),
            str(tmp_path / "consumer.c"),
            f"-L{decoy}",
            "-l:libsqlite3.so",
            "-Wl,--disable-new-dtags",
            f"-Wl,-rpath,{decoy}",
        ],
        check=True,
    )
    alias = tmp_path / "private" / "libsqlite3.so"
    alias.parent.mkdir()
    shutil.copy2(library, alias)
    assert runtime.sha(alias) == runtime.sha(library)
    assert runtime.sha(alias) != runtime.sha(decoy / "libsqlite3.so")
    environment = {**os.environ, "LD_LIBRARY_PATH": str(alias.parent)}
    environment.pop("LD_PRELOAD", None)
    with pytest.raises(RuntimeError, match="alias bypassed"):
        runtime.probe_alias_binding(consumer, "sqlite3_compatibility_fixture", alias, environment)
    environment["LD_PRELOAD"] = str(alias)
    assert runtime.probe_alias_binding(
        consumer, "sqlite3_compatibility_fixture", alias, environment
    ) == str(alias)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import ctypes; assert ctypes.CDLL({str(consumer)!r}).fixture() == 42",
        ],
        env=environment,
        check=True,
        capture_output=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "error", [OSError("launch failed"), __import__("subprocess").TimeoutExpired("probe", 30)]
)
def test_binding_probe_wraps_launch_and_timeout_errors(monkeypatch, tmp_path, error):
    path = tmp_path / "placeholder"
    path.write_bytes(b"placeholder")

    def fail(*args, **kwargs):
        assert args[0][0] == sys.executable
        assert kwargs["start_new_session"] is True
        raise error

    monkeypatch.setattr(runtime.subprocess, "Popen", fail)
    with pytest.raises(RuntimeError, match="could not execute|timed out") as caught:
        runtime.probe_alias_binding(path, "sqlite3_fixture", path)
    assert "import ctypes" not in str(caught.value)


def test_global_preload_lookup_without_consumer_relocation_cannot_pass(tmp_path, elf_binding):  # noqa: F811
    import shutil
    import subprocess

    library, _, consumer, _ = elf_binding
    source = tmp_path / "unrelated.c"
    source.write_text("int fixture(void) { return 0; }\n")
    subprocess.run(
        [shutil.which("cc"), "-shared", "-fPIC", "-o", str(consumer), str(source)], check=True
    )
    alias = tmp_path / "alias" / "libsqlite3.so"
    alias.parent.mkdir()
    shutil.copy2(library, alias)
    environment = {**os.environ, "LD_PRELOAD": str(alias)}
    with pytest.raises(RuntimeError, match="consumer relocation unverified"):
        runtime.probe_alias_binding(consumer, "sqlite3_compatibility_fixture", alias, environment)


@pytest.mark.parametrize("force_global_fallback", [False, True])
def test_probe_dladdr_explicit_libdl_and_global_fallback(
    monkeypatch, deployed_runtime, force_global_fallback
):
    import subprocess

    _, model = deployed_runtime
    actual_popen = subprocess.Popen

    def run_with_loader_observation(command, **kwargs):
        instrument = """
import ctypes
original_cdll = ctypes.CDLL
libdl_attempted = False
def observed_cdll(name, *args, **kwargs):
    global libdl_attempted
    if name == "libdl.so.2":
        libdl_attempted = True
        if FORCE_FALLBACK:
            raise OSError("controlled libdl unavailability")
    return original_cdll(name, *args, **kwargs)
ctypes.CDLL = observed_cdll
""".replace("FORCE_FALLBACK", repr(force_global_fallback))
        instrument += command[2] + "\nassert libdl_attempted\n"
        return actual_popen([*command[:2], instrument, *command[3:]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", run_with_loader_observation)
    environment = {
        **os.environ,
        "LD_PRELOAD": str(model.alias_path),
        "LD_LIBRARY_PATH": str(model.alias_path.parent),
    }
    assert runtime.probe_alias_binding(
        model.consumer_path, model.required_sqlite_symbols[0], model.alias_path, environment
    ) == str(model.alias_path)


@pytest.mark.parametrize("outcome", ["timeout", "failure", "success"])
def test_probe_reaps_owned_forking_constructor_group(tmp_path, monkeypatch, elf_binding, outcome):  # noqa: F811
    import ctypes
    import shutil
    import subprocess
    import time

    library, _, consumer, _ = elf_binding
    alias = tmp_path / "alias" / "libsqlite3.so"
    alias.parent.mkdir()
    shutil.copy2(library, alias)
    pid_file = tmp_path / "descendant.pid"
    source = tmp_path / "forking.c"
    source.write_text(
        "#include <unistd.h>\n#include <stdio.h>\n"
        "extern int sqlite3_compatibility_fixture(void);\n"
        "int fixture(void) { return sqlite3_compatibility_fixture(); }\n"
        "__attribute__((constructor)) static void start(void) {\n"
        "pid_t pid = fork(); if (pid == 0) { for (;;) pause(); }\n"
        f'FILE *f = fopen("{pid_file}", "w"); fprintf(f, "%ld", (long)pid); fclose(f);\n'
        + ("for (;;) pause();\n" if outcome == "timeout" else "")
        + "}\n"
    )
    subprocess.run(
        [
            shutil.which("cc"),
            "-shared",
            "-fPIC",
            "-o",
            str(consumer),
            str(source),
            f"-L{alias.parent}",
            "-l:libsqlite3.so",
        ],
        check=True,
    )
    actual_popen = subprocess.Popen

    def short_probe(command, **kwargs):
        process = actual_popen(command, **kwargs)
        actual_wait = process.wait

        def wait(timeout=None):
            assert timeout in (None, 30)
            return actual_wait(timeout=0.5 if timeout is not None else None)

        process.wait = wait
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", short_probe)
    environment = {**os.environ, "LD_PRELOAD": str(alias), "LD_LIBRARY_PATH": str(alias.parent)}
    # Adopt the killed grandchild so this test can reap and prove its exit,
    # rather than depending on the container's PID 1 zombie handling.
    libc = ctypes.CDLL(None)
    old_subreaper = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(old_subreaper), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    descendant = None
    try:
        symbol = "sqlite3_missing" if outcome == "failure" else "sqlite3_compatibility_fixture"
        if outcome == "success":
            assert runtime.probe_alias_binding(consumer, symbol, alias, environment) == str(alias)
        else:
            with pytest.raises(RuntimeError, match="timed out|probe failed"):
                runtime.probe_alias_binding(consumer, symbol, alias, environment)
        descendant = int(pid_file.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            waited, status = os.waitpid(descendant, os.WNOHANG)
            if waited:
                assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == 9
                descendant = None
                break
            time.sleep(0.01)
        assert descendant is None, "probe descendant survived group cleanup"
    finally:
        if descendant is not None:
            os.kill(descendant, 9)
            os.waitpid(descendant, 0)
        assert libc.prctl(36, old_subreaper.value, 0, 0, 0) == 0
