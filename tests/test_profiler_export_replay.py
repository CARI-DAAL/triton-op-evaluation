"""Offline process/actual CSV-parser checks, never native profiling acceptance."""

import builtins
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/replay_profiler_export.py"
spec = importlib.util.spec_from_file_location("profiler_export_replay_test", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import tracked export replay")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def capture(tmp_path):
    directory = tmp_path / "original"
    root = directory / "sample-01" / "native_ascend_pt" / "PROF_captured"
    raw = root / "device_8/data/ffts_profile.data.8.slice_0"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"original captured bytes")
    marker = root / "device_8/data/all_file.complete"
    marker.write_bytes(b"")
    report = directory / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "profiler-latency-probe-v1",
                "physical_device": 8,
                "artifacts": [
                    {
                        "path": str(path.relative_to(directory)),
                        "bytes": path.stat().st_size,
                        "sha256": replay.sha(path),
                    }
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                ],
            }
        )
    )
    return report, replay.sha(report), raw


def fake_msprof(tmp_path, monkeypatch, code):
    executable = tmp_path / "msprof"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "root = Path(sys.argv[2].split('=', 1)[1])\n"
        "assert sys.argv[1] == '--export=on'\n"
        "assert (root / 'device_8/data/all_file.complete').exists()\n"
        "assert (root / 'device_8/data/ffts_profile.data.8.slice_0').read_bytes() "
        "== b'original captured bytes'\n" + code
    )
    executable.chmod(0o755)
    monkeypatch.setattr(replay.shutil, "which", lambda name: str(executable))
    return executable


def evidence(output):
    return json.loads((output / "report.json").read_bytes())


def test_exact_retained_copy_rc_zero_with_no_csv_is_empty_export(tmp_path, monkeypatch, capture):
    report, digest, raw = capture
    fake_msprof(tmp_path, monkeypatch, "print('export completed without operator data')\n")
    output = tmp_path / "replay"
    assert replay.replay(report, digest, output) == 1
    result = evidence(output)
    assert result["state"] == "empty_export"
    assert result["returncode"] == 0
    assert result["export_completed"] is True
    assert result["original_artifacts_unchanged"] is True
    assert result["raw_latency"] == "None"
    assert result["cann_csv_observations"] == []
    assert raw.read_bytes() == b"original captured bytes"
    assert (output / "export.stdout").read_text() == "export completed without operator data\n"


def test_cann_text_data_does_not_claim_kernel_details_or_eight_new_measurements(
    tmp_path, monkeypatch, capture
):
    report, digest, _ = capture
    fake_msprof(
        tmp_path,
        monkeypatch,
        "(root / 'op_summary.csv').write_text('Op Name,Task Duration(us)\\nmatmul,20\\n')\n",
    )
    output = tmp_path / "replay"
    assert replay.replay(report, digest, output) == 0
    result = evidence(output)
    assert result["state"] == "exported_text_data"
    assert result["cann_csv_observations"][0]["data_rows"] == 1
    assert result["latency_ms"] is None
    assert result["kernel_details_parseable"] is False
    assert result["executes_npu_operations"] is False
    assert result["is_independent_sample"] is result["admission_evidence"] is False


def test_actual_service_kernel_csv_formula_is_used_without_torch_or_triton_imports(
    tmp_path, monkeypatch, capture
):
    report, digest, _ = capture
    fake_msprof(
        tmp_path,
        monkeypatch,
        "(root / 'kernel_details.csv').write_text('Name,Duration(us)\\n' "
        "+ 'matmul,1000\\n' * 5 + 'matmul,20\\n' * 20)\n",
    )
    original_import = builtins.__import__

    def import_without_npu(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "triton"}:
            raise AssertionError("Offline parsing must not import a device stack")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_npu)
    output = tmp_path / "replay"
    assert replay.replay(report, digest, output) == 0
    result = evidence(output)
    assert result["operators"] == {"matmul": 20.0}
    assert result["latency_ms"] == 0.02  # Actual last-20 mean, excludes five warmups.
    assert result["original_artifacts_unchanged"] is True


def test_hash_tamper_refuses_before_export_and_preserves_original(tmp_path, monkeypatch, capture):
    report, digest, raw = capture
    raw.write_bytes(b"tampered")
    monkeypatch.setattr(replay.shutil, "which", lambda *_: pytest.fail("must not resolve/run tool"))
    with pytest.raises(ValueError, match="does not match"):
        replay.replay(report, digest, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_extra_artifact_refuses_unpinned_capture(tmp_path, capture):
    report, digest, raw = capture
    (raw.parent / "extra").write_text("unrecorded")
    with pytest.raises(ValueError, match="file set"):
        replay.verified_capture(report, digest)


def test_timeout_retains_executed_pid_signal_and_partial_stdout(tmp_path, monkeypatch, capture):
    report, digest, _ = capture
    fake_msprof(
        tmp_path, monkeypatch, "import time\nprint('started', flush=True)\ntime.sleep(30)\n"
    )
    output = tmp_path / "replay"
    assert replay.replay(report, digest, output, timeout_seconds=1) == 1
    result = evidence(output)
    assert result["state"] == "timeout"
    assert result["pid"] > 0
    assert result["process_group"] == result["pid"]
    assert result["returncode"] == -15
    assert result["signals_sent"] == ["SIGTERM"]
    assert result["export_completed"] is False
    assert result["original_artifacts_unchanged"] is True
    assert (output / "export.stdout").read_bytes() == b"started\n"


def test_running_attempt_and_actual_pid_are_persisted_before_wait(tmp_path, monkeypatch, capture):
    report, digest, _ = capture
    fake_msprof(tmp_path, monkeypatch, "print('complete')\n")
    actual_run = replay.run_export
    output = tmp_path / "replay"

    def inspect(command, timeout_seconds, on_started, env):
        initial = evidence(output)
        assert initial["state"] == "running"
        assert initial["command"] == command

        def inspect_started(pid):
            on_started(pid)
            running = evidence(output)
            assert running["state"] == "running"
            assert running["pid"] == running["process_group"] == pid

        return actual_run(command, timeout_seconds, inspect_started, env)

    monkeypatch.setattr(replay, "run_export", inspect)
    assert replay.replay(report, digest, output) == 1


def test_existing_replay_attempt_is_never_overwritten(tmp_path, monkeypatch, capture):
    report, digest, _ = capture
    fake_msprof(tmp_path, monkeypatch, "print('unused')\n")
    output = tmp_path / "previous"
    output.mkdir()
    (output / "old.json").write_text("old attempt")
    with pytest.raises(FileExistsError):
        replay.replay(report, digest, output)
    assert (output / "old.json").read_text() == "old attempt"


@pytest.fixture
def elf_binding(tmp_path):
    """Build real controlled ELF files; this is not the CANN binary or NPU acceptance."""
    compiler = shutil.which("cc")
    if compiler is None or shutil.which("readelf") is None:
        pytest.skip("ELF seam test requires the local C compiler and readelf")
    stub = tmp_path / "link-only"
    stub.mkdir()
    sqlite_source = tmp_path / "sqlite.c"
    sqlite_source.write_text("int sqlite3_compatibility_fixture(void) { return 42; }\n")
    source = tmp_path / "consumer.c"
    source.write_text(
        "extern int sqlite3_compatibility_fixture(void);\n"
        "int fixture(void) { return sqlite3_compatibility_fixture(); }\n"
    )
    library = tmp_path / "libsqlite3.so.0"
    consumer = tmp_path / "consumer.so"
    for command in (
        [compiler, "-shared", "-fPIC", "-o", str(stub / "libsqlite3.so"), str(sqlite_source)],
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wl,-soname,libsqlite3.so.0",
            "-o",
            str(library),
            str(sqlite_source),
        ],
        [
            compiler,
            "-shared",
            "-fPIC",
            "-o",
            str(consumer),
            str(source),
            f"-L{stub}",
            "-l:libsqlite3.so",
        ],
    ):
        subprocess.run(command, check=True, capture_output=True)
    return library, replay.sha(library), consumer, replay.sha(consumer)


def test_private_exact_elf_alias_changes_only_export_child_loader_and_restores_data(
    tmp_path, monkeypatch, capture, elf_binding
):
    report, digest, _ = capture
    library, library_sha, consumer, consumer_sha = elf_binding
    fake_msprof(
        tmp_path,
        monkeypatch,
        "import ctypes\n"
        "try:\n"
        f"    loaded = ctypes.CDLL({str(consumer)!r})\n"
        "except OSError as error:\n"
        "    print(str(error), file=sys.stderr)\n"
        "else:\n"
        "    assert loaded.fixture() == 42\n"
        "    (root / 'op_summary.csv').write_text('Op Name,Task Duration(us)\\nmatmul,20\\n')\n",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/controlled/original")
    monkeypatch.setenv("TRITON_EVAL_LD_LIBRARY_PATH", "/immutable/anchor")
    assert replay.replay(report, digest, tmp_path / "without-alias") == 1
    output = tmp_path / "with-alias"
    assert (
        replay.replay(
            report,
            digest,
            output,
            sqlite_library=library,
            sqlite_library_sha256=library_sha,
            sqlite_consumer=consumer,
            sqlite_consumer_sha256=consumer_sha,
        )
        == 0
    )
    result = evidence(output)
    assert result["state"] == "exported_text_data"
    assert result["sqlite_alias"]["required_sqlite_symbols"] == ["sqlite3_compatibility_fixture"]
    assert result["sqlite_alias"]["target_soname"] == "libsqlite3.so.0"
    assert set(result["environment_diff"]) == {"LD_LIBRARY_PATH"}
    assert replay.os.environ["LD_LIBRARY_PATH"] == "/controlled/original"
    assert replay.os.environ["TRITON_EVAL_LD_LIBRARY_PATH"] == "/immutable/anchor"
    assert result["sqlite_sources_unchanged"] is True
    assert result["original_artifacts_unchanged"] is True
    assert replay.sha(output / "private-library-alias/libsqlite3.so") == library_sha
    assert result["kernel_details_parseable"] is False


def test_sqlite_binding_refuses_unpinned_or_incomplete_sources(tmp_path, capture, elf_binding):
    report, digest, _ = capture
    library, library_sha, consumer, consumer_sha = elf_binding
    with pytest.raises(ValueError, match="explicit source pins"):
        replay.replay(report, digest, tmp_path / "invalid", sqlite_library=library)
    with pytest.raises(ValueError, match="explicit pin"):
        replay.inspect_sqlite_binding(library, "0" * 64, consumer, consumer_sha)
    assert not (tmp_path / "invalid").exists()
    assert replay.sha(library) == library_sha


def test_actual_elf_metadata_refuses_target_without_the_required_sqlite_exports(
    tmp_path, elf_binding
):
    library, _library_sha, consumer, consumer_sha = elf_binding
    replacement_source = tmp_path / "empty.c"
    replacement_source.write_text("int unrelated(void) { return 0; }\n")
    subprocess.run(
        [
            shutil.which("cc"),
            "-shared",
            "-fPIC",
            "-Wl,-soname,libsqlite3.so.0",
            "-o",
            str(library),
            str(replacement_source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="lacks required symbols"):
        replay.inspect_sqlite_binding(library, replay.sha(library), consumer, consumer_sha)
