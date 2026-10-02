#!/usr/bin/env python3
"""Replay one retained profiler export without executing another NPU operation.

The captured report and every original artifact are SHA-pinned before copying.
Run the identical msprof TEXT export on that copy, leaving completion markers
and raw slices intact. Success only means CANN exported nonempty text rows;
it is never eight new reference measurements or device admission evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sqlite_binding(
    library: Path, digest: str, consumer: Path, consumer_digest: str
) -> dict:
    """Inspect actual ELF metadata and required symbols; never load either ELF."""
    library = library.resolve(strict=True)
    consumer = consumer.resolve(strict=True)
    if sha(library) != digest or sha(consumer) != consumer_digest:
        raise ValueError("SQLite library/consumer SHA does not match the explicit pin")

    def elf(path):
        observed = {}
        for name, flag in (("header", "-h"), ("dynamic", "-d"), ("symbols", "--dyn-syms")):
            result = subprocess.run(
                ["readelf", "--wide", flag, str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                env={**os.environ, "LC_ALL": "C"},
            )
            observed[name] = result.stdout
        return observed

    target = elf(library)
    source = elf(consumer)

    def abi(header):
        return {
            key: match.group(1).strip()
            for key in ("Class", "Data", "Machine")
            if (match := re.search(rf"^\s*{key}:\s*(.+)$", header, re.MULTILINE))
        }

    target_abi = abi(target["header"])
    if len(target_abi) != 3 or target_abi != abi(source["header"]):
        raise ValueError("SQLite target and CANN consumer ELF class/data/machine differ")
    if "Shared library: [libsqlite3.so]" not in source["dynamic"]:
        raise ValueError("Pinned CANN consumer does not require the missing libsqlite3.so")
    if "Library soname: [libsqlite3.so.0]" not in target["dynamic"]:
        raise ValueError("Expected the existing libsqlite3.so.0 library, without replacement")

    def sqlite_symbols(text, undefined):
        result = set()
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 8 and (fields[6] == "UND") == undefined:
                symbol = fields[7].split("@")[0]
                if symbol.startswith("sqlite3_"):
                    result.add(symbol)
        return result

    required = sqlite_symbols(source["symbols"], True)
    available = sqlite_symbols(target["symbols"], False)
    if not required or not required <= available:
        raise ValueError(
            f"Existing SQLite library lacks required symbols: {sorted(required - available)}"
        )
    return {
        "library_path": str(library),
        "library_sha256": digest,
        "consumer_path": str(consumer),
        "consumer_sha256": consumer_digest,
        "elf_abi": target_abi,
        "needed": "libsqlite3.so",
        "target_soname": "libsqlite3.so.0",
        "required_sqlite_symbols": sorted(required),
        "readelf_observations": {"library": target, "consumer": source},
    }


def verified_capture(report_path: Path, report_sha256: str) -> tuple[dict, list[tuple[Path, str]]]:
    if sha(report_path) != report_sha256:
        raise ValueError("Captured report SHA does not match the pinned attempt")
    report = json.loads(report_path.read_bytes())
    if report.get("schema_version") != "profiler-latency-probe-v1":
        raise ValueError("Expected an original tracked profiler probe report")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Captured report contains no immutable artifacts")
    pinned = []
    seen = set()
    for row in artifacts:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Invalid or duplicate relative artifact path")
        seen.add(str(relative))
        path = report_path.parent / relative
        if any(part.is_symlink() for part in [path, *path.parents] if part != Path("/")):
            raise ValueError("Captured artifact cannot resolve through a symlink")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Captured artifact does not match its pin: {relative}")
        pinned.append((Path(relative), row["sha256"]))
    actual = {
        str(path.relative_to(report_path.parent))
        for path in report_path.parent.rglob("*")
        if path.is_file() and path != report_path
    }
    if actual != seen:
        raise ValueError("Captured artifact file set differs from its pinned manifest")
    return report, pinned


def run_export(
    command: list[str],
    timeout_seconds: int,
    on_started: Callable[[int], None] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Own a separate process group; timeout is an executed, retained attempt."""
    started = time.monotonic_ns()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=env
    )
    signals = []
    timed_out = False
    try:
        if on_started is not None:
            on_started(process.pid)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        signals.append("SIGTERM")
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            signals.append("SIGKILL")
            stdout, stderr = process.communicate()
    return {
        "pid": process.pid,
        "process_group": process.pid,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "signals_sent": signals,
        "elapsed_ns": time.monotonic_ns() - started,
        "stdout": stdout,
        "stderr": stderr,
    }


def replay(
    capture_report: Path,
    report_sha256: str,
    output: Path,
    timeout_seconds: int = 45,
    sqlite_library: Path | None = None,
    sqlite_library_sha256: str | None = None,
    sqlite_consumer: Path | None = None,
    sqlite_consumer_sha256: str | None = None,
) -> int:
    if not 1 <= timeout_seconds <= 60:
        raise ValueError("Offline export timeout must be between one and sixty seconds")
    capture_report = capture_report.resolve()
    original, pinned = verified_capture(capture_report, report_sha256)
    sqlite_args = (sqlite_library, sqlite_library_sha256, sqlite_consumer, sqlite_consumer_sha256)
    sqlite = None
    if any(sqlite_args):
        if not all(sqlite_args):
            raise ValueError("SQLite diagnostic alias requires all four explicit source pins")
        sqlite = inspect_sqlite_binding(*sqlite_args)
    output = output.resolve()
    if output == capture_report.parent or capture_report.parent in output.parents:
        raise ValueError("Replay output must be outside the original capture")
    executable = shutil.which("msprof")
    if executable is None:
        raise RuntimeError("Source the factual CANN environment before offline replay")
    executable = Path(executable).resolve(strict=True)
    # Device info is nested below PROF_*; identify the actual capture by the
    # CANN directory component rather than inventing an output layout.
    prof_dirs = {
        Path(*relative.parts[: index + 1])
        for relative, _ in pinned
        for index, part in enumerate(relative.parts)
        if part.startswith("PROF_")
    }
    if len(prof_dirs) != 1:
        raise ValueError("Offline replay requires exactly one retained CANN capture")
    prof_dir = prof_dirs.pop()
    output.mkdir(parents=True, exist_ok=False)
    copy_root = output / "capture"
    for relative, digest in pinned:
        destination = copy_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(capture_report.parent / relative, destination)
        if sha(destination) != digest:
            raise RuntimeError("Copy did not preserve the pinned capture bytes")
    copied_prof = copy_root / prof_dir
    command = [str(executable), "--export=on", f"--output={copied_prof}"]
    evidence: dict[str, Any] = {
        "schema_version": "profiler-offline-export-replay-v1",
        "is_independent_sample": False,
        "admission_evidence": False,
        "executes_npu_operations": False,
        "state": "running",
        "original_report_path": str(capture_report),
        "original_report_sha256": report_sha256,
        "original_physical_device": original.get("physical_device"),
        "original_source_sha256": original.get("source_sha256"),
        "replay_source_sha256": {
            name: sha(ROOT / name)
            for name in (
                "scripts/replay_profiler_export.py",
                "triton_evaluation/engine/benchmark.py",
            )
        },
        "msprof_path": str(executable),
        "msprof_sha256": sha(executable),
        "command": command,
        "timeout_seconds": timeout_seconds,
        "started_unix_ns": time.time_ns(),
    }
    child_env = None
    if sqlite is not None:
        alias_dir = output / "private-library-alias"
        alias_dir.mkdir()
        alias = alias_dir / "libsqlite3.so"
        shutil.copy2(sqlite["library_path"], alias)
        if sha(alias) != sqlite["library_sha256"]:
            raise RuntimeError("Private SQLite alias copy does not match the pinned library")
        old_ld = os.environ.get("LD_LIBRARY_PATH")
        new_ld = str(alias_dir) + (os.pathsep + old_ld if old_ld else "")
        child_env = {**os.environ, "LD_LIBRARY_PATH": new_ld}
        evidence["sqlite_alias"] = {**sqlite, "private_alias_path": str(alias)}
        evidence["environment_diff"] = {
            "LD_LIBRARY_PATH": {
                "original_present": old_ld is not None,
                "original_sha256": hashlib.sha256((old_ld or "").encode()).hexdigest(),
                "child_sha256": hashlib.sha256(new_ld.encode()).hexdigest(),
                "child_prefix": str(alias_dir),
            }
        }
    else:
        evidence["environment_diff"] = {}
    target = output / "report.json"

    def persist() -> None:
        target.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")

    def started(pid: int) -> None:
        evidence.update(pid=pid, process_group=pid)
        persist()

    persist()
    try:
        result = run_export(command, timeout_seconds, on_started=started, env=child_env)
        (output / "export.stdout").write_bytes(result.pop("stdout"))
        (output / "export.stderr").write_bytes(result.pop("stderr"))
        evidence.update(result)
        evidence["export_completed"] = not result["timed_out"]
        csv_observations = []
        for path in sorted(copied_prof.rglob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = csv.reader(stream)
                header = next(rows, [])
                count = sum(1 for row in rows if row)
            csv_observations.append(
                {"path": str(path.relative_to(copy_root)), "header": header, "data_rows": count}
            )
        evidence["cann_csv_observations"] = csv_observations
        evidence["cann_text_data_present"] = any(row["data_rows"] for row in csv_observations)
        from triton_evaluation.engine.benchmark import parse_operator_latency

        operators, latency = parse_operator_latency(str(copy_root), 5, 20, keep_res=True)
        evidence["raw_latency"] = repr(latency)
        evidence["operators"] = operators
        valid = bool(
            not isinstance(latency, bool)
            and isinstance(latency, Real)
            and math.isfinite(latency)
            and latency > 0
        )
        evidence["latency_ms"] = float(latency) if valid else None
        # CANN export alone does not re-run torch_npu KernelViewParser, which
        # originally creates kernel_details.csv. Keep those distinct outcomes.
        evidence["kernel_details_parseable"] = valid
        evidence["state"] = (
            "exported_text_data"
            if result["returncode"] == 0 and evidence["cann_text_data_present"]
            else "empty_export"
            if result["returncode"] == 0
            else "export_failed"
        )
        if result["timed_out"]:
            evidence["state"] = "timeout"
    except Exception as error:
        evidence.update(state="error", error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        _, after = verified_capture(capture_report, report_sha256)
        evidence["original_artifacts_unchanged"] = after == pinned
        if sqlite is not None:
            unchanged = (
                sha(Path(sqlite["library_path"])) == sqlite["library_sha256"]
                and sha(Path(sqlite["consumer_path"])) == sqlite["consumer_sha256"]
            )
            evidence["sqlite_sources_unchanged"] = unchanged
            if not unchanged:
                evidence.update(state="error", error="Pinned SQLite source changed during replay")
        evidence["finished_unix_ns"] = time.time_ns()
        evidence["artifacts"] = [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != target
        ]
        persist()
    print(json.dumps({key: evidence.get(key) for key in ("state", "returncode", "latency_ms")}))
    return 0 if evidence["state"] == "exported_text_data" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", required=True, type=Path)
    parser.add_argument("--capture-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sqlite-library", type=Path)
    parser.add_argument("--sqlite-library-sha256")
    parser.add_argument("--sqlite-consumer", type=Path)
    parser.add_argument("--sqlite-consumer-sha256")
    args = parser.parse_args()
    return replay(
        args.capture_report,
        args.capture_report_sha256,
        args.output,
        sqlite_library=args.sqlite_library,
        sqlite_library_sha256=args.sqlite_library_sha256,
        sqlite_consumer=args.sqlite_consumer,
        sqlite_consumer_sha256=args.sqlite_consumer_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
