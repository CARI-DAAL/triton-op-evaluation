"""Pinned service-owned compatibility for CANN's existing SQLite dependency.

Only the synchronous engine profiler context derives a private loader prefix.
Neither the operator environment nor the six deployment anchors are rewritten.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_ENV = "TRITON_EVAL_PROFILER_RUNTIME_MANIFEST"
DIGEST_ENV = "TRITON_EVAL_PROFILER_RUNTIME_SHA256"
RUNTIME_ENV_KEYS = frozenset({MANIFEST_ENV, DIGEST_ENV})
_ENV_LOCK = threading.Lock()
SHA_PATTERN = r"^[0-9a-f]{64}$"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sqlite_binding(
    library: Path, digest: str, consumer: Path, consumer_digest: str
) -> dict[str, Any]:
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


class ProfilerRuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["profiler-runtime-v1"] = "profiler-runtime-v1"
    derivation: Literal["prepend-alias-parent-v1"] = "prepend-alias-parent-v1"
    library_path: Path
    library_sha256: str = Field(pattern=SHA_PATTERN)
    consumer_path: Path
    consumer_sha256: str = Field(pattern=SHA_PATTERN)
    msprof_path: Path
    msprof_sha256: str = Field(pattern=SHA_PATTERN)
    alias_path: Path
    alias_sha256: str = Field(pattern=SHA_PATTERN)
    elf_observations_path: Path
    elf_observations_sha256: str = Field(pattern=SHA_PATTERN)
    elf_abi: dict[str, str]
    required_sqlite_symbols: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_binding(self):
        for key in (
            "library_path",
            "consumer_path",
            "msprof_path",
            "alias_path",
            "elf_observations_path",
        ):
            value = getattr(self, key)
            if not value.is_absolute() or ".." in value.parts:
                raise ValueError("Profiler runtime dependency paths must be absolute")
        if self.alias_path.name != "libsqlite3.so" or self.alias_sha256 != self.library_sha256:
            raise ValueError("Profiler alias must contain the exact existing SQLite bytes")
        if set(self.elf_abi) != {"Class", "Data", "Machine"}:
            raise ValueError("Profiler runtime requires observed ELF class/data/machine")
        symbols = self.required_sqlite_symbols
        if sorted(set(symbols)) != symbols or any(
            not name.startswith("sqlite3_") for name in symbols
        ):
            raise ValueError("Required SQLite symbols must be a sorted unique observed set")
        return self


def prepare_runtime(
    output: Path,
    library: Path,
    library_sha256: str,
    consumer: Path,
    consumer_sha256: str,
    msprof: Path,
    msprof_sha256: str,
) -> Path:
    """Create a fresh immutable artifact, without installing or loading libraries."""
    binding = inspect_sqlite_binding(library, library_sha256, consumer, consumer_sha256)
    msprof = msprof.resolve(strict=True)
    if sha(msprof) != msprof_sha256:
        raise ValueError("msprof bytes differ from the explicit runtime pin")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "lib"
    directory.mkdir()
    alias = directory / "libsqlite3.so"
    shutil.copyfile(binding["library_path"], alias)
    if sha(alias) != library_sha256:
        raise RuntimeError("Profiler alias copy differs from the existing SQLite bytes")
    observations = output / "elf-observations.json"
    observations.write_text(json.dumps(binding["readelf_observations"], indent=2) + "\n")
    manifest = ProfilerRuntimeManifest(
        library_path=binding["library_path"],
        library_sha256=library_sha256,
        consumer_path=binding["consumer_path"],
        consumer_sha256=consumer_sha256,
        msprof_path=msprof,
        msprof_sha256=msprof_sha256,
        alias_path=alias,
        alias_sha256=library_sha256,
        elf_observations_path=observations,
        elf_observations_sha256=sha(observations),
        elf_abi=binding["elf_abi"],
        required_sqlite_symbols=binding["required_sqlite_symbols"],
    )
    target = output / "manifest.json"
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    for path in (alias, observations, target):
        path.chmod(0o444)
    directory.chmod(0o555)
    output.chmod(0o555)
    return target


def read_runtime(
    config: str, digest: str, check_resolution: bool = True
) -> ProfilerRuntimeManifest:
    target = Path(config)
    if not target.is_absolute() or sha(target) != digest:
        raise ValueError("Profiler runtime manifest differs from its operator pin")
    model = ProfilerRuntimeManifest.model_validate_json(target.read_bytes())
    if (
        model.alias_path != target.parent / "lib" / "libsqlite3.so"
        or model.elf_observations_path != target.parent / "elf-observations.json"
    ):
        raise ValueError("Profiler artifact paths must stay in the prepared manifest directory")
    for field in ("library", "consumer", "msprof", "alias", "elf_observations"):
        path = getattr(model, field + "_path")
        if not path.is_file() or sha(path) != getattr(model, field + "_sha256"):
            raise ValueError(f"Profiler runtime {field} differs from its pinned bytes")
    for path in (target, model.alias_path, model.elf_observations_path):
        if path.resolve() != path or stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(
                "Profiler manifest, alias and ELF observations must be read-only files"
            )
    for directory in {target.parent, model.alias_path.parent}:
        if stat.S_IMODE(directory.stat().st_mode) & 0o222:
            raise ValueError("Profiler artifact directories must be read-only")
    if check_resolution:
        resolved_msprof = shutil.which("msprof")
        if resolved_msprof is None or Path(resolved_msprof).resolve() != model.msprof_path:
            raise ValueError("Actual msprof resolution differs from the pinned CANN tool")
    return model


def load_runtime() -> ProfilerRuntimeManifest | None:
    config, digest = os.environ.get(MANIFEST_ENV), os.environ.get(DIGEST_ENV)
    if config is None and digest is None:
        return None
    if config is None or digest is None:
        raise ValueError("Profiler runtime requires its manifest and explicit SHA together")
    return read_runtime(config, digest)


def describe_runtime() -> dict[str, Any]:
    model = load_runtime()
    if model is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "manifest_path": os.environ[MANIFEST_ENV],
        "manifest_sha256": os.environ[DIGEST_ENV],
        **model.model_dump(mode="json"),
        "base_ld_sha256": hashlib.sha256(
            os.environ.get("LD_LIBRARY_PATH", "").encode()
        ).hexdigest(),
    }


def check_synchronous_handler(handler) -> None:
    parameter = inspect.signature(handler).parameters.get("async_mode")
    if parameter is None or parameter.default is not False:
        raise ValueError("Pinned profiler runtime requires the original synchronous trace callback")


@contextmanager
def profiler_export_runtime():
    """Hold the prefix through synchronous callback completion in one check child.

    The main-thread/nonblocking guard refuses concurrent or nested captures in
    this process. Normal service concurrency uses separate engine processes.
    """
    model = load_runtime()
    if model is None:
        yield {"enabled": False}
        return
    if threading.current_thread() is not threading.main_thread() or not _ENV_LOCK.acquire(False):
        raise RuntimeError(
            "Configured profiler runtime requires one main-thread capture per process"
        )
    previous = os.environ.get("LD_LIBRARY_PATH")
    config, digest = os.environ[MANIFEST_ENV], os.environ[DIGEST_ENV]
    evidence = {
        "enabled": True,
        "manifest_path": config,
        "manifest_sha256": digest,
        "helper_sha256": sha(Path(__file__)),
        "helper_path": str(Path(__file__).resolve()),
        "service_package_path": str(Path(__file__).resolve().parent),
        "alias_path": str(model.alias_path),
        "alias_sha256": model.alias_sha256,
        "library_path": str(model.library_path),
        "library_sha256": model.library_sha256,
        "consumer_path": str(model.consumer_path),
        "consumer_sha256": model.consumer_sha256,
        "msprof_path": str(model.msprof_path),
        "msprof_sha256": model.msprof_sha256,
        "process_id": os.getpid(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "physical_device": os.environ.get(
            "TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID", os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        ),
        "reference_device_policy": os.environ.get(
            "TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first"
        ),
        "main_thread": True,
        "previous_ld_sha256": hashlib.sha256((previous or "").encode()).hexdigest(),
        "callback_scope_completed": False,
    }
    try:
        os.environ["LD_LIBRARY_PATH"] = str(model.alias_path.parent) + (
            os.pathsep + previous if previous else ""
        )
        evidence["effective_ld_prefix"] = str(model.alias_path.parent)
        evidence["effective_ld_sha256"] = hashlib.sha256(
            os.environ["LD_LIBRARY_PATH"].encode()
        ).hexdigest()
        yield evidence
        read_runtime(config, digest)
        evidence["callback_scope_completed"] = True
    finally:
        if previous is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous
        evidence["base_ld_restored"] = os.environ.get("LD_LIBRARY_PATH") == previous
        _ENV_LOCK.release()
