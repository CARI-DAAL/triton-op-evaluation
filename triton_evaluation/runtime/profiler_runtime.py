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
import signal
import stat
import subprocess
import sys
import tempfile
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
CANN_ENV_KEYS = frozenset(
    {
        "ASCEND_AICPU_PATH",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_TOOLKIT_HOME",
        "TOOLCHAIN_HOME",
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    }
)
CANN_PATH_KEYS = frozenset({"PATH", "LD_LIBRARY_PATH", "PYTHONPATH"})
CANN_REQUIRED_LIBRARIES = (
    "lib64/libascendcl.so",
    "lib64/libruntime.so",
    "lib64/libacl_op_compiler.so",
    "lib64/libnnopbase.so",
)


class CannRuntimeEnvironment(BaseModel):
    """Observed output of the pinned vendor script, not arbitrary overlay input."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["cann-runtime-environment-v1"] = "cann-runtime-environment-v1"
    cann_root: Path
    set_env_path: Path
    set_env_sha256: str = Field(pattern=SHA_PATTERN)
    environment: dict[str, str]
    directories: dict[str, dict[str, Any]]
    runtime_files: dict[str, str]
    preparation: dict[str, Any]

    @model_validator(mode="after")
    def check_environment(self):
        root = self.cann_root
        if (
            not root.is_absolute()
            or ".." in root.parts
            or not re.fullmatch(r"cann[-_][0-9][A-Za-z0-9._-]*", root.name)
            or self.set_env_path != root / "set_env.sh"
        ):
            raise ValueError("CANN environment requires its canonical versioned vendor root")
        if self.environment.keys() != CANN_ENV_KEYS:
            raise ValueError("CANN environment must contain exactly the fixed eight vendor keys")
        for key in CANN_ENV_KEYS - CANN_PATH_KEYS:
            suffix = {"ASCEND_OPP_PATH": "opp", "TOOLCHAIN_HOME": "toolkit"}.get(key, "")
            if self.environment[key] != str(root / suffix):
                raise ValueError("CANN scalar environment differs from its pinned vendor root")
        driver = root.parent / "driver"
        for key in CANN_PATH_KEYS:
            for item in self.environment[key].split(os.pathsep):
                if not item:
                    continue
                path = Path(item)
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or not (
                        path == root
                        or root in path.parents
                        or path == driver
                        or driver in path.parents
                        or path
                        in {Path("/usr/bin"), Path("/usr/sbin"), Path("/bin"), Path("/sbin")}
                    )
                ):
                    raise ValueError("CANN script produced a path outside its fixed runtime roots")
        for path, digest in self.runtime_files.items():
            if not Path(path).is_absolute() or not re.fullmatch(SHA_PATTERN, digest):
                raise ValueError("CANN runtime file requires an absolute path and SHA pin")
        for relative in CANN_REQUIRED_LIBRARIES:
            library = (root / relative).resolve(strict=True)
            if root not in library.parents or str(library) not in self.runtime_files:
                raise ValueError("CANN closure must pin every fixed required runtime library")
        return self


def _cann_directory_requirements(root: Path) -> dict[Path, bool]:
    required = (
        "lib64",
        "opp",
        "tools/ccec_compiler/bin",
        "tools/profiler/bin",
        "python/site-packages",
        "opp/built-in/op_impl/ai_core/tbe",
        "toolkit",
    )
    optional = (
        "lib64/plugin/opskernel",
        "lib64/plugin/nnengine",
        "devlib",
        "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
        "tools/ascend_system_advisor/asys",
        "tools/show_kernel_debug_data",
        "tools/msobjdump",
    )
    result = {root / path: True for path in required}
    result.update({root / path: False for path in optional})
    result.update(
        {root.parent / "driver" / path: True for path in ("lib64", "lib64/common", "lib64/driver")}
    )
    return result


def capture_cann_environment(
    source: Path, digest: str, msprof: Path, consumer: Path
) -> CannRuntimeEnvironment:
    """Trusted preparation only: source existing bytes in a clean, narrow environment."""
    source = source.resolve(strict=True)
    root = source.parent
    if source.name != "set_env.sh" or sha(source) != digest:
        raise ValueError("CANN set_env source differs from its explicit pin")
    if (
        msprof != root / "tools/profiler/bin/msprof"
        or consumer != root / "tools/profiler/profiler_tool/analysis/lib64/msprof_analysis.so"
    ):
        raise ValueError("CANN set_env, msprof and consumer must use one canonical vendor root")
    directories = {}
    for path, required in _cann_directory_requirements(root).items():
        exists = path.is_dir()
        if required and not exists:
            raise ValueError(f"CANN required runtime directory is absent: {path}")
        if exists and not (
            root in path.resolve().parents
            or root.parent / "driver" in path.resolve().parents
            or path.resolve() == root.parent / "driver"
        ):
            raise ValueError("CANN directory escapes the fixed vendor/driver roots")
        directories[str(path)] = {
            "required": required,
            "exists": exists,
            "resolved_path": str(path.resolve()) if exists else None,
        }
    clean = {
        "PATH": "/usr/bin:/bin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONPATH": "",
        "LD_LIBRARY_PATH": os.pathsep.join(
            str(root.parent / "driver" / path) for path in ("lib64", "lib64/common", "lib64/driver")
        ),
    }
    command = [
        "/usr/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        'set +u; source "$1" >&2 || exit $?; /usr/bin/env -0',
        "cann-runtime-prepare",
        str(source),
    ]
    observed = subprocess.run(command, env=clean, capture_output=True, timeout=30, check=True)
    values = dict(
        item.decode().split("=", 1) for item in observed.stdout.split(b"\0") if b"=" in item
    )
    environment = {key: values[key] for key in sorted(CANN_ENV_KEYS) if key in values}
    files = {str(source): digest}
    for relative in (
        "version.info",
        "version.cfg",
        "compiler/version.info",
        "opp/version.info",
        "lib64/libascendcl.so",
        "lib64/libruntime.so",
        "lib64/libascend_runtime.so",
        "lib64/libacl_op_compiler.so",
        "lib64/libnnopbase.so",
    ):
        candidate = root / relative
        if candidate.is_file():
            actual = candidate.resolve(strict=True)
            if root not in actual.parents:
                raise ValueError("CANN runtime dependency escapes the canonical vendor root")
            files[str(actual)] = sha(actual)
    if sha(source) != digest:
        raise ValueError("CANN set_env source changed during preparation")
    return CannRuntimeEnvironment(
        cann_root=root,
        set_env_path=source,
        set_env_sha256=digest,
        environment=environment,
        directories=directories,
        runtime_files=files,
        preparation={
            "command": command,
            "returncode": observed.returncode,
            "clean_environment": clean,
            "stderr_sha256": hashlib.sha256(observed.stderr).hexdigest(),
            "imports_npu": False,
        },
    )


def read_cann_environment(model: ProfilerRuntimeManifest) -> CannRuntimeEnvironment | None:
    if model.cann_environment_path is None:
        return None
    if sha(model.cann_environment_path) != model.cann_environment_sha256:
        raise ValueError("CANN environment artifact differs from its pinned bytes")
    environment = CannRuntimeEnvironment.model_validate_json(
        model.cann_environment_path.read_bytes()
    )
    root = environment.cann_root
    if (
        model.cann_set_env_path != environment.set_env_path
        or model.cann_set_env_sha256 != environment.set_env_sha256
        or model.msprof_path != root / "tools/profiler/bin/msprof"
        or model.consumer_path
        != root / "tools/profiler/profiler_tool/analysis/lib64/msprof_analysis.so"
    ):
        raise ValueError("CANN environment and profiler dependency roots/pins differ")
    expected_dirs = _cann_directory_requirements(root)
    if environment.directories.keys() != {str(path) for path in expected_dirs}:
        raise ValueError("CANN directory observations do not cover the fixed runtime set")
    for path, required in expected_dirs.items():
        observed = environment.directories[str(path)]
        if observed.get("required") is not required or (required and not path.is_dir()):
            raise ValueError(f"CANN required runtime directory is absent or mismatched: {path}")
        if path.is_dir() != observed.get("exists") or (
            path.is_dir() and str(path.resolve()) != observed.get("resolved_path")
        ):
            raise ValueError("CANN observed directory conditions changed")
    if environment.runtime_files.get(str(environment.set_env_path)) != environment.set_env_sha256:
        raise ValueError("CANN runtime closure must pin its vendor source")
    for path, digest in environment.runtime_files.items():
        candidate = Path(path)
        if (
            candidate.resolve(strict=True) != candidate
            or root not in candidate.parents
            or sha(candidate) != digest
        ):
            raise ValueError("CANN runtime dependency differs from its canonical pinned bytes")
    return environment


def derive_cann_environment(
    model: ProfilerRuntimeManifest, original: dict[str, str]
) -> dict[str, str]:
    """Use only pinned vendor observations; retain non-CANN anchored entries."""
    cann = read_cann_environment(model)
    if cann is None:
        return {
            "PATH": str(model.msprof_path.parent)
            + (os.pathsep + original["PATH"] if original.get("PATH") else "")
        }
    root = cann.cann_root

    def is_cann_entry(item: str) -> bool:
        path = Path(item)
        try:
            first = path.relative_to(root.parent).parts[0]
        except (ValueError, IndexError):
            return False
        return first == "cann" or first.startswith(("cann-", "cann_"))

    effective = dict(cann.environment)
    for key in CANN_PATH_KEYS:
        # Empty path components expose cwd; the clean captured vendor prefix
        # has no user paths and the anchored non-CANN suffix retains its order.
        captured = [part for part in effective[key].split(os.pathsep) if part]
        retained = [
            part
            for part in original.get(key, "").split(os.pathsep)
            if part and not is_cann_entry(part)
        ]
        effective[key] = os.pathsep.join(dict.fromkeys([*captured, *retained]))
    return effective


def actual_cann_environment(model: ProfilerRuntimeManifest) -> dict[str, Any] | None:
    cann = read_cann_environment(model)
    if cann is None:
        return None
    actual = {key: os.environ.get(key, "") for key in sorted(CANN_ENV_KEYS)}
    for key in CANN_ENV_KEYS - CANN_PATH_KEYS:
        if actual[key] != cann.environment[key]:
            raise ValueError("Actual child CANN scalar differs from the pinned vendor environment")
    for key in CANN_PATH_KEYS:
        expected = [item for item in cann.environment[key].split(os.pathsep) if item]
        observed = [item for item in actual[key].split(os.pathsep) if item]
        if any(item not in observed for item in expected):
            raise ValueError("Actual child CANN paths omit the pinned vendor environment")
        # No deleted/other version root may survive the derived fixed mapping.
        for item in observed:
            try:
                first = Path(item).relative_to(cann.cann_root.parent).parts[0]
            except (ValueError, IndexError):
                continue
            if (
                first == "cann" or first.startswith(("cann-", "cann_"))
            ) and first != cann.cann_root.name:
                raise ValueError("Actual child CANN paths still contain another version root")
    return {
        "cann_root": str(cann.cann_root),
        "cann_set_env_path": str(cann.set_env_path),
        "cann_set_env_sha256": cann.set_env_sha256,
        "cann_environment_path": str(model.cann_environment_path),
        "cann_environment_sha256": model.cann_environment_sha256,
        "effective_environment": actual,
        "effective_environment_sha256": hashlib.sha256(
            json.dumps(actual, sort_keys=True).encode()
        ).hexdigest(),
        "directories": cann.directories,
        "runtime_files": cann.runtime_files,
    }


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


def probe_alias_binding(
    consumer: Path, symbol: str, alias: Path, environment: dict | None = None
) -> str:
    """Preflight eager consumer relocation and provider dladdr in a fresh child.

    This does not prove the binding of every subsequent vendor subprocess.
    """
    consumer = consumer.resolve(strict=True)
    alias = alias.resolve(strict=True)
    child_env = dict(os.environ if environment is None else environment)
    child_env.update(LD_BIND_NOW="1", LD_DEBUG="bindings", LC_ALL="C")
    child_env.pop("LD_DEBUG_OUTPUT", None)
    code = r"""
import ctypes, json, os, sys
consumer = ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW)
print("TRITON_EVAL_RELOCATIONS_COMPLETE", file=sys.stderr, flush=True)
try:
    function = getattr(ctypes.CDLL(None), sys.argv[2])
except AttributeError:
    function = getattr(consumer, sys.argv[2])
address = ctypes.cast(function, ctypes.c_void_p)
class Info(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
try:
    loader = ctypes.CDLL("libdl.so.2")
except OSError:
    loader = ctypes.CDLL(None)
loader.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
loader.dladdr.restype = ctypes.c_int
info = Info()
if not loader.dladdr(address, ctypes.byref(info)) or not info.name:
    raise RuntimeError("dladdr could not identify the consumer symbol provider")
print(json.dumps(os.path.realpath(os.fsdecode(info.name))))
"""
    try:
        command = [sys.executable, "-c", code, str(consumer), symbol]
        # Files avoid inherited pipe descriptors keeping communicate() alive
        # after a constructor forks. Own the whole group, even on success.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command, env=child_env, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                process.wait(timeout=30)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            result = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout.read().decode(errors="replace"),
                stderr.read().decode(errors="replace"),
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "SQLite alias binding probe timed out after 30 seconds; check consumer dependencies"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"SQLite alias binding probe could not execute ({type(error).__name__}); "
            "check the interpreter and pinned consumer dependencies"
        ) from error
    if result.returncode:
        diagnostic = "\n".join(
            line for line in result.stderr.splitlines() if not re.match(r"\s*\d+:", line)
        )[-1000:]
        raise RuntimeError(
            "SQLite alias binding probe failed; check pinned consumer dependencies: " + diagnostic
        )
    try:
        provider = Path(json.loads(result.stdout.strip())).resolve(strict=True)
    except (ValueError, OSError) as error:
        raise RuntimeError(
            "SQLite alias binding probe produced invalid provider evidence"
        ) from error
    bindings = re.findall(
        r"binding file (.*?) \[\d+\] to (.*?) \[\d+\]:.*?symbol [`']" + re.escape(symbol) + r"'",
        result.stderr.split("TRITON_EVAL_RELOCATIONS_COMPLETE", 1)[0],
    )
    providers = [
        Path(target).resolve() for source, target in bindings if Path(source).resolve() == consumer
    ]
    if provider != alias or not providers or any(target != alias for target in providers):
        raise RuntimeError(
            f"Pinned SQLite alias bypassed or consumer relocation unverified: consumer={consumer}, "
            f"symbol={symbol}, expected={alias}, lookup={provider}, relocations={providers}. "
            "Check competing RPATH/preloads and glibc loader support before capturing."
        )
    return str(provider)


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
    cann_set_env_path: Path | None = None
    cann_set_env_sha256: str | None = Field(default=None, pattern=SHA_PATTERN)
    cann_environment_path: Path | None = None
    cann_environment_sha256: str | None = Field(default=None, pattern=SHA_PATTERN)

    @model_validator(mode="after")
    def check_binding(self):
        quartet = [
            self.cann_set_env_path,
            self.cann_set_env_sha256,
            self.cann_environment_path,
            self.cann_environment_sha256,
        ]
        if any(value is not None for value in quartet) and any(value is None for value in quartet):
            raise ValueError("CANN source and environment artifact require all four pins together")
        for key in (
            "library_path",
            "consumer_path",
            "msprof_path",
            "alias_path",
            "elf_observations_path",
            "cann_set_env_path",
            "cann_environment_path",
        ):
            value = getattr(self, key)
            if value is None:
                continue
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
    cann_set_env: Path | None = None,
    cann_set_env_sha256: str | None = None,
) -> Path:
    """Create a fresh immutable artifact, without installing or loading libraries."""
    binding = inspect_sqlite_binding(library, library_sha256, consumer, consumer_sha256)
    msprof = msprof.resolve(strict=True)
    if sha(msprof) != msprof_sha256:
        raise ValueError("msprof bytes differ from the explicit runtime pin")
    if (cann_set_env is None) != (cann_set_env_sha256 is None):
        raise ValueError("CANN source requires its path and explicit SHA together")
    cann = None
    if cann_set_env is not None and cann_set_env_sha256 is not None:
        cann = capture_cann_environment(
            cann_set_env, cann_set_env_sha256, msprof, Path(binding["consumer_path"])
        )
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
    cann_artifact = None
    if cann is not None:
        cann_artifact = output / "cann-environment.json"
        cann_artifact.write_text(cann.model_dump_json(indent=2) + "\n")
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
        cann_set_env_path=cann.set_env_path if cann is not None else None,
        cann_set_env_sha256=cann.set_env_sha256 if cann is not None else None,
        cann_environment_path=cann_artifact,
        cann_environment_sha256=sha(cann_artifact) if cann_artifact is not None else None,
    )
    target = output / "manifest.json"
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    for path in (alias, observations, target):
        path.chmod(0o444)
    if cann_artifact is not None:
        cann_artifact.chmod(0o444)
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
        or (
            model.cann_environment_path is not None
            and model.cann_environment_path != target.parent / "cann-environment.json"
        )
    ):
        raise ValueError("Profiler artifact paths must stay in the prepared manifest directory")
    for field in ("library", "consumer", "msprof", "alias", "elf_observations"):
        path = getattr(model, field + "_path")
        if not path.is_file() or sha(path) != getattr(model, field + "_sha256"):
            raise ValueError(f"Profiler runtime {field} differs from its pinned bytes")
    for field in ("cann_set_env", "cann_environment"):
        path = getattr(model, field + "_path")
        if path is not None and (
            not path.is_file() or sha(path) != getattr(model, field + "_sha256")
        ):
            raise ValueError(f"Profiler runtime {field} differs from its pinned bytes")
    artifacts = [target, model.alias_path, model.elf_observations_path]
    if model.cann_environment_path is not None:
        artifacts.append(model.cann_environment_path)
    for path in artifacts:
        if path.resolve() != path or stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(
                "Profiler manifest, alias and ELF observations must be read-only files"
            )
    for directory in {target.parent, model.alias_path.parent}:
        if stat.S_IMODE(directory.stat().st_mode) & 0o222:
            raise ValueError("Profiler artifact directories must be read-only")
    read_cann_environment(model)
    if check_resolution:
        resolved_msprof = shutil.which("msprof")
        if resolved_msprof is None or Path(resolved_msprof).resolve() != model.msprof_path:
            raise ValueError("Actual msprof resolution differs from the pinned CANN tool")
        actual_cann_environment(model)
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
    selected = shutil.which("msprof")
    if selected is None:
        raise ValueError("Actual msprof disappeared from the pinned runtime PATH")
    resolved = Path(selected).resolve(strict=True)
    return {
        "enabled": True,
        "manifest_path": os.environ[MANIFEST_ENV],
        "manifest_sha256": os.environ[DIGEST_ENV],
        "resolved_msprof_path": str(resolved),
        "resolved_msprof_sha256": sha(resolved),
        "cann_runtime": actual_cann_environment(model),
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
    previous_preload = os.environ.get("LD_PRELOAD")
    config, digest = os.environ[MANIFEST_ENV], os.environ[DIGEST_ENV]
    evidence = {}
    try:
        evidence.update(
            {
                "enabled": True,
                "manifest_path": config,
                "manifest_sha256": digest,
                "helper_sha256": sha(Path(__file__)),
                "helper_path": str(Path(__file__).resolve()),
                "service_package_path": str(Path(__file__).resolve().parents[1]),
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
                    "TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID",
                    os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                ),
                "reference_device_policy": os.environ.get(
                    "TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first"
                ),
                "main_thread": True,
                "cann_runtime": actual_cann_environment(model),
                "previous_ld_sha256": hashlib.sha256((previous or "").encode()).hexdigest(),
                "callback_scope_completed": False,
                "previous_preload_sha256": hashlib.sha256(
                    (previous_preload or "").encode()
                ).hexdigest(),
            }
        )
        os.environ["LD_LIBRARY_PATH"] = str(model.alias_path.parent) + (
            os.pathsep + previous if previous else ""
        )
        evidence["effective_ld_prefix"] = str(model.alias_path.parent)
        evidence["effective_ld_sha256"] = hashlib.sha256(
            os.environ["LD_LIBRARY_PATH"].encode()
        ).hexdigest()
        os.environ["LD_PRELOAD"] = str(model.alias_path) + (
            " " + previous_preload if previous_preload else ""
        )
        evidence["effective_preload"] = os.environ["LD_PRELOAD"]
        evidence["effective_preload_sha256"] = hashlib.sha256(
            os.environ["LD_PRELOAD"].encode()
        ).hexdigest()
        probe_symbol = model.required_sqlite_symbols[0]
        provider = probe_alias_binding(model.consumer_path, probe_symbol, model.alias_path)
        evidence["alias_binding"] = {
            "consumer_path": str(model.consumer_path),
            "probe_symbol": probe_symbol,
            "provider_path": provider,
            "alias_path": str(model.alias_path),
            "verified": True,
        }
        yield evidence
        read_runtime(config, digest)
        evidence["callback_scope_completed"] = True
    finally:
        if previous is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous
        if previous_preload is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = previous_preload
        evidence["base_preload_restored"] = os.environ.get("LD_PRELOAD") == previous_preload
        evidence["base_ld_restored"] = os.environ.get("LD_LIBRARY_PATH") == previous
        _ENV_LOCK.release()
