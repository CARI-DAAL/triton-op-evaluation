"""Opt-in, fail-closed Linux boundary for each engine invocation.

The manifest is operator-owned deployment configuration, never candidate input.
It lists only immutable runtime dependencies. The engine receives its own attempt
directory, its assigned devices and a fresh process/network namespace.
"""

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..reference_cache import CACHE_ENV_KEYS, DEVICE_ENV, STATE_ENV

CONFIG_ENV = "TRITON_EVAL_ENGINE_SANDBOX_CONFIG"
DIGEST_ENV = "TRITON_EVAL_ENGINE_SANDBOX_SHA256"
RUNTIME_ENV = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_AICPU_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_CUSTOM_OPP_PATH",
        "ASCEND_GLOBAL_LOG_LEVEL",
        "ASCEND_SLOG_PRINT_TO_STDOUT",
        "TOOLCHAIN_HOME",
        "TBE_IMPL_PATH",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        *CACHE_ENV_KEYS,
        "TRITON_EVAL_REFERENCE_DEVICE_POLICY",
    }
)


class EngineSandbox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    readonly_mounts: list[tuple[Path, Path]]
    environment: dict[str, str] = Field(default_factory=dict)
    shared_npu_devices: list[Path] = Field(default_factory=list)
    executable: Path = Path("/usr/bin/bwrap")
    reference_cache_state_dir: Path | None = None

    @model_validator(mode="after")
    def check_manifest(self):
        if not self.executable.is_absolute():
            raise ValueError("sandbox executable must be absolute")
        if self.environment.keys() - RUNTIME_ENV:
            raise ValueError("sandbox environment must contain only runtime variables")
        forbidden = {Path(p) for p in ("/", "/home", "/root", "/tmp", "/run", "/dev", "/proc")}
        for source, destination in self.readonly_mounts:
            if not source.is_absolute() or not destination.is_absolute():
                raise ValueError("runtime mounts must be absolute")
            if ".." in source.parts or ".." in destination.parts:
                raise ValueError("runtime mounts must not contain parent traversal")
            if source.resolve() in forbidden or destination in forbidden:
                raise ValueError("whole host state directories are not runtime mounts")
        cache_keys = self.environment.keys() & CACHE_ENV_KEYS
        if cache_keys and cache_keys != CACHE_ENV_KEYS:
            raise ValueError("sandbox requires every reference cache deployment variable")
        if cache_keys:
            if self.reference_cache_state_dir is None:
                raise ValueError(
                    "sandbox cache invalidation needs an explicit writable state mount"
                )
            state = self.reference_cache_state_dir
            if not state.is_absolute() or ".." in state.parts or state.resolve() in forbidden:
                raise ValueError("cache state mount must be an absolute dedicated directory")
            if self.environment[STATE_ENV] != str(state):
                raise ValueError("cache state environment must identify its declared state mount")
            if any(
                state == source or state in source.parents or source in state.parents
                for source, _ in self.readonly_mounts
            ):
                raise ValueError("cache state mount must not overlap immutable runtime mounts")
        elif self.reference_cache_state_dir is not None:
            raise ValueError("cache state mount requires cache configuration")
        for device in self.shared_npu_devices:
            if device not in {
                Path("/dev/davinci_manager"),
                Path("/dev/devmm_svm"),
                Path("/dev/hisi_hdc"),
            }:
                raise ValueError("only shared Ascend control devices may be configured")
        return self

    def wrap(self, command: list, cwd: Path, device_id: int | None) -> tuple[list, dict]:
        work = cwd.resolve(strict=True)
        if work in {Path(p) for p in ("/", "/home", "/root", "/tmp", "/run")}:
            raise ValueError("engine needs a dedicated attempt directory")
        args = [
            str(self.executable),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
        ]
        for source, destination in self.readonly_mounts:
            source = source.resolve(strict=True)
            if source == work or source in work.parents or work in source.parents:
                raise ValueError("runtime mounts must not expose attempt state")
            if destination == work or destination in work.parents or work in destination.parents:
                raise ValueError("runtime mounts must not overlay attempt state")
            args += ["--ro-bind", str(source), str(destination)]
        if self.reference_cache_state_dir is not None:
            state = self.reference_cache_state_dir.resolve(strict=True)
            if state == work or state in work.parents or work in state.parents:
                raise ValueError("cache state mount must not overlap attempt state")
            args += ["--bind", str(state), str(self.reference_cache_state_dir)]
        # Match host runtime aliases without exposing their containing directories.
        for alias in ("/bin", "/sbin", "/lib", "/lib64"):
            path = Path(alias)
            if path.is_symlink():
                args += ["--symlink", str(path.readlink()), alias]
        args += ["--proc", "/proc", "--dev", "/dev"]
        if device_id is not None:
            if device_id < 0:
                raise ValueError("NPU device id must be nonnegative")
            for device in [*self.shared_npu_devices, Path(f"/dev/davinci{device_id}")]:
                if not device.exists():
                    raise FileNotFoundError(device)
                args += ["--dev-bind", str(device), str(device)]
        home, cache = work / ".engine-home", work / ".engine-cache"
        home.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        args += ["--bind", str(work), str(work), "--chdir", str(work), "--clearenv"]
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            **self.environment,
            "HOME": str(home),
            "XDG_CACHE_HOME": str(cache),
            "TMPDIR": "/tmp",
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "ASCEND_PROCESS_LOG_PATH": str(home / "ascend-log"),
            # The worker identity and device node retain the physical assignment.
            # Ascend renumbers the sole device inside a fresh mount namespace to
            # logical zero; reusing a physical id here rejects every card > 0.
            "ASCEND_RT_VISIBLE_DEVICES": "" if device_id is None else "0",
            # Preserve the host physical assignment while the runtime uses logical zero.
            DEVICE_ENV: "" if device_id is None else str(device_id),
        }
        for key, value in sorted(env.items()):
            args += ["--setenv", key, value]
        return [*args, "--remount-ro", "/", "--", *command], {"PATH": "/usr/bin:/bin"}


def isolate_command(command: list, cwd, device_id: int | None, env: dict) -> tuple[list, dict]:
    config = os.environ.get(CONFIG_ENV)
    expected_digest = os.environ.get(DIGEST_ENV)
    if config is None:
        if expected_digest is not None:
            raise ValueError("sandbox manifest path is required with its digest")
        # Existing deployments remain explicitly host-executed until configured.
        return command, env
    if expected_digest is None:
        raise ValueError("sandbox manifest digest is required for isolated execution")
    manifest = Path(config).read_bytes()
    if hashlib.sha256(manifest).hexdigest() != expected_digest:
        raise ValueError("sandbox manifest digest differs from the pinned deployment")
    sandbox = EngineSandbox.model_validate_json(manifest)
    return sandbox.wrap(command, Path(cwd), device_id)
