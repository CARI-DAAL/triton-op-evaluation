"""Opt-in, fail-closed Linux boundary for each engine invocation.

The manifest is operator-owned deployment configuration, never candidate input.
It lists only immutable runtime dependencies. The engine receives its own attempt
directory, its assigned devices and a fresh process/network namespace.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .profiler_runtime import (
    CANN_ENV_KEYS,
    derive_cann_environment,
    read_cann_environment,
    read_runtime,
)
from .profiler_runtime import DIGEST_ENV as PROFILER_DIGEST_ENV
from .profiler_runtime import MANIFEST_ENV as PROFILER_MANIFEST_ENV
from .profiler_runtime import RUNTIME_ENV_KEYS as PROFILER_RUNTIME_ENV_KEYS
from .reference_cache import (
    BINDING_ENV,
    CACHE_ENV,
    CACHE_ENV_KEYS,
    DEVICE_ENV,
    STATE_ENV,
    CacheIndex,
    CacheIndexV2,
)
from .reference_cache import DIGEST_ENV as CACHE_DIGEST_ENV
from .reference_timing_reuse import POLICY_ENV as TIMING_REUSE_POLICY_ENV
from .reference_timing_reuse import POLICY_SHA_ENV as TIMING_REUSE_SHA_ENV
from .reference_timing_reuse import (
    POLICY_ENV_KEYS as TIMING_REUSE_ENV_KEYS,
)
from .reference_timing_reuse import TimingReusePolicy as ValidatedTimingReusePolicy

CONFIG_ENV = "TRITON_EVAL_ENGINE_SANDBOX_CONFIG"
DIGEST_ENV = "TRITON_EVAL_ENGINE_SANDBOX_SHA256"
OVERLAY_CONFIG_ENV = "TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_CONFIG"
OVERLAY_DIGEST_ENV = "TRITON_EVAL_ENGINE_RUNTIME_OVERLAY_SHA256"
OVERLAY_ENV_KEYS = frozenset({OVERLAY_CONFIG_ENV, OVERLAY_DIGEST_ENV})
REFERENCE_POLICY_ENV = "TRITON_EVAL_REFERENCE_DEVICE_POLICY"
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
        *PROFILER_RUNTIME_ENV_KEYS,
        *TIMING_REUSE_ENV_KEYS,
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
    # Dedicated writable state mount for the validated timing-reuse sentinel's
    # monotone drift markers; required exactly when the policy pair is set.
    timing_reuse_state_dir: Path | None = None

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
        profiler_keys = self.environment.keys() & PROFILER_RUNTIME_ENV_KEYS
        if profiler_keys and profiler_keys != PROFILER_RUNTIME_ENV_KEYS:
            raise ValueError("sandbox requires both pinned profiler runtime variables")
        if profiler_keys:
            config = self.environment[PROFILER_MANIFEST_ENV]
            model = read_runtime(
                config, self.environment[PROFILER_DIGEST_ENV], check_resolution=False
            )
            required = [
                # Preserve real 0555 artifact directory modes in the child;
                # file-only binds would create writable-looking 0755 parents.
                Path(config).parent,
                model.library_path,
                model.consumer_path,
                model.msprof_path,
            ]
            cann = read_cann_environment(model)
            if cann is not None:
                required += [cann.cann_root, cann.cann_root.parent / "driver"]
            for path in required:
                visible = any(
                    (path == destination or destination in path.parents)
                    and (source / path.relative_to(destination)).resolve() == path.resolve()
                    for source, destination in self.readonly_mounts
                )
                if not visible:
                    raise ValueError(
                        f"profiler runtime path needs an explicit read-only mount: {path}"
                    )
        cache_keys = self.environment.keys() & CACHE_ENV_KEYS
        if cache_keys and cache_keys != CACHE_ENV_KEYS:
            raise ValueError("sandbox requires every reference cache deployment variable")
        reuse_keys = self.environment.keys() & TIMING_REUSE_ENV_KEYS
        if reuse_keys and reuse_keys != TIMING_REUSE_ENV_KEYS:
            raise ValueError("sandbox requires both validated timing-reuse variables")
        if reuse_keys and self.timing_reuse_state_dir is None:
            raise ValueError(
                "sandbox timing reuse needs an explicit writable state mount"
            )
        if not reuse_keys and self.timing_reuse_state_dir is not None:
            raise ValueError("timing-reuse state mount requires the policy pair")
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

    def wrap(
        self, command: list[str], cwd: Path, device_id: int | None
    ) -> tuple[list[str], dict[str, str]]:
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
        # /proc and /dev are namespace foundations, not late mounts: bwrap
        # applies argv in order, so a /dev emitted after a /dev-subtree bind
        # (reference cache under /dev/shm) shadows that bind inside the child.
        args += ["--proc", "/proc", "--dev", "/dev"]
        if device_id is not None:
            if device_id < 0:
                raise ValueError("NPU device id must be nonnegative")
            for device in [*self.shared_npu_devices, Path(f"/dev/davinci{device_id}")]:
                if not device.exists():
                    raise FileNotFoundError(device)
                args += ["--dev-bind", str(device), str(device)]
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
        if self.timing_reuse_state_dir is not None:
            reuse_state = self.timing_reuse_state_dir.resolve(strict=True)
            if reuse_state == work or reuse_state in work.parents or work in reuse_state.parents:
                raise ValueError("timing-reuse state mount must not overlap attempt state")
            args += ["--bind", str(reuse_state), str(self.timing_reuse_state_dir)]
        # Match host runtime aliases without exposing their containing directories.
        for alias in ("/bin", "/sbin", "/lib", "/lib64"):
            path = Path(alias)
            if path.is_symlink():
                args += ["--symlink", str(path.readlink()), alias]
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


class EngineRuntimeOverlay(BaseModel):
    """Pinned sparse input; runtime/source mounts are derived by service code."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["engine-runtime-overlay-v1"] = "engine-runtime-overlay-v1"
    environment: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_environment(self):
        allowed = CACHE_ENV_KEYS | PROFILER_RUNTIME_ENV_KEYS | TIMING_REUSE_ENV_KEYS
        if self.environment.keys() - allowed:
            raise ValueError("runtime overlay permits only profiler and reference-cache variables")
        for group in (CACHE_ENV_KEYS, PROFILER_RUNTIME_ENV_KEYS, TIMING_REUSE_ENV_KEYS):
            present = self.environment.keys() & group
            if present and present != group:
                raise ValueError("runtime overlay requires complete profiler/cache variable groups")
        return self


def read_runtime_overlay(config: str, digest: str) -> EngineRuntimeOverlay:
    path = Path(config)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("runtime overlay must be an absolute service-owned file without aliases")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("runtime overlay differs from its explicit SHA pin")
    return EngineRuntimeOverlay.model_validate_json(raw)


def _covered(path: Path, mounts: list[tuple[Path, Path]]) -> bool:
    return any(
        (path == destination or destination in path.parents)
        and (source / path.relative_to(destination)).resolve() == path.resolve()
        for source, destination in mounts
    )


def apply_runtime_overlay(
    sandbox: EngineSandbox, overlay: EngineRuntimeOverlay
) -> tuple[EngineSandbox, dict[str, Any]]:
    """Derive narrow bindings without mutating the anchored base manifest."""
    environment = dict(sandbox.environment)
    for key, value in overlay.environment.items():
        if key in environment and environment[key] != value:
            raise ValueError("runtime overlay cannot override an existing anchored value")
        if key in os.environ and os.environ[key] != value:
            raise ValueError("runtime overlay differs from the worker's declared runtime")
        environment[key] = value
    # The immutable worker anchor retains its original value. Benchmark's
    # cache binding reads it from the child environment (verify has a CLI
    # argument), so --clearenv must not silently select a different default.
    policy = os.environ.get(
        REFERENCE_POLICY_ENV, environment.get(REFERENCE_POLICY_ENV, "cpu-first")
    )
    if policy not in {"cpu-first", "npu-strict"}:
        raise ValueError("runtime overlay requires the existing valid reference policy anchor")
    if REFERENCE_POLICY_ENV in environment and environment[REFERENCE_POLICY_ENV] != policy:
        raise ValueError("runtime overlay cannot override an existing reference policy anchor")
    environment[REFERENCE_POLICY_ENV] = policy
    mounts = list(sandbox.readonly_mounts)
    required: list[Path] = []
    cann_evidence: dict[str, Any] | None = None
    if overlay.environment.keys() & PROFILER_RUNTIME_ENV_KEYS:
        config = environment[PROFILER_MANIFEST_ENV]
        runtime = read_runtime(config, environment[PROFILER_DIGEST_ENV], check_resolution=False)
        environment.update(derive_cann_environment(runtime, environment))
        cann = read_cann_environment(runtime)
        if cann is not None:
            required += [cann.cann_root, cann.cann_root.parent / "driver"]
            cann_evidence = {
                "cann_root": str(cann.cann_root),
                "cann_set_env_path": str(runtime.cann_set_env_path),
                "cann_set_env_sha256": runtime.cann_set_env_sha256,
                "cann_environment_path": str(runtime.cann_environment_path),
                "cann_environment_sha256": runtime.cann_environment_sha256,
                "directories": cann.directories,
            }
        required += [
            Path(config).parent,
            runtime.library_path,
            runtime.consumer_path,
            runtime.msprof_path,
        ]
    state = sandbox.reference_cache_state_dir
    reuse_state: Path | None = None
    if overlay.environment.keys() & TIMING_REUSE_ENV_KEYS:
        # The explicit policy pair travels through --clearenv only via this
        # pinned overlay: verify the artifact bytes, bind exactly what the
        # policy declares, and never inherit anything else from the parent.
        policy_path = Path(environment[TIMING_REUSE_POLICY_ENV])
        for path in (policy_path,):
            if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
                raise ValueError("timing-reuse policy path must be absolute without aliases")
        policy_bytes = policy_path.read_bytes()
        if hashlib.sha256(policy_bytes).hexdigest() != environment[TIMING_REUSE_SHA_ENV]:
            raise ValueError("timing-reuse policy differs from its explicit SHA pin")
        policy = ValidatedTimingReusePolicy.model_validate_json(policy_bytes)
        reuse_state = Path(policy.source_state_dir)
        declared = {
            "cache_dir": Path(policy.source_cache_dir),
            "index_path": Path(policy.source_index_path),
            "state_dir": reuse_state,
        }
        for name, path in declared.items():
            if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
                raise ValueError(f"timing-reuse policy {name} must be absolute without aliases")
        index_raw = declared["index_path"].read_bytes()
        if hashlib.sha256(index_raw).hexdigest() != policy.source_index_sha256:
            raise ValueError("timing-reuse source index differs from the policy declaration")
        index = CacheIndex.model_validate_json(index_raw)
        if index.deployment.digest != policy.source_deployment_sha256:
            raise ValueError("timing-reuse source index deployment differs from the policy")
        if not declared["cache_dir"].is_dir():
            raise ValueError("timing-reuse cache root must be a dedicated artifact directory")
        for path in (declared["cache_dir"] / "cases", declared["cache_dir"] / "blobs"):
            if path.exists():
                if not path.is_dir() or path.resolve(strict=True) != path:
                    raise ValueError("timing-reuse cache namespaces must not be aliases")
                required.append(path)
        required.append(declared["index_path"])
        required.append(policy_path)
        if not reuse_state.is_dir() or any(
            path.is_symlink()
            or (
                path.is_dir()
                and (path.parent != reuse_state or not re.fullmatch(r"[0-9a-f]{64}", path.name))
            )
            or (
                path.is_file()
                and (
                    path.parent.parent != reuse_state
                    or not re.fullmatch(r"card-[0-9]+\.invalid\.json", path.name)
                )
            )
            for path in reuse_state.rglob("*")
        ):
            raise ValueError("runtime overlay requires a dedicated timing-reuse state directory")
        for config, digest in index.deployment.runtime_files.items():
            path = Path(config)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("timing-reuse cache dependency differs from its pin")
            required.append(path)
    if overlay.environment.keys() & CACHE_ENV_KEYS:
        cache_root = Path(environment[CACHE_ENV])
        index_path = Path(environment[BINDING_ENV])
        state = Path(environment[STATE_ENV])
        for path in (cache_root, index_path, state):
            if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
                raise ValueError("runtime overlay cache paths must be absolute without aliases")
        raw = index_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != environment[CACHE_DIGEST_ENV]:
            raise ValueError("runtime overlay cache index differs from its pin")
        # Model-bound (v2) indexes are pinned exactly like v1: same digest
        # pin, same deployment integrity; entries are only relocated by
        # NPU-model workload identity instead of the card index.
        if json.loads(raw).get("schema_version") == "reference-cache-index-v2":
            index = CacheIndexV2.model_validate_json(raw)
        else:
            index = CacheIndex.model_validate_json(raw)
        if not cache_root.is_dir():
            raise ValueError("runtime overlay cache root must be a dedicated artifact directory")
        # Expose only the two content-addressed artifact namespaces, rather
        # than the containing operator directory or its neighboring evidence.
        for path in (cache_root / "cases", cache_root / "blobs"):
            if path.exists():
                if not path.is_dir() or path.resolve(strict=True) != path:
                    raise ValueError(
                        "runtime overlay cache artifact namespaces must not be aliases"
                    )
                required.append(path)
        required.append(index_path)
        if not state.is_dir() or any(
            path.is_symlink()
            or (
                path.is_dir()
                and (path.parent != state or not re.fullmatch(r"[0-9a-f]{64}", path.name))
            )
            or (
                path.is_file()
                and (
                    path.parent.parent != state
                    or not re.fullmatch(r"card-[0-9]+\.invalid\.json", path.name)
                )
            )
            for path in state.rglob("*")
        ):
            raise ValueError("runtime overlay requires a dedicated reference-cache state directory")
        for config, digest in index.deployment.runtime_files.items():
            path = Path(config)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("runtime overlay cache dependency differs from its pin")
            required.append(path)
    # Source is the executing service package, never a caller-supplied path.
    package = Path(__file__).absolute().parents[1]
    if package.resolve(strict=True) != package:
        raise ValueError("runtime overlay service source must not traverse aliases")
    source_files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    if not source_files or any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError("runtime overlay needs factual service source without file aliases")
    required.append(package)
    # One tracked service-owned diagnostic is allowed alongside actual engine
    # scripts; its path is derived here and never supplied by the overlay.
    probe = package.parent / "scripts" / "probe_profiler_latency.py"
    if probe.is_file():
        if probe.resolve(strict=True) != probe:
            raise ValueError("service-owned profiler probe must not traverse aliases")
        required.append(probe)
    environment["PYTHONPATH"] = str(package.parent) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    if cann_evidence is not None:
        cann_evidence["effective_environment"] = {
            key: environment[key] for key in sorted(CANN_ENV_KEYS)
        }
        cann_evidence["effective_environment_sha256"] = hashlib.sha256(
            json.dumps(cann_evidence["effective_environment"], sort_keys=True).encode()
        ).hexdigest()
    for path in required:
        if not _covered(path, mounts):
            mounts.append((path, path))
    derived = EngineSandbox(
        readonly_mounts=mounts,
        environment=environment,
        shared_npu_devices=sandbox.shared_npu_devices,
        executable=sandbox.executable,
        reference_cache_state_dir=state,
        timing_reuse_state_dir=reuse_state,
    )
    return derived, {
        "service_package_path": str(package),
        "service_source_sha256": source_files,
        "effective_sandbox_sha256": hashlib.sha256(derived.model_dump_json().encode()).hexdigest(),
        "effective_environment_sha256": hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest(),
        "effective_environment_keys": sorted(environment),
        "readonly_mounts": [[str(a), str(b)] for a, b in derived.readonly_mounts],
        "reference_cache_state_dir": str(state) if state is not None else None,
        "timing_reuse_state_dir": str(reuse_state) if reuse_state is not None else None,
        "timing_reuse_policy_sha256": (
            environment[TIMING_REUSE_SHA_ENV]
            if TIMING_REUSE_SHA_ENV in environment
            else None
        ),
        "cann_runtime": cann_evidence,
    }


def isolate_command(
    command: list[str], cwd: Path, device_id: int | None, env: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    config = os.environ.get(CONFIG_ENV)
    expected_digest = os.environ.get(DIGEST_ENV)
    if config is None:
        if expected_digest is not None or any(key in os.environ for key in OVERLAY_ENV_KEYS):
            raise ValueError("sandbox manifest path is required with its digest")
        # Existing deployments remain explicitly host-executed until configured.
        return command, env
    if expected_digest is None:
        raise ValueError("sandbox manifest digest is required for isolated execution")
    manifest = Path(config).read_bytes()
    if hashlib.sha256(manifest).hexdigest() != expected_digest:
        raise ValueError("sandbox manifest digest differs from the pinned deployment")
    sandbox = EngineSandbox.model_validate_json(manifest)
    overlay_path, overlay_digest = (
        os.environ.get(OVERLAY_CONFIG_ENV),
        os.environ.get(OVERLAY_DIGEST_ENV),
    )
    if overlay_path is None and overlay_digest is None:
        if any(
            key in os.environ and sandbox.environment.get(key) != os.environ[key]
            for key in CACHE_ENV_KEYS | PROFILER_RUNTIME_ENV_KEYS | TIMING_REUSE_ENV_KEYS
        ):
            raise ValueError("declared runtime needs a pinned engine overlay before --clearenv")
        return sandbox.wrap(command, Path(cwd), device_id)
    if overlay_path is None or overlay_digest is None:
        raise ValueError("runtime overlay requires config and SHA together")
    overlay = read_runtime_overlay(overlay_path, overlay_digest)
    if any(
        key in os.environ
        and key not in overlay.environment
        and sandbox.environment.get(key) != os.environ[key]
        for key in CACHE_ENV_KEYS | PROFILER_RUNTIME_ENV_KEYS | TIMING_REUSE_ENV_KEYS
    ):
        raise ValueError("runtime overlay must include every declared worker runtime group")
    sandbox, evidence = apply_runtime_overlay(sandbox, overlay)
    package = Path(evidence["service_package_path"])
    configured_engine = os.environ.get("TRITON_EVAL_ENGINE_DIR")
    if configured_engine is not None and Path(configured_engine).resolve() != package / "engine":
        raise ValueError("runtime overlay refuses an engine directory from another service source")
    if len(command) > 1 and str(command[1]).endswith(".py"):
        script = Path(command[1]).resolve(strict=True)
        probe = package.parent / "scripts" / "probe_profiler_latency.py"
        attempt = Path(cwd).resolve(strict=True)
        if script.parent != package / "engine" and script != probe and script.parent != attempt:
            raise ValueError("runtime overlay refuses a script from another service source")
        if Path(command[1]).absolute() != script:
            raise ValueError("runtime overlay refuses a script reached through aliases")
        evidence["engine_script_path"] = str(script)
        evidence["engine_script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    wrapped, outer = sandbox.wrap(command, Path(cwd), device_id)
    evidence.update(
        schema_version="engine-runtime-boundary-v1",
        state="prepared",
        executed=False,
        base_manifest_path=config,
        base_manifest_sha256=expected_digest,
        overlay_path=overlay_path,
        overlay_sha256=overlay_digest,
        sandbox_helper_path=str(Path(__file__)),
        sandbox_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command_sha256=hashlib.sha256(json.dumps(wrapped).encode()).hexdigest(),
    )
    (Path(cwd) / f"engine-runtime-boundary-{time.time_ns()}.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    return wrapped, outer
