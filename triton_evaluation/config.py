import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The evaluation engine (verify / internal benchmark / profile / validate) is
# vendored inside the package: this service is self-contained. `engine_dir`
# defaults to the bundled engine; the env override exists only to point at a
# different engine build.
_DEFAULT_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")

# Whole-run engine budgets. The verify cap bounds the ENTIRE verify run (all
# cases), not one subject — 900s was too tight for multi-case/large-kernel
# workloads (see the K3 intermittent-failure report). The benchmark constant
# doubles as the Settings default; both are deployment-tunable via env.
VERIFY_BUDGET_CAP_SECONDS = 1500
BENCHMARK_TIMEOUT_SECONDS = 600

_EVALUATOR_BUILD_PATTERN = r"^(sha256:[0-9a-f]{64}|git:[0-9a-f]{40})$"
_ALGORITHM_VERSION_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"


class Settings(BaseSettings):
    """Service configuration. Fields are read from ``TRITON_EVAL_<FIELD>`` env vars."""

    model_config = SettingsConfigDict(env_prefix="TRITON_EVAL_", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    # Isolated durable state for the v2.1-only release.
    redis_prefix: str = Field(default="triton-eval:v2.1", min_length=1)
    # Change this namespace when the fixed hardware or timing runtime changes.
    baseline_cache_namespace: str = Field(default="torch-baseline-v1", min_length=1)
    engine_dir: str = _DEFAULT_ENGINE_DIR
    reference_device_policy: Literal["cpu-first", "npu-strict"] = "cpu-first"
    reference_diagnostics_dir: Path | None = None
    # Optional reference-cache settings are pinned as one deployment group.
    reference_cache_dir: Path | None = None
    reference_cache_binding: Path | None = None
    reference_cache_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_cache_state_dir: Path | None = None
    # Sampled-validated historical timing reuse (timing evidence only), pinned
    # as one artifact path + SHA256 pair like the reference cache anchors.
    reference_timing_reuse_policy: Path | None = None
    reference_timing_reuse_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    # Service-owned immutable loader compatibility.
    profiler_runtime_manifest: Path | None = None
    profiler_runtime_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    engine_runtime_overlay_config: Path | None = None
    engine_runtime_overlay_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Compile-only attempts use CPU capacity; mixed checks stay on one NPU worker.
    cpu_workers: int = Field(default=1, ge=0, le=32)
    api_poll_interval_seconds: float = Field(default=0.25, ge=0.05, le=5)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=11907, ge=1, le=65535)
    api_tls_port: int | None = Field(default=None, ge=1, le=65535)
    api_tls_host: str = "127.0.0.1"
    api_tls_certfile: Path | None = None
    api_tls_keyfile: Path | None = None

    # --- durable-core lifecycle (evaluation API v2) ---
    # Terminal resources and idempotency records live at least one day, and the
    # recovery window must fit inside the retention window.
    retention_seconds: int = Field(default=7 * 24 * 3600, ge=86400)
    max_recovery_age_seconds: int = Field(default=24 * 3600, gt=0)
    lease_seconds: int = Field(default=30, gt=0)
    heartbeat_interval_seconds: float = Field(default=10.0, gt=0)
    maintenance_interval_seconds: float = Field(default=1.0, ge=0.05, le=5)
    queue_poll_interval_seconds: float = Field(default=1.0, ge=0.05, le=5)

    # --- engine whole-run budgets (deployment-tunable) ---
    verify_budget_cap_seconds: int = Field(default=VERIFY_BUDGET_CAP_SECONDS, gt=60)
    benchmark_timeout_seconds: int = Field(default=BENCHMARK_TIMEOUT_SECONDS, ge=10)

    # --- NPU visibility hold ---
    # Megabytes of device memory each worker allocates at startup and keeps
    # for its lifetime so `npu-smi info` shows the worker even while idle
    # (without this, only the per-job engine subprocesses briefly appear).
    # 0 disables the hold.
    device_hold_mb: int = Field(default=64, ge=0)

    # --- admission quotas; trial and trusted buckets stay independent ---
    trial_rate_limit_per_minute: int = Field(default=60, gt=0)
    trusted_rate_limit_per_minute: int = Field(default=60, gt=0)
    trial_max_inflight: int = Field(default=4, gt=0)
    trusted_max_inflight: int = Field(default=4, gt=0)
    session_max_inflight: int = Field(default=2, gt=0)
    # Rollout bucket: server-declared RL traffic (routes._build_command maps
    # declared callers here). Contract-invisible; grade never lands here.
    rollout_rate_limit_per_minute: int = Field(default=600, gt=0)
    rollout_max_inflight: int = Field(default=256, gt=0)
    rollout_caller_ids: list[str] = Field(default_factory=list)

    # Card health loop (spec §10): poison quarantine + canary/drain/recovery.
    poison_death_threshold: int = Field(default=2, ge=2)
    card_affinity_avoidance: bool = Field(default=True)

    # Opt in to bounded retries when an engine failure produced no check data.
    # Missing output cannot establish whether a candidate ran or crashed.
    engine_failure_requeue: bool = Field(default=False)

    # Canary + recovery ladder (spec §10).
    canary_budget_seconds: int = Field(default=120, gt=0)
    canary_interval_seconds: float = Field(default=60.0, gt=0)
    card_reset_command: str = Field(default="", max_length=500)  # "{card}" substituted
    card_recovery_max_step: int = Field(default=2, ge=1, le=3)  # 1 cleanup, 2 reset, 3 restart
    card_recovery_retry_seconds: float = Field(default=300.0, gt=0)
    card_canary_on_startup: bool = Field(default=True)

    # Syntax/import verdict reuse scoped to candidate source and toolchain.
    # Backend JIT failures and completed evaluation results are never cached.
    verdict_cache_enabled: bool = Field(default=True)
    verdict_cache_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    # Worker precheck: subprocess syntax/import classification within an attempt.
    precheck_budget_seconds: int = Field(default=60, gt=0)
    cann_fingerprint: str = Field(default="", pattern=r"^[A-Za-z0-9._:-]{0,128}$")

    # --- deployment-attested worker environment ---
    worker_hardware_model: str = Field(default="Ascend910", min_length=1)
    worker_driver_version: str = Field(default="26.0.rc1", min_length=1)
    worker_runtime_version: str = Field(default="CANN-9.1.1", min_length=1)
    worker_toolchain_version: str = Field(
        default=(
            "CANN 9.1.1 / python 3.10 / torch 2.9.0 / torch_npu 2.9.0.post8 / triton_ascend 3.2.0"
        ),
        min_length=1,
    )

    # --- evaluator provenance pinned at admission ---
    evaluator_build: str = Field(default="git:" + "0" * 40, pattern=_EVALUATOR_BUILD_PATTERN)
    correctness_algorithm_version: str = Field(
        default="correctness-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )
    decoy_algorithm_version: str = Field(default="decoy-v1", pattern=_ALGORITHM_VERSION_PATTERN)
    benchmark_algorithm_version: str = Field(
        # Scalar Torch baseline timing may be reused across candidate evaluations;
        # each measured case records whether that timing came from the cache.
        default="geometric-mean-baseline-v2", pattern=_ALGORITHM_VERSION_PATTERN
    )
    profiler_algorithm_version: str = Field(
        default="profiler-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )

    # Mounted caller capability registry.
    capability_secret_path: Path | None = None

    @property
    def effective_correctness_algorithm_version(self) -> str:
        """Reference placement is part of the correctness protocol identity."""
        version = self.correctness_algorithm_version
        if self.reference_device_policy == "npu-strict":
            version += ":npu-strict-v1"
        if self.reference_cache_sha256 is not None:
            version += ":cache-v1:" + self.reference_cache_sha256
        return version

    @model_validator(mode="after")
    def _durable_core_windows(self):
        overlay_values = (self.engine_runtime_overlay_config, self.engine_runtime_overlay_sha256)
        if any(value is not None for value in overlay_values) and not all(
            value is not None for value in overlay_values
        ):
            raise ValueError("engine runtime overlay config and SHA must be configured together")
        if self.engine_runtime_overlay_config is not None and (
            not self.engine_runtime_overlay_config.is_absolute()
            or ".." in self.engine_runtime_overlay_config.parts
        ):
            raise ValueError("engine runtime overlay config must be absolute without traversal")
        profiler_values = (self.profiler_runtime_manifest, self.profiler_runtime_sha256)
        if any(value is not None for value in profiler_values) and not all(
            value is not None for value in profiler_values
        ):
            raise ValueError("profiler runtime manifest and SHA must be configured together")
        if (
            self.profiler_runtime_manifest is not None
            and not self.profiler_runtime_manifest.is_absolute()
        ):
            raise ValueError("profiler runtime manifest must be absolute")
        tls = (self.api_tls_port, self.api_tls_certfile, self.api_tls_keyfile)
        if any(value is not None for value in tls) and not all(value is not None for value in tls):
            raise ValueError("native HTTPS requires port, certfile and keyfile together")
        if self.api_tls_port is not None and self.api_tls_port == self.api_port:
            raise ValueError("native HTTP and HTTPS must use different ports")
        cache_values = (
            self.reference_cache_dir,
            self.reference_cache_binding,
            self.reference_cache_sha256,
            self.reference_cache_state_dir,
        )
        if any(value is not None for value in cache_values) and any(
            value is None for value in cache_values
        ):
            raise ValueError("all four reference cache deployment settings are required")
        reuse_values = (self.reference_timing_reuse_policy, self.reference_timing_reuse_sha256)
        if any(value is not None for value in reuse_values) and not all(
            value is not None for value in reuse_values
        ):
            raise ValueError(
                "timing reuse policy and pinned SHA256 must be configured together"
            )
        if self.reference_timing_reuse_policy is not None and (
            not self.reference_timing_reuse_policy.is_absolute()
            or ".." in self.reference_timing_reuse_policy.parts
        ):
            raise ValueError("timing reuse policy path must be absolute without traversal")
        if len(self.effective_correctness_algorithm_version) > 128:
            raise ValueError("effective correctness algorithm version exceeds 128 characters")
        if self.max_recovery_age_seconds > self.retention_seconds:
            raise ValueError("max_recovery_age_seconds must not exceed retention_seconds")
        if self.heartbeat_interval_seconds >= self.lease_seconds:
            raise ValueError("heartbeat_interval_seconds must be less than lease_seconds")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        """Explicit constructor used by entrypoints; BaseSettings reads the env itself."""
        return cls()
