import os
from datetime import datetime
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The evaluation engine (verify / internal benchmark / profile / validate) is
# vendored inside the package: this service is self-contained. `engine_dir`
# defaults to the bundled engine; the env override exists only to point at a
# different engine build.
_DEFAULT_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
BENCHMARK_TIMEOUT_SECONDS = 300

_EVALUATOR_BUILD_PATTERN = r"^(sha256:[0-9a-f]{64}|git:[0-9a-f]{40})$"
_ALGORITHM_VERSION_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"


class Settings(BaseSettings):
    """Service configuration. Fields are read from ``TRITON_EVAL_<FIELD>`` env vars."""

    model_config = SettingsConfigDict(env_prefix="TRITON_EVAL_", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    # Fixed v2 durable-core keyspace root; v1 keys live outside this prefix.
    redis_prefix: str = Field(default="triton-eval:v2", min_length=1)
    engine_dir: str = _DEFAULT_ENGINE_DIR
    queue_key: str = "eval:queue"
    task_key_prefix: str = "eval:task:"
    session_key_prefix: str = "eval:session:"
    task_ttl_seconds: int = 24 * 3600
    dequeue_block_seconds: int = 5
    reaper_grace_seconds: int = 120
    api_poll_interval_seconds: float = Field(default=0.25, ge=0.05, le=5)

    # --- durable-core lifecycle (evaluation API v2) ---
    # Terminal resources and idempotency records live at least one day, and the
    # recovery window must fit inside the retention window.
    retention_seconds: int = Field(default=7 * 24 * 3600, ge=86400)
    max_recovery_age_seconds: int = Field(default=24 * 3600, gt=0)
    lease_seconds: int = Field(default=30, gt=0)
    heartbeat_interval_seconds: float = Field(default=10.0, gt=0)
    maintenance_interval_seconds: float = Field(default=1.0, ge=0.05, le=5)

    # --- admission quotas; trial and trusted buckets stay independent ---
    trial_rate_limit_per_minute: int = Field(default=60, gt=0)
    trusted_rate_limit_per_minute: int = Field(default=60, gt=0)
    trial_max_inflight: int = Field(default=4, gt=0)
    trusted_max_inflight: int = Field(default=4, gt=0)
    session_max_inflight: int = Field(default=2, gt=0)

    # --- deployment-attested worker environment ---
    worker_hardware_model: str = Field(default="Ascend910B3", min_length=1)
    worker_driver_version: str = Field(default="24.1.0", min_length=1)
    worker_runtime_version: str = Field(default="8.0.RC3", min_length=1)
    worker_toolchain_version: str = Field(default="CANN 8.0.RC3", min_length=1)

    # --- evaluator provenance pinned at admission ---
    evaluator_build: str = Field(
        default="git:" + "0" * 40, pattern=_EVALUATOR_BUILD_PATTERN
    )
    correctness_algorithm_version: str = Field(
        default="correctness-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )
    decoy_algorithm_version: str = Field(
        default="decoy-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )
    benchmark_algorithm_version: str = Field(
        default="geometric-mean-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )
    profiler_algorithm_version: str = Field(
        default="profiler-v1", pattern=_ALGORITHM_VERSION_PATTERN
    )

    # --- mounted capability secret and trusted release catalog ---
    capability_secret_path: Path | None = None
    catalog_root: Path | None = None
    catalog_manifest_path: Path | None = None

    # --- legacy synchronous API window (sunset semantics arrive with Task 11) ---
    legacy_api_enabled: bool = False
    legacy_api_sunset_at: datetime | None = None

    @model_validator(mode="after")
    def _durable_core_windows(self):
        if self.max_recovery_age_seconds > self.retention_seconds:
            raise ValueError("max_recovery_age_seconds must not exceed retention_seconds")
        if self.heartbeat_interval_seconds >= self.lease_seconds:
            raise ValueError("heartbeat_interval_seconds must be less than lease_seconds")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        """Explicit constructor used by entrypoints; BaseSettings reads the env itself."""
        return cls()
