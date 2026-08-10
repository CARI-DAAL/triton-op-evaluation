import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The evaluation engine (verify / internal benchmark / profile / validate) is
# vendored inside the package: this service is self-contained. `engine_dir`
# defaults to the bundled engine; the env override exists only to point at a
# different engine build.
_DEFAULT_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
BENCHMARK_TIMEOUT_SECONDS = 300


class Settings(BaseSettings):
    """Service configuration. Fields are read from ``TRITON_EVAL_<FIELD>`` env vars."""

    model_config = SettingsConfigDict(env_prefix="TRITON_EVAL_", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    engine_dir: str = _DEFAULT_ENGINE_DIR
    queue_key: str = "eval:queue"
    task_key_prefix: str = "eval:task:"
    session_key_prefix: str = "eval:session:"
    task_ttl_seconds: int = 24 * 3600
    dequeue_block_seconds: int = 5
    reaper_grace_seconds: int = 120
    api_poll_interval_seconds: float = Field(default=0.25, ge=0, le=5)

    @classmethod
    def from_env(cls) -> "Settings":
        """Explicit constructor used by entrypoints; BaseSettings reads the env itself."""
        return cls()
