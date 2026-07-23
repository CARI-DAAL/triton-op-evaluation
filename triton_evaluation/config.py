import os
from dataclasses import dataclass

# The evaluation engine (verify / benchmark / validate) is vendored inside the
# package: this service is self-contained. `engine_dir` defaults to the bundled
# engine; the env override exists only to point at a different engine build.
_DEFAULT_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    engine_dir: str = _DEFAULT_ENGINE_DIR
    queue_key: str = "eval:queue"
    task_key_prefix: str = "eval:task:"
    task_ttl_seconds: int = 24 * 3600
    dequeue_block_seconds: int = 5
    reaper_grace_seconds: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=os.environ.get("TRITON_EVAL_REDIS_URL", cls.redis_url),
            engine_dir=os.environ.get("TRITON_EVAL_ENGINE_DIR", cls.engine_dir),
            task_ttl_seconds=int(os.environ.get("TRITON_EVAL_TASK_TTL", cls.task_ttl_seconds)),
            dequeue_block_seconds=int(os.environ.get("TRITON_EVAL_DEQUEUE_BLOCK", cls.dequeue_block_seconds)),
            reaper_grace_seconds=int(os.environ.get("TRITON_EVAL_REAPER_GRACE", cls.reaper_grace_seconds)),
        )
