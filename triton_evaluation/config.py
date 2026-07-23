import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    skill_scripts_dir: str = ""
    queue_key: str = "eval:queue"
    task_key_prefix: str = "eval:task:"
    task_ttl_seconds: int = 24 * 3600
    dequeue_block_seconds: int = 5
    reaper_grace_seconds: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=os.environ.get("TRITON_EVAL_REDIS_URL", cls.redis_url),
            skill_scripts_dir=os.environ.get("TRITON_EVAL_SKILL_SCRIPTS_DIR", cls.skill_scripts_dir),
            task_ttl_seconds=int(os.environ.get("TRITON_EVAL_TASK_TTL", cls.task_ttl_seconds)),
            dequeue_block_seconds=int(os.environ.get("TRITON_EVAL_DEQUEUE_BLOCK", cls.dequeue_block_seconds)),
            reaper_grace_seconds=int(os.environ.get("TRITON_EVAL_REAPER_GRACE", cls.reaper_grace_seconds)),
        )
