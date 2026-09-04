"""Compile verdict cache (spec §8, Plan D-on_card).

Digest-keyed dedup of deterministic compile-class failures: a kernel that
failed compilation once fails identically until the toolchain changes. Fed by
worker-observed engine outcomes and the admission pre-check; read at admission
to short-circuit before queueing. Advisory only — every failure path degrades
to normal queueing.
"""
import importlib.metadata
import logging

from triton_evaluation.broker import EvaluationStore, _scoped_digest

logger = logging.getLogger(__name__)

_TARGET = "ascend-npu"


def _triton_ascend_version() -> str:
    try:
        return importlib.metadata.version("triton-ascend")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def digest_for(kernel_code: str, impl_name: str, cann_fingerprint: str) -> str:
    return _scoped_digest(
        kernel_code, impl_name, _triton_ascend_version(), cann_fingerprint, _TARGET
    )


def get(store: EvaluationStore, digest: str) -> str | None:
    try:
        record = store._r.hgetall(store._key(f"verdict:{digest}"))
    except Exception as error:  # noqa: BLE001 — advisory; never block admission
        logger.warning("verdict cache read failed (%s)", type(error).__name__)
        return None
    if not record:
        return None
    verdict = record.get(b"verdict")
    return verdict.decode("utf-8") if verdict else None


def put(store: EvaluationStore, digest: str, verdict: str, ttl_seconds: int) -> None:
    try:
        key = store._key(f"verdict:{digest}")
        store._r.hset(key, mapping={"verdict": verdict})
        store._r.expire(key, ttl_seconds)
    except Exception as error:  # noqa: BLE001 — advisory; never block a commit
        logger.warning("verdict cache write failed (%s)", type(error).__name__)
