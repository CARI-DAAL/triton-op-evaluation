"""Advisory syntax/import precheck verdict cache.

A pass only skips repeated prechecks; it never supplies an evaluation result.
Backend JIT failures depend on evaluation inputs and must not enter this cache.
Reuse requires identical candidate source, implementation name, Python/runtime
package versions, and an explicit deployment CANN fingerprint. Cache failures
degrade to normal prechecking and evaluation.
"""
import importlib.metadata
import logging
import sys
from typing import Literal

import redis

from triton_evaluation.broker import EvaluationStore, _scoped_digest

logger = logging.getLogger(__name__)

_TARGET = "ascend-npu"
_RUNTIME_PACKAGES = ("torch", "torch-npu", "triton", "triton-ascend")
Verdict = Literal["compile_error", "precheck_passed"]


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def digest_for(kernel_code: str, impl_name: str, cann_fingerprint: str) -> str | None:
    """Scope reuse to a known toolchain without importing its runtime modules.

    An absent CANN fingerprint disables reuse: deployment defaults cannot
    establish whether two hosts or successive installations use the same CANN.
    """
    if not cann_fingerprint:
        return None
    return _scoped_digest(
        "precheck-verdict-v2", kernel_code, impl_name,
        sys.implementation.name, sys.version,
        *(_package_version(package) for package in _RUNTIME_PACKAGES),
        cann_fingerprint, _TARGET,
    )


def get(store: EvaluationStore, digest: str | None) -> Verdict | None:
    if digest is None:
        return None
    try:
        record = store._r.hgetall(store._key(f"verdict:{digest}"))
        verdict = record.get(b"verdict")
        if verdict == b"compile_error":
            return "compile_error"
        if verdict == b"precheck_passed":
            return "precheck_passed"
    except Exception as error:  # noqa: BLE001 — advisory; never block admission
        logger.warning("verdict cache read failed (%s)", type(error).__name__)
        return None
    return None


def put(store: EvaluationStore, digest: str | None,
        verdict: Verdict, ttl_seconds: int) -> Verdict | None:
    """Publish a verdict with its TTL and return the effective cached verdict.

    Definitive failures take precedence over syntax/import passes. A late pass
    cannot replace a failure or extend its lifetime. Redis failure is advisory;
    callers still use any definitive failure they directly observed.
    """
    if digest is None:
        return None
    try:
        key = store._key(f"verdict:{digest}")
        with store._r.pipeline() as pipe:
            if verdict == "precheck_passed":
                pipe.watch(key)
                if pipe.hget(key, "verdict") == b"compile_error":
                    return "compile_error"
                pipe.multi()
            pipe.hset(key, mapping={"verdict": verdict})
            pipe.expire(key, ttl_seconds)
            pipe.execute()
        return verdict
    except redis.WatchError:
        # Another publisher won. Dropping a redundant pass is safe; reading
        # the winner also lets admission observe a concurrent compile failure.
        return get(store, digest)
    except Exception as error:  # noqa: BLE001 — advisory; never block a commit
        logger.warning("verdict cache write failed (%s)", type(error).__name__)
        return None
