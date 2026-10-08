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

from triton_evaluation.serialization import sha256_hex

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
    return sha256_hex("\0".join((
        "precheck-verdict-v2", kernel_code, impl_name,
        sys.implementation.name, sys.version,
        *(_package_version(package) for package in _RUNTIME_PACKAGES),
        cann_fingerprint, _TARGET,
    )).encode("utf-8"))


class VerdictCache:
    """Fail-open Redis adapter for scoped precheck verdicts."""

    def __init__(self, client: redis.Redis, *, prefix: str):
        self._client = client
        self._prefix = prefix

    def get(self, digest: str | None) -> Verdict | None:
        if digest is None:
            return None
        try:
            record = self._client.hgetall(f"{self._prefix}:verdict:{digest}")
            verdict = record.get(b"verdict")
            if verdict == b"compile_error":
                return "compile_error"
            if verdict == b"precheck_passed":
                return "precheck_passed"
        except Exception as error:  # noqa: BLE001 — advisory; never block admission
            logger.warning("verdict cache read failed (%s)", type(error).__name__)
            return None
        return None

    def put(self, digest: str | None,
            verdict: Verdict, ttl_seconds: int) -> Verdict | None:
        """Publish a verdict with its TTL and return the effective cached verdict.

        Definitive failures take precedence over syntax/import passes. A late pass
        cannot replace a failure or extend its lifetime. Redis failure is advisory;
        callers still use any definitive failure they directly observed.
        """
        if digest is None:
            return None
        try:
            key = f"{self._prefix}:verdict:{digest}"
            with self._client.pipeline() as pipe:
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
            return self.get(digest)
        except Exception as error:  # noqa: BLE001 — advisory; never block a commit
            logger.warning("verdict cache write failed (%s)", type(error).__name__)
            return None
