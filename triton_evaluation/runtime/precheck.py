"""Attempt-owned syntax/import precheck with scoped verdict reuse.

A scoped Redis verdict can avoid repeating syntax/import work; a miss executes
in a process group owned by the attempt's lease and deadline. Only definitive
evidence is cached. Unavailable evidence fails the requested compile check;
lost leases abort the attempt.
"""
import json
import logging
import re
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from triton_evaluation.config import Settings
from triton_evaluation.evaluators.base import LeaseLostError, remaining_seconds, run_script
from triton_evaluation.runtime import verdict_cache
from triton_evaluation.service.broker import EvaluationStore

logger = logging.getLogger(__name__)

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"


def run_precheck(
    store: EvaluationStore, settings: Settings, kernel_code: str, impl_name: str,
    *, deadline_at: datetime | None = None,
    lease_valid: Callable[[], bool] | None = None, device_id: int | None = None,
) -> verdict_cache.Verdict | None:
    """Return explicit syntax/import evidence; None means no definitive verdict.

    Every invocation belongs to an explicitly requested compile check. Cached
    passes establish only this same scoped precheck, never backend JIT
    compilation or an evaluation result.
    """
    if lease_valid is not None and not lease_valid():
        raise LeaseLostError("lease lost before syntax/import precheck")
    if deadline_at is not None and remaining_seconds(deadline_at) <= 0:
        return None
    digest = verdict_cache.digest_for(kernel_code, impl_name, settings.cann_fingerprint)
    if settings.verdict_cache_enabled:
        cached = verdict_cache.get(store, digest)
        if cached is not None:
            return cached
    try:
        with tempfile.TemporaryDirectory(prefix="precheck_") as work:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", impl_name)
            module_path = Path(work) / f"candidate_{safe_name}.py"
            module_path.write_text(kernel_code, encoding="utf-8")
            proc = run_script(
                [sys.executable, str(_ENGINE_DIR / "precheck_module.py"),
                 str(module_path), impl_name],
                cwd=work, device_id=device_id, deadline_at=deadline_at,
                lease_valid=lease_valid, budget_seconds=settings.precheck_budget_seconds,
            )
    except OSError as error:
        logger.warning("precheck unavailable (%s)", type(error).__name__)
        return None
    if proc.outcome == "fenced" or (lease_valid is not None and not lease_valid()):
        raise LeaseLostError("lease lost during syntax/import precheck")
    if proc.timed_out:
        return None
    line = next((s for s in reversed(proc.stdout.splitlines()) if s.startswith("VERDICT ")), None)
    if proc.exit_code != 0 or line is None:
        logger.warning("precheck produced no verdict (exit=%s)", proc.exit_code)
        return None
    try:
        verdict = json.loads(line[len("VERDICT "):])
    except json.JSONDecodeError:
        return None
    if (not isinstance(verdict, dict) or verdict.get("inconclusive")
            or "stage" not in verdict or verdict["stage"] not in (None, "syntax", "import")):
        return None
    if verdict.get("stage") in ("syntax", "import"):
        if settings.verdict_cache_enabled:
            verdict_cache.put(store, digest, "compile_error",
                              settings.verdict_cache_ttl_seconds)
        return "compile_error"
    if settings.verdict_cache_enabled:
        effective = verdict_cache.put(store, digest, "precheck_passed",
                                      settings.verdict_cache_ttl_seconds)
        if effective == "compile_error":
            return "compile_error"
    return "precheck_passed"
