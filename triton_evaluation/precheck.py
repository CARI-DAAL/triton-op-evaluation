"""Admission pre-check runner (spec §8, Plan D-on_card).

Cache-first classification of candidate compile failures: a Redis verdict hit
short-circuits without spawning anything; a miss runs the subprocess entry with
a budget and caches definitive verdicts. Advisory and fail-open throughout.
"""
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from triton_evaluation import verdict_cache
from triton_evaluation.broker import EvaluationStore
from triton_evaluation.config import Settings

logger = logging.getLogger(__name__)

_ENGINE_DIR = Path(__file__).parent / "engine"


def run_precheck(
    store: EvaluationStore, settings: Settings, kernel_code: str, impl_name: str,
    *, force: bool = False,
) -> verdict_cache.Verdict | None:
    """Return explicit syntax/import evidence; None means no definitive verdict.

    ``force`` executes an explicitly requested compile check even when optional
    admission prechecking is disabled. Cached passes are evidence of this same
    scoped precheck, never of backend JIT compilation or an evaluation result.
    """
    if not force and not settings.compile_precheck_enabled and not settings.verdict_cache_enabled:
        return None
    digest = verdict_cache.digest_for(kernel_code, impl_name, settings.cann_fingerprint)
    if settings.verdict_cache_enabled:
        cached = verdict_cache.get(store, digest)
        if cached is not None:
            return cached
    if not force and not settings.compile_precheck_enabled:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="precheck_") as work:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", impl_name)
            module_path = Path(work) / f"candidate_{safe_name}.py"
            module_path.write_text(kernel_code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(_ENGINE_DIR / "precheck_module.py"),
                 str(module_path), impl_name],
                capture_output=True, text=True, timeout=settings.precheck_budget_seconds,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("precheck unavailable (%s)", type(error).__name__)
        return None
    line = next((s for s in reversed(proc.stdout.splitlines()) if s.startswith("VERDICT ")), None)
    if proc.returncode != 0 or line is None:
        logger.warning("precheck produced no verdict (exit=%s)", proc.returncode)
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
