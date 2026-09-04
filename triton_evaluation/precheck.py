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


def _digest_for(kernel_code: str, impl_name: str, cann_fingerprint: str) -> str:
    return verdict_cache.digest_for(kernel_code, impl_name, cann_fingerprint)


def _classify_module(module_path: str) -> dict:
    """Direct subprocess-free classification for tests: delegates to the entry
    in-process (safe: the module arg is a temp file this process wrote)."""
    import contextlib
    import io

    from triton_evaluation.engine import precheck_module

    argv = sys.argv
    try:
        sys.argv = ["precheck_module", module_path, "candidate"]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            precheck_module.main()
    finally:
        sys.argv = argv
    line = next(s for s in buffer.getvalue().splitlines() if s.startswith("VERDICT "))
    return json.loads(line[len("VERDICT "):])


def run_precheck(store: EvaluationStore, settings: Settings,
                 kernel_code: str, impl_name: str) -> str | None:
    """Returns "compile_error" only on a definitive verdict; None = proceed."""
    if not settings.compile_precheck_enabled and not settings.verdict_cache_enabled:
        return None
    digest = _digest_for(kernel_code, impl_name, settings.cann_fingerprint)
    if settings.verdict_cache_enabled:
        cached = verdict_cache.get(store, digest)
        if cached is not None:
            return cached if cached == "compile_error" else None
    if not settings.compile_precheck_enabled:
        return None
    with tempfile.TemporaryDirectory(prefix="precheck_") as work:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", impl_name)
        module_path = Path(work) / f"candidate_{safe_name}.py"
        module_path.write_text(kernel_code)
        try:
            proc = subprocess.run(
                [sys.executable, str(_ENGINE_DIR / "precheck_module.py"),
                 str(module_path), impl_name],
                capture_output=True, text=True, timeout=settings.precheck_budget_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.warning("precheck unavailable (%s)", type(error).__name__)
            return None
    line = next((s for s in proc.stdout.splitlines() if s.startswith("VERDICT ")), None)
    if proc.returncode != 0 or line is None:
        logger.warning("precheck produced no verdict (exit=%s)", proc.returncode)
        return None
    try:
        verdict = json.loads(line[len("VERDICT "):])
    except json.JSONDecodeError:
        return None
    if verdict.get("inconclusive"):
        return None
    if verdict.get("stage") in ("syntax", "import"):
        if settings.verdict_cache_enabled:
            verdict_cache.put(store, digest, "compile_error",
                              settings.verdict_cache_ttl_seconds)
        return "compile_error"
    if settings.verdict_cache_enabled:
        verdict_cache.put(store, digest, "compiled", settings.verdict_cache_ttl_seconds)
    return None
