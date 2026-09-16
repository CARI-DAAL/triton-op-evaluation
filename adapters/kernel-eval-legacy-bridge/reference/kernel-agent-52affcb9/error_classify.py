"""Error classification + numeric-diff extraction for kernel_eval feedback.

Triton is JIT: the evaluator's ``compiled`` flag only reflects that the module
imported; real Triton compile errors (``min_dot_size`` asserts, constexpr
misuse, dtype errors in ``tl`` ops, ptxas failures) fire at the FIRST kernel
launch and come back as ``"Kernel execution failed: <traceback>"`` with
``compiled=True`` — which the taxonomy then mis-buckets as ``incorrect``
(so ``compile_fail`` ≈ 0 is an artifact, not a signal). ``classify_error``
re-classifies from the traceback text so the resulting ``error_class`` still
reads ``jit_compile_error`` for diagnostics (the ``compiled=False`` demotion
itself is driven by the grader's own ``CompilationError`` failure tag — see
``kernel_agent.rewards.triton_eval_client.map_verify``).

``extract_numeric_detail`` is the best-effort precision quantifier: it parses
``max_abs_err`` / ``mean_abs_err`` / ``pass_rate`` out of the grader's
correctness message (the evaluator's ``_check_accuracy_allclose`` /
``outputs_diff`` / ``max_difference`` metadata) so the agent can tell
"off by a little (precision/accumulation)" from "off by a lot (logic bug)".
Fields it cannot parse stay ``None`` — never invent numbers.

The in-sandbox CLI script (``kernel_agent/tools/kernel_eval/kernel_eval``) no
longer carries an inline copy of this module — it POSTs to the driver-side
grade handler and prints the handler's rendered observation. ``classify_error``
is called from ``kernel_agent.rewards.triton_eval_client.map_verify`` (the
response→verdict mapper, both the agent's in-loop grade and model-only
scoring); the session facade's own rendering
(``kernel_agent/eval/session_server.py::_render_verify``) then reads the
already-classified ``error_class`` off the verdict dict. This module's
functions are exercised directly by ``tests/kernel_agent/test_error_classify.py``.
"""

from __future__ import annotations

import re

ERROR_CLASSES = (
    "jit_compile_error",
    "runtime_error",
    "numerical_mismatch",
    "shape_mismatch",
    "nan_inf",
    "other",
)

_JIT_MARKERS = (
    "min_dot_size",               # tl.dot dim < 16 assert (triton frontend)
    "compilationerror",           # triton.compiler.errors.CompilationError
    "ptxaserror",
    "outofresources",             # triton.runtime.errors.OutOfResources
    "kernel compilation failed",  # evaluator compiled=False error_message
    "triton/compiler/",           # traceback frames inside the Triton compiler
    "triton/language/",           # frontend semantic asserts (dtype/shape in tl ops)
)

_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def classify_error(traceback_text: str | None) -> str:
    """Classify a grader error/traceback text into one taxonomy class.

    Pure text heuristic — order matters: JIT markers first (a JIT failure is
    wrapped in "Kernel execution failed: ..."), then the validation-specific
    messages (NaN/Inf, shape, numeric), then generic runtime, else "other".
    Empty/unknown text → "other" (callers must not change their verdict on it).
    """
    t = (traceback_text or "").strip()
    if not t:
        return "other"
    low = t.lower()
    if any(m in low for m in _JIT_MARKERS) or (
        "ptxas" in low and ("error" in low or "fatal" in low)
    ):
        return "jit_compile_error"
    if (
        "nan position mismatch" in low
        or "inf position mismatch" in low
        or "inf sign mismatch" in low
    ):
        return "nan_inf"
    if "shape mismatch" in low:
        return "shape_mismatch"
    if (
        "output mismatch" in low
        or "produced incorrect results" in low
        or "allclose" in low
        or "max_abs_err" in low
        or "max_difference" in low
    ):
        return "numerical_mismatch"
    if (
        "kernel execution failed" in low
        or "cuda error" in low
        or "illegal memory access" in low
        or "device-side assert" in low
        or "traceback (most recent call last)" in low
    ):
        return "runtime_error"
    return "other"


def extract_numeric_detail(text: str | None) -> dict:
    """Best-effort parse of numeric-diff detail from grader error text.

    Returns {"max_abs_err", "mean_abs_err", "pass_rate"} — floats when found,
    None when not (honest labelling: parse, never invent). pass_rate is a
    fraction in [0, 1] (the grader prints a percentage).
    """
    out = {"max_abs_err": None, "mean_abs_err": None, "pass_rate": None}
    if not text:
        return out
    # allclose message / outputs_diff dict: max_abs_err=1.2e-4 or 'max_abs_err': 0.12
    m = re.search(r"max_abs_err['\"]?\s*[=:]\s*" + _NUM, text)
    if not m:  # pipeline metadata variant: 'max_difference': ['0.123456']
        m = re.search(r"max_difference['\"]?\s*[=:]\s*\[?\s*['\"]?" + _NUM, text)
    if m:
        out["max_abs_err"] = float(m.group(1))
    m = re.search(r"mean_abs_err['\"]?\s*[=:]\s*" + _NUM, text)
    if not m:
        m = re.search(r"avg_difference['\"]?\s*[=:]\s*\[?\s*['\"]?" + _NUM, text)
    if m:
        out["mean_abs_err"] = float(m.group(1))
    m = re.search(r"pass_rate['\"]?\s*[=:]\s*" + _NUM + r"\s*%", text)
    if m:
        out["pass_rate"] = float(m.group(1)) / 100.0
    return out


def diag_text(d: dict) -> str:
    """Fold the response's error_message + diagnostic metadata into one text.

    A numerical mismatch arrives as error_message="Kernel produced incorrect
    results" with the detail (max_difference/avg_difference/diff) only in
    metadata — classification and numeric extraction must see both.
    """
    md = d.get("metadata") or {}
    parts = [str(d.get("error_message") or d.get("error") or "")]
    for k in (
        "compilation_error",
        "runtime_error",
        "correctness_issue",
        "max_difference",
        "avg_difference",
        "diff",
    ):
        v = md.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    return "\n".join(p for p in parts if p)
