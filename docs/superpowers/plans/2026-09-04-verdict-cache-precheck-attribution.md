# Verdict Cache + Admission Short-Circuit + Attribution Widening (Plan D-on_card) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repeated identical kernels (RL best-of-n/replay) stop burning card time on deterministic compile failures; first-time syntax/import failures are caught at admission without touching a card; the engine stops misattributing first-launch JIT compile failures as runtime errors.

**Architecture (smallest viable per spike verdict `on_card`):** A Redis verdict cache keyed by `sha256(kernel_code + impl_name + triton-ascend version + CANN fingerprint + target)` feeds two writers — the worker (engine outcomes: compile_error terminals) and a cheap admission-time subprocess pre-check (syntax + module import only, ~100 ms) — and one reader: admission short-circuits cached compile_error verdicts into an admission-time terminal synthesized byte-identically to the engine's compile_error shape. Positive verdicts only skip re-checking; kernels always still execute on-card. Frontend/backend-class compile errors are covered by the engine-feedback loop, not a new compile stage. No compile worker, no dual-lane state machine.

**Tech Stack:** Python 3.11, fakeredis + FakeClock in tests; the pre-check subprocess runs in the deployment conda env (lazy torch import).

**Spec:** `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§4/§5/§8; spike verdict note in §8)

## Global Constraints

- **Contract-invisible**: no `contracts/` or `api/contract.py` changes; synthesized terminals must validate against the frozen 2.0 models and match `contracts/evaluation-api/2.0/fixtures/resource-completed-compile-error.json` shape (correctness `status:"succeeded"` carrying `data.outcome="compile_error"` + `domain_failure{code:"SUBJECT_COMPILE_ERROR", phase:"compile", case_idx:null}`; dependent checks skipped `DEPENDENCY_NOT_PASSED`).
- **Candidate code never executes in the API process**: the pre-check runs as a subprocess with a budget, exactly like `validate_triton_impl`.
- **Engine is vendored (CANN license, ruff-excluded)**: minimal diff, name-based duck typing (no new triton imports).
- **Fail-open everywhere advisory**: precheck unavailable/inconclusive → normal queueing; cache write failures swallowed; only a definitive compile-class verdict short-circuits.
- **v1 scope cut (deliberate)**: no frontend-compile precheck stage (spike's ASTSource interception) — first-occurrence value small, engine feedback covers repeats; no synthesis for requests with `profile` in checks (engine really runs profile; shape wouldn't byte-match) — those queue normally.
- Full suite green each task except the 5 pre-existing environmental `referencing` failures; ruff line-length 100.

---

### Task 1: Settings + verdict cache + worker engine-feedback writer

**Files:**
- Modify: `triton_evaluation/config.py` (verdict block)
- Create: `triton_evaluation/verdict_cache.py`
- Modify: `triton_evaluation/worker.py` (`_run_attempt` commit region)
- Test: `tests/test_verdict_cache.py` (create); append to `tests/test_worker.py`

**Interfaces:**
- Consumes: `broker._scoped_digest`, `sha256_hex` (contract_artifacts), worker's `validated` result in scope at worker.py:467-478.
- Produces:
  - Settings: `verdict_cache_enabled: bool = True`, `verdict_cache_ttl_seconds: int = 604800`, `cann_fingerprint: str = ""` (pattern `^[A-Za-z0-9._:-]{0,128}$`).
  - `triton_evaluation/verdict_cache.py`:
    - `digest_for(kernel_code: str, impl_name: str, cann_fingerprint: str) -> str` — `_scoped_digest(kernel_code, impl_name, triton_ascend_version, cann_fingerprint, "ascend-npu")`; version via `importlib.metadata.version("triton-ascend")` lazily, `"unavailable"` on failure.
    - `get(store, digest) -> str | None` — returns `"compile_error"` or None; swallows Redis errors.
    - `put(store, digest, verdict: str, ttl_seconds: int) -> None` — HSET `verdict:{digest}` {verdict, at_ms} + EXPIRE; swallows Redis errors.
  - Worker: after a successful `commit_terminal` (return value True — capture it) where `validated.result.correctness` is a succeeded-status check with `data.outcome == "compile_error"`, write the verdict cache from the canonical request's `kernel_code`/`triton_impl_name`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verdict_cache.py`:

```python
"""Verdict cache: digest stability, get/put round-trip, worker feedback writer."""
import json

from conftest import make_caller, make_command, make_completed_result

from triton_evaluation import verdict_cache
from triton_evaluation.broker import _scoped_digest


def test_digest_is_stable_and_scoped():
    d1 = verdict_cache.digest_for("code-a", "triton_ascend_impl", "")
    d2 = verdict_cache.digest_for("code-a", "triton_ascend_impl", "")
    d3 = verdict_cache.digest_for("code-b", "triton_ascend_impl", "")
    assert d1 == d2 and d1 != d3
    assert d1 == _scoped_digest(
        "code-a", "triton_ascend_impl",
        verdict_cache._triton_ascend_version(), "", "ascend-npu",
    )


def test_put_get_roundtrip_and_expiry(store, redis_client, clock, settings):
    digest = verdict_cache.digest_for("k", "i", "")
    assert verdict_cache.get(store, digest) is None
    verdict_cache.put(store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    assert verdict_cache.get(store, digest) == "compile_error"
    key = f"triton-eval:v2:verdict:{digest}"
    assert redis_client.ttl(key) > 0
    clock.advance_ms(settings.verdict_cache_ttl_seconds * 1000 + 1)
    assert verdict_cache.get(store, digest) is None  # expired


def test_get_put_swallow_redis_errors(store, mocker):
    boom = mocker.patch.object(store._r, "hgetall", side_effect=Exception("down"))
    assert verdict_cache.get(store, "x") is None
    boom.side_effect = None
    boom2 = mocker.patch.object(store._r, "hset", side_effect=Exception("down"))
    verdict_cache.put(store, "x", "compile_error", 60)  # must not raise
```

(Use unittest.mock, not pytest-mock. If `_triton_ascend_version` naming differs, expose whatever private helper digest_for uses and test through `digest_for` only — drop the second assert rather than test privates.)

Append to `tests/test_worker.py`:

```python
def test_compile_error_terminal_feeds_verdict_cache(store, settings, clock,
                                                    make_command):
    from triton_evaluation import verdict_cache
    from conftest import make_completed_result
    result = make_completed_result(settings)
    result.correctness.data.outcome = "compile_error"
    result.correctness.data.passed = False
    result.correctness.data.domain_failure = {
        "case_idx": None, "code": "SUBJECT_COMPILE_ERROR",
        "message": "candidate compilation failed", "phase": "compile",
    }
    worker = build_worker(store, settings, StubEvaluator(result=result), clock)
    cmd = make_command()
    store.create_or_replay(cmd)
    assert worker.run_once() is True
    request = json.loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"], request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    assert verdict_cache.get(store, digest) == "compile_error"


def test_correct_terminal_does_not_feed_cache(store, settings, clock, make_command):
    from triton_evaluation import verdict_cache
    from conftest import make_completed_result
    worker = build_worker(store, settings,
                          StubEvaluator(result=make_completed_result(settings)), clock)
    cmd = make_command()
    store.create_or_replay(cmd)
    assert worker.run_once() is True
    request = json.loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"], request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    assert verdict_cache.get(store, digest) is None
```

(Verify `make_completed_result`'s shape: if its correctness data uses a model instance, mutate via model_copy or set attributes as the object allows — adapt minimally and disclose. `json` may already be imported in the file; add if needed.)

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_verdict_cache.py tests/test_worker.py -k verdict or feed -q`
Expected: import errors (`triton_evaluation.verdict_cache` missing), AttributeError on settings.

- [ ] **Step 3: Settings**

After the card-health block in `triton_evaluation/config.py`:

```python
    # Compile verdict cache (spec §8, Plan D-on_card): digest-keyed dedup of
    # deterministic compile failures; fed by engine outcomes + admission precheck.
    verdict_cache_enabled: bool = Field(default=True)
    verdict_cache_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    cann_fingerprint: str = Field(default="", pattern=r"^[A-Za-z0-9._:-]{0,128}$")
```

- [ ] **Step 4: verdict_cache.py**

```python
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
```

- [ ] **Step 5: Worker feedback writer**

In `triton_evaluation/worker.py` `_run_attempt` (worker.py:467-478): capture the commit result and feed the cache. Replace the tail:

```python
            validated = EvaluationResult.model_validate(
                result.model_dump(mode="json", exclude_none=False),
                context={"requested_checks": request.checks},
            )
            # False means another winner already terminalized the resource.
            committed = self.store.commit_terminal(lease, "completed", validated)
            if committed and self.settings.verdict_cache_enabled:
                self._record_compile_verdict(lease, validated)
```

New private method:

```python
    def _record_compile_verdict(self, lease, validated) -> None:
        """Engine-authoritative compile verdicts feed the admission cache."""
        from triton_evaluation import verdict_cache

        correctness = validated.correctness
        if correctness is None or correctness.status != "succeeded":
            return
        data = correctness.data
        if data is None or getattr(data, "outcome", None) != "compile_error":
            return
        try:
            payload = json.loads(lease.canonical_request)
            digest = verdict_cache.digest_for(
                payload["kernel_code"],
                payload.get("triton_impl_name", "triton_ascend_impl"),
                self.settings.cann_fingerprint,
            )
            verdict_cache.put(
                self.store, digest, "compile_error",
                self.settings.verdict_cache_ttl_seconds,
            )
        except (KeyError, ValueError):
            logger.warning("compile verdict feedback skipped (unparseable request)")
```

(`json` is already imported at worker.py module scope — verify; `from triton_evaluation import verdict_cache` locally to keep worker import graph torch-free.)

- [ ] **Step 6: Run + full suite + commit**

Run: `python3 -m pytest tests/test_verdict_cache.py tests/test_worker.py -q`, then full suite; `ruff check triton_evaluation/ tests/test_verdict_cache.py tests/test_worker.py`.

```bash
git add triton_evaluation/config.py triton_evaluation/verdict_cache.py \
        triton_evaluation/worker.py tests/test_verdict_cache.py tests/test_worker.py
git commit -m "feat: compile verdict cache + engine-outcome feedback writer"
```

---

### Task 2: Engine attribution widening — first-launch CompilationError → compile phase

**Files:**
- Modify: `triton_evaluation/engine/verify.py` (exception class, phase table, `_run_candidate_forward`)
- Test: `tests/test_verify_engine.py` (append)

**Interfaces:**
- Consumes: existing `_CandidateError` class family (verify.py:54-86), `_CANDIDATE_ERROR_PHASES` table (verify.py:1154-1159), `_try_run_case` handler (verify.py:1162-1214).
- Produces: `_CandidateCompileError` (phase `"compile"`); first-launch exceptions whose type is named `CompilationError` from a `triton.*` module are attributed phase compile instead of execute. Shaping maps phase compile → outcome `compile_error` + `SUBJECT_COMPILE_ERROR` unchanged (shaping.py:362-379 key on phase).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_verify_engine.py` (follow the file's existing fake-torch/monkeypatch pattern, ~lines 11-17 and 92-130):

```python
def test_first_launch_compilation_error_is_attributed_compile(monkeypatch, tmp_path):
    """A triton CompilationError raised lazily at first kernel launch is a
    compile-phase failure, not an execute-phase runtime error (spec §8/Plan D)."""
    from triton_evaluation.engine import verify

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    class CompilationError(Exception):
        pass

    CompilationError.__module__ = "triton.compiler.errors"

    def fake_import_module(name, package=None):
        if name.endswith("_triton_ascend_impl"):
            module = ModuleType(name)

            class ModelNew:
                def __init__(self, *_a, **_k):
                    pass

                def forward(self, *_inputs):
                    raise CompilationError("at 2:14: NameError: undefined_name")

            module.ModelNew = ModelNew
            return module
        raise AssertionError(name)

    monkeypatch.setattr(verify.importlib, "import_module", fake_import_module)
    (tmp_path / "addmm_triton_ascend_impl.py").write_text("# stub")
    passed, total, _out = _run_verify_main(monkeypatch, tmp_path)
    assert (passed, total) == (0, 1)
    result = json.loads((tmp_path / "verify_result.json").read_text())
    failure = result["cases"][0]
    assert failure["candidate_error"]["phase"] == "compile"
    assert failure["error_type"] == "CompilationError"
```

(Adapt to how the file's existing tests invoke the engine — the existing compile-error test at ~91-130 is the template; reuse its helper for running verify main and reading verify_result.json. The phase assertion is the load-bearing part; if the result file's case field names differ, match the existing test's accessors.)

- [ ] **Step 2: Run it — verify it fails**

Run: `python3 -m pytest tests/test_verify_engine.py -k compilation_error -q`
Expected: FAIL — `candidate_error.phase == "execute"` today.

- [ ] **Step 3: Engine change**

In `triton_evaluation/engine/verify.py`:

1. Next to the other candidate error classes (~line 69):

```python
class _CandidateCompileError(Exception):
    """Candidate JIT compilation failed at first launch (phase compile)."""
```

2. Extend the phase table (~lines 1154-1159) — insert FIRST so the lookup hits it:

```python
_CANDIDATE_ERROR_PHASES = (
    (_CandidateCompileError, "compile"),
    (_CandidateInitializeError, "initialize"),
    (_CandidateExecuteError, "execute"),
    (_CandidateExecutionTimeout, "execution_timeout"),
)
```

3. In `_run_candidate_forward` (~1008-1033): both `except Exception as exc` branches (with and without budget) become:

```python
    except Exception as exc:
        if _is_compile_error(exc):
            raise _CandidateCompileError(
                f"candidate compilation raised {type(exc).__name__}"
            ) from exc
        raise _CandidateExecuteError(
            f"candidate execution raised {type(exc).__name__}"
        ) from exc
```

4. Duck-typing helper (module level, near the error classes):

```python
def _is_compile_error(exc: BaseException) -> bool:
    """First-launch JIT compile failures surface as triton's CompilationError,
    raised lazily inside the first kernel launch. Duck-typed by name/module so
    the engine stays importable without triton (engine runs on CPU workers too)."""
    cls = type(exc)
    return cls.__name__ == "CompilationError" and cls.__module__.startswith("triton")
```

- [ ] **Step 4: Run + full suite + commit**

Run: `python3 -m pytest tests/test_verify_engine.py -q`, then full suite; ruff excluded for engine/ but run it on tests.

```bash
git add triton_evaluation/engine/verify.py tests/test_verify_engine.py
git commit -m "feat(engine): attribute first-launch CompilationError to compile phase"
```

---

### Task 3: Admission short-circuit — precheck subprocess + synthesis + terminal-at-admission

**Files:**
- Modify: `triton_evaluation/config.py` (precheck block)
- Create: `triton_evaluation/engine/precheck_module.py` (subprocess entry)
- Create: `triton_evaluation/precheck.py` (runner)
- Modify: `triton_evaluation/terminal_results.py` (`synthesize_compile_error_result`)
- Modify: `triton_evaluation/broker.py` (`create_terminal_at_admission` + keyspace doc line)
- Modify: `triton_evaluation/api/routes.py` (wiring after `_require_ready`)
- Test: `tests/test_precheck.py`, `tests/test_admission_shortcircuit.py` (create)

**Interfaces:**
- Consumes: Task 1 verdict cache; `terminal_results` synthesis patterns; broker digest/key conventions.
- Produces:
  - Settings: `compile_precheck_enabled: bool = True`, `precheck_budget_seconds: int = 60 (gt=0)`.
  - `engine/precheck_module.py` — run as `python precheck_module.py <module_path> <impl_name>` in the deployment env: ast.parse → import → `get_input_groups()` count → JSON `{"stage": null, "total_cases": N}` or `{"stage": "syntax"|"import", "error": <bounded excerpt>}`; exit 0 always (verdict in JSON).
  - `precheck.py`: `run_precheck(store, settings, kernel_code, impl_name) -> str | None` — cache lookup first (hit → verdict); miss + enabled → subprocess with budget → cache write on definitive verdict → return `"compile_error" | None`.
  - `terminal_results.synthesize_compile_error_result(*, requested_checks, failure_message, total_cases, evaluator_build, algorithms, manifest_id, problem_id) -> EvaluationResult` — byte-shape per the fixture (see Global Constraints).
  - `broker.create_terminal_at_admission(command, result) -> CreateResourceResult` — replay-aware, rate-limited, terminal-completed at revision 1, no queue/inflight writes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_precheck.py`:

```python
"""Admission pre-check: syntax/import classification, cache integration."""
from triton_evaluation import precheck


def test_syntax_error_classified(tmp_path):
    module = tmp_path / "bad_mod.py"
    module.write_text("def broken(:\n    return 1\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] == "syntax"


def test_import_error_classified(tmp_path):
    module = tmp_path / "bad_mod.py"
    module.write_text("import definitely_not_a_real_pkg_xyz\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] == "import"


def test_good_module_passes(tmp_path):
    module = tmp_path / "good_mod.py"
    module.write_text("class ModelNew:\n    def __init__(self, *a, **k):\n        pass\n")
    verdict = precheck._classify_module(str(module))
    assert verdict["stage"] is None


def test_run_precheck_caches_pass_verdict(store, settings, redis_client):
    settings = settings.model_copy(update={"verdict_cache_enabled": True})
    good_source = "class ModelNew:\n    def __init__(self, *a, **k):\n        pass\n"
    digest = precheck._digest_for(good_source, "triton_ascend_impl", settings.cann_fingerprint)
    assert precheck.run_precheck(store, settings, good_source, "triton_ascend_impl") is None
    # second call must be a cache hit without a subprocess
    from unittest import mock
    with mock.patch.object(precheck.subprocess, "run") as run_mock:
        verdict = precheck.run_precheck(store, settings, good_source, "triton_ascend_impl")
    assert verdict is None  # pass verdict cached; not compile_error
    run_mock.assert_not_called()
    from triton_evaluation import verdict_cache
    assert verdict_cache.get(store, digest) is not None


def test_run_precheck_negative_verdict_shortcircuits(store, settings):
    settings = settings.model_copy(update={"verdict_cache_enabled": True})
    bad_source = "def broken(:\n    return 1\n"
    assert precheck.run_precheck(store, settings, bad_source, "triton_ascend_impl") == "compile_error"
    # cached: repeat hits cache, still compile_error, no subprocess
    from unittest import mock
    with mock.patch.object(precheck.subprocess, "run") as run_mock:
        assert precheck.run_precheck(store, settings, bad_source, "triton_ascend_impl") == "compile_error"
    run_mock.assert_not_called()
```

(The load-bearing parts: the pass-verdict test proves a repeat admission is a cache hit with zero subprocess spawns; the negative test proves a syntax verdict short-circuits and caches. Assert cached passes through `verdict_cache.get(...) is not None` rather than a literal name.)

Create `tests/test_admission_shortcircuit.py`:

```python
"""Admission-time terminal synthesis for cached compile_error verdicts."""
from types import SimpleNamespace

import pytest

from conftest import make_caller, make_request

from triton_evaluation import terminal_results, verdict_cache
from triton_evaluation.api import routes
from triton_evaluation.api.contract import EvaluationResult


def _command_for(settings, kernel_code="import torch\n"):
    caller = make_caller()
    request = make_request(kernel_code=kernel_code)
    from triton_evaluation.admission import admit
    from triton_evaluation.catalog import ReleaseCatalog
    admission = admit(request, caller, ReleaseCatalog({}))
    return routes._build_command(
        SimpleNamespace(settings=settings), caller, "s1", "k1", admission, "trial"
    )


def test_synthesize_matches_frozen_compile_error_shape(settings):
    result = terminal_results.synthesize_compile_error_result(
        requested_checks=["correctness", "performance"],
        failure_message="candidate compilation failed",
        total_cases=8,
        evaluator_build=settings.evaluator_build,
        algorithms=SimpleNamespace(
            correctness=settings.correctness_algorithm_version,
            decoy=settings.decoy_algorithm_version,
            benchmark=settings.benchmark_algorithm_version,
            profiler=settings.profiler_algorithm_version,
        ),
        manifest_id=None,
        problem_id=None,
    )
    validated = EvaluationResult.model_validate(
        result.model_dump(mode="json", exclude_none=False),
        context={"requested_checks": ["correctness", "performance"]},
    )
    correctness = validated.correctness
    assert correctness.status == "succeeded"
    assert correctness.data.outcome == "compile_error"
    assert correctness.data.passed is False
    assert correctness.data.passed_cases == 0
    assert correctness.data.total_cases == 8
    assert correctness.data.domain_failure.code == "SUBJECT_COMPILE_ERROR"
    assert correctness.data.domain_failure.phase == "compile"
    assert correctness.data.domain_failure.case_idx is None
    assert validated.performance.status == "skipped"
    assert validated.performance.reason.code == "DEPENDENCY_NOT_PASSED"
    assert validated.provenance.grade_eligible is False


def test_create_terminal_at_admission(store, settings, redis_client, clock,
                                      make_command, make_worker):
    from conftest import make_completed_result
    cmd = make_command()
    result = terminal_results.synthesize_compile_error_result(
        requested_checks=list(cmd.admission.requested_checks),
        failure_message="candidate compilation failed",
        total_cases=1,
        evaluator_build=cmd.evaluator_build,
        algorithms=cmd.algorithms,
        manifest_id=None,
        problem_id=None,
    )
    outcome = store.create_terminal_at_admission(cmd, result)
    assert outcome.decision == "created"
    resource = outcome.resource
    assert resource.state == "completed"
    assert resource.revision == 1
    assert resource.started_at is not None and resource.finished_at is not None
    assert resource.result.correctness.data.outcome == "compile_error"
    # never queued: no ready membership, no inflight held
    assert redis_client.zscore("triton-eval:v2:ready:exec:interactive",
                               cmd.evaluation_id.encode()) is None
    # idempotent replay returns the same terminal resource
    replay = store.create_terminal_at_admission(cmd, result)
    assert replay.decision == "replayed"
    assert replay.resource.evaluation_id == resource.evaluation_id
    # normal claiming sees nothing
    assert store.claim_next(make_worker()) is None


def test_shortcircuit_wiring_skips_queue(store, settings, clock, make_command):
    """End-to-end at the broker+precheck level: cached verdict -> terminal."""
    cmd = make_command()
    request = __import__("json").loads(cmd.admission.canonical_request)
    digest = verdict_cache.digest_for(
        request["kernel_code"],
        request.get("triton_impl_name", "triton_ascend_impl"),
        settings.cann_fingerprint,
    )
    verdict_cache.put(store, digest, "compile_error", settings.verdict_cache_ttl_seconds)
    # admission path consults the cache before create_or_replay
    from triton_evaluation import precheck
    assert precheck.run_precheck(store, settings, request["kernel_code"],
                                 request.get("triton_impl_name", "triton_ascend_impl")) == "compile_error"
```

(`make_completed_result` import in the second test is unused if synthesis replaces it — drop as needed; `algorithms=cmd.algorithms` works since CreateResourceCommand carries AlgorithmVersions, and synthesize expects the same duck shape. Adapt imports to repo conventions.)

- [ ] **Step 2: Run them — verify they fail**

Run: `python3 -m pytest tests/test_precheck.py tests/test_admission_shortcircuit.py -q`
Expected: import errors (`precheck`, `synthesize_compile_error_result`, `create_terminal_at_admission` missing).

- [ ] **Step 3: Precheck subprocess entry**

`triton_evaluation/engine/precheck_module.py`:

```python
"""Admission pre-check subprocess entry (spec §8, Plan D-on_card).

Classifies a candidate module without any NPU: syntax parse, then import, then
input-case counting. Never launches kernels. Always exits 0; the verdict is the
last stdout line as JSON. Stdlib-only so it runs in any worker/API env.
"""
import ast
import importlib.util
import json
import sys


def main() -> int:
    module_path, impl_name = sys.argv[1], sys.argv[2]
    try:
        source = open(module_path, encoding="utf-8").read()
    except OSError as error:
        _emit({"stage": "import", "error": f"unreadable module: {error}"})
        return 0
    try:
        ast.parse(source)
    except SyntaxError as error:
        _emit({"stage": "syntax", "error": f"SyntaxError: {error}"})
        return 0
    try:
        spec = importlib.util.spec_from_file_location(f"{impl_name}_candidate", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 — any import failure is a verdict
        _emit({"stage": "import", "error": f"{type(error).__name__}: {error}"[:400]})
        return 0
    total_cases = 0
    try:
        if hasattr(module, "get_input_groups"):
            total_cases = len(module.get_input_groups())
        elif hasattr(module, "get_inputs"):
            total_cases = 1
    except Exception:  # noqa: BLE001 — case counting is best effort
        total_cases = 0
    _emit({"stage": None, "total_cases": total_cases})
    return 0


def _emit(verdict: dict) -> None:
    print("VERDICT " + json.dumps(verdict))


if __name__ == "__main__":
    sys.exit(main())
```

Note: this imports the candidate module **in a subprocess** — torch import inside the candidate is expected in the deployment env; on a torch-less host the import fails → stage "import" verdict. That is wrong for dev-box classification of good modules (no torch → false compile_error!). Guard: if the import failure is specifically `ModuleNotFoundError` for torch/torch_npu/triton, emit `{"stage": None, "total_cases": 0, "inconclusive": True}` instead — precheck.py treats inconclusive as pass-through (fail-open). Add that check in the except block.

- [ ] **Step 4: precheck.py runner**

```python
"""Admission pre-check runner (spec §8, Plan D-on_card).

Cache-first classification of candidate compile failures: a Redis verdict hit
short-circuits without spawning anything; a miss runs the subprocess entry with
a budget and caches definitive verdicts. Advisory and fail-open throughout.
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
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
    from triton_evaluation.engine import precheck_module

    argv = sys.argv
    try:
        sys.argv = ["precheck_module", module_path, "candidate"]
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            precheck_module.main()
    finally:
        sys.argv = argv
    line = next(l for l in buffer.getvalue().splitlines() if l.startswith("VERDICT "))
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
        module_path = Path(work) / f"candidate_{impl_name}.py"
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
    line = next((l for l in proc.stdout.splitlines() if l.startswith("VERDICT ")), None)
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
```

- [ ] **Step 5: Synthesis in terminal_results.py**

```python
def synthesize_compile_error_result(
    *, requested_checks, failure_message, total_cases,
    evaluator_build, algorithms, manifest_id, problem_id,
) -> EvaluationResult:
    """Admission-time compile failure, byte-shaped like the engine's own
    compile_error terminal (contracts .../resource-completed-compile-error.json):
    correctness succeeds as a carrier with outcome compile_error; dependent
    checks skip DEPENDENCY_NOT_PASSED."""
    requested = [check for check in _CHECK_ORDER if check in requested_checks]
    correctness = {
        "status": "succeeded",
        "data": {
            "decoy": False,
            "domain_failure": {
                "case_idx": None,
                "code": "SUBJECT_COMPILE_ERROR",
                "message": failure_message,
                "phase": "compile",
            },
            "failures": [],
            "outcome": "compile_error",
            "passed": False,
            "passed_cases": 0,
            "total_cases": int(total_cases),
        },
        "error": None,
        "reason": None,
    }
    checks = {}
    for check in requested:
        if check == "correctness":
            checks[check] = correctness
        else:
            checks[check] = {
                "status": "skipped",
                "data": None,
                "error": None,
                "reason": {
                    "code": "DEPENDENCY_NOT_PASSED",
                    "depends_on": _SKIP_DEPENDS_ON[check],
                    "message": "correctness did not pass",
                },
            }
    return EvaluationResult.model_validate(
        {
            **checks,
            "provenance": {
                "schema_version": "2.0",
                "evaluator_build": evaluator_build,
                "correctness_algorithm": algorithms.correctness,
                "decoy_algorithm": algorithms.decoy,
                "benchmark_algorithm": algorithms.benchmark,
                "profiler_algorithm": algorithms.profiler,
                "execution_environment": None,
                "grade_eligible": False,
                "manifest_id": manifest_id,
                "problem_id": problem_id,
            },
        },
        context={"requested_checks": requested},
    )
```

(Match the provenance construction in `synthesize_orchestration_result` exactly — read it first and mirror field names/order; `"profile"` unrequested must be absent from the dump per contract context rules, which model_validate with the context enforces.)

- [ ] **Step 6: Broker create_terminal_at_admission**

Keyspace doc header: add `verdict:{sha256}  hash {verdict}, TTL = verdict_cache_ttl_seconds` and amend the TTL sentence to note the verdict cache as a second expiring class.

```python
    def create_terminal_at_admission(
        self, command: CreateResourceCommand, result: EvaluationResult
    ) -> CreateResourceResult:
        """Admission-time terminal for a definitive compile verdict: durable and
        idempotent like create_or_replay, but never queued — no ready/deadlines
        entries, no inflight held, state completed at revision 1."""
        caller_digest = _digest_text(command.caller_id)
        idempotency_key = self._key(
            f"idempotency:{_scoped_digest(command.caller_id, command.idempotency_key)}"
        )
        sequence_key = self._key("queue-sequence")
        rate_limit, _ = self._quota_limits(command.traffic_class)
        while True:
            now_ms = self._clock()
            rate_key = self._key(
                f"rate:{command.traffic_class}:{caller_digest}:{now_ms // _MINUTE_MS}"
            )
            pipeline = self._r.pipeline()
            try:
                pipeline.watch(idempotency_key, rate_key, sequence_key)
                record = pipeline.hgetall(idempotency_key)
                if record:
                    return self._replay(record, command, pipeline)
                if int(pipeline.get(rate_key) or 0) >= rate_limit:
                    return CreateResourceResult("rate_limited", None)
                resource, mapping, _deadline = self._new_queued_resource(
                    command, idempotency_key,
                    _scoped_digest(command.caller_id, command.session_id), now_ms,
                )
                finished = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
                terminal = EvaluationResource.model_validate(
                    {
                        **resource.model_dump(exclude_none=False),
                        "revision": 1,
                        "state": "completed",
                        "started_at": finished,
                        "updated_at": finished,
                        "finished_at": finished,
                        "result": result.model_dump(mode="json", exclude_none=False),
                    }
                )
                expiry_ms = now_ms + self._s.retention_seconds * 1000
                mapping = {
                    **mapping,
                    "resource": canonical_json_bytes(
                        terminal.model_dump(mode="json", exclude_none=False)
                    ),
                    "updated_at_ms": str(now_ms),
                }
                pipeline.multi()
                pipeline.hset(
                    self._key(f"resource:{command.evaluation_id}"), mapping=mapping
                )
                pipeline.expire(
                    self._key(f"resource:{command.evaluation_id}"),
                    self._s.retention_seconds,
                )
                pipeline.hset(
                    idempotency_key,
                    mapping={
                        "evaluation_id": command.evaluation_id,
                        "request_digest": command.admission.request_digest,
                    },
                )
                pipeline.expire(idempotency_key, self._s.retention_seconds)
                pipeline.zadd(
                    self._key("state:completed"),
                    {command.evaluation_id: expiry_ms},
                )
                pipeline.incr(rate_key)
                pipeline.expire(rate_key, _RATE_WINDOW_EXPIRY_SECONDS)
                pipeline.execute()
                return CreateResourceResult("created", terminal)
            except redis.WatchError:
                continue
            finally:
                pipeline.reset()
```

Watch ordering vs `_replay`: check `_replay`'s signature/return (it returns CreateResourceResult) — match how `create_or_replay` returns its replay result and mirror it exactly.

- [ ] **Step 7: Routes wiring**

In `submit_evaluation` after `outcome = await asyncio.to_thread(context.store.create_or_replay, command)` is TOO LATE — the resource is already queued. Insert BEFORE the `create_or_replay` call (after `_require_ready`):

```python
    if settings flags:  # context.settings.compile_precheck_enabled or verdict_cache_enabled
        payload = json.loads(command.admission.canonical_request)
        verdict = await asyncio.to_thread(
            precheck.run_precheck, context.store, context.settings,
            payload["kernel_code"],
            payload.get("triton_impl_name", "triton_ascend_impl"),
        )
        if verdict == "compile_error" and "profile" not in admission.requested_checks:
            result = synthesize_compile_error_result(
                requested_checks=list(admission.requested_checks),
                failure_message="candidate compilation failed (admission precheck)",
                total_cases=1,
                evaluator_build=command.evaluator_build,
                algorithms=command.algorithms,
                manifest_id=(payload.get("grade_context") or {}).get("manifest_id"),
                problem_id=(payload.get("grade_context") or {}).get("problem_id"),
            )
            outcome = await asyncio.to_thread(
                context.store.create_terminal_at_admission, command, result
            )
        else:
            outcome = await asyncio.to_thread(context.store.create_or_replay, command)
    else:
        outcome = await asyncio.to_thread(context.store.create_or_replay, command)
```

(`json` import at routes module scope — verify/add. Keep the existing decision-error mapping untouched: it already handles `replayed`/`conflict`/`rate_limited`. total_cases=1 at admission because the precheck's per-case counting is subprocess-side; the message notes precheck provenance.)

- [ ] **Step 8: Run + full suite + commit**

Run: `python3 -m pytest tests/test_precheck.py tests/test_admission_shortcircuit.py -q`, then full suite; `ruff check triton_evaluation/ tests/test_precheck.py tests/test_admission_shortcircuit.py` (engine/ excluded).

```bash
git add triton_evaluation/config.py triton_evaluation/engine/precheck_module.py \
        triton_evaluation/precheck.py triton_evaluation/terminal_results.py \
        triton_evaluation/broker.py triton_evaluation/api/routes.py \
        tests/test_precheck.py tests/test_admission_shortcircuit.py
git commit -m "feat: admission compile short-circuit (precheck + verdict cache + synthesis)"
```

---

### Task 4: Docs — env example, README, spec note

**Files:**
- Modify: `docker.env.example`, `README.md`, `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§8 实现注记)

- [ ] **Step 1: env example** — append:

```bash
# Compile verdict cache + admission precheck (spec §8, Plan D-on_card).
# TRITON_EVAL_VERDICT_CACHE_ENABLED=true
# TRITON_EVAL_VERDICT_CACHE_TTL_SECONDS=604800
# TRITON_EVAL_COMPILE_PRECHECK_ENABLED=true
# TRITON_EVAL_PRECHECK_BUDGET_SECONDS=60
# CANN version fingerprint (invalidates cached verdicts on toolchain upgrades):
# TRITON_EVAL_CANN_FINGERPRINT=9.0.0.0430
```

- [ ] **Step 2: README** — one paragraph near the rollout/card-health notes:

```markdown
Compile verdict cache (spec §8): deterministic compile failures (syntax/import
class via an admission-time precheck subprocess; all compile-class via
worker-observed engine outcomes) are cached by a toolchain-scoped digest for
`TRITON_EVAL_VERDICT_CACHE_TTL_SECONDS` (default 7 d) and short-circuited at
admission into a terminal identical in shape to an on-card compile failure —
no queue slot, no card time. Set `TRITON_EVAL_CANN_FINGERPRINT` to your CANN
version so toolchain upgrades invalidate stale verdicts.
```

- [ ] **Step 3: spec note** — append to the §8 spike-verdict block:

```markdown
**Plan D-on_card 实现注记(2026-09-04)**:v1 落地三项——①verdict cache(digest = kernel + impl + triton-ascend 版本 + CANN 指纹 + target;worker 引擎反馈 + 准入预检双写;负判定短路为准入期终态,正判定只免复检);②准入静态预检(子进程 ast.parse + import,~100ms,覆盖 syntax/import 两类;frontend/backend 类由引擎反馈环覆盖,ASTSource 前端编译预检阶段刻意不做——首现收益小,重复样本已由反馈环兜住);③引擎归因拓宽(首次 launch 的 triton CompilationError 按名/模块 duck-type 识别,归因 compile 期而非 execute 期)。请求含 profile check 时不短路(引擎真实运行 profile,合成形状无法逐字节对齐)。
```

- [ ] **Step 4: Run + commit**

Full suite green (5 pre-existing environmental failures excepted); ruff clean.

```bash
git add docker.env.example README.md docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md
git commit -m "docs: verdict cache + precheck deployment notes and spec errata"
```

---

## Self-Review Notes (author check, completed at plan-write time)

- **Spec coverage (§8 on_card 形态):** verdict cache(双写、负缓存、TTL、指纹失效)、无卡预检(syntax/import 两类,spike 判据 3 证明可无卡检出且判定一致)、归因拓宽(fail_frontend 实证 gap)、共享 TRITON_CACHE_DIR(spike 证明无效,compose 挂载不变,不另设任务)。刻意不做:ASTSource 前端编译预检、compile lane/worker、`running→queued` 转移——均被 spike 裁决或 YAGNI 排除。
- **Shape fidelity:** synthesis 镜像 fixture `resource-completed-compile-error.json`(correctness succeeded-carrier、DEPENDENCY_NOT_PASSED、decoy false、domain_failure 三字段),经 `EvaluationResult.model_validate(context=...)` 强制契约规则。
- **Type consistency:** `digest_for` 名跨 verdict_cache/precheck/worker/tests 一致;`run_precheck`/`create_terminal_at_admission`/`synthesize_compile_error_result` 签名跨任务一致。
- **Known test-infra friction:** `make_completed_result` 的可变性(engine feedback 测试需要改 outcome)在 Step 1 标注了适应指引;routes 布线的集成级断言靠 `test_admission_shortcircuit.py` 的 broker+precheck 级测试兜住,routes 薄分支(一次 if)不单测(与既有 _build_command 测试策略一致)。
