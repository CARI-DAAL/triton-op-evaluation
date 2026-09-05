# Event Timing + Contract v2.1 (Plan E) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship contract v2.1 additively — `compile` check, `purpose=rollout`, `performance.method` (event default) + `measurement_method` on results, formal `SUSPECTED_POISON_CANDIDATE` — while the frozen 2.0 tree's bytes and digests stay bit-identical, plus the event-timing measurement path and the 2.0 sunset time-box.

**Architecture (chosen over parallel model classes):** ONE runtime model set in `api/contract.py`, extended additively (new fields Optional/None-default; new Literal members on version-gated unions). Per-version runtime validators reject 2.0 requests carrying 2.1 features. Serialization is versioned: a `strip_v21_fields` helper pops 2.1-only keys when `schema_version == "2.0"`, applied at every canonical-bytes boundary. The published 2.0 artifact tree keeps generating from a **frozen snapshot module** (`api/_frozen_v20.py`, verbatim copy of today's models) so its bytes/digests never move; the 2.1 tree generates from the live models. Startup/entrypoint/start.sh verify BOTH roots. The 2.0 sunset reuses the legacy-route time-box pattern (published constant + exact-instant + ≤90-day window); after sunset 2.0 requests get 400 INVALID_REQUEST.

**Tech Stack:** pydantic 2 (discriminated unions, model validators, `ValidationInfo` context), rfc8785 canonical JSON, pytest + fakeredis.

**Spec:** `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§6 v2.1, §9 event timing)

## Global Constraints

- **The 2.0 published tree is byte-frozen**: `contracts/evaluation-api/2.0/*` digests MUST NOT change (cross-repo consumers pin them). Any 2.0-tree byte change = plan defect.
- 2.0 responses must never carry 2.1-only fields (serialization strips them; tests assert stripped 2.0 dumps validate against the frozen 2.0 models).
- Version follows the request: `EvaluationResource.schema_version` is set from the request at admission; all constructors default "2.0".
- `schema_version` is part of the canonical request bytes (idempotency digest) — 2.0/2.1 replays never collide; do not "fix" this.
- Engine is vendored (ruff-excluded); minimal diff; event path is additive alongside the profiler path.
- 2.0 default measurement stays `profiler` (frozen absolute values); 2.1 default is `event` (spec §9).
- Full suite green each task except the 5 pre-existing environmental `referencing` failures; ruff line-length 100 on non-engine files.

---

### Task 1: Runtime contract extension + versioned serialization

**Files:**
- Modify: `triton_evaluation/api/contract.py` (extensions + validators + strip helper)
- Create: `triton_evaluation/api/_frozen_v20.py` (verbatim snapshot of the pre-extension models)
- Test: `tests/test_contract_v21.py` (create)

**Interfaces:**
- Consumes: existing contract models.
- Produces (consumed by Tasks 2-4):
  - `Check = Literal["correctness", "performance", "profile", "compile"]`; `Purpose = Literal["trial", "grade", "rollout"]`; `SchemaVersion = Literal["2.0", "2.1"]`.
  - `EvaluationRequest.schema_version: SchemaVersion`; `_normalize_and_couple` extended: 2.0 ⇒ checks ⊆ old three, purpose ≠ rollout, `performance.method is None`; performance⇔correctness coupling unchanged; `compile` independent; `checks` sorting order `("compile", "correctness", "performance", "profile")`.
  - `PerformanceParams.method: Literal["event", "profiler"] | None = None`.
  - `CompileOutcome` StrictModel: `outcome: Literal["compiled", "compile_error"]`, `compiler_stage: Literal["frontend", "backend"] | None = None`, `error_excerpt: str | None = None` (max 400 chars), `duration_ms: float | None = None`.
  - `PerformanceCase.measurement_method: Literal["npu_event_v1", "profiler_kernel_sum_v1"] | None = None`; `PerformanceResult.measurement_method` same, with validator: overall set ⇒ all measured cases share it; `UnmeasuredPerformanceCase` unchanged.
  - `EvaluationResult.compile: CompileCheckOutcome | None` where the result's check-union gains a `CheckSucceeded[CompileOutcome]` slot keyed `"compile"` (discriminated by status; `EvaluationResult._requested_checks_filled` accepts "compile").
  - `strip_v21_fields(dumped: dict) -> dict` — removes `compile` (from result), `measurement_method` (result + per-case), recursive into `result`; used for 2.0 canonical bytes.
  - `dump_for_version(model, version) -> bytes` — `model_dump(mode="json", exclude_none=False)` → strip if "2.0" → `canonical_json_bytes`.

- [ ] **Step 1: Create the frozen snapshot FIRST (before touching contract.py)**

```bash
cp triton_evaluation/api/contract.py triton_evaluation/api/_frozen_v20.py
```

Then edit ONLY the module docstring first line of `_frozen_v20.py` to: `"""Byte-frozen snapshot of the evaluation-api 2.0 models (published tree source). Do not edit; the 2.0 artifact digests pin these bytes' generated output."""` — and rename the module's internal model references are unnecessary (it's self-contained; nothing imports it except the artifact generator and tests).

Commit this snapshot alone: `git add triton_evaluation/api/_frozen_v20.py && git commit -m "chore: freeze 2.0 contract model snapshot for artifact generation"`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_contract_v21.py`:

```python
"""Contract v2.1: additive extensions, per-version purity, serialization stripping."""
import pytest

from triton_evaluation.api import _frozen_v20
from triton_evaluation.api.contract import (
    EvaluationRequest,
    EvaluationResource,
    EvaluationResult,
    PerformanceParams,
)


def _request(**overrides):
    payload = {
        "schema_version": "2.1",
        "op_name": "addmm",
        "reference_code": "import torch",
        "kernel_code": "import torch",
        "checks": ["correctness"],
    }
    payload.update(overrides)
    return payload


def test_v21_accepts_compile_check_and_rollout():
    request = EvaluationRequest.model_validate(_request(
        purpose="rollout", checks=["compile", "correctness"]))
    assert request.checks == ["compile", "correctness"]
    assert request.purpose == "rollout"


def test_v20_rejects_compile_check():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(_request(schema_version="2.0",
                                                  checks=["compile"]))


def test_v20_rejects_rollout_purpose():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(_request(schema_version="2.0",
                                                  purpose="rollout"))


def test_v20_rejects_performance_method():
    with pytest.raises(ValueError, match="2.0"):
        EvaluationRequest.model_validate(_request(
            schema_version="2.0", checks=["correctness", "performance"],
            performance={"method": "event"}))


def test_v21_defaults_method_and_accepts_profiler():
    request = EvaluationRequest.model_validate(_request(
        checks=["correctness", "performance"]))
    assert request.performance.method is None  # effective default resolved later
    request2 = EvaluationRequest.model_validate(_request(
        checks=["correctness", "performance"], performance={"method": "profiler"}))
    assert request2.performance.method == "profiler"


def _result_with_compile(version="2.1"):
    return {
        "compile": {"status": "succeeded",
                    "data": {"outcome": "compiled"},
                    "error": None, "reason": None},
        "correctness": {
            "status": "succeeded",
            "data": {"decoy": False, "domain_failure": None, "failures": [],
                     "outcome": "correct", "passed": True,
                     "passed_cases": 1, "total_cases": 1},
            "error": None, "reason": None,
        },
        "performance": None,
        "profile": None,
        "provenance": {
            "schema_version": version,
            "evaluator_build": "git:" + "a" * 40,
            "correctness_algorithm": "ca", "decoy_algorithm": "da",
            "benchmark_algorithm": "ba", "profiler_algorithm": "pa",
            "execution_environment": None, "grade_eligible": False,
            "manifest_id": None, "problem_id": None,
        },
    }


def test_result_accepts_compile_slot_and_strips_for_v20():
    from triton_evaluation.api.contract import dump_for_version
    validated = EvaluationResult.model_validate(
        _result_with_compile(),
        context={"requested_checks": ["compile", "correctness"]},
    )
    raw = validated.model_dump(mode="json", exclude_none=False)
    assert raw["compile"] is not None
    stripped = dump_for_version(validated, "2.0")
    import json
    assert json.loads(stripped).get("compile") is None
    # the stripped 2.0 dump must still validate against the FROZEN 2.0 models
    frozen = _frozen_v20.EvaluationResult.model_validate(
        json.loads(stripped), context={"requested_checks": ["correctness"]})


def test_performance_measurement_method_consistency():
    from triton_evaluation.api.contract import PerformanceResult
    base = {
        "aggregation": "geometric_mean_v1",
        "overall_speedup_vs_reference": 1.5,
        "selected_cases": 1, "measured_cases": 1, "coverage": 1.0,
        "measurement_method": "npu_event_v1",
        "per_case": [{
            "case_idx": 0, "input_shapes": [["n"]],
            "implementation_latency_ms": 1.0, "reference_latency_ms": 1.5,
            "speedup_vs_reference": 1.5,
            "measurement_method": "npu_event_v1",
        }],
        "unmeasured_cases": [],
    }
    assert PerformanceResult.model_validate(base).measurement_method == "npu_event_v1"
    bad = {**base, "per_case": [{**base["per_case"][0],
                                "measurement_method": "profiler_kernel_sum_v1"}]}
    with pytest.raises(ValueError):
        PerformanceResult.model_validate(bad)
```

- [ ] **Step 3: Run them — verify they fail**

Run: `python3 -m pytest tests/test_contract_v21.py -q` → collection/validation failures (no dump_for_version, no compile slot).

- [ ] **Step 4: Extend contract.py**

Precise deltas (all additive; do not reformat untouched code):

1. Near `Check`/`Purpose` literals (~lines 17-27):

```python
SchemaVersion = Literal["2.0", "2.1"]
Purpose = Literal["trial", "grade", "rollout"]
Check = Literal["correctness", "performance", "profile", "compile"]
_CHECK_ORDER = ("compile", "correctness", "performance", "profile")
MeasurementMethod = Literal["npu_event_v1", "profiler_kernel_sum_v1"]
```

2. `EvaluationRequest.schema_version: SchemaVersion` (was `Literal["2.0"]`).

3. `PerformanceParams` gains `method: Literal["event", "profiler"] | None = None`.

4. `_normalize_and_couple` (extend the existing validator, keep every current rule): after current logic add:

```python
        if self.schema_version == "2.0":
            if "compile" in self.checks:
                raise ValueError("schema_version 2.0 does not support the compile check")
            if self.purpose == "rollout":
                raise ValueError("schema_version 2.0 does not support purpose rollout")
            if self.performance is not None and self.performance.method is not None:
                raise ValueError("schema_version 2.0 does not support performance.method")
```

5. `CompileOutcome` model + a compile check outcome union member. The check envelope types are generic (`CheckSucceeded[T]`); add near the outcome models (~after CorrectnessResult, following existing style):

```python
class CompileOutcome(StrictModel):
    """Result of the v2.1 compile check (spec §6)."""

    outcome: Literal["compiled", "compile_error"]
    compiler_stage: Literal["frontend", "backend"] | None = None
    error_excerpt: Annotated[str, AfterValidator(utf8_len(400))] | None = None
    duration_ms: float | None = Field(default=None, ge=0)
```

6. `EvaluationResult` gains `compile: CheckSucceeded[CompileOutcome] | None = None` and `_requested_checks_filled` must treat "compile" as a legal requested slot (read the existing validator and extend its check-name set; unrequested ⇒ null rule applies as today).

7. `PerformanceCase.measurement_method: MeasurementMethod | None = None`; `PerformanceResult.measurement_method: MeasurementMethod | None = None` plus a validator that when overall is set, every measured per_case equals it.

8. Serialization helpers (module level, near canonical imports):

```python
_V21_ONLY_RESULT_KEYS = ("compile",)


def strip_v21_fields(dumped: dict) -> dict:
    """Remove v2.1-only keys for a 2.0 canonical serialization."""
    result = dumped.get("result")
    if isinstance(result, dict):
        for key in _V21_ONLY_RESULT_KEYS:
            result.pop(key, None)
        performance = result.get("performance")
        if isinstance(performance, dict):
            performance.pop("measurement_method", None)
            for case in performance.get("per_case") or []:
                if isinstance(case, dict):
                    case.pop("measurement_method", None)
    return dumped


def dump_for_version(model, version: str) -> bytes:
    dumped = model.model_dump(mode="json", exclude_none=False)
    if version == "2.0":
        dumped = strip_v21_fields(dumped)
    return canonical_json_bytes(dumped)
```

(The helper mutates the dumped dict — callers must pass a fresh dump; document that. `canonical_json_bytes` is already imported in contract.py? If not, import from contract_artifacts — check the existing import block.)

9. `EvaluationResource.schema_version: SchemaVersion` (was Literal["2.0"]); `Provenance.schema_version` and `ErrorEnvelope.schema_version` stay `Literal["2.0"]` in this task (envelope/provenance versioning lands with the artifact tree in Task 4 — note: EvaluationResult.provenance schema stays "2.0" even for 2.1 resources in v1; the 2.1 tree's provenance literal widens in Task 4. Keep a comment.)

- [ ] **Step 5: Run tests + full suite + commit**

Full suite must be green with UNCHANGED contracts/evaluation-api/2.0 bytes (the frozen snapshot guarantees generation identity — verified by existing artifact tests in Task 4 when the generator switches; for this task, `python scripts/generate_contract_artifacts.py --check` must still pass since the generator still uses the live models... WAIT: after extending contract.py, the live models generate DIFFERENT 2.0-tree bytes. So the generator MUST switch to the frozen snapshot in THIS task. Do it now: in `triton_evaluation/api/contract_artifacts.py`, change the model import to the frozen snapshot (`from ._frozen_v20 import ErrorEnvelope, EvaluationRequest, EvaluationResource`) and run `python scripts/generate_contract_artifacts.py --check` — it must report NO differences (proves snapshot fidelity). Then the full suite (test_contract_artifacts, test_cross_repo_contract, test_portable_schema_probe must all stay green — they now exercise the frozen snapshot path).

```bash
git add triton_evaluation/api/contract.py triton_evaluation/api/_frozen_v20.py \
        triton_evaluation/api/contract_artifacts.py tests/test_contract_v21.py
git commit -m "feat(contract): v2.1 additive runtime models + frozen 2.0 snapshot generation"
```

---

### Task 2: Event timing engine path + shaping projection

**Files:**
- Modify: `triton_evaluation/engine/benchmark.py` (event path in `measure_single`, `--measurement-method` CLI, per-case/overall `measurement_method` in result JSON)
- Modify: `triton_evaluation/evaluators/benchmark.py` (pass method through CLI)
- Modify: `triton_evaluation/evaluators/shaping.py` (project measurement_method; effective default by request version)
- Test: `tests/test_benchmark_event_stats.py` (create, pure statistics); `tests/test_shaping.py` (append); NPU: `tests/test_event_timing_npu.py` (create, marked npu)

**Interfaces:**
- Consumes: Task 1 contract fields.
- Produces: engine JSON `measurement_method` per per_shape_result + top-level; `method` CLI arg (`event`|`profiler`, default `profiler`); evaluator resolves effective method: request version 2.1 + params.method None → `event`; else given/`profiler`.

- [ ] **Step 1: Pure statistics tests**

Create `tests/test_benchmark_event_stats.py`:

```python
"""Event-timing statistics: median latency and CV stability gate (pure)."""
import pytest

from triton_evaluation.engine import benchmark


def test_median_and_cv_basic():
    assert benchmark._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert benchmark._median([5.0]) == 5.0
    assert round(benchmark._cv([1.0, 1.0, 1.0]), 6) == 0.0
    assert benchmark._cv([1.0, 2.0, 3.0]) > 0.3


def test_stable_gate_accepts_low_cv():
    assert benchmark._stable_latency([0.31, 0.30, 0.32, 0.31], cv_threshold=0.3) == 0.31


def test_stable_gate_rejects_high_cv():
    assert benchmark._stable_latency([0.10, 0.30, 0.90], cv_threshold=0.3) is None
```

- [ ] **Step 2: Run — verify they fail** (helpers missing).

- [ ] **Step 3: Engine helpers + event path**

In `triton_evaluation/engine/benchmark.py`:

```python
def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _cv(values: list[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return float("inf")
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5 / mean


def _stable_latency(samples_ms: list[float], cv_threshold: float) -> float | None:
    """Median of the iteration samples; None when the spread is unstable
    (surfaces as NO_STABLE_MEASUREMENT through the existing None-latency path)."""
    if len(samples_ms) < 3 or _cv(samples_ms) > cv_threshold:
        return None
    return round(_median(samples_ms), 4)
```

Event measurement in `measure_single` (branch on `ctx.method`):

```python
def _measure_event(test_fn, warmup: int, repeats: int, cv_threshold: float):
    """torch.npu.Event per-iteration timing (spec §9). Events are stream-ordered
    device timestamps; the test_fn's internal syncs do not pollute the window."""
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    for _ in range(warmup):
        test_fn()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        test_fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return _stable_latency(samples, cv_threshold)
```

`MeasureContext` gains `method: str = "profiler"` and `cv_threshold: float = 0.3`; `measure_single` branches: method == "event" → `latency_ms = _measure_event(test_fn, ctx.warmup, ctx.repeats, ctx.cv_threshold)` and `operators = None`; else the existing profiler path unchanged. `peak_memory` collection unchanged (both paths).

CLI (`main`, ~1077): accept `--measurement-method {event,profiler}` default `profiler` and `--cv-threshold` float default 0.3; thread into config/measure_single. `result_to_dict`: per-shape entry gains `measurement_method` ("npu_event_v1" if method event and latency not None else "profiler_kernel_sum_v1"); top-level gains the same for the aggregate.

- [ ] **Step 4: Evaluator passthrough + shaping projection**

`triton_evaluation/evaluators/benchmark.py`: resolve effective method — the evaluator's run() receives the request; parse `performance.method` and request schema_version from the stored canonical request (same JSON parse pattern as elsewhere): effective = params.method or ("event" if schema_version == "2.1" else "profiler"); add `--measurement-method {effective}` to the engine cmd.

`shaping.py`: `_ShapeMeasurement`/per-case projection gains `measurement_method` mapped from the engine per-shape value (validate against the MeasurementMethod literal; drop/None if absent for robustness with older engine JSON); overall `PerformanceResult.measurement_method` = top-level engine value when measured_cases > 0. Guard: for 2.0-versioned requests strip the field at serialization (Task 1's dump_for_version handles the wire; shaping may always set it — the strip makes 2.0 wire-safe).

- [ ] **Step 5: Shaping tests + commit**

Append to `tests/test_shaping.py`: a benchmark engine-JSON fixture with `measurement_method` fields projects into per-case + overall (round-trip through EvaluationResult model_validate with context). Run full suite; ruff on non-engine files.

```bash
git add triton_evaluation/engine/benchmark.py triton_evaluation/evaluators/benchmark.py \
        triton_evaluation/evaluators/shaping.py tests/test_benchmark_event_stats.py tests/test_shaping.py
git commit -m "feat(benchmark): event-timing measurement path + measurement_method projection"
```

- [ ] **Step 6: NPU-only test**

Create `tests/test_event_timing_npu.py` (pytestmark npu; skipif no torch_npu): run `_measure_event` over a trivial torch op on device npu with warmup=2 repeats=10 → latency not None, > 0; and a high-CV synthetic (interleave a cpu sleep? simplest: monkeypatched samples via directly testing _stable_latency already done — the NPU test only proves the Event mechanics work on real hardware).

---

### Task 3: compile check semantics + poison version-awareness

> **Erratum(2026-09-05,Task 1 评审裁决,执行原 Step 2 之前必须完成)**:
>
> **E-1 序列化改造(2.0 线合规,部署级阻断项)**:Task 1 后原始 `canonical_json_bytes(model_dump(...))` 站点在 2.0 响应上发出 `compile: null` / `measurement_method: null`,**违反已发布 2.0 schema**(additionalProperties: false)。把以下站点改为 `dump_for_version(model, model.schema_version)`(EvaluationResult 自身无 schema_version 字段,剥离开关一律跟随外层资源版本):
> - `triton_evaluation/broker.py` 四处资源/结果字节(~296-297、~385-386、~601-602、~789-790)
> - `triton_evaluation/api/routes.py:289、313`(GET 响应)
> - `triton_evaluation/worker.py:~472`(committed result 字节;~150 saved outcomes 是 per-check 载荷,核实其 dump 形状——per-check outcome 的 `data` 若含 v21 键则需同改,否则不动)
> 新增测试:构造 2.0 版终态资源,其响应 JSON 通过 `contracts/evaluation-api/2.0/schema.json` 的 jsonschema 校验(test deps 已含 jsonschema;参考 tests/contract_consumers/portable_schema_probe.py);同构 2.1 断言待到 Task 4 生成 2.1 树后补。
>
> **E-2 schema-version 回显**:`broker._new_queued_resource` 的 `schema_version="2.0"` 硬编码改为 `payload.get("schema_version", "2.0")`;`terminal_results` 两个合成函数加 `version: str = "2.0"` 参数(调用方传存储/请求版本;毒丸路径传解析出的 canonical 请求版本)。否则 2.1 响应被按 2.0 剥掉 v21 字段。
>
> **E-3 rollout 准入门语义(500 隐患)**:2.1 `purpose="rollout"` 请求过模型校验但 `admit()` 当 grade 调 `catalog.bind(None, ...)` → AttributeError → 未捕获 500。最小修法:`admission.py admit()` 中 rollout 与 trial 同待遇——要求 `trial:submit` scope、**不做 catalog 绑定**;routes 的 `catalog_binding_outcomes_total` bump 保持仅 grade;`broker.get_visible` owner 读权限 purpose rollout → `trial:read`;`routes._build_command` 的 is_rollout 扩展为 `purpose == "rollout" or caller.caller_id in settings.rollout_caller_ids`(spec §7:2.1 purpose=rollout 默认解析 rollout class;2.0 在模型层已拒绝 rollout)。测试:2.1 rollout 请求经 admit() 不绑定 catalog;无 trial:submit 的 caller 403;有 scope 时 traffic_class == "rollout"。

**Files:**
- Modify: `triton_evaluation/broker.py` (E-1 序列化改造;E-2 回显;原:precheck verdict stored on resource; version-aware poison synthesis; `result.compile` assembly helpers)
- Modify: `triton_evaluation/api/routes.py` (E-1 序列化改造;E-3 rollout 解析;原:compile-only terminal at admission; precheck result travels)
- Modify: `triton_evaluation/worker.py` (E-1 committed-result 序列化)
- Modify: `triton_evaluation/admission.py` (E-3 rollout scope/绑定语义)
- Modify: `triton_evaluation/terminal_results.py` (E-2 version 参数;原:compile slot in syntheses)
- Modify: `triton_evaluation/evaluators/evaluation.py` (on-card assembly of result.compile)
- Test: `tests/test_compile_check.py` (create);`tests/test_contract_v21.py` 追加 E-1 jsonschema 测试与 E-3 测试

**Interfaces:**
- Consumes: Tasks 1-2; Plan D precheck/verdict cache.
- Produces: `checks:["compile"]` executes via the admission precheck (never queues, never touches a card); `["compile", x,...]` stores the precheck verdict on the resource hash (`precheck_verdict`, `precheck_excerpt`, `precheck_duration_ms`); terminal assembly fills `result.compile`; poison quarantine uses `SUSPECTED_POISON_CANDIDATE` (retryable False) for 2.1 requests, WORKER_LOST for 2.0.

- [ ] **Step 1: Failing tests** — `tests/test_compile_check.py`:

```python
"""v2.1 compile check: admission-time execution via precheck, result assembly."""
import json

from conftest import make_caller, make_request

from triton_evaluation import precheck as precheck_mod
from triton_evaluation.api.contract import EvaluationRequest


def _v21_request(**overrides):
    payload = {
        "schema_version": "2.1", "op_name": "addmm",
        "reference_code": "import torch", "kernel_code": "import torch",
        "checks": ["compile"],
    }
    payload.update(overrides)
    return EvaluationRequest.model_validate(payload)


def test_compile_only_request_never_queues(store, settings, clock, make_command,
                                           make_worker):
    cmd = make_command(request=_v21_request())
    outcome = precheck_mod.run_precheck(store, settings,
        json.loads(cmd.admission.canonical_request)["kernel_code"],
        "triton_ascend_impl")
    # compiled verdict -> a compile-only terminal is assembled at admission
    assert outcome is None  # good module passes precheck
    # (terminal assembly assertions live in the routes-level test below)
```

(The heavy lifting tests: extend `tests/test_admission_shortcircuit.py` — a 2.1 compile-only request with a seeded compile_error verdict returns 200 terminal with `result.compile.data.outcome == "compile_error"` and correctness null; a compiled verdict + checks ["compile","correctness"] queues normally and the committed result carries `result.compile.data.outcome == "compiled"` (worker path via StubEvaluator + precheck verdict field on the resource); a 2.1 poisoned resource (two suspicious deaths with claimed_card) terminals with `correctness.error.code == "SUSPECTED_POISON_CANDIDATE"` and retryable False, while the identical 2.0 resource terminals with WORKER_LOST.)

- [ ] **Step 2: Implementation**

1. Broker `_new_queued_resource` mapping gains `precheck_verdict=""`, `precheck_excerpt=""`, `precheck_duration_ms=""` (advisory, empty = none).
2. Routes: for 2.1 requests with "compile" in checks — run `precheck.run_precheck` at admission (already invoked for cache reasons; capture verdict + duration); write the three fields via a new advisory broker method `note_precheck_result(evaluation_id, verdict, excerpt, duration_ms)` after `create_or_replay` returns created (not for replays). If verdict == "compile_error" and checks == ["compile"] → assemble terminal immediately via `create_terminal_at_admission` with a result whose `compile` slot carries outcome compile_error (+ excerpt, stage "frontend", duration) and nothing else requested. If verdict == "compile_error" and other checks requested → the existing short-circuit already fires for non-profile... reconcile: extend the existing short-circuit synthesis to ALSO fill the compile slot for 2.1 (2.0 responses strip it automatically via dump_for_version).
3. Worker `_run_attempt`/`evaluators/evaluation.py`: after a successful on-card run where "compile" was requested, set `result.compile` = CheckSucceeded(CompileOutcome(outcome="compiled")) unless correctness outcome == "compile_error" (then outcome compile_error, stage "backend", excerpt from domain_failure message). Assemble via the contract models; validate with the requested_checks context (which now includes "compile").
4. Poison: `_recover_one_lease` poison path reads `json.loads(stored[b"canonical_request"])["schema_version"]`; for "2.1" synthesize with a NEW terminal_results helper `synthesize_poison_result(..., code="SUSPECTED_POISON_CANDIDATE", retryable=False, message=...)`; 2.0 keeps the current WORKER_LOST shape verbatim.

- [ ] **Step 3: Run + full suite + commit**

```bash
git add triton_evaluation/broker.py triton_evaluation/api/routes.py \
        triton_evaluation/terminal_results.py triton_evaluation/evaluators/evaluation.py \
        tests/test_compile_check.py tests/test_admission_shortcircuit.py
git commit -m "feat: v2.1 compile check (admission precheck execution) + poison code"
```

---

### Task 4: Dual artifact trees + 2.0 sunset + fixtures + docs

**Files:**
- Modify: `triton_evaluation/api/contract_artifacts.py` (parameterized builder: v20 from `_frozen_v20`, v21 from live models; `CONTRACT_ROOT_V21`, `SCHEMA_ID_V21`)
- Modify: `scripts/generate_contract_artifacts.py` (write/check BOTH roots)
- Modify: `triton_evaluation/api/app.py`, `docker/entrypoint.sh`, `scripts/start.sh` (verify both roots)
- Modify: `triton_evaluation/config.py` (v20 sunset fields + validator mirroring legacy pattern)
- Modify: `triton_evaluation/api/routes.py` (post-sunset 2.0 rejection)
- Create: `contracts/evaluation-api/2.1/*` (generated)
- Test: extend `tests/test_contract_artifacts.py` (EXPECTED_FIXTURES_V21), `tests/test_cross_repo_contract.py` (2.1 digest pinning), `tests/test_contract_v21.py` (sunset validator)
- Modify: README, docker.env.example

- [ ] **Step 1: Parameterize the generator**

`contract_artifacts.py`: extract `build_artifacts(models_module, schema_version, schema_id)` — v20 call passes `_frozen_v20` + `"2.0"` + the existing URN; add `CONTRACT_ROOT_V21 = Path("contracts/evaluation-api/2.1")`, `SCHEMA_ID_V21 = "urn:triton-evaluation:evaluation-api:2.1:schema"`. v21 fixtures: extend `_fixtures()` with four new builders — `request-trial-rollout.json` (purpose rollout), `request-trial-compile-only.json` (checks ["compile"]), `resource-completed-compile-check.json` (result.compile outcome compiled + measurement_method on performance), `resource-failed-poison.json` (2.1 poison terminal, code SUSPECTED_POISON_CANDIDATE). Keep every existing 2.0 fixture byte-identical (they generate from the frozen snapshot — `--check` proves it).

`generate_contract_artifacts.py`: `--write` writes both roots; `--check` verifies both.

- [ ] **Step 2: Sunset mechanism**

`config.py` (mirror the legacy pattern exactly):

```python
V20_SUNSET_AT = datetime(2027, 3, 31, 0, 0, 0, tzinfo=timezone.utc)  # published instant; tighten per release plan
```

plus fields `v20_sunset_at: datetime | None = None` with validator rules identical in shape to legacy: enabling... actually inverse — 2.0 is DEFAULT-ON today; the time-box needs `v20_sunset_at` always set in production? Smallest viable: sunset is enforcement-time, not enable-time: validator requires `v20_sunset_at == V20_SUNSET_AT` whenever set; routes rejects 2.0 requests with `now > sunset` (monkeypatchable `_utcnow`); deploys set the env var to arm it; unset = unarmed (pre-sunset dev). Window check: `(sunset - now).days > 90` rejected — but that forces quarterly re-release; per spec "窗口有上限" keep 90d but allow the constant to move per release (it's a published constant, like LEGACY_SUNSET_AT). Tests mirror tests/test_legacy_routes.py's two patterns.

- [ ] **Step 3: Dual verification**

app.py lifespan: verify both roots, set `artifacts_verified` only if both clean. entrypoint.sh `verify_readiness` + start.sh heredoc: same dual check. Run `python scripts/generate_contract_artifacts.py --write` then `--check`; full suite.

- [ ] **Step 4: Docs**

README: v2.1 section (new fields, measurement methods + cross-method incomparability, compile check semantics, 2.0 sunset date). docker.env.example: `TRITON_EVAL_V20_SUNSET_AT`. Spec §6/§9 errata notes.

```bash
git add -A contracts/ triton_evaluation/ scripts/ tests/ README.md docker.env.example
git commit -m "feat(contract): publish v2.1 artifact tree + 2.0 sunset time-box"
```

---

## Self-Review Notes (author check, completed at plan-write time)

- **2.0 byte-frozen guarantee**: the snapshot module is copied BEFORE any contract.py edit (Task 1 Step 1) and the generator switches to it in the same task; `--check` passing after the runtime extension is the fidelity proof. test_cross_repo_contract's on-disk digest tests then pin the tree forever.
- **Additive-only on the wire**: every new request field is Optional/None-default or rejected for 2.0 by validators; every new response field is stripped for 2.0 by dump_for_version. Task 1's strip test round-trips through the frozen models.
- **Schema-version echo**: resources take the request's version at admission (broker reads payload["schema_version"], defaulting "2.0"); Provenance/ErrorEnvelope stay "2.0" literals in v1 (documented; the 2.1 tree generation tolerates this since schema derives from the request/resource models only).
- **Known hard parts** flagged for implementers: discriminated-union extension for the compile slot (Task 1 Step 4.6 — follow `_requested_checks_filled`'s existing pattern); the fake benchmark fixture's minimal shape (Task 2 tests must build richer per-case dicts where shaping is exercised); sunset validator asymmetry (2.0 default-on → time-box is enforcement-time, documented in Step 2).
- **Pre-existing failures**: the 5 environmental `referencing` failures; on torch-less hosts the Task 2 NPU test skips by marker.
