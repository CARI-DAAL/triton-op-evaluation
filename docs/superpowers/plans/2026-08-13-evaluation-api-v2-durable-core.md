# Evaluation API v2 Durable Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the frozen Evaluation API 2.0 contract and implement a crash-durable, authenticated, idempotent `EvaluationResource` whose worker attempts are deadline-bound and fenced.

**Architecture:** FastAPI owns only authentication, admission, short waiting, and resource reads. One Redis instance is the durable source of truth for idempotency records, resources, the FIFO ready set, attempts, leases, worker heartbeats, and bounded metrics; every state transition uses Redis `WATCH`/`MULTI` compare-and-set and Redis AOF `appendfsync always`. Workers parse the stored canonical request, run the existing private engines, shape their JSON exactly once into strict public models, and commit through an attempt token that stale workers cannot reuse. The published schema and fixtures are generated from those models with RFC 8785 canonical bytes and are the only cross-repository contract.

**Tech Stack:** Python 3.10–3.12; `fastapi==0.139.2`, `uvicorn==0.51.0`, `redis==8.0.1`, `pydantic==2.13.4`, `pydantic-settings==2.14.2`, `pytest==9.1.1`, `fakeredis==2.37.0`, `httpx==0.28.1`, `ruff==0.16.0`, `jsonschema==4.26.0`, `rfc8785==0.1.4`; Redis 7 AOF; Docker Compose; pure-stdlib deployment preflight.

## Global Constraints

- The sole design authority is `docs/specs/2026-08-06-evaluation-api-v2-design.md`. Do not copy behavior from the deleted synchronous plan or reinterpret the older July documents as v2 requirements.
- This batch ends at Phase 0 contract publication plus the durable Evaluation resource core. Do not add cancellation, priorities, dynamic manifest APIs, workflow DAGs, multi-host disaster recovery, TLS termination, tenant administration, Redis Cluster, or a general identity platform.
- Preserve the vendored engine algorithms unless a task explicitly changes a private JSON field needed for coarse diagnostics. Never reformat `triton_evaluation/engine/` wholesale.
- Public request models are strict and reject unknown fields and coercion. Public response readers in other repositories are tolerant within major version 2, but this service's output validation remains strict.
- Never log or meter source code, test inputs, credentials, absolute paths, internal addresses, or full session IDs. Public errors are single-line, byte-bounded, redacted `ErrorEnvelope` values.
- `X-Session-ID` is correlation and a secondary quota key only. Authorization, visibility, idempotency, and primary quotas use the authenticated `caller_id`.
- The service promises durable at-least-once attempts and one fenced terminal winner. It must never claim exactly-once execution.
- Redis `WATCH`/`MULTI` is the one transaction mechanism for this batch. Do not introduce Lua scripts, a second queue, or an in-memory fallback. A Redis error before an accepted transaction is acknowledged fails admission; it never creates a best-effort resource.
- A nonterminal resource and its idempotency record have no Redis expiry. The terminal transition atomically applies the configured retention, which is at least 86,400 seconds and at least `max_recovery_age_seconds`.
- Every task starts with a focused RED test, demonstrates the expected failure, implements only that task's behavior, demonstrates GREEN, and makes the listed small commit. Do not combine task commits.
- Before Task 1, require `git status --short` to contain no uncommitted file except this plan. If spec/doc closure edits remain, stop for the repository owner to checkpoint them; never stage or absorb pre-existing work into a task commit.
- Do not modify the canonical design spec while executing this plan. Documentation changes are restricted to the operational and migration documents named below.
- Commands run from repository root: `/home/yubaifeng/e84381970/projects/triton-op-evaluation/.claude/worktrees/feat-evaluation-api-v2`.
- Task 1 begins by replacing every direct runtime/test/dev dependency range in `pyproject.toml` with the exact reviewed versions in its setup step, then creates the ignored, isolated `.venv` with no system-site packages. Task 2 adds the two contract-publication dependencies only after its intentional missing-import RED. Direct dependencies and tools are exact and asserted; transitive packages remain resolver-managed and are not represented as a fully locked environment.
- Except for `/usr/bin/python3` creating `.venv`, every Python, pytest, Ruff, generator, and release-gate command in this plan uses `.venv/bin/python`. Never fall back to a shared interpreter after bootstrap.
- Deterministic CPU CI is a merge gate. Target-NPU calibration and crash/fencing soak are a production-release gate, not a reason to weaken or skip CPU tests.

---

### Task 1: Freeze strict request, result, resource, and error models

**Files:**
- Modify: `pyproject.toml`
- Modify: `triton_evaluation/api/contract.py`
- Modify: `tests/test_contract.py`
- Create: `tests/test_contract_invariants.py`

**Interfaces:**

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )

class EvaluationRequest(StrictModel):
    schema_version: Literal["2.0"]
    purpose: Literal["trial", "grade"] = "trial"
    op_name: str = Field(min_length=1, max_length=128)
    reference_code: str
    kernel_code: str
    triton_impl_name: str = Field(default="triton_ascend_impl", min_length=1, max_length=128)
    checks: list[Check] = Field(min_length=1, max_length=3)
    non_compute: bool = False
    deadline_seconds: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    grade_context: GradeContext | None = None

class EvaluationResource(StrictModel):
    schema_version: Literal["2.0"]
    evaluation_id: str
    revision: int = Field(ge=1)
    state: Literal["queued", "running", "completed", "failed", "timed_out"]
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    deadline_at: datetime
    request_digest: Digest
    reference_digest: Digest
    result: EvaluationResult | None
```

`contract.py` remains the only public model import surface. It must also export all named models in design sections 3.4 and 4: `ErrorInfo`, `ErrorEnvelope`, `PerformanceParams`, `ProfileParams`, `GradeContext`, four discriminated check outcome variants, `SkipInfo`, correctness/domain diagnostic models, performance models, both profile branches, `ExecutionEnvironment`, `Provenance`, and `EvaluationResult`. Use `typing.Generic` rather than Python 3.12 generic syntax so Python 3.10 remains supported.

- [ ] Freeze the direct dependency baseline and bootstrap the isolated executor before writing Task 1 tests. Replace the current ranges with exactly this `pyproject.toml` content; deliberately do not add `rfc8785` or `jsonschema` until Task 2's missing-import RED has run:

```toml
dependencies = [
    "fastapi==0.139.2",
    "uvicorn==0.51.0",
    "redis==8.0.1",
    "pydantic==2.13.4",
    "pydantic-settings==2.14.2",
]

[project.optional-dependencies]
test = ["pytest==9.1.1", "fakeredis==2.37.0", "httpx==0.28.1"]
dev = ["ruff==0.16.0"]
```

Run these commands once from the clean reviewed worktree. Stop rather than reuse or delete a pre-existing `.venv`:

```bash
test ! -e .venv
/usr/bin/python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version'
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,dev]'
.venv/bin/python -c 'from importlib.metadata import version; expected={"fastapi":"0.139.2","uvicorn":"0.51.0","redis":"8.0.1","pydantic":"2.13.4","pydantic-settings":"2.14.2","pytest":"9.1.1","fakeredis":"2.37.0","httpx":"0.28.1","ruff":"0.16.0"}; actual={name:version(name) for name in expected}; assert actual == expected, (actual, expected)'
.venv/bin/python -c 'from importlib.util import find_spec; assert find_spec("rfc8785") is None; assert find_spec("jsonschema") is None'
```

Expected: the supported-version check and exact nine-version assertion exit 0, both contract-publication modules remain absent for Task 2 RED, and `git status --short` shows only the reviewed plan plus the intentional `pyproject.toml` Task 1 change. `.venv` is ignored.

- [ ] Replace the old contract tests with RED parameterized tests for strict scalar types, unknown request fields, missing/duplicate/invalid check combinations, parameter/check coupling, trial/grade context coupling, 2 MiB UTF-8 source limits, 128 case indices, nonstandard JSON `NaN`/`Infinity`, and finite ordinary floats.

```python
@pytest.mark.parametrize(
    "patch",
    [
        {"deadline_seconds": "900"},
        {"non_compute": 1},
        {"checks": ["performance"]},
        {"checks": ["correctness", "correctness"]},
        {"checks": ["correctness"], "performance": {"warmup": 1, "repeats": 2}},
        {"purpose": "trial", "grade_context": GRADE_CONTEXT},
        {"purpose": "grade", "grade_context": None},
    ],
)
def test_request_rejects_invalid_contract_combinations(valid_request, patch):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid_request | patch)
```

- [ ] Add RED invariant matrices covering all five resource states, requested versus unrequested outcomes, four check outcome discriminants, five correctness outcomes, domain failure code/phase/case rules, maximum eight sorted samples, finite sentinel encoding, performance geometric-mean/coverage arithmetic, and op/deep profile count/ratio rules. Explicitly assert that serialized models contain neither `mean_abs_err` nor `throughput`.
- [ ] Run the focused tests and confirm they fail against the current permissive synchronous contract.

```bash
.venv/bin/python -m pytest tests/test_contract.py tests/test_contract_invariants.py -q
```

Expected: failures include missing `schema_version`, accepted coercion, old terminal status names, old divergence kinds, and missing resource/check outcome models.

- [ ] Implement `StrictModel` and every design model. Use model validators for cross-field rules; a shared `utf8_len()` validator enforces byte rather than code-point limits, and `redacted_text()` rejects CR/LF and enforces 256-byte skip or 512-byte domain/error messages. Use `Annotated[float, Field(allow_inf_nan=False)]` for ordinary floats and the explicit string union for non-finite samples.
- [ ] Make `EvaluationRequest` normalize check order to `correctness`, `performance`, `profile` and case indices to ascending order only after rejecting duplicates. Omitted requested parameter objects become explicit defaults; unrequested parameter objects must be absent or `null`, never silently discarded.
- [ ] Validate resource lifecycle shape: queued/running have no result or finish time; terminal states have both; `started_at` is optional only for queued deadline; each requested check is non-null and each unrequested check is null. Pass requested checks as Pydantic validation context when reading a resource.
- [ ] Run the focused suite and static lint.

```bash
.venv/bin/python -m pytest tests/test_contract.py tests/test_contract_invariants.py -q
.venv/bin/python -m ruff check triton_evaluation/api/contract.py tests/test_contract.py tests/test_contract_invariants.py
```

Expected: all focused tests pass and Ruff emits no diagnostics.

- [ ] Commit the contract model boundary.

```bash
git add pyproject.toml triton_evaluation/api/contract.py tests/test_contract.py tests/test_contract_invariants.py
git commit -m "feat(contract): freeze evaluation api 2.0 models"
```

### Task 2: Generate and pin reproducible contract artifacts

**Files:**
- Modify: `pyproject.toml`
- Create: `triton_evaluation/api/contract_artifacts.py`
- Create: `scripts/generate_contract_artifacts.py`
- Create: `tests/test_contract_artifacts.py`
- Create: `contracts/evaluation-api/2.0/schema.json`
- Create: `contracts/evaluation-api/2.0/fixtures/index.json`
- Create: `contracts/evaluation-api/2.0/fixtures/request-trial-minimal.json`
- Create: `contracts/evaluation-api/2.0/fixtures/request-grade-full.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-queued.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-running.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-correct.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-wrong.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-decoy.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-compile-error.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-runtime-error.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-execution-timeout.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-performance-profile-op.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-completed-profile-deep-partial.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-failed.json`
- Create: `contracts/evaluation-api/2.0/fixtures/resource-timed-out.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-invalid-request.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-unauthenticated.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-forbidden.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-manifest-binding-failed.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-not-found.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-idempotency-conflict.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-limit-exceeded.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-rate-limited.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-not-ready.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-queue-unavailable.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-worker-lost.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-evaluator-execution-failed.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-invalid-engine-output.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-deadline-exceeded.json`
- Create: `contracts/evaluation-api/2.0/fixtures/error-internal.json`
- Create: `contracts/evaluation-api/2.0/contract.json`

**Interfaces:**

```python
CONTRACT_ROOT = Path("contracts/evaluation-api/2.0")

def canonical_json_bytes(value: object) -> bytes:
    return rfc8785.dumps(value)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def build_artifacts() -> dict[PurePosixPath, bytes]:
    """Return every publication path and its final bytes without writing."""

def write_artifacts(root: Path) -> None:
    """Atomically replace only files returned by build_artifacts()."""

def verify_artifacts(root: Path) -> list[str]:
    """Return sorted missing, unexpected, or byte-different relative paths."""
```

The contract-generator toolchain is frozen to Task 1's `pydantic==2.13.4`, plus `rfc8785==0.1.4` in project dependencies and `jsonschema==4.26.0` in the test extra. Pydantic owns the generated JSON Schema bytes, `rfc8785.dumps()` is the only JSON canonicalizer, and jsonschema validates the three published consumer entrypoints. `json.dumps(sort_keys=True)` is not RFC 8785 and must not be used for publication or digest input. All direct runtime/test/dev dependencies remain at Task 1's exact reviewed versions; this task does not claim a transitive dependency lock.

- [ ] Add RED RFC 8785 tests before installing the dependency. Cover Appendix B representations for `333333333.33333329 -> b"333333333.3333333"`, `1e+30 -> b"1e+30"`, `4.50 -> b"4.5"`, `2e-3 -> b"0.002"`, and `1e-27 -> b"1e-27"`; after canonicalizing an object with shuffled keys, assert `list(json.loads(encoded).keys()) == ["\r", "1", "\u0080", "ö", "€", "😀", "דּ"]`, the RFC 8785 UTF-16 order; cover invalid surrogate and non-finite rejection.
- [ ] Add RED artifact tests that assert the exact four publication path groups, exact `contract.json` keys, recomputable digests, POSIX-relative fixture paths, a sorted unique fixture inventory, and `verify_artifacts()` detecting one changed byte and one unexpected file. Resolve and validate each consumer entrypoint independently: request at `schema.json#/$defs/EvaluationRequest`, resource at `schema.json#/$defs/EvaluationResource`, and error at `schema.json#/$defs/ErrorEnvelope`; do not validate consumers against an ambiguous top-level union.

The fixture inventory is fixed to these names:

```text
request-trial-minimal.json
request-grade-full.json
resource-queued.json
resource-running.json
resource-completed-correct.json
resource-completed-wrong.json
resource-completed-decoy.json
resource-completed-compile-error.json
resource-completed-runtime-error.json
resource-completed-execution-timeout.json
resource-completed-performance-profile-op.json
resource-completed-profile-deep-partial.json
resource-failed.json
resource-timed-out.json
error-invalid-request.json
error-unauthenticated.json
error-forbidden.json
error-manifest-binding-failed.json
error-not-found.json
error-idempotency-conflict.json
error-limit-exceeded.json
error-rate-limited.json
error-not-ready.json
error-queue-unavailable.json
error-worker-lost.json
error-evaluator-execution-failed.json
error-invalid-engine-output.json
error-deadline-exceeded.json
error-internal.json
```

Together these fixtures cover every resource state, every correctness domain outcome, every check outcome discriminant, dependency and deadline skips, both profile discriminants, performance coverage, and every initial stable error code. `index.json` itself is not an indexed fixture.

- [ ] Run tests once and confirm import/artifact failures.

```bash
.venv/bin/python -m pytest tests/test_contract_artifacts.py -q
```

Expected: collection fails because `rfc8785` and `contract_artifacts` do not exist.

- [ ] Keep Task 1's exact direct dependency pins unchanged, add `rfc8785==0.1.4` to project dependencies, add `jsonschema==4.26.0` to the existing test extra, reinstall both test and dev extras, and verify the complete direct dependency/tool set.

Do not add another Pydantic entry or loosen any Task 1 pin. `jsonschema==4.26.0` is a test/portable-consumer dependency for Draft 2020-12 entrypoint validation; it is not a service runtime dependency. The final direct dependency groups are exactly:

```toml
dependencies = [
    "fastapi==0.139.2",
    "uvicorn==0.51.0",
    "redis==8.0.1",
    "pydantic==2.13.4",
    "pydantic-settings==2.14.2",
    "rfc8785==0.1.4",
]

[project.optional-dependencies]
test = ["pytest==9.1.1", "fakeredis==2.37.0", "httpx==0.28.1", "jsonschema==4.26.0"]
dev = ["ruff==0.16.0"]
```

```bash
.venv/bin/python -m pip install -e '.[test,dev]'
.venv/bin/python -c 'from importlib.metadata import version; expected={"fastapi":"0.139.2","uvicorn":"0.51.0","redis":"8.0.1","pydantic":"2.13.4","pydantic-settings":"2.14.2","rfc8785":"0.1.4","pytest":"9.1.1","fakeredis":"2.37.0","httpx":"0.28.1","jsonschema":"4.26.0","ruff":"0.16.0"}; actual={name:version(name) for name in expected}; assert actual == expected, (actual, expected)'
```

Expected: editable install succeeds and the exact eleven-version assertion exits 0. The three contract-generator packages are Pydantic 2.13.4, RFC 8785 0.1.4, and jsonschema 4.26.0; no direct Task 1 pin changed.

- [ ] Implement the pure builder. Produce one Draft 2020-12 `schema.json` with stable `$id`, sorted `$defs`, and a top-level `oneOf` that references `#/$defs/EvaluationRequest`, `#/$defs/EvaluationResource`, and `#/$defs/ErrorEnvelope`. Those three `$defs` JSON Pointers are the frozen consumer entrypoints; each closure must include every nested referenced definition. Serialize schema and every fixture as RFC 8785 bytes with no trailing newline.
- [ ] Define `fixtures/index.json` as RFC 8785 bytes of this exact shape, where paths are relative to `contracts/evaluation-api/2.0` and entries are lexicographically sorted by `name`:

```python
index_object = {
    "schema_version": "2.0",
    "fixtures": [
        {
            "name": fixture_name,
            "path": fixture_path.as_posix(),
            "sha256": sha256_hex(artifact_bytes[fixture_path]),
        }
        for fixture_name, fixture_path in sorted(fixture_paths.items())
    ],
}
index_bytes = canonical_json_bytes(index_object)
```

The shown entry illustrates the fixed field shape; the generated index contains all 29 names above. Its digest is `sha256(canonical_json_bytes(index_object))`, which is also the SHA-256 of the exact `index.json` file bytes.
- [ ] Define `contract.json` with exactly these five keys and no others:

```python
contract_bytes = canonical_json_bytes(
    {
        "evaluation_api_schema_version": "2.0",
        "evaluation_api_schema_digest": sha256_hex(schema_bytes),
        "evaluation_api_golden_fixture_digest": sha256_hex(index_bytes),
        "json_schema_path": "schema.json",
        "golden_fixture_index_path": "fixtures/index.json",
    }
)
```

- [ ] Make the CLI accept only `--write` or `--check`. `--write` uses same-directory temporary files plus `os.replace`; `--check` never writes and exits 1 with sorted relative paths if the generated file set or bytes differ.
- [ ] Generate once, then prove deterministic rebuild and model round-trip.

```bash
.venv/bin/python scripts/generate_contract_artifacts.py --write
.venv/bin/python scripts/generate_contract_artifacts.py --check
.venv/bin/python -m pytest tests/test_contract.py tests/test_contract_invariants.py tests/test_contract_artifacts.py -q
git diff --exit-code -- contracts/evaluation-api/2.0
```

Expected: `--check` reports `contract artifacts are current`; all tests pass; the final diff command is empty because the second generation did not change bytes.

- [ ] Commit the Phase 0 publication as a reviewable unit.

```bash
git add pyproject.toml triton_evaluation/api/contract_artifacts.py scripts/generate_contract_artifacts.py tests/test_contract_artifacts.py contracts/evaluation-api/2.0
git commit -m "feat(contract): publish evaluation api 2.0 artifacts"
```

### Task 3: Authenticate capabilities and validate canonical grade admission

**Files:**
- Modify: `triton_evaluation/config.py`
- Modify: `tests/conftest.py`
- Create: `triton_evaluation/security.py`
- Create: `triton_evaluation/catalog.py`
- Create: `triton_evaluation/admission.py`
- Create: `tests/test_security.py`
- Create: `tests/test_catalog.py`
- Create: `tests/test_admission.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Caller:
    caller_id: str
    caller_class: Literal["trial", "trusted", "observer"]
    scopes: frozenset[Literal[
        "trial:submit", "trial:read", "grade:submit", "grade:read", "observer:read"
    ]]

class CapabilityRegistry:
    @classmethod
    def from_secret_file(cls, path: Path) -> "CapabilityRegistry":
        """Reject duplicate token digests/scopes and conflicting caller definitions."""

    def authenticate(self, authorization: str | None) -> Caller:
        """Parse exactly one Bearer token and compare SHA-256 digests in constant time."""

class ReleaseCatalog:
    @classmethod
    def load(cls, root: Path, manifest_path: Path) -> "ReleaseCatalog":
        """Validate immutable unique catalog entries and reference bytes."""

    def bind(self, context: GradeContext, reference_code: str) -> CatalogBinding:
        """Require exact manifest/problem/reference triple and body digest."""

@dataclass(frozen=True)
class CanonicalAdmission:
    canonical_request: bytes
    request_digest: str
    reference_digest: str
    requested_checks: Sequence[Check]
    catalog_binding: CatalogBinding | None

def admit(request: EvaluationRequest, caller: Caller, catalog: ReleaseCatalog) -> CanonicalAdmission:
    """Authorize purpose, validate grade binding, and canonicalize the model."""
```

Capability secret JSON is fixed as `{"capabilities":[{"token":"opaque","caller_id":"stable","caller_class":"trial","scopes":["trial:submit","trial:read"]}]}`. It is a mounted secret, never a response schema or dynamic API. The catalog JSON is fixed as `{"schema_version":"1.0","entries":[{"manifest_id":"release-1","problem_id":"addmm-1","reference_digest":"64-hex","reference_path":"references/addmm-1.py"}]}`.

Caller IDs match `^[A-Za-z0-9._:-]{1,128}$`. Trial callers may hold only `trial:submit` and `trial:read`; trusted callers may hold `grade:submit`, `grade:read`, and `observer:read`; observer callers must hold `observer:read` and may additionally receive `grade:submit` and `grade:read`. At least one scope is required. In this batch, `observer:read` is the explicit global read policy across trial and grade resources; it never grants submit by implication.

- [ ] Write RED tests for missing/malformed/wrong Bearer credentials, scope denial, constant-time digest lookup behavior, duplicate capability tokens/scopes, conflicting definitions for one caller, and secret-file mode broader than `0600` in production validation. Permit two rotation tokens for one caller only when caller class and scopes are identical. Assert no login or token-minting route is implied by these classes.
- [ ] Write RED catalog tests for unique manifest identity, unique triples, duplicate manifest rebinding, missing files, digest mismatch, absolute paths, `..` traversal, symlink escape, and a valid UTF-8 artifact beneath the read-only root.
- [ ] Write RED admission tests proving request canonicalization collapses defaults/check order/case order but preserves source bytes; excludes wait/session/headers; isolates trial and grade scopes; rejects trial grade context and grade without context; and independently checks both catalog triple and body reference digest.
- [ ] Run the tests and confirm missing modules.

```bash
.venv/bin/python -m pytest tests/test_security.py tests/test_catalog.py tests/test_admission.py -q
```

Expected: collection fails for the three new modules.

- [ ] Expand `Settings` with strict positive retention, recovery, lease, heartbeat, maintenance/poll interval, quota, worker-environment, algorithm-version, secret/catalog path, evaluator build, and legacy-window fields. Validate `retention_seconds >= 86400`, `max_recovery_age_seconds <= retention_seconds`, `heartbeat_interval_seconds < lease_seconds`, intervals in `[0.05, 5]`, and all rate/inflight limits are nonzero. Require evaluator build to match `sha256:[0-9a-f]{64}` or `git:[0-9a-f]{40}` and each algorithm version to match `[A-Za-z0-9._:-]{1,128}`.
- [ ] Implement capability loading. Store only token SHA-256 bytes in memory after parsing; compare each configured digest with `secrets.compare_digest`; return the generic `UNAUTHENTICATED` public error for every credential failure. Never include token text in exceptions or `repr`.
- [ ] Implement catalog loading with `Path.resolve(strict=True)`, require the resolved file to remain under the resolved catalog root, reject non-files and non-UTF-8 artifacts, and hash exact file bytes. A manifest ID may have multiple problems but the same `(manifest_id, problem_id)` can appear once and cannot rebind.
- [ ] Implement `admit()`. It calls `request.model_dump(mode="json", exclude_none=False)`, then `canonical_json_bytes()` after model validators have ordered checks/cases. It computes lowercase SHA-256 over those exact bytes and over `reference_code.encode("utf-8")`. It requires `trial:submit` for trial and `grade:submit` plus catalog binding for grade.
- [ ] Run focused tests and lint.

```bash
.venv/bin/python -m pytest tests/test_security.py tests/test_catalog.py tests/test_admission.py -q
.venv/bin/python -m ruff check triton_evaluation/config.py triton_evaluation/security.py triton_evaluation/catalog.py triton_evaluation/admission.py tests/conftest.py tests/test_security.py tests/test_catalog.py tests/test_admission.py
```

Expected: all tests pass; no source, token, reference path, or full session value appears in captured logs/errors.

- [ ] Commit admission primitives.

```bash
git add triton_evaluation/config.py triton_evaluation/security.py triton_evaluation/catalog.py triton_evaluation/admission.py tests/conftest.py tests/test_security.py tests/test_catalog.py tests/test_admission.py
git commit -m "feat(admission): enforce capability and manifest binding"
```

### Task 4: Atomically create, replay, and read durable resources

**Files:**
- Modify: `triton_evaluation/models.py`
- Modify: `triton_evaluation/broker.py`
- Modify: `tests/test_broker.py`
- Create: `tests/test_idempotency.py`
- Create: `tests/test_resource_visibility.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CreateResourceCommand:
    evaluation_id: str
    caller_id: str
    caller_class: Literal["trial", "trusted", "observer"]
    traffic_class: Literal["trial", "trusted"]
    session_id: str
    idempotency_key: str
    admission: CanonicalAdmission
    evaluator_build: str
    algorithms: AlgorithmVersions

@dataclass(frozen=True)
class CreateResourceResult:
    decision: Literal["created", "replayed", "conflict", "rate_limited", "limit_exceeded"]
    resource: EvaluationResource | None

class EvaluationStore:
    def create_or_replay(self, command: CreateResourceCommand) -> CreateResourceResult:
        """Atomically write idempotency, resource, FIFO entry, deadline, and quotas."""

    def get_visible(self, evaluation_id: str, caller: Caller) -> EvaluationResource | None:
        """Return owner-readable or observer-readable resource; hide all other records."""
```

Use binary redis-py connections (`decode_responses=False`) so canonical request bytes are never decoded and re-encoded. The fixed keyspace under `settings.redis_prefix`, default `triton-eval:v2`, is:

```text
resource:{evaluation_id}                         Redis hash
idempotency:{sha256(caller_id + NUL + key)}      Redis hash
ready                                             sorted set, score=monotonic queue sequence
queue-sequence                                    integer
deadlines                                         sorted set, score=deadline epoch milliseconds
state:{queued|running|completed|failed|timed_out} sorted sets of evaluation IDs
inflight:caller:{caller_class}:{sha256(caller_id)} integer
inflight:session:{sha256(caller_id + NUL + session_id)} integer
rate:{caller_class}:{sha256(caller_id)}:{minute} integer with 120-second expiry
```

The five state indexes are sorted sets: queued/running scores are the absolute deadline in epoch milliseconds; terminal scores are the retention-expiry instant. This permits gauges to exclude expired terminal resources before asynchronous index cleanup. The idempotency hash stores only `evaluation_id` and `request_digest`; exact-byte replay comparison reads the single canonical request copy from the watched resource hash. The resource hash stores owner, purpose, session digest, idempotency record key, exact canonical request bytes, request/reference digests, requested checks JSON, public resource JSON, evaluator build, four algorithm versions, attempt count, current attempt/token/check fields, private partial check outcomes, and epoch-millisecond timestamps. Source is not copied into any index, metrics, or logs. `traffic_class` is derived from purpose (`trial` for trial and `trusted` for grade), so an observer explicitly granted `grade:submit` consumes trusted rather than observer capacity.

- [ ] Replace old queue/session tests with RED tests for initial revision/state/timestamps, FIFO sequence, exact canonical request storage, same-key replay, conflicts, same key across callers, default/check/case canonical equivalence, and no expiry on queued resource/idempotency keys.
- [ ] Add a thread-barrier RED test with 16 concurrent identical creates. Assert one `created`, 15 `replayed`, one resource ID, one ready member, one sequence increment, and one increment of each inflight counter.
- [ ] Add RED quota tests proving idempotent replay bypasses new-request rate/inflight charging, caller and session limits are both enforced, and trial/trusted keys and configured capacities remain independent.
- [ ] Add RED visibility tests: owner with purpose-appropriate read scope succeeds; same caller without read scope, a different caller, unknown ID, and guessed session all return `None`; `observer:read` succeeds without changing ownership.
- [ ] Run focused tests and observe failures from the old list-based broker.

```bash
.venv/bin/python -m pytest tests/test_broker.py tests/test_idempotency.py tests/test_resource_visibility.py -q
```

Expected: failures show missing `EvaluationStore`, non-atomic idempotency, old uppercase task states, and session-derived access.

- [ ] Replace `Job`/`TaskStatus` storage types with immutable command/result/attempt records. Keep legacy HTTP request/response models out of `models.py`; Task 11 places them in its isolated compatibility module.
- [ ] Implement a production Redis clock from `TIME` and inject a deterministic clock in tests; all persisted epoch timestamps come from that store clock, while HTTP waiting alone uses process monotonic time. Implement `create_or_replay()` as a retrying optimistic transaction. First `WATCH` the idempotency, quota, and queue-sequence keys; when an idempotency record exists, also watch/read its resource and compare both digest and its exact canonical bytes, not digest alone. A missing mapped resource is a storage failure, never permission to reuse the key. For a new record, read the watched sequence, calculate its successor, and issue one `MULTI/EXEC` that stores the successor and all admission writes. Retry only `redis.WatchError`; propagate connection/storage failures.
- [ ] Create the public queued resource before serializing it, with `revision=1`, one shared `accepted_at` for `created_at`/`updated_at`, `deadline_at=accepted_at+deadline_seconds`, and `result=null`. Generate IDs as `ev_` plus 32 lowercase UUID4 hex characters.
- [ ] Store no TTL for any nonterminal record. The rate-window key alone expires after 120 seconds. Validate configured retention before constructing the store.
- [ ] Implement visibility from stored `caller_id` and purpose: owners require `trial:read` or `grade:read` matching the resource purpose; observers require `observer:read`. Unknown and invisible return the same `None` without a distinguishing metric label.
- [ ] Run lint, then run the concurrency suites five times without an optional pytest plugin.

```bash
.venv/bin/python -m ruff check triton_evaluation/models.py triton_evaluation/broker.py tests/test_broker.py tests/test_idempotency.py tests/test_resource_visibility.py
for run in 1 2 3 4 5; do .venv/bin/python -m pytest tests/test_broker.py tests/test_idempotency.py tests/test_resource_visibility.py -q || exit 1; done
```

Expected: every run passes; resource, idempotency, and queue cardinalities remain one in each race.

- [ ] Commit atomic resource admission.

```bash
git add triton_evaluation/models.py triton_evaluation/broker.py tests/test_broker.py tests/test_idempotency.py tests/test_resource_visibility.py
git commit -m "feat(storage): create durable idempotent resources"
```

### Task 5: Enforce the state machine, leases, deadlines, and fencing

**Files:**
- Modify: `triton_evaluation/broker.py`
- Modify: `triton_evaluation/models.py`
- Create: `triton_evaluation/terminal_results.py`
- Create: `tests/test_resource_state_machine.py`
- Create: `tests/test_fencing.py`
- Create: `tests/test_recovery.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AttemptLease:
    evaluation_id: str
    attempt_id: str
    fencing_token: int
    expected_revision: int
    deadline_at_ms: int
    canonical_request: bytes
    evaluator_build: str
    algorithms: AlgorithmVersions

class EvaluationStore:
    claim_next(worker: WorkerIdentity) -> AttemptLease | None
    renew_lease(lease: AttemptLease) -> bool
    set_active_check(lease: AttemptLease, check: Check) -> bool
    save_check_outcome(lease: AttemptLease, check: Check, outcome: CheckOutcome) -> bool
    commit_terminal(lease: AttemptLease, state: TerminalState, result: EvaluationResult) -> bool
    recover_expired_leases(limit: int = 100) -> RecoveryCounts
    expire_due_resources(limit: int = 100) -> int
    purge_expired_terminal_indexes(limit: int = 100) -> int
    heartbeat(worker: WorkerIdentity) -> None
    has_fresh_compatible_worker(compatibility: WorkerCompatibility) -> bool
    note_fenced_write() -> None
```

Add `running-leases` as a sorted set scored by lease expiry and `worker-heartbeats` as a hash plus a `worker-heartbeat-expiry` sorted set. Attempt IDs are `at_` plus 32 lowercase UUID4 hex characters. Worker IDs are process-random `wk_` IDs and never encode host/card identity.

- [ ] Add a RED transition table that accepts only `queued -> running`, `queued -> failed|timed_out`, and `running -> completed|failed|timed_out`. Assert terminal immutability, monotonic visible revision, synchronized `updated_at`, first-attempt-only `started_at`, and complete terminal results.
- [ ] Add RED fencing races: duplicate claim, lease renewal with wrong attempt/token/revision, expired lease replacement, old attempt finishing before and after replacement, two terminal commits, and deadline watchdog racing a worker commit. Assert exactly one terminal result and no terminal rewrite.
- [ ] Add RED recovery tests for queued deadline, running deadline, worker crash below attempt limit, worker crash at attempt limit, delivery after deadline, duplicate ready members, Redis reconnect, and API process recreation against the same fake Redis data.
- [ ] Assert fenced internal progress writes persist completed check outcomes without exposing a nonterminal public `result`. Synthesized deadline/orchestration results retain valid persisted earlier outcomes, set the active or first-missing check to `timeout`/`failed`, skip later requested checks with `RESOURCE_DEADLINE`/`PRIOR_CHECK_FAILED`, and leave unrequested checks null. These results use admission-pinned provenance and no execution environment if no attempt produced data.
- [ ] Run the new suites and confirm missing transition methods.

```bash
.venv/bin/python -m pytest tests/test_resource_state_machine.py tests/test_fencing.py tests/test_recovery.py -q
```

Expected: collection or calls fail because the store has creation only.

- [ ] Implement claim with `WATCH` on the ready set and selected resource. A queued claim atomically removes the ready member, changes public state to running, increments revision, sets the first `started_at`, increments attempt count and fencing token, installs the current attempt and lease, and moves state indexes. A replacement-ready running resource keeps state/revision/started time and consumes the higher token reserved by the reaper.
- [ ] Implement renewal as a transaction conditioned on running state, exact attempt ID, exact token, expected public revision, and `now < deadline_at`. Renewal changes only internal lease expiry and does not increment the public revision.
- [ ] Implement `set_active_check()` and `save_check_outcome()` with the same fence conditions. Store strict serialized partial outcomes privately, never expose them through queued/running resources, and use them only to assemble a complete terminal result. A replacement below the attempt limit clears partial progress and reruns the pure evaluation from the beginning; a terminal reaper/watchdog retains whatever the current attempt durably committed.
- [ ] Implement terminal commit as one conditioned transaction that updates public resource/result, revision/timestamps/state indexes, removes lease/deadline entries, decrements both inflight counters without going below zero, gives both resource and idempotency records the same retention TTL, and scores the terminal state index at that expiry instant. Reject stale writes with `False`; never throw away the persisted winner.
- [ ] Implement reaping. Before deadline and below `max_attempts`, clear current attempt, increment the fencing token, mark `replacement_pending=1`, reinsert the same evaluation ID once into ready with a fresh sequence, and leave public state running. At the attempt limit, atomically commit a `WORKER_LOST` failed result. This evaluation workload is retry-safe; there is no per-check side effect registry in this batch.
- [ ] Implement the deadline watchdog for both queued and running resources. It increments the fence, removes ready/lease membership, commits state `timed_out`, and builds a `DEADLINE_EXCEEDED` outcome without inventing a candidate `execution_timeout`.
- [ ] Implement bounded terminal-index cleanup by selecting terminal sorted-set members whose expiry score is at or before store time and removing only members whose resource hash has expired. API and worker maintenance loops may run cleanup concurrently; operations are idempotent and must not delete a live resource.
- [ ] Implement heartbeat compatibility using exact evaluator build and four algorithm version strings. Prune expired heartbeat fields during readiness checks; do not count stale or incompatible workers.
- [ ] Run all store/state tests, including five dependency-free repetitions of races.

```bash
.venv/bin/python -m pytest tests/test_broker.py tests/test_idempotency.py tests/test_resource_state_machine.py tests/test_fencing.py tests/test_recovery.py -q
for run in 1 2 3 4 5; do .venv/bin/python -m pytest tests/test_fencing.py -q || exit 1; done
```

Expected: all tests pass and each race reports one terminal winner.

- [ ] Commit state and fencing behavior.

```bash
git add triton_evaluation/broker.py triton_evaluation/models.py triton_evaluation/terminal_results.py tests/test_resource_state_machine.py tests/test_fencing.py tests/test_recovery.py
git commit -m "feat(runner): add leased fenced resource transitions"
```

### Task 6: Shape private engine JSON into typed outcomes and provenance

**Files:**
- Modify: `triton_evaluation/engine/verify.py`
- Modify: `triton_evaluation/evaluators/base.py`
- Modify: `triton_evaluation/evaluators/verify.py`
- Modify: `triton_evaluation/evaluators/benchmark.py`
- Modify: `triton_evaluation/evaluators/profile.py`
- Modify: `triton_evaluation/evaluators/evaluation.py`
- Modify: `triton_evaluation/evaluators/shaping.py`
- Create: `triton_evaluation/execution_environment.py`
- Modify: `tests/test_shaping.py`
- Modify: `tests/test_evaluation_evaluator.py`
- Create: `tests/fixtures/engine/verify-correct.json`
- Create: `tests/fixtures/engine/verify-wrong-value.json`
- Create: `tests/fixtures/engine/verify-shape-mismatch.json`
- Create: `tests/fixtures/engine/verify-non-finite-mismatch.json`
- Create: `tests/fixtures/engine/verify-decoy.json`
- Create: `tests/fixtures/engine/verify-candidate-compile-error.json`
- Create: `tests/fixtures/engine/verify-candidate-initialize-error.json`
- Create: `tests/fixtures/engine/verify-candidate-execute-error.json`
- Create: `tests/fixtures/engine/verify-candidate-execution-timeout.json`
- Create: `tests/fixtures/engine/verify-ambiguous-timeout.json`
- Create: `tests/fixtures/engine/benchmark-mixed-coverage.json`
- Create: `tests/fixtures/engine/benchmark-invalid-measurement.json`
- Create: `tests/fixtures/engine/profile-op-complete.json`
- Create: `tests/fixtures/engine/profile-op-partial.json`
- Create: `tests/fixtures/engine/profile-deep-partial.json`
- Create: `tests/fixtures/engine/profile-invalid-counts.json`
- Create: `tests/test_grade_boundary.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class EngineCheckResult:
    status: Literal["succeeded", "failed", "timeout"]
    raw: dict[str, object] | None
    error: ErrorInfo | None

shape_correctness(raw: dict[str, object], decoy: bool) -> CorrectnessResult
shape_performance(raw: dict[str, object]) -> PerformanceResult
shape_profile(raw: dict[str, object]) -> ProfileResult

class ProgressSink:
    begin(check: Check) -> bool
    save(check: Check, outcome: CheckOutcome) -> bool

build_execution_environment(settings: Settings) -> ExecutionEnvironment

class EvaluationEvaluator:
    def run(
        self,
        request: EvaluationRequest,
        work_dir: Path,
        device_id: int,
        deadline_at: datetime,
        lease_valid: Callable[[], bool],
        execution_environment: ExecutionEnvironment,
        pinned: PinnedProvenance,
        progress: ProgressSink,
    ) -> EvaluationResult:
        """Run correctness, conditional performance, then independent profile."""
```

The only service-side grade decision is `Provenance.grade_eligible`; this repository must not create `GradeRecord` or numeric reward. Coordinator-owned mapping is verified from published terminal fixtures plus consumer-owned cancellation fixtures: subject wrong/decoy/compile/runtime/execution-timeout maps to scored numeric zero; evaluator/resource failure maps to unscored; correct with required performance unavailable maps to unscored `REQUIRED_PERFORMANCE_UNAVAILABLE`; coordinator cancel maps to immutable unscored `COORDINATOR_CANCELED` with null scalar and `retryable=false`. A pre-formal cancel has all formal Evaluation fields null. A post-request/ID cancel preserves every already-durable formal identity/evidence field but may have null result/provenance; the cancellation CAS winner cannot be rewritten by a late terminal resource. Only a record derived from a trusted terminal formal Evaluation requires the complete ID/request/result/provenance field set.

- [ ] Replace shaping tests with RED private-protocol fixtures for correct, numeric wrong, shape mismatch, non-finite mismatch, decoy, candidate compile error, candidate initialize/execute error, authoritatively attributed candidate execution timeout, un-attributable process timeout, benchmark measured/unmeasured cases, and op/deep complete/partial/failed profile cases.
- [ ] Add RED rejection tests for every private model: extra key, missing case identity, duplicate case, contradictory counts, ordinary NaN/Inf, more than eight samples, unsorted samples, compressed string input shapes, invalid latency/ratio/percentage, and raw traceback/path in a message.
- [ ] Add RED orchestration tests proving the fixed order, fenced `begin`/`save` around every check, performance only after `status=succeeded` plus `outcome=correct`, profile after wrong/decoy, profile skipped only when infrastructure made the environment untrustworthy, and all requested slots filled on any terminal path.
- [ ] Add RED grade-boundary tests against the generated fixtures. Trial is never eligible; grade requires manifest/problem, completed state, all required provenance, and trusted admission. Tests must explicitly distinguish `grade_eligible` from `scored`.
- [ ] Run the focused tests and observe old flat payload failures.

```bash
.venv/bin/python -m pytest tests/test_shaping.py tests/test_evaluation_evaluator.py tests/test_grade_boundary.py -q
```

Expected: failures reference old divergence names, `mean_abs_err`, string shapes, missing check discriminants, and missing provenance.

- [ ] Add an engine-private attribution object to verify failures at the point where the engine can prove candidate origin. Its allowed values are candidate compile, initialize, execute, and candidate execution timeout. Reference import/run failures, outer subprocess timeouts, missing output, and ambiguous exits remain evaluator errors and never receive candidate attribution.
- [ ] Replace the shaping layer's permissive dictionary access with private `StrictModel` parsers matching the checked-in engine fixtures. Reject unmodeled keys. Map only coarse `shape_mismatch`, `value_mismatch`, and `non_finite_mismatch`; convert sample non-finite values to `"inf"`, `"-inf"`, or `"nan"`; rank finite absolute differences first in descending order, then sentinel differences in lexical order, and break all ties by row-major index; cap at eight; omit `mean_abs_err`.
- [ ] Compute performance from per-case raw measurements rather than trusting the engine aggregate: structured `InputShape`, positive reference/implementation latency, per-case speedup, unmeasured reasons, coverage, and `exp(sum(log(speedup))/n)` under `geometric_mean_v1`. Omit throughput.
- [ ] Shape profile into the `mode` discriminated union, enforce counts and unique case indices, normalize the six ratio names, use the fixed bound enum, and carry typed per-case `ErrorInfo` only where the contract permits it.
- [ ] Build provenance from admission-pinned build/algorithm versions and the actual worker environment. The worker requires deployment-attested `hardware_model`, `driver_version`, `runtime_version`, and `toolchain_version` settings, fixes `hardware_count=1` because each process exposes one card, and reads optional torch/torch-npu/triton versions with `importlib.metadata.version()` inside that worker process. The target-NPU gate compares attested values with its calibrated environment. Set `grade_eligible=true` only for a completed, manifest-bound grade with complete reward-relevant provenance. For a decoy, resolve the trusted reference input count, emit `outcome="wrong"`, `decoy=true`, `passed_cases=total_cases`, and no mismatch failures. Never include hostname, card serial, source, or input.
- [ ] Run focused tests, engine regression tests, and lint only the non-vendored files.

```bash
.venv/bin/python -m pytest tests/test_shaping.py tests/test_evaluation_evaluator.py tests/test_grade_boundary.py tests/test_verify_engine.py tests/test_benchmark_gate.py tests/test_profile_op.py tests/test_profile_deep.py -q
.venv/bin/python -m ruff check triton_evaluation/evaluators triton_evaluation/execution_environment.py tests/test_shaping.py tests/test_evaluation_evaluator.py tests/test_grade_boundary.py
```

Expected: all deterministic tests pass; no stable output contains experimental divergence kinds, raw traceback, `mean_abs_err`, or `throughput`.

- [ ] Prove implementation conforms to the frozen publication without rewriting it.

```bash
.venv/bin/python scripts/generate_contract_artifacts.py --check
.venv/bin/python -m pytest tests/test_contract_artifacts.py -q
```

Expected: the check and tests pass with no byte change. Any mismatch stops execution and returns to a reviewed Task 1/2 correction; Task 6 must not rewrite a published contract.

- [ ] Commit shaping against the frozen artifacts.

```bash
git add triton_evaluation/engine/verify.py triton_evaluation/evaluators triton_evaluation/execution_environment.py tests/fixtures/engine tests/test_shaping.py tests/test_evaluation_evaluator.py tests/test_grade_boundary.py
git commit -m "feat(evaluator): shape typed outcomes and provenance"
```

### Task 7: Run attempts with continuous leases and discard fenced work

**Files:**
- Modify: `triton_evaluation/worker.py`
- Modify: `run_worker.py`
- Modify: `triton_evaluation/evaluators/base.py`
- Modify: `tests/test_worker.py`
- Create: `tests/test_worker_crash_recovery.py`

**Interfaces:**

```python
class LeaseKeeper:
    __enter__() -> LeaseKeeper
    is_valid() -> bool
    stop() -> None
    __exit__(exc_type, exc, traceback) -> None

class Worker:
    def run_once(self) -> bool:
        """Claim one attempt, maintain lease/heartbeat, shape, and fenced-commit it."""

def run_script(
    cmd: list[str],
    cwd: Path,
    device_id: int,
    deadline_at: datetime,
    lease_valid: Callable[[], bool],
) -> ScriptOutcome:
    """Kill the whole process group on deadline or lost lease."""
```

- [ ] Write RED worker tests for no work, normal completion, candidate failure completion, evaluator failure, resource deadline, lease loss during a subprocess, Redis renewal outage, stale terminal commit, heartbeat freshness, incompatible worker build, temporary-directory cleanup, and worker-loop backoff.
- [ ] Add a RED crash recovery test that claims an attempt, abandons the first `Worker` instance, advances the injected clock beyond its lease, runs recovery, lets a second worker commit, and then proves the first token cannot mutate the resource.
- [ ] Run and confirm the old BRPOP worker fails the new protocol.

```bash
.venv/bin/python -m pytest tests/test_worker.py tests/test_worker_crash_recovery.py -q
```

Expected: failures show missing lease keeper, heartbeat, fenced commit, and durable claim APIs.

- [ ] Implement `LeaseKeeper` as one daemon thread per active worker attempt. It heartbeats and renews at `heartbeat_interval_seconds`; any false renewal or Redis exception permanently clears validity. Joining it is bounded to two heartbeat intervals. The outer worker loop also heartbeats while idle, so readiness represents claim-capable workers rather than only busy workers.
- [ ] Change `run_script()` from one long blocking `communicate()` call to bounded polling. If the absolute resource deadline expires, kill the process group and return resource-deadline status. If the lease becomes invalid, kill the group and return fenced status. Neither case may be classified as candidate execution timeout.
- [ ] Parse the stored canonical request with `EvaluationRequest.model_validate_json`, verify its recomputed digest before materialization, and use an internal safe module stem `eval_` plus the first 24 request-digest hex characters for filenames/import arguments. Do not interpolate public `op_name` into filesystem paths.
- [ ] Pass a fenced progress sink into the evaluator; reject the attempt immediately if beginning or saving a check returns false. On evaluator success, validate the complete `EvaluationResult` before `commit_terminal(completed)`. On evaluator protocol failure, construct a redacted failed result. On resource deadline, attempt a timed-out commit; accept a false result as evidence that another winner already terminalized it.
- [ ] If lease validity is lost, do not build or commit a terminal result. Increment only the bounded `fenced_write_rejected` metric and let reaper/replacement own recovery.
- [ ] On an empty claim, run one bounded batch each of deadline expiry, lease recovery, and terminal-index cleanup, then wait `queue_poll_interval_seconds` on a stoppable event. More than one worker may perform this maintenance because every mutation is fenced/CAS; never busy-spin an empty queue.
- [ ] Run worker, evaluator, fencing, and recovery tests.

```bash
.venv/bin/python -m pytest tests/test_worker.py tests/test_worker_crash_recovery.py tests/test_fencing.py tests/test_recovery.py tests/test_evaluation_evaluator.py -q
.venv/bin/python -m ruff check triton_evaluation/worker.py triton_evaluation/evaluators/base.py run_worker.py tests/test_worker.py tests/test_worker_crash_recovery.py
```

Expected: all tests pass; captured logs contain IDs/error codes but no source, credential, full session, absolute work path, or traceback.

- [ ] Commit the durable worker loop.

```bash
git add triton_evaluation/worker.py triton_evaluation/evaluators/base.py run_worker.py tests/test_worker.py tests/test_worker_crash_recovery.py
git commit -m "feat(worker): maintain leases and honor fencing"
```

### Task 8: Expose authenticated durable POST and GET semantics

**Files:**
- Modify: `triton_evaluation/api/app.py`
- Modify: `triton_evaluation/api/dependencies.py`
- Modify: `triton_evaluation/api/routes.py`
- Modify: `triton_evaluation/api/__init__.py`
- Modify: `run_api.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_api_v2.py`
- Create: `triton_evaluation/api/errors.py`
- Create: `triton_evaluation/api/middleware.py`
- Create: `triton_evaluation/api/auth.py`
- Create: `tests/test_http_errors.py`
- Create: `tests/test_durable_http.py`
- Create: `tests/test_http_security.py`

**Interfaces:**

```text
POST /evaluations?wait_seconds=0..60
  auth: require_submit_caller
  input: raw Request + X-Session-ID + Idempotency-Key
  output: 200 terminal EvaluationResource or 202 nonterminal EvaluationResource + Location

GET /evaluations/{evaluation_id}
  auth: require_read_caller
  output: 200 visible EvaluationResource or indistinguishable 404 ErrorEnvelope
```

```python
@dataclass(frozen=True)
class ApiError(Exception):
    http_status: int
    info: ErrorInfo
    headers: Mapping[str, str] = field(default_factory=dict)
```

Freeze the initial code policy; a code's phase may be narrowed to its failing check but its retryable value may not vary by call site:

| Code | HTTP when pre-resource | Phase | Retryable |
|---|---:|---|---|
| `INVALID_REQUEST` | 400 or 422 | request | false |
| `UNAUTHENTICATED` | 401 | auth | false |
| `FORBIDDEN` | 403 | auth | false |
| `MANIFEST_BINDING_FAILED` | 403 | admission | false |
| `NOT_FOUND` | 404 | request | false |
| `IDEMPOTENCY_CONFLICT` | 409 | admission | false |
| `LIMIT_EXCEEDED` | 413 | request | false |
| `RATE_LIMITED` | 429 | admission | true |
| `NOT_READY` | 503 | admission | true |
| `QUEUE_UNAVAILABLE` | 503 | queue | true |
| `WORKER_LOST` | resource outcome only | execution | true |
| `EVALUATOR_EXECUTION_FAILED` | resource outcome only | execution | true |
| `INVALID_ENGINE_OUTPUT` | resource outcome only | shaping | false |
| `DEADLINE_EXCEEDED` | resource outcome only | deadline | false |
| `INTERNAL` | 500 | storage or execution | false |

`check` is null before execution and names the affected check in a resource outcome. `Retry-After: 1` is fixed for 429/503 in this batch.

`create_app()` receives one `AppContext` containing settings, store, capability registry, catalog, and metrics dependency. It verifies artifact bytes and loads security/catalog data during lifespan before serving. The lifespan runs a stoppable maintenance task that performs bounded deadline expiry, lease recovery, and terminal-index cleanup every `maintenance_interval_seconds`, so queued deadlines still terminalize when workers are absent; duplicate worker maintenance remains CAS-safe. It never constructs a development credential unless `development_mode=true` and a separate development secret file is configured.

- [ ] Write RED endpoint tests for POST created/replayed queued as 202 plus exact `Location`, terminal replay as 200, `wait_seconds` terminal as 200, wait expiry as 202 without state mutation, GET for queued/running/each terminal state, `Retry-After: 1` on nonterminal GET, client disconnect followed by GET, and ambiguous POST response recovered with the same key/request.
- [ ] Write RED header/body tests for missing/invalid auth, session regex, idempotency visible-ASCII rule, missing content type, malformed JSON, nonstandard constants, actual body over 4 MiB with and without `Content-Length`, each source over 2 MiB, invalid wait range, scope denial, manifest binding failure, cross-caller GET, and a forged shared session.
- [ ] Write a parameterized RED test that exercises every non-2xx path, including FastAPI validation, and validates the response as `ErrorEnvelope`. Assert stable status/code/phase/retryable mappings, `WWW-Authenticate: Bearer` on 401, `Retry-After` on 429/503, request ID propagation, and absence of submitted source/token/path text.
- [ ] Run focused HTTP tests and confirm failures against the synchronous route.

```bash
.venv/bin/python -m pytest tests/test_api.py tests/test_api_v2.py tests/test_http_errors.py tests/test_durable_http.py tests/test_http_security.py -q
```

Expected: failures show synchronous timeout synthesis, no GET/auth/idempotency, old payload, and FastAPI's default validation envelope.

- [ ] Add request-ID and body-limit middleware. Accept an inbound ID only if it matches `[A-Za-z0-9._:-]{1,64}`; otherwise generate `rq_` plus 32 UUID4 hex characters. Stream-count actual request bytes and stop above 4,194,304 before parsing. Store only the request ID in `request.state`.
- [ ] Authenticate before JSON parsing by keeping the route body as raw `Request`. Require `application/json`; decode UTF-8 with `json.loads(parse_constant=reject_nonstandard_number)`, then strict-model validate. Map syntax and cross-model combination errors to 400, field/schema errors to 422, and size errors to 413.
- [ ] Validate `X-Session-ID` with `^[A-Za-z0-9._:-]{1,128}$` and `Idempotency-Key` by byte values `0x21..0x7e`, length 1..128. Do not strip either header. Transport wait and headers never enter `admit()`.
- [ ] Before new admission, require the ready check's storage/schema/worker conditions. An idempotent replay remains recoverable during a transient no-worker condition: check the idempotency record first, return its resource if found, and apply readiness only when the key is absent.
- [ ] Implement POST status selection. Poll through `asyncio.to_thread(store.get_visible)` until terminal, monotonic wait budget, or `request.is_disconnected()`. Waiting never calls deadline terminalization and never cancels the resource. Set `Location` on every 202.
- [ ] Implement GET with owner/observer visibility; invalidly shaped, unknown, and invisible IDs all use the same 404 envelope. Return the persisted public resource without reparsing private request data into the response.
- [ ] Install exception handlers for `ApiError`, request validation, HTTP errors, and uncaught exceptions. Use the exact design mappings and redact at the handler boundary. Once a resource exists, execution errors are persisted outcomes, not HTTP 5xx.
- [ ] Run HTTP tests and lint.

```bash
.venv/bin/python -m pytest tests/test_api.py tests/test_api_v2.py tests/test_http_errors.py tests/test_durable_http.py tests/test_http_security.py -q
.venv/bin/python -m ruff check triton_evaluation/api run_api.py tests/test_api.py tests/test_api_v2.py tests/test_http_errors.py tests/test_durable_http.py tests/test_http_security.py
```

Expected: all tests pass; all non-2xx payloads validate as schema 2.0 `ErrorEnvelope`; waiting tests leave stored states unchanged.

- [ ] Commit the durable HTTP boundary.

```bash
git add triton_evaluation/api run_api.py tests/test_api.py tests/test_api_v2.py tests/test_http_errors.py tests/test_durable_http.py tests/test_http_security.py
git commit -m "feat(api): expose durable authenticated evaluations"
```

### Task 9: Add dependency-aware readiness and bounded internal metrics

**Files:**
- Modify: `triton_evaluation/broker.py`
- Modify: `triton_evaluation/worker.py`
- Modify: `triton_evaluation/evaluators/shaping.py`
- Modify: `triton_evaluation/api/routes.py`
- Modify: `triton_evaluation/api/dependencies.py`
- Create: `triton_evaluation/metrics.py`
- Create: `tests/test_health.py`
- Create: `tests/test_metrics.py`

**Interfaces:**

```text
class RedisMetrics:
    increment(name: str, labels: Mapping[str, str], amount: int = 1) -> None
    observe(name: str, labels: Mapping[str, str], value: float) -> None
    render_openmetrics() -> bytes

GET /livez -> constant 200 process-liveness object
GET /readyz -> 200 ready object or 503 ErrorEnvelope
GET /internal/metrics -> observer-authenticated OpenMetrics bytes
```

Metrics use Redis hashes so API and worker processes share one bounded view. Counter fields are `metric_name|sorted_label_key=value`; histogram observations use fixed buckets and `_count`/`_sum`. Reject metric names, label names, and label values not present in a compile-time allowlist. Never accept arbitrary caller/session/evaluation values as labels.

- [ ] Write RED health tests proving `/livez` performs no store, Redis, catalog, worker, or NPU call; `/readyz` requires verified contract bytes, a successful ephemeral storage write/delete, Redis reachability, and at least one fresh compatible heartbeat. Assert its public success/failure bodies reveal no counts, cards, queue, build string, or addresses.
- [ ] Write RED metrics tests for required POST/auth/idempotency/limit dimensions; resource state/terminal/duration/deadline dimensions; check/skip/error/domain/decoy/grade-eligible/performance rejection dimensions; worker heartbeat/lease/retry/fence/CAS/reaper dimensions; and shaping/catalog/build-algorithm compatibility dimensions.
- [ ] Add RED leakage/cardinality tests that attempt source, token, full session, evaluation ID, arbitrary caller ID, absolute path, and unknown labels. Assert rejection or aggregation to the bounded `caller_class` values `trial|trusted|observer`.
- [ ] Run tests and confirm endpoints/registry are absent.

```bash
.venv/bin/python -m pytest tests/test_health.py tests/test_metrics.py -q
```

Expected: failures show no split liveness/readiness and no metrics implementation.

- [ ] Implement `/livez` as a constant response generated without touching `AppContext` dependencies. Implement readiness with one random Redis probe key using `SET NX PX 5000` followed by `DELETE`, local artifact digest verification, and compatible worker freshness; map failure to 503 `NOT_READY` with no component detail.
- [ ] Implement the allowlisted Redis metrics registry. Use fixed duration buckets `0.01, 0.1, 1, 5, 15, 60, 300, 900, 3600` seconds and coverage buckets `0.0, 0.25, 0.5, 0.75, 0.9, 1.0`. A metrics write failure is logged as a redacted operational warning and never changes a resource transition result.
- [ ] Instrument exact decision points rather than inferring from logs. Queued/running gauges use state-index cardinality; terminal gauges use `ZCOUNT` from current store time through positive infinity so expired resources are excluded before cleanup. Terminal transition durations are observed in the same winning code path; fenced losers increment only reject/CAS counters.
- [ ] Protect `/internal/metrics` with `observer:read` and render deterministic OpenMetrics 1.0 text with `Content-Type: application/openmetrics-text; version=1.0.0; charset=utf-8` and a final `# EOF`. This endpoint is operational/internal; it is not added to the cross-repository JSON contract. It must not render queue depth, card busy/idle state, or worker/evaluation/session IDs.
- [ ] Note in code comments and tests that `GradeRecord scored/unscored` metrics belong to Eval Hub because this service does not create GradeRecords. This service exposes purpose, grade eligibility, domain outcome, and check availability needed for Hub's metrics.
- [ ] Run health/metrics plus worker/API regressions.

```bash
.venv/bin/python -m pytest tests/test_health.py tests/test_metrics.py tests/test_worker.py tests/test_api_v2.py tests/test_http_security.py -q
.venv/bin/python -m ruff check triton_evaluation/metrics.py triton_evaluation/api triton_evaluation/broker.py triton_evaluation/worker.py tests/test_health.py tests/test_metrics.py
```

Expected: all tests pass; readiness failures are uniform; metrics cardinality remains within the allowlist.

- [ ] Commit operations signals.

```bash
git add triton_evaluation/metrics.py triton_evaluation/api triton_evaluation/broker.py triton_evaluation/worker.py triton_evaluation/evaluators/shaping.py tests/test_health.py tests/test_metrics.py
git commit -m "feat(ops): add readiness and bounded metrics"
```

### Task 10: Make production Redis and network deployment fail closed

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker-compose.npu0-test.yml`
- Modify: `docker/entrypoint.sh`
- Modify: `docker.env.example`
- Modify: `.env.example`
- Modify: `scripts/start.sh`
- Modify: `tests/test_deployment.py`
- Create: `scripts/deployment_preflight.py`
- Create: `scripts/compose.sh`
- Create: `tests/test_deployment_preflight.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class DeploymentConfig:
    network_name: str
    gateway: ipaddress.IPv4Address
    api_host: ipaddress.IPv4Address
    proxy_host: ipaddress.IPv4Address
    api_port: int
    evaluation_url: str
    capability_file: Path
    catalog_root: Path
    catalog_manifest: Path
    redis_image: str
    evaluator_build: str

def inspect_deployment(env: Mapping[str, str], runner: CommandRunner) -> DeploymentConfig:
    """Perform every pure-stdlib and fakeable Docker preflight check."""

def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 only when starting this exact rendered deployment is safe."""
```

Production remains host-networked. Redis binds only `127.0.0.1:6379`. The API binds only the one IPv4 gateway reported by Docker network `kernel-agent-sandbox-net`, on fixed production port `11907`; loopback, wildcard, IPv6 wildcard, physical/public addresses, multiple gateways, and an environment URL mismatch are fatal.

The host deployment environment names are frozen as `TRITON_EVAL_API_HOST`, `TRITON_EVAL_API_PORT`, `PROXY_HOST`, `EVALUATION_URL`, `NO_PROXY`, `TRITON_EVAL_CAPABILITY_FILE`, `TRITON_EVAL_CATALOG_ROOT`, and `TRITON_EVAL_CATALOG_MANIFEST`. `PROXY_HOST`, the URL host, API host, and inspected gateway must be byte-for-byte the same canonical IPv4 text; `EVALUATION_URL` is exactly `http://{gateway}:11907` with no credentials, path, query, or fragment; `NO_PROXY` contains the gateway as a complete comma-delimited entry.

- [ ] Invert the old deployment tests to RED: require the exact multi-platform Redis image `docker.io/library/redis:7.4.10-alpine3.21@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2`, Redis `--appendonly yes --appendfsync always --aof-use-rdb-preamble yes --save ""`, a named `/data` volume, loopback bind, host network, and no Docker socket mount. Require API host to equal the inspected bridge gateway rather than loopback or wildcard. Reject every tag-only Redis reference, another digest, and a rendered platform that is absent from the pinned index.
- [ ] Add fake-Docker RED tests for missing network, zero/multiple/non-IPv4 gateways, bind mismatch, port mismatch, evaluation/proxy URL mismatch, missing proxy bypass, rendered non-host network, non-loopback Redis, mutable/tag-only or wrong-digest Redis, absent AOF/`appendfsync always`, missing volume, missing/mutable eval image, evaluator-build mismatch, and preflight subprocess failure.
- [ ] Add secret/catalog RED tests for missing/unreadable files, capability mode with any group/world bit, duplicate token/caller/scope configuration, writable catalog mount, manifest identity/triple duplicates, missing reference, digest mismatch, traversal/symlink escape, and output redaction. Seed fake secret/reference contents and assert neither appears in stdout/stderr.
- [ ] Add wrapper RED tests proving `up`, `start`, and `restart` do not invoke `docker compose` when preflight exits nonzero, and invoke preflight exactly once before Compose when valid. `down`, `logs`, `ps`, and `config` remain read/control commands and do not claim startup validation.
- [ ] Run deployment tests and observe failures from ephemeral Redis/loopback API.

```bash
.venv/bin/python -m pytest tests/test_deployment.py tests/test_deployment_preflight.py -q
```

Expected: assertions fail on `appendonly no`, absent volume/preflight, and loopback API bind.

- [ ] Implement preflight using only stdlib modules and `docker network inspect`, `docker image inspect`, and `docker compose config --format json`. Parse command output as JSON; never invoke a shell or pull an image. Validate `EVALUATION_URL`, `NO_PROXY`, rendered env/port/network/volumes, the exact Redis image string and resolved `RepoDigests`, mounted secret/catalog read-only flags, and the same capability/catalog semantics as production loaders. A tag-only image, digest mismatch, unavailable platform, or locally resolved image whose RepoDigests omit the pinned digest fails before Compose.
- [ ] Make `scripts/compose.sh` accept one Compose action and remaining safe arguments as an array. For `up|start|restart`, run preflight first, obtain the already-validated gateway from its machine-readable result, export `TRITON_EVAL_API_HOST`, then `exec docker compose`. Never use `eval` or print environment secrets.
- [ ] Change Redis to the exact reviewed Redis 7.4.10 index digest plus this durable command and volume:

```yaml
image: docker.io/library/redis:7.4.10-alpine3.21@sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2
command:
  - redis-server
  - --bind
  - 127.0.0.1
  - --port
  - "6379"
  - --save
  - ""
  - --appendonly
  - "yes"
  - --appendfsync
  - always
  - --aof-use-rdb-preamble
  - "yes"
volumes:
  - triton-eval-redis-v2:/data
```

The pinned multi-platform OCI index was resolved on 2026-08-13 with `docker buildx imagetools inspect redis:7.4.10-alpine3.21`; reviewing a Redis update requires recording a new exact tag+index digest and rerunning deployment/AOF tests. Declare top-level `triton-eval-redis-v2`. Explain in Compose that `EXEC` acknowledgement under `appendfsync always` is the first-2xx durability boundary; do not add a weaker in-memory development fallback to production Compose.
- [ ] Replace the mutable eval-service `:latest` reference with deployment input `TRITON_EVAL_IMAGE`, required to be `registry/repository@sha256:<64-lowercase-hex>` with no tag-only default. Require `TRITON_EVAL_EVALUATOR_BUILD` to equal that exact immutable image reference (or, for an explicitly reviewed non-image dev build, exact `git:<40-lowercase-hex>`); pass it as the sole `evaluator_build` setting used at admission and in Provenance. Preflight inspects the local image and rejects absent/mismatched RepoDigests, missing build value, or rendered service/build disagreement. The NPU report records the same value and tests mutate it independently.
- [ ] Mount capability file, catalog manifest, and catalog references read-only into the eval container; pass only their container paths. Update entrypoint/start scripts to fail before API/workers if artifact verification, secret/catalog load, immutable eval/Redis image verification, evaluator-build equality, or bind validation fails.
- [ ] Run deployment tests, shell syntax checks, and rendered Compose assertions.

```bash
.venv/bin/python -m pytest tests/test_deployment.py tests/test_deployment_preflight.py -q
bash -n scripts/compose.sh scripts/start.sh docker/entrypoint.sh
.venv/bin/python scripts/deployment_preflight.py --help
```

Expected: tests and syntax checks pass; help exits 0 without Docker access; exact Redis/eval image and evaluator-build fixtures pass, while mutable/wrong-digest fixtures fail without executing Compose or revealing seeded values.

- [ ] Commit durable deployment controls.

```bash
git add docker-compose.yml docker-compose.npu0-test.yml docker/entrypoint.sh docker.env.example .env.example scripts/start.sh scripts/deployment_preflight.py scripts/compose.sh tests/test_deployment.py tests/test_deployment_preflight.py
git commit -m "feat(deploy): require durable redis and gateway preflight"
```

### Task 11: Preserve a time-boxed legacy adapter while consumers migrate

**Files:**
- Modify: `triton_evaluation/api/app.py`
- Modify: `triton_evaluation/config.py`
- Modify: `README.md`
- Modify: `docs/ascend-docker-deploy.md`
- Modify: `scripts/test_eval_service.py`
- Modify: `tests/test_smoke_script.py`
- Create: `triton_evaluation/api/legacy_contract.py`
- Create: `triton_evaluation/api/legacy_routes.py`
- Create: `tests/test_legacy_routes.py`

**Interfaces:**

```text
POST /verify             legacy request -> one trial EvaluationResource
POST /profile            legacy request -> one trial EvaluationResource
GET  /tasks/{task_id}    evaluation_id alias -> legacy task projection
```

The compatibility window is explicitly configured with `legacy_routes_enabled=true` and `legacy_routes_sunset_at=2026-10-15T00:00:00Z`. Enabling without that exact published UTC instant, after that instant, or with a window over 90 days fails startup/preflight. Common Authorization, session, and idempotency headers remain mandatory; compatibility does not weaken v2 security.

- [ ] Restore the old route behavior only in RED adapter tests. Verify `/verify` translates `measure_speed` into correctness plus optional performance; `/profile` translates mode/case selection; `/tasks/{task_id}` maps evaluation states to `QUEUED|RUNNING|SUCCEEDED|FAILED|TIMEOUT`; all return the old field names without source or private results.
- [ ] Add RED tests that adapters require a trial capability and common headers, cannot submit grade, share the same store/FIFO/fencing path, do not import evaluator/broker internals, emit legacy usage metrics, and disappear when disabled or expired.
- [ ] Add RED boundary tests for legacy timeout values above v2's 3600-second maximum, performance dependency, profile case translation, idempotent replay, invisible task IDs, and terminal check projection. Reject unrepresentable values; never clamp or reinterpret them.
- [ ] Run tests and confirm the routes are currently absent.

```bash
.venv/bin/python -m pytest tests/test_legacy_routes.py tests/test_smoke_script.py -q
```

Expected: legacy route requests return 404 and smoke-client assumptions fail.

- [ ] Put all prior-major Pydantic shapes and state mapping in `legacy_contract.py`; core `contract.py`, storage records, and worker remain v2-only. `legacy_routes.py` calls the same `admit()` and `EvaluationStore.create_or_replay()` APIs as `/evaluations` and never imports service implementation from another repository.
- [ ] Return the persisted `evaluation_id` as legacy `task_id`. Legacy POST returns immediately after durable admission; legacy GET projects only representable terminal data. A failed projection returns a redacted legacy failure rather than leaking a raw v2 object.
- [ ] Gate router inclusion at app startup with the frozen sunset instant and add a bounded `legacy_route_requests_total{route=verify|profile|tasks}` metric. Do not implement automatic route deletion in this batch.
- [ ] Update the smoke client to send Bearer/session/idempotency headers, accept POST 200/202, persist `evaluation_id`, follow `Location` with bounded backoff, and safely replay the original POST only when no response was obtained.
- [ ] Update README and deployment docs with the durable API flow, capability classes, artifact paths/digests, Redis persistence, gateway bind, exact compatibility removal date, rollback preservation of Redis data, and the statement that client HTTP timeout covers only `wait_seconds` plus transport margin.
- [ ] Run adapter/smoke/API tests and lint.

```bash
.venv/bin/python -m pytest tests/test_legacy_routes.py tests/test_smoke_script.py tests/test_api_v2.py tests/test_durable_http.py -q
.venv/bin/python -m ruff check triton_evaluation/api/legacy_contract.py triton_evaluation/api/legacy_routes.py scripts/test_eval_service.py tests/test_legacy_routes.py tests/test_smoke_script.py
```

Expected: all tests pass; legacy and v2 requests create the same durable resource type; docs publish `2026-10-15T00:00:00Z` consistently.

- [ ] Commit the bounded migration surface.

```bash
git add triton_evaluation/api/app.py triton_evaluation/api/legacy_contract.py triton_evaluation/api/legacy_routes.py triton_evaluation/config.py README.md docs/ascend-docker-deploy.md scripts/test_eval_service.py tests/test_legacy_routes.py tests/test_smoke_script.py
git commit -m "feat(migration): add time-boxed legacy adapters"
```

### Task 12: Add end-to-end fault gates, a portable schema consumer, and CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_profile_npu_integration.py`
- Create: `.github/workflows/evaluation-api-v2.yml`
- Create: `tests/test_end_to_end_durable.py`
- Create: `tests/test_fault_injection.py`
- Create: `tests/test_cross_repo_contract.py`
- Create: `tests/contract_consumers/portable_schema_probe.py`
- Create: `tests/test_portable_schema_probe.py`
- Create: `tests/test_npu_release_calibration.py`
- Create: `scripts/npu_release_gate.py`

**Interfaces:**

```text
.venv/bin/python scripts/npu_release_gate.py \
  --service-url http://172.18.0.1:11907 \
  --capability-file /run/secrets/triton-eval-grade \
  --catalog-manifest /srv/triton-eval-release/catalog.json \
  --cases /srv/triton-eval-release/calibration-cases.json \
  --output /srv/triton-eval-release/reports/evaluation-api-2.0.json
```

The report records contract digests, evaluator build, algorithm versions, driver/runtime/toolchain versions, pass/fail counts, benchmark coefficient-of-variation checks, coverage/geometric-mean recomputation, profile count/ratio validation, and crash/fencing soak result. It does not record credentials, sources, inputs, hostname, device serials, or full session IDs.

- [ ] Add RED end-to-end tests that use the real FastAPI app plus shared fake Redis: authenticated POST, worker claim, typed terminal commit, GET recovery, same-key replay, API object recreation, worker replacement, deadline, and cross-caller hiding.
- [ ] Add RED failpoint tests immediately before and after `EXEC` acknowledgement for admission and terminal commit, API restart during wait, Redis connection loss during claim/renew/commit, duplicated ready delivery, and terminal CAS race. Assert either no resource was accepted or the same resource is recoverable; never accept an unqueryable ID.
- [ ] Add RED cross-repository artifact tests that load only `contract.json`, `schema.json`, and fixture index/files, recompute both pinned digests, and exercise every grade mapping case. Add consumer-owned cancel fixtures for (1) cancel before formal POST with every formal field null, (2) cancel after request or `evaluation_id` persistence retaining exactly the already-durable identity/evidence while result/provenance may be null, and (3) a late terminal Evaluation losing the cancel CAS and leaving the immutable `COORDINATOR_CANCELED`, null scalar, nontrainable, nonretryable disposition unchanged. Mutating one schema byte, fixture byte, index path, or digest must fail closed.
- [ ] Add a portable schema probe using Python stdlib `argparse`, `hashlib`, strict `json`, `time`, and `urllib`, plus the sole narrow direct dependency `jsonschema==4.26.0`. Test it in an isolated subprocess against a local `http.server`; inspect imported modules and permit only stdlib, `jsonschema`, and jsonschema's resolver runtime dependency `referencing`. Assert it never imports Eval Hub, FastAPI, Pydantic, Redis, or `triton_evaluation`, and contains no handwritten Evaluation field/type validator.
- [ ] Add `npu` and `release_gate` pytest markers. Build NPU calibration cases for correct, decoy, each coarse divergence, candidate compile/runtime/execution timeout, benchmark stability/coverage, op/deep profile counts/ratios, and worker kill/replacement fencing.
- [ ] Run new CPU tests and confirm the absent gates.

```bash
.venv/bin/python -m pytest tests/test_end_to_end_durable.py tests/test_fault_injection.py tests/test_cross_repo_contract.py tests/test_portable_schema_probe.py -q
```

Expected: collection fails for the new suites and probe.

- [ ] Implement failpoint hooks as test-only injected callables around transaction execution; production defaults are immutable no-op functions and cannot be configured from requests or environment.
- [ ] Implement the portable schema probe without copying wire models. After digest verification, load the published root schema and build three separate `Draft202012Validator` instances whose schemas are `$ref` wrappers for the absolute root `$id` plus `#/$defs/EvaluationRequest`, `#/$defs/EvaluationResource`, and `#/$defs/ErrorEnvelope`; register the unmodified root with `referencing.Registry().with_resource(root_id, Resource.from_contents(schema))`. Validate the outgoing object through the request validator before POST. Decode response bytes as strict UTF-8, reject duplicate keys and non-finite constants during stdlib JSON parsing, then validate 2xx bodies only through the resource validator and non-2xx bodies only through the error validator. Never validate a consumer against the top-level union or a handwritten field/type table. Persist `evaluation_id` only after a POST 200/202 resource also passes its ID/status rules (200 terminal; 202 queued/running with exact `Location: /evaluations/{evaluation_id}`); an ErrorEnvelope never supplies an ID. After a valid ID, poll only GET with capped exponential backoff; replay POST only after a transport failure with no response and no durable ID, using the original bytes and key.
- [ ] Implement the NPU gate as an orchestrator of the marked tests plus HTTP crash soak. It exits nonzero on any missing case, skipped calibration, digest mismatch, unstable benchmark, invalid profile, or fenced-write failure. Write the report atomically only after all checks complete; a report with `production_ready=false` is the only output on failure.
- [ ] Add a CPU CI workflow for Python 3.10, 3.11, and 3.12. Each job creates its own `.venv`, installs the exact `.[test,dev]` direct dependency set, runs the same eleven-version assertion as Task 2, checks generated bytes, runs Ruff and all non-NPU tests with skipped tests treated as a reviewed list, compiles the package, and proves no command changed generated files.

```yaml
- uses: actions/setup-python@v6
  with:
    python-version: ${{ matrix.python-version }}
- run: python -m venv .venv
- run: .venv/bin/python -m pip install -e '.[test,dev]'
- run: .venv/bin/python -c 'from importlib.metadata import version; expected={"fastapi":"0.139.2","uvicorn":"0.51.0","redis":"8.0.1","pydantic":"2.13.4","pydantic-settings":"2.14.2","rfc8785":"0.1.4","pytest":"9.1.1","fakeredis":"2.37.0","httpx":"0.28.1","jsonschema":"4.26.0","ruff":"0.16.0"}; actual={name:version(name) for name in expected}; assert actual == expected, (actual, expected)'
- run: .venv/bin/python scripts/generate_contract_artifacts.py --check
- run: .venv/bin/python -m ruff check .
- run: .venv/bin/python -m pytest -m 'not npu' -q
- run: .venv/bin/python -m compileall -q triton_evaluation scripts
- run: git diff --exit-code -- contracts/evaluation-api/2.0
```

- [ ] Run the entire deterministic gate locally.

```bash
.venv/bin/python scripts/generate_contract_artifacts.py --check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -m 'not npu' -q
.venv/bin/python -m compileall -q triton_evaluation scripts
git diff --exit-code -- contracts/evaluation-api/2.0
```

Expected: every command exits 0. Any existing environment-specific skip is listed in the test output and explicitly reviewed; contract, durability, auth, fault, deployment, and framework-free suites do not skip.

- [ ] On the target Ascend host, run the production release gate after deploying the candidate build.

```bash
.venv/bin/python scripts/npu_release_gate.py --service-url http://172.18.0.1:11907 --capability-file /run/secrets/triton-eval-grade --catalog-manifest /srv/triton-eval-release/catalog.json --cases /srv/triton-eval-release/calibration-cases.json --output /srv/triton-eval-release/reports/evaluation-api-2.0.json
```

Expected: exit 0 and an atomic report with `production_ready=true`. If target infrastructure is unavailable, the implementation may merge after CPU gates but must not be labeled or deployed production-ready.

- [ ] Commit deterministic CI and release gates.

```bash
git add pyproject.toml .github/workflows/evaluation-api-v2.yml tests/test_end_to_end_durable.py tests/test_fault_injection.py tests/test_cross_repo_contract.py tests/contract_consumers/portable_schema_probe.py tests/test_portable_schema_probe.py tests/test_profile_npu_integration.py tests/test_npu_release_calibration.py scripts/npu_release_gate.py
git commit -m "test(v2): gate durability contracts and npu release"
```

## Complete Execution Order

1. Task 1 freezes the typed schema; no storage or route work starts before it is GREEN.
2. Task 2 publishes schema/fixtures/digests. Record the two generated digest values from `contract.json` in the release notes used by Hub and Harness owners.
3. Task 3 freezes caller, catalog, and canonical request identity.
4. Task 4 creates the only durable resource/idempotency/queue admission path.
5. Task 5 adds every legal transition, terminal synthesis, deadline, recovery, and fence.
6. Task 6 replaces the private-engine shaping boundary and proves it conforms byte-for-byte to the Task 2 publication. Any artifact mismatch returns to reviewed Tasks 1–2; later tasks may not alter contract bytes.
7. Task 7 attaches the worker to the fenced protocol.
8. Task 8 exposes POST/GET and the unified error envelope.
9. Task 9 adds liveness, readiness, and bounded internal metrics.
10. Task 10 makes Redis/network/secret/catalog deployment durable and fail closed.
11. Task 11 enables the explicitly time-boxed legacy bridge and migrates the local smoke client/docs.
12. Task 12 runs the full CPU gate, then the separate target-NPU production gate.

Within this batch, execute tasks strictly in order. Parallelize only independent RED fixture authoring inside one task; do not allow two tasks to edit the same contract/storage files concurrently.

## Recovery and Rollback Points

| Checkpoint | Recoverable state | Rollback rule |
|---|---|---|
| After Task 2 | Contract-only publication, no runtime depends on it | A contract correction is a new reviewed Task 1/2 commit and new digests; never silently replace published bytes after a consumer pins them. |
| After Task 4 | Queued v2 records under isolated `triton-eval:v2` keys | Revert the candidate application while preserving Redis/AOF volume. The old application ignores the v2 prefix; do not delete accepted resources. |
| After Task 5 | Store can terminalize/recover without HTTP or NPU | Run watchdog/reaper from the tested maintenance entrypoint before resuming workers. Never edit resource hashes manually. |
| After Task 7 | Worker attempts may be running | Stop new claims, leave Redis/AOF intact, wait for leases to expire, deploy the prior compatible worker only if build/algorithm pins match; otherwise let the candidate reaper fail/retry safely. |
| After Task 8 | Durable client-visible IDs exist | Roll API back only to a build that can read schema 2.0 resource records. Keep the gateway and Redis addresses unchanged so clients can GET existing IDs. |
| After Task 10 | Durable Compose owns the Redis volume | `scripts/compose.sh down` must not remove volumes. Restore the prior image/config, rerun preflight, and start against the preserved volume. Never use `down -v` in this runbook. |
| During consumer rollout | Hub/Harness may have pinned final digests | Roll back the consumer independently while keeping legacy routes enabled. Do not roll contract bytes backward or generate new evaluation IDs for known idempotency keys. |
| NPU gate failure | CPU implementation may be merged but not released | Keep `production_ready=false`, retain the report, correct implementation/calibration, and rerun the entire NPU gate. Do not waive failed taxonomy, benchmark, profile, or fencing cases. |

## Spec Requirement to Task Coverage

| Canonical spec requirement | Implementation tasks | Blocking evidence |
|---|---|---|
| Strict schema 2.0, invariants, finite values | 1, 6 | `test_contract*`, `test_shaping.py` |
| Reproducible schema/golden fixtures and SHA-256 | 2, 6, 12 | artifact `--check`, cross-repo mutation tests |
| Caller capability, scopes, session/idempotency headers | 3, 8 | security/admission/HTTP security tests |
| Trusted immutable grade catalog and reference binding | 3, 10 | catalog and deployment-preflight tests |
| Canonical request digest and caller-scoped idempotency | 3, 4 | admission and concurrent idempotency tests |
| 200/202 wait, Location, GET recovery, hidden resources | 4, 8 | durable HTTP and visibility tests |
| Unified error envelope and stable mapping | 1, 8 | parameterized non-2xx contract tests |
| Monotonic resource FSM and immutable terminal | 5 | state-machine and CAS race tests |
| Deadline, leases, replacement, fencing, at-least-once | 5, 7, 12 | fencing/recovery/crash/failpoint tests |
| Crash-persistent Redis before first accepted 2xx | 4, 5, 10, 12 | atomic transaction tests, AOF preflight, fault tests |
| Typed correctness/performance/profile shaping | 1, 6 | engine fixture and orchestration tests |
| Subject failure versus infrastructure/cancel/unscored boundary | 6, 12 | grade-boundary, three-phase cancel, and cross-repo fixture tests |
| Provenance and grade eligibility | 3, 6 | grade admission/provenance tests |
| Live, ready, bounded metrics and redaction | 9 | health, metrics, leakage tests |
| Gateway bind, loopback Redis, secret/catalog preflight | 10 | fake-Docker/preflight/shell tests |
| Time-boxed legacy migration | 11 | legacy adapter and smoke tests |
| Deterministic CPU and target-NPU release gates | 12 | CI workflow and signed-off NPU report |

## Cross-Repository Publication and Blocking Gates

The only files copied or fetched by Eval Hub, framework-free Harness, or a standalone trusted coordinator are:

```text
contracts/evaluation-api/2.0/contract.json
contracts/evaluation-api/2.0/schema.json
contracts/evaluation-api/2.0/fixtures/index.json
```

The 29 exact fixture paths are the Task 2 list in `contracts/evaluation-api/2.0/fixtures/`; consumers fetch only entries named and hashed by `index.json`.

- `contract.json.evaluation_api_schema_digest` pins exact `schema.json` bytes.
- `contract.json.evaluation_api_golden_fixture_digest` pins exact RFC 8785 `fixtures/index.json` bytes; the index then pins every named fixture's exact bytes by relative path and SHA-256.
- JSON Schema consumers resolve request at `schema.json#/$defs/EvaluationRequest`, resource at `schema.json#/$defs/EvaluationResource`, and error at `schema.json#/$defs/ErrorEnvelope`; they never guess a top-level branch.
- Eval Hub deployment is blocked until its own contract test pins both Task 2 digests; persists `evaluation_id` immediately after, and only after, a POST 200/202 body passes strict JSON plus the exact `$defs/EvaluationResource` validator and ID/status/`Location` consistency; proves non-2xx `$defs/ErrorEnvelope` responses never supply an ID; recovers through GET or the original idempotent POST when no response/ID exists; resolves reference from the same trusted release catalog; and implements the frozen GradeRecord mapping without converting unscored to zero. Its cancel-race tests must cover pre-formal null fields, post-ID evidence retention, `retryable=false`, and late Evaluation responses losing the cancellation CAS.
- Framework-free Harness deployment is blocked until its own portable `jsonschema==4.26.0` consumer test pins both digests, resolves all three exact `$defs`, uses a trial capability, preserves the original request/idempotency key across recovery, distinguishes resource/check/domain states, and imports no Hub/service framework package or copied Evaluation wire model. Standalone cancellation uses the same three-phase GradeRecord invariant as Hub.
- A standalone trusted coordinator is blocked on the same catalog/digest/provenance and grade-boundary tests as Hub.
- The first production v2 deployment is blocked on Task 12 CPU GREEN, deployment preflight GREEN, and a target-NPU report with `production_ready=true`.
- Removal of `/verify`, `/profile`, and `/tasks/{task_id}` is outside this implementation batch. It is blocked until Hub and Harness are deployed on the pinned final digests, stored-artifact reads are verified, compatibility metrics remain acceptable through the published window, and a separate removal review approves the change.

## Final Verification Before Handoff

- [ ] Run the complete deterministic gate exactly once from a clean process environment.

```bash
.venv/bin/python scripts/generate_contract_artifacts.py --check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -m 'not npu' -q
.venv/bin/python -m compileall -q triton_evaluation scripts
bash -n scripts/compose.sh scripts/start.sh docker/entrypoint.sh
git diff --check
git diff --exit-code -- contracts/evaluation-api/2.0
```

Expected: every command exits 0 and artifact bytes remain unchanged.

- [ ] Inspect scope and contract consistency.

```bash
rg -n 'mean_abs_err|throughput|exactly-once|dynamic manifest|0\.0\.0\.0|--appendonly[[:space:]]+no' triton_evaluation contracts tests docker-compose.yml
rg -n 'evaluation_api_schema_version|evaluation_api_schema_digest|evaluation_api_golden_fixture_digest|json_schema_path|golden_fixture_index_path' contracts/evaluation-api/2.0/contract.json
git status --short
```

Expected: the first search returns only explicit rejection/assertion text, the second returns all five frozen keys, and `git status --short` is empty after the final task commit.

- [ ] Record the final contract values and gate results in the implementation handoff without editing the design spec.

```bash
.venv/bin/python -c 'import json; print(json.dumps(json.load(open("contracts/evaluation-api/2.0/contract.json", encoding="utf-8")), indent=2, sort_keys=True))'
```

Expected: the handoff includes schema version 2.0, the exact schema digest, the exact fixture-index digest, CPU gate result, and NPU gate result or explicit non-production status.
