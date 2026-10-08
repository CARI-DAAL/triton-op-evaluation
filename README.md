# triton-evaluation

A FastAPI service that evaluates Triton-Ascend kernels against a reference:
correctness, performance, profiling, and syntax/import checks. Submit three fields:
`torch_code`, `triton_code`, and `cases`. Both source strings define `run(*args)`;
cases describe positional inputs. No task registration or dataset is required.

An accepted evaluation is a durable resource. Workers execute it under a lease,
publish typed evidence, and recover unfinished work after failures. Clients can
reconnect and retrieve the result from the instance that accepted it. Each host
owns its Redis state, CPU workers and assigned NPU cards.

## API

On a configured, running instance:

- **`/docs`** — interactive API documentation.
- **`/openapi.json`** — generated request, response and authentication schemas.
- **`POST /evaluations`** — submit or replay an evaluation.
- **`GET /evaluations/{evaluation_id}`** — recover its current state and results.
- **`/livez`** and **`/readyz`** — process health and execution readiness.

`POST /verify`, `/benchmark` and `/profile` accept the same three-field input
and select the requested checks. All routes use the same capability credential
and return durable evaluation resources. See [operation shortcuts](docs/usage/kernel-eval.md).

A minimal request reads the two implementations from local files:

```python
from pathlib import Path

request = {
    "torch_code": Path("torch_impl.py").read_text(encoding="utf-8"),
    "triton_code": Path("triton_impl.py").read_text(encoding="utf-8"),
    "cases": [{"inputs": [{"shape": [1024], "dtype": "float32"}]}],
}
```

Each implementation receives one generated tensor.
`POST /evaluations` defaults to correctness and performance. The
[request example](docs/usage/evaluations.md) includes authentication, submission
and result recovery.

[Submit and recover an evaluation](docs/usage/evaluations.md) explains credentials,
idempotency, asynchronous results and request examples. Python models and route
declarations define the HTTP interface. FastAPI generates `/openapi.json` from
those definitions by default, and `/docs` uses it to display the interactive API.
Ordinary evaluation requests go directly to their endpoint without fetching the
schema first. The service does not need a schema file on disk to start. See
[FastAPI's OpenAPI explanation](https://fastapi.tiangolo.com/tutorial/first-steps/#openapi)
and [response models](https://fastapi.tiangolo.com/tutorial/response-model/).

## Develop

Python 3.10+; run from the repository root. CPU tests require no Ascend device.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'
python -m pytest -m 'not npu' -q
python -m ruff check .
```

See [Contributing](#contributing) for change verification and
[tests/README.md](tests/README.md) for consumer and NPU checks.

## Deploy

Use [the deployment guide](docs/operations/deployment.md) with
[`docker.env.example`](docker.env.example) and [`docker-compose.yml`](docker-compose.yml).
For host execution, use [`.env.example`](.env.example) and `scripts/start.sh`.
Both paths require configured credentials, durable Redis, and the
appropriate Ascend runtime. API documentation is served on the instance's own URL.

## Repository

| Path | Purpose |
| --- | --- |
| [`triton_evaluation/`](triton_evaluation/) | Shared configuration, profiler and reference-cache policies |
| [`triton_evaluation/api/`](triton_evaluation/api/) | HTTP models, routes and generated OpenAPI |
| [`triton_evaluation/service/`](triton_evaluation/service/) | Admission, durable scheduling, credentials and terminal evidence |
| [`triton_evaluation/runtime/`](triton_evaluation/runtime/) | Workers, card health, execution environment and precheck caching |
| [`triton_evaluation/evaluators/`](triton_evaluation/evaluators/) | Check orchestration and adapters to the engine |
| [`triton_evaluation/engine/`](triton_evaluation/engine/) | Bundled evaluation engine |
| [`tests/`](tests/) | Behavior, HTTP compatibility, fixtures and hardware gates |
| [`scripts/`](scripts/README.md) | API/worker entrypoints, preparation, deployment and diagnostic commands |
| [`docker/`](docker/), `Dockerfile`, `docker-compose.yml` | Container entrypoint, image and deployment |
| [`docs/`](docs/README.md) | Usage, operations, architecture and decisions |

The source package stays at the root; process entrypoints live under `scripts/`.
Test data and historical evidence are not part of the running service.

## Get help

Use [issues](https://github.com/CARI-DAAL/triton-op-evaluation/issues) for reproducible
bugs and [pull requests](https://github.com/CARI-DAAL/triton-op-evaluation/pulls) for
reviewed changes. Include the relevant revision, runtime and reproduction steps.

## Contributing

Read the [README](README.md) for setup and the [documentation index](docs/README.md)
for the part you are changing. Use [Domain vocabulary](#domain-vocabulary) for domain terms.

### Validate a change

Install `.[test,dev]` in a virtual environment. Run focused tests while working,
then the same CPU checks used by CI:

```bash
python -m ruff check .
python -m pytest -m 'not npu' -q
python -m compileall -q triton_evaluation scripts
git diff --check
```

CI covers Python 3.10–3.12. NPU behavior needs the separate
[hardware checks](tests/README.md); a passing CPU suite is not a throughput or
device-correctness measurement.

### Keep ownership explicit

- HTTP models and routes generate OpenAPI. Verify `/openapi.json` when changing
  request bodies, responses, authentication, headers or status codes.
- Admission persists identity before candidate execution. Execution and result
  publication belong to a fenced attempt.
- Preserve observed evidence. An infrastructure failure is not a candidate
  correctness verdict, and missing evidence cannot establish success.
- Test through the interface callers use. Keep reusable examples and historical
  regression bytes in `tests/fixtures/`, separate from runtime code.
- Runtime and measurement changes require explicit evaluator identity and NPU
  validation. Source files pinned by an active timing policy need a new policy
  before deployment; broad formatting changes are not a runtime migration.

The bundled engine has third-party license headers and is excluded from general
Ruff formatting. Make engine edits deliberately, preserving attribution and
verifying the affected behavior. See [third-party source](#third-party-source).

### Keep the repository useful

Put user workflows in `docs/usage/`, operator procedures in `docs/operations/`,
current implementation invariants in `docs/architecture/`, and lasting trade-offs
in `docs/adr/`. Keep one authoritative page for each subject and link to it.
Completed plans, local deployment snapshots and one-off experiments belong in
Git history or external run artifacts. Retain a fixture when it proves behavior
that an active test or current explanation depends on.

The source package remains at the root, with API and worker entrypoints under
`scripts/`. Layout changes must improve ownership or distribution behavior and
update their consumers together.

### Submit for review

Describe the concrete problem, resulting behavior, verification and remaining
limits. Keep unrelated changes separate. Update the relevant operating instructions
and API models with the implementation. Use synthetic reproductions; keep tokens,
private source, device dumps and local environment files outside the repository.

The documentation layout follows GitHub's guidance for a
[focused README](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
and [discoverable contribution instructions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors).
The [PyPA layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
describes the trade-offs behind source-layout changes.

## Domain vocabulary

**Evaluation instance**:
The authority that admits an evaluation and owns its execution, durable
identity, physical cards and recoverable evidence.

**Evaluation handle**:
The information identifying an evaluation and its owning instance, sufficient
to recover that same logical evaluation without submitting duplicate work.

**Admission**:
The durable decision to accept one logical evaluation with pinned identity,
quotas and deadline. Admission does not execute candidate code.

**Execution class**:
The capacity needed by an evaluation: CPU for syntax/import checks alone, or
NPU for evaluations containing device checks. One attempt owns all requested checks.

**Check evidence**:
A typed observation produced by a requested check in an attempt. Missing
evidence cannot establish success; a candidate failure can still be valid evidence.

**Precheck verdict**:
Evidence from parsing and importing a candidate under an identified toolchain.
It does not prove correctness or successful device compilation for every input.

**EvaluationResource**:
A recoverable evaluation with a stable identity, a deadline, and a lifecycle
independent of the caller's connection.

**Attempt**:
One worker's leased execution of an EvaluationResource. A replacement attempt
belongs to the same evaluation and must not overwrite a newer result.

**Attempt card provenance**:
Optional knowledge of the physical card assigned to one attempt. It cannot
describe a replacement attempt, even when both belong to the same evaluation.

**Card affinity avoidance**:
A retry policy that avoids the last known failed card. It is independent of
recording the card on which the current attempt runs.

**Card health**:
The assessment of whether a worker's assigned NPU card is fit to execute another
attempt. It is distinct from the correctness of a candidate kernel.

**Canary**:
A known-good kernel used to assess card health independently of the candidate.

**Drained worker**:
A worker that has suspended new attempts until its card passes a canary.
It is not available execution capacity while drained.

**Suspicious ending**:
An attempt ending without trustworthy execution evidence, such as a lost lease
or an evaluator crash. A normal candidate failure is not a suspicious ending.

**Recovery ladder**:
An ordered sequence of increasingly disruptive card recovery actions, each
followed by a canary to determine whether recovery succeeded.

### Operator inputs

**Operator request**:
A PyTorch implementation, a Triton implementation and input cases for comparing
their behavior under requested checks.

**Case**:
One ordered list of positional inputs and a deterministic seed. Tensor inputs
are generated from their shape, dtype and optional range; scalars pass through.

**Operation shortcut**:
An HTTP endpoint that selects correctness, performance or profiling while using
the same request input, admission process and durable result lifecycle.

## Third-party source

### Evaluation engine

`triton_evaluation/engine/` contains code originating from
`cannbot-skills/ops/triton-op-verifier`, as recorded in its package documentation.
The upstream headers identify Huawei Technologies Co., Ltd. and the
CANN Open Software License Agreement Version 2.0. Preserve those headers when
modifying or redistributing the files. The repository currently does not include
the full CANN license text referenced by those headers.

This section records the bundled engine attribution. No repository-wide license has been
selected for the remaining project source.
