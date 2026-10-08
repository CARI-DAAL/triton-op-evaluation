# triton-evaluation

Evaluate Triton-Ascend operators against a PyTorch reference for correctness,
performance and profiling. Supply **PyTorch code, Triton code and input cases**;
both implementations expose `run(*args)`.

Use the bundled CLI from a developer machine or coding agent, or call the HTTP
API directly. The evaluation service runs separately on an Ascend host. Accepted
work is stored in Redis and can be recovered by ID after a client disconnect.

[Quick start](#quick-start) · [API](#api) · [Deployment](#deploy) ·
[Development](#contributing) · [Documentation](#documentation)

## Quick start

You need Python 3.10+, a [running service](docs/operations/deployment.md), and a
caller capability file with submission and read access. The CLI uses only the
Python standard library; the client machine needs no Torch, Triton or NPU stack.

Run the following from the repository root with `torch_impl.py` and
`triton_impl.py` ready. This example supplies one tensor argument to each
implementation's `run` function:

```bash
export EVALUATION_URL=http://10.0.0.11:11907
export EVALUATION_CAPABILITY_FILE=/path/to/caller-token
TRITON_EVAL_CLI=plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py

cat > cases.json <<'JSON'
[{"inputs": [{"shape": [1024], "dtype": "float32"}], "seed": 0}]
JSON

python3 "$TRITON_EVAL_CLI" verify \
  --torch torch_impl.py --triton triton_impl.py --cases cases.json \
  --save-handle evaluation-handle.json
```

The CLI waits for a result, writes JSON to stdout and progress to stderr. Use
`benchmark` for correctness plus timing, or `profile` for profiling. Client
wait expiry does not cancel accepted work; recover it on the same instance:

```bash
# Set EVALUATION_ID to the ID returned by the service.
python3 "$TRITON_EVAL_CLI" get "$EVALUATION_ID"
```

See the [client guide](plugins/triton-eval/README.md) for configuration and the
[skill](plugins/triton-eval/skills/triton-eval/SKILL.md) for agent usage. The
plugin follows the standard `plugin.json` and `skills/` layout and is independent
of the server deployment.

## API

All submission endpoints accept the same three required fields:

| Field | Contents |
| --- | --- |
| `torch_code` | PyTorch reference source defining `run(*args)` |
| `triton_code` | Triton implementation source defining `run(*args)` |
| `cases` | Ordered input arguments and an optional seed for each case |

Inputs may be JSON scalars or tensor descriptions with `shape`, `dtype` and an
optional value range. [The request guide](docs/usage/evaluations.md) covers input
validation, optional execution settings and complete HTTP examples.

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Check correctness |
| `POST /benchmark` | Check correctness, then measure performance |
| `POST /profile` | Collect profiling evidence |
| `POST /evaluations` | Select checks, including syntax/import prechecks |
| `GET /evaluations/{evaluation_id}` | Read or recover an evaluation |

`/evaluations` defaults to correctness and performance. Evaluation endpoints
use `Authorization: Bearer`. Provide an `Idempotency-Key`
when a submission may need retrying. The operation shortcuts share the same
request format, authentication and durable result lifecycle.

A POST returns **200** for a terminal resource or **202** with a recovery
`Location`. GET returns the current resource, including pending states. Inspect
the per-check results: `completed` means execution finished, not necessarily
that the candidate passed. See [recovery and upgrades](docs/operations/recovery.md).

Each running instance also exposes:

- `/docs` — interactive API documentation.
- `/openapi.json` — the API description generated from models and routes.
- `/livez` and `/readyz` — liveness and execution readiness.
- `/internal/metrics` — metrics for authenticated observers.

## Deploy

Each Ascend host runs an API, durable Redis, CPU workers for import prechecks,
and NPU workers for device execution. Workers own leased attempts and publish
results through fenced writes. Clients connect to the instance that accepted
their evaluation.

| Deployment | Configuration and entrypoint |
| --- | --- |
| Docker Compose | [`docker.env.example`](docker.env.example), [`docker-compose.yml`](docker-compose.yml) and `scripts/compose.sh` |
| Host processes | [`.env.example`](.env.example), `scripts/start.sh` and `scripts/stop.sh` |

Follow the [deployment guide](docs/operations/deployment.md) to configure caller
credentials, private networking, Redis durability and the Ascend runtime. The
[upgrade guide](docs/operations/recovery.md#upgrading) explains client migration
and recovery of existing work. Runtime files with pinned source paths or hashes
require rebuilt deployment manifests when those files change.

## Contributing

### Set up and validate

Use Python 3.10+ and run commands from the repository root. CPU checks need no
Ascend hardware; CI runs them on Python 3.10, 3.11 and 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

python -m ruff check .
python -m pytest -m 'not npu' -q
python -m compileall -q triton_evaluation scripts plugins
git diff --check
```

The [test guide](tests/README.md) lists focused checks, consumer integration and
NPU validation. CPU test results do not establish device correctness or timing.

### Prepare a change

- Keep each change focused and describe its behavior, validation and limits.
- Update models, routes and OpenAPI coverage together when changing the HTTP
  interface. Admission must persist identity before candidate execution.
- Preserve check evidence and lease fencing. Infrastructure failures and missing
  evidence must not become successful correctness verdicts.
- Validate runtime and measurement changes on the target NPU with an explicit
  evaluator build. Rebuild source-pinned policies before deployment.
- Keep reusable regression inputs in `tests/fixtures/`, user workflows in
  `docs/usage/`, operations in `docs/operations/`, implementation invariants in
  `docs/architecture/`, and decisions in `docs/adr/`. Link to one maintained page
  per subject; completed experiments belong in Git history or external artifacts.

The bundled engine is excluded from general Ruff formatting. Preserve its
copyright headers and verify deliberate engine changes; see
[third-party source](#third-party-source).

Submit [issues](https://github.com/CARI-DAAL/triton-op-evaluation/issues) with a
revision, runtime and reproducible example, or open a
[pull request](https://github.com/CARI-DAAL/triton-op-evaluation/pulls) with the
relevant checks. Keep credentials, private source and device dumps out of the
repository.

## Repository layout

| Path | Responsibility |
| --- | --- |
| [`triton_evaluation/api/`](triton_evaluation/api/) | HTTP models, routes, authentication and OpenAPI |
| [`triton_evaluation/service/`](triton_evaluation/service/) | Admission, durable scheduling and terminal evidence |
| [`triton_evaluation/runtime/`](triton_evaluation/runtime/) | Workers, device health, isolation, profiling and caches |
| [`triton_evaluation/evaluators/`](triton_evaluation/evaluators/) | Check orchestration and result shaping |
| [`triton_evaluation/engine/`](triton_evaluation/engine/) | Bundled evaluation engine |
| [`config.py`](triton_evaluation/config.py), [`serialization.py`](triton_evaluation/serialization.py) | Shared configuration and canonical serialization |
| [`plugins/triton-eval/`](plugins/triton-eval/) | Portable skill and standalone HTTP client |
| [`scripts/`](scripts/README.md) | Process entrypoints, deployment and diagnostics |
| [`docker/`](docker/), [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml) | Container image and host deployment |
| [`tests/`](tests/README.md) | Protocol, execution, recovery and hardware checks |
| [`docs/`](docs/README.md) | Usage, operations, architecture and decisions |

## Documentation

- **Use:** [Operator requests](docs/usage/evaluations.md),
  [operation shortcuts](docs/usage/kernel-eval.md),
  [client plugin](plugins/triton-eval/README.md).
- **Operate:** [Deployment](docs/operations/deployment.md),
  [recovery and upgrades](docs/operations/recovery.md),
  [profiler runtime](docs/operations/profiler-runtime.md),
  [reference caches](docs/operations/reference-cache.md).
- **Understand:** [Execution flow](docs/architecture/kernel-evaluation.md),
  [sandbox](docs/architecture/engine-sandbox.md),
  [card health](docs/architecture/card-health.md),
  [domain vocabulary](#domain-vocabulary).

The [documentation index](docs/README.md) includes the remaining architecture
notes and decisions.

## Domain vocabulary

<details>
<summary>Expand the domain vocabulary</summary>

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

</details>

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
