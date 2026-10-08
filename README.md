# triton-evaluation

Measure or profile a PyTorch or Triton-Ascend operator, or compare a Triton
implementation against PyTorch for correctness and speedup. Supply **input cases
and at least one implementation**; each implementation exposes `run(*args)`.

Use the bundled CLI from a developer machine or coding agent, or call the HTTP
API directly. The evaluation service runs separately on an Ascend host. Accepted
work is stored in Redis and can be recovered by ID after a client disconnect.

[Quick start](#quick-start) · [API](#api) · [Deployment](#deploy) ·
[Development](#contributing) · [Documentation](#documentation)

## Quick start

You need Python 3.10+, a [running service](docs/operations/deployment.md), and a
caller capability file with submission and read access. The CLI uses only the
Python standard library; the client machine needs no Torch, Triton or NPU stack.

The example uses a local service listening on `127.0.0.1:11907`, the API's
default bind address. For a remote service, use its reachable hostname or IP.
Inside a container, `127.0.0.1` refers to that container. Compose deployments
use their explicitly configured private host address. The CLI requires an
explicit URL through `EVALUATION_URL` or `--url`.

Run the following from the repository root with `torch_impl.py` and
`triton_impl.py` ready. This example supplies one tensor argument to each
implementation's `run` function:

```bash
export EVALUATION_URL=http://127.0.0.1:11907
export EVALUATION_CAPABILITY_FILE=/path/to/caller-token
TRITON_EVAL_CLI=plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py

cat > cases.json <<'JSON'
[{"inputs": [{"shape": [1024], "dtype": "float32"}], "seed": 0}]
JSON

python3 "$TRITON_EVAL_CLI" verify \
  --torch torch_impl.py --triton triton_impl.py --cases cases.json \
  --save-handle evaluation-handle.json
```

The comparison above needs both implementations. To measure just one, keep the
same cases and supply only its source:

```bash
python3 "$TRITON_EVAL_CLI" benchmark --torch torch_impl.py --cases cases.json
python3 "$TRITON_EVAL_CLI" benchmark --triton triton_impl.py --cases cases.json
```

Use `profile` with either source selection for profiling. A single-implementation
measurement reports latency and `correctness_verified: false`; it cannot establish
correctness or a speedup. `verify` requires both implementations. Supplying both
to `benchmark` retains correctness checks and comparative timing.

The CLI waits for a result, writes its JSON resource to stdout and progress to
stderr. A successful measurement exits zero while explicitly noting
`correctness not verified` for a single implementation. Client wait expiry does
not cancel accepted work; recover it on the same instance:

```bash
# Set EVALUATION_ID to the ID returned by the service.
python3 "$TRITON_EVAL_CLI" get "$EVALUATION_ID"
```

See the [client guide](plugins/triton-eval/README.md) for configuration and the
[skill](plugins/triton-eval/skills/triton-eval/SKILL.md) for agent usage. The
plugin is independent of the server deployment. Its directory keeps portable
metadata, client compatibility manifests and skill content separate:

```text
plugins/triton-eval/              # Repository location; this is the plugin root
├── plugin.json                  # Canonical portable manifest
├── .codex-plugin/plugin.json    # Codex compatibility metadata
├── .claude-plugin/plugin.json   # Claude Code metadata
└── skills/triton-eval/          # Shared skill and bundled CLI
    ├── SKILL.md
    └── scripts/triton_eval.py
```

The hidden directories contain metadata only; the [client guide](plugins/triton-eval/README.md)
links the official format and open-source examples behind this layout.

## API

All submission endpoints use the same fields. `cases` is required, together
with at least one of `torch_code` or `triton_code`:

| Field | Contents |
| --- | --- |
| `torch_code` | PyTorch source defining `run(*args)`; optional when Triton source is supplied |
| `triton_code` | Triton source defining `run(*args)`; optional when PyTorch source is supplied |
| `cases` | Ordered input arguments and an optional seed for each case |

Inputs may be JSON scalars or tensor descriptions with `shape`, `dtype` and an
optional value range. [The request guide](docs/usage/evaluations.md) covers input
validation, optional execution settings and complete HTTP examples.

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Check correctness; requires both implementations |
| `POST /benchmark` | Measure one implementation, or check correctness and speedup with both |
| `POST /profile` | Collect profiling evidence |
| `POST /evaluations` | Select checks, including syntax/import prechecks |
| `GET /evaluations/{evaluation_id}` | Read or recover an evaluation |

`/evaluations` defaults to performance for one implementation, or correctness
and performance for both. Evaluation endpoints
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

Key directories and configuration files:

```text
.
├── triton_evaluation/
│   ├── api/                    # HTTP interface and OpenAPI
│   ├── service/                # Admission, scheduling and results
│   ├── runtime/                # Workers, isolation, profiling and caches
│   ├── evaluators/             # Check orchestration and result shaping
│   ├── engine/                 # Bundled evaluation engine
│   ├── config.py               # Shared configuration
│   └── serialization.py        # Canonical serialization
├── plugins/
│   └── triton-eval/            # Portable client plugin
│       ├── plugin.json         # Standard plugin manifest
│       └── skills/             # Skill instructions and standalone CLI
├── scripts/                    # Entrypoints, deployment and diagnostics
├── tests/                      # Protocol, execution and recovery checks
├── docs/                       # Usage, operations and architecture
├── docker/                     # Container entrypoint
├── Dockerfile                  # Service image build
├── docker-compose.yml          # Host deployment
└── pyproject.toml              # Dependencies and tooling
```

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
Input cases and one or both implementations. A single implementation supports
measurement and profiling; supplying PyTorch and Triton supports correctness
checks and comparative performance.

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
