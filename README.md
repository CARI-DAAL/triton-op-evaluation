# triton-evaluation

Evaluate Triton-Ascend operators against a PyTorch reference for correctness,
performance and profiling. Supply an **input definition, named workloads and
optional Triton source**. The input format follows the pinned NPUKernelBench
protocol; the reference lives in `definition.reference`. Implementations expose
`run(*args)` in the order declared by `definition.inputs`.

The evaluation service runs separately on an Ascend host. Accepted work is stored
in Redis and can be recovered by ID after a client disconnect.

[Quick start](#quick-start) · [API](#api) · [Deployment](#deploy) ·
[Development](#contributing) · [Documentation](#documentation)

## Quick start

You need Python 3.10+, a [running service](docs/operations/deployment.md), and a
caller capability file with submission/read access. The bundled CLI uses only
the Python standard library. Set its URL to an address reachable from the client;
`127.0.0.1` refers to the client host or its container.

Prepare `definition.json` (or use an official file directly):

```json
{
  "name": "add_one",
  "axes": {},
  "inputs": {"x": {}},
  "outputs": {},
  "reference": "def run(x): return x + 1"
}
```

Create `workload.jsonl`, one workload per line:

```json
{"uuid":"case-1","inputs":{"x":{"type":"random","shape":[1024],"dtype":"float32"}}}
```

With a Triton implementation exporting `run(x)` in `triton_impl.py`:

```bash
export EVALUATION_URL=http://127.0.0.1:11907
export EVALUATION_CAPABILITY_FILE=/path/to/caller-token
TRITON_EVAL_CLI=plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py
python3 "$TRITON_EVAL_CLI" benchmark   --definition definition.json --workloads workload.jsonl --triton triton_impl.py   --save-handle evaluation-handle.json
```

Omit `--triton` for Torch-only measurement. Add `--triton-only` to measure a
Triton implementation without a reference comparison. `verify` requires both
implementations; `profile` supports either source selection. Single-source
measurements are explicitly correctness-unverified and report no speedup.

The CLI waits for a result and writes JSON to stdout. Client wait expiry does not
cancel accepted work; recover it on the original instance:

```bash
python3 "$TRITON_EVAL_CLI" get "$EVALUATION_ID"
```

See the [client guide](plugins/triton-eval/README.md) and
[skill](plugins/triton-eval/skills/triton-eval/SKILL.md) for retry, recovery and
plugin packaging. Both client compatibility manifests share the same skill.

## API

All submission endpoints accept one input protocol:

| Field | Contents |
| --- | --- |
| `definition` | Ordered named inputs, axes, optional metadata and optional Torch `reference` source |
| `workloads` | Nonempty array of official workload objects, including named `inputs` |
| `triton_code` | Optional Triton source defining `run(*args)` |
| `seed` | Shared per-workload RNG seed; defaults to the official value `200` |
| `custom_inputs_code` | Optional explicit custom-generator source; otherwise use the reference's declared entrypoint |

At least one of `definition.reference` and `triton_code` is required. Supported
inputs include random tensors, scalars, tensor lists and custom generation.
Input generation executes on the assigned NPU after admission. The
[request guide](docs/usage/evaluations.md) specifies distributions, axes,
unsupported external resources and validation limits.

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Correctness comparison; requires both implementations |
| `POST /benchmark` | Single-source timing or correctness plus comparative timing |
| `POST /profile` | Profiling evidence |
| `POST /evaluations` | Explicit selection of checks, including compile prechecks |
| `GET /evaluations/{evaluation_id}` | Recover the admitted resource |

All evaluation routes use `Authorization: Bearer`. `Idempotency-Key` is optional;
when supplied, retry the same request against its original instance. A submission
returns 200 for a terminal resource or 202 with a recovery `Location`. Inspect
each check result: a completed evaluation can establish that a candidate is wrong.

FastAPI generates `/openapi.json` and `/docs` from request/response models.
`/livez`, `/readyz` and observer-authenticated `/internal/metrics` expose health
and observability.

Comparison timing reuses Torch baselines by default. Set
`performance.use_baseline_cache=false` (CLI: `--no-baseline-cache`) for a fresh
baseline that leaves the cache unchanged. Input definitions, generation policy,
workloads and seed participate in reference identity; candidate changes can reuse
that baseline. Single-source timing always measures fresh.

## Deploy

Each Ascend host owns its API, durable Redis, CPU workers and NPU workers.

| Deployment | Configuration and entrypoint |
| --- | --- |
| Docker Compose | `docker/.env.example`, `docker/compose.yaml`, `scripts/compose.sh` |
| Host processes | `.env.example`, `scripts/start.sh`, `scripts/stop.sh` |

Follow the [deployment guide](docs/operations/deployment.md) and the
[upgrade procedure](docs/operations/recovery.md#upgrading). The previous
`torch_code`/positional `cases` submission shape and `--torch`/`--cases` CLI flags
are retired. Deploy matching API, worker and client versions, drain older work,
and renew runtime source pins when upgrading.

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
│   ├── operator_input.py       # Operator request, cases and translation
│   ├── config.py               # Shared configuration
│   └── serialization.py        # Canonical serialization
├── plugins/
│   └── triton-eval/            # Portable client plugin
│       ├── plugin.json         # Standard plugin manifest
│       └── skills/             # Skill instructions and standalone CLI
├── scripts/                    # Entrypoints, deployment and diagnostics
├── tests/                      # Protocol, execution and recovery checks
├── docs/                       # Usage, operations and architecture
├── docker/                     # Container build and deployment
│   ├── Dockerfile              # Service image build
│   ├── compose.yaml            # Host deployment
│   ├── .env.example            # Deployment settings template
│   └── entrypoint.sh           # Container startup
├── .dockerignore               # Exclusions for the repository build context
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
One workload resolved for execution: concrete axes and every definition input
in declared order. Tensor inputs are generated from their spec and the request
seed; scalars pass through.
_Avoid_: test case, shape

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

### Input generation

The input-generation policy is adapted from Ascend's NPUKernelBench at commit
`0607a29d231cde384c9dfdb8a39f54f91837c327`, under Apache-2.0. Its
[license](triton_evaluation/runtime/INPUT_GENERATION_LICENSE) is included in the
runtime package. The independent parity tests retain the corresponding upstream
function source and attribution.
