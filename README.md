# triton-evaluation

A FastAPI service that evaluates Triton-Ascend kernels against a reference:
correctness, performance, profiling, and syntax/import checks.

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

The optional task-tool interface exposes `/verify`, `/benchmark` and `/profile`
against an official task catalog. See [tool usage](docs/usage/kernel-eval.md).

[Submit and recover an evaluation](docs/usage/evaluations.md) explains credentials,
idempotency, asynchronous results and request examples. Python models and route
declarations define the HTTP interface; generated schemas are not deployment inputs.

## Develop

Python 3.10+; run from the repository root. CPU tests require no Ascend device.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'
python -m pytest -m 'not npu' -q
python -m ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for change verification and
[tests/README.md](tests/README.md) for consumer and NPU checks.

## Deploy

Use [the deployment guide](docs/operations/deployment.md) with
[`docker.env.example`](docker.env.example) and [`docker-compose.yml`](docker-compose.yml).
For host execution, use [`.env.example`](.env.example) and `scripts/start.sh`.
Both paths require configured credentials, a catalog, durable Redis, and the
appropriate Ascend runtime. API documentation is served on the instance's own URL.

## Repository

| Path | Purpose |
| --- | --- |
| [`triton_evaluation/`](triton_evaluation/) | HTTP interface, evaluation logic, durable scheduling and bundled engine |
| [`tests/`](tests/) | Behavior, HTTP compatibility, fixtures and hardware gates |
| [`scripts/`](scripts/README.md) | Supported preparation, deployment and diagnostic commands |
| [`docker/`](docker/), `Dockerfile`, `docker-compose.yml` | Container entrypoint, image and deployment |
| [`docs/`](docs/README.md) | Usage, operations, architecture and decisions |
| [`GLOSSARY.md`](GLOSSARY.md) | Shared domain vocabulary |

The source package stays at the root alongside its entrypoints. Test data and
historical evidence are not part of the running service.

## Contribute and get help

Use [issues](https://github.com/CARI-DAAL/triton-op-evaluation/issues) for reproducible
bugs and [pull requests](https://github.com/CARI-DAAL/triton-op-evaluation/pulls) for
reviewed changes. Include the relevant revision, runtime and reproduction steps.

See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled source attribution.
A repository-wide license has not been declared.
