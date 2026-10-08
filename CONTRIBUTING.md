# Contributing

Read the [README](README.md) for setup and the [documentation index](docs/README.md)
for the part you are changing. Use [GLOSSARY.md](GLOSSARY.md) for domain terms.

## Validate a change

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

## Keep ownership explicit

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
verifying the affected behavior. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Keep the repository useful

Put user workflows in `docs/usage/`, operator procedures in `docs/operations/`,
current implementation invariants in `docs/architecture/`, and lasting trade-offs
in `docs/adr/`. Keep one authoritative page for each subject and link to it.
Completed plans, local deployment snapshots and one-off experiments belong in
Git history or external run artifacts. Retain a fixture when it proves behavior
that an active test or current explanation depends on.

The root layout is intentional: this is a deployed service with explicit
entrypoints, not a new library packaging project. A layout change must improve
ownership or distribution behavior and update its consumers together.

## Submit for review

Describe the concrete problem, resulting behavior, verification and remaining
limits. Keep unrelated changes separate. Update the relevant operating instructions
and API models with the implementation. Use synthetic reproductions; keep tokens,
private source, device dumps and local environment files outside the repository.

The documentation layout follows GitHub's guidance for a
[focused README](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
and [discoverable contribution instructions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors).
The [PyPA layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
describes the trade-offs behind source-layout changes.
