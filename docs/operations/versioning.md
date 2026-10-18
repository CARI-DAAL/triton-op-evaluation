# Release identity

Callers and operators must be able to say exactly which service, client and
build answered a request. [`triton_evaluation/versioning.py`](../../triton_evaluation/versioning.py)
is the single place those identities are defined; the API, the workers and
[`scripts/release_manifest.py`](../../scripts/release_manifest.py) share it, so a
declared identity always describes implemented behaviour.

## The identities

| Identity | Source | Meaning |
| --- | --- | --- |
| `service_version` | `pyproject.toml`, reported by the installed distribution | Service release number, also OpenAPI's `info.version` |
| `client_version` | the three plugin manifests and the CLI's `__version__` | Client release number, published independently of the service |
| `api_versions` | the version module | HTTP contracts this build implements; today exactly `legacy` |
| `storage` | the version module | Persisted formats this build reads and writes; today exactly `legacy` |
| `evaluator_build` | deployment input | `sha256:<digest>` of the deployed image, or a reviewed `git:<commit>` |
| `source_digest` | the installed package | Digest of the Python sources this process is running |
| `default_protocol` | not implemented | Stays `null` until the evaluation-protocol manifest exists |

`legacy` names today's frozen behaviour. Nothing here declares a negotiated
contract, a protocol digest or a storage migration that does not exist yet;
those arrive with their own tickets and their own tests.

## Release manifest

The manifest describes one release: its versions, its sources and its declared
dependency pins. Generation is deterministic -- the same checkout and the same
revision always produce the same canonical bytes.

Build identity is bound *after* the artifact exists, so the image never has to
contain its own digest:

```bash
python -m scripts.release_manifest generate --output release-manifest.json
docker build -f docker/Dockerfile -t "${REGISTRY}/triton-eval:${VERSION}" .
docker push "${REGISTRY}/triton-eval:${VERSION}"            # prints the digest
python -m scripts.release_manifest bind \
  --manifest release-manifest.json \
  --image "${REGISTRY}/triton-eval@sha256:<digest>" \
  --output release-manifest.json
```

For a host deployment there is no image; bind the reviewed source build
instead: `bind --build "git:$(git rev-parse HEAD)"`.

Keep the bound manifest with the release. Compose mounts it read-only at
`/opt/triton-eval/release-manifest.json`; host deployments point
`TRITON_EVAL_RELEASE_MANIFEST` at it.

## Verified startup

`TRITON_EVAL_RELEASE_MODE` selects what a formal startup path requires:

- `release` verifies the installed service version, the source digest, the
  installed dependency versions and the bound build against the manifest. Any
  mismatch -- a tampered manifest, a swapped dependency, changed sources, an
  unbound or all-zero build -- refuses to start and names the failing
  conditions. Release mode never degrades into development mode.
- `dev` is the default. It starts without a manifest and reports its identity as
  unverified; test fixtures and in-process assembly use it and say so. A
  release-mode failure never falls back to it, so a deployment that must be
  verified sets `release` explicitly -- Compose always does, and preflight
  rejects a Compose deployment that does not.

Both deployment paths verify through the same module: the API at
`scripts/run_api.py` (the module uvicorn imports) and each worker at
`scripts/run_worker.py`. Compose deployments additionally run
`scripts/deployment_preflight.py`, which requires release mode, the read-only
manifest mount and a manifest bound to the exact image being deployed.

Check the released artifacts at any time:

```bash
python -m scripts.release_manifest verify --manifest /srv/triton-eval/release-manifest.json
python -m scripts.release_manifest consistency   # declared version sources agree
```

`consistency` is the CI gate: `pyproject.toml`, the installed distribution
metadata, a release tag on `HEAD`, the three plugin manifests and the CLI must
all agree before a release is published.

## Discovery

`GET /version` answers any authenticated caller with the running service
version, the implemented contracts, the build, the storage versions, the
implemented capabilities and the verification status. Identity is
process-local, so it answers while no worker can serve a request, and it
reports no deployment configuration.

The bundled client reports its own version offline with
`triton_eval.py --version` -- no service, token or service package needed --
and queries a deployment with `triton_eval.py version --url ... --capability-file ...`.

## Raising a version

| Change | Raise |
| --- | --- |
| Incompatible HTTP behaviour | service major, and a new API contract identity |
| Compatible service feature | service minor |
| Service fix | service patch; a fix that changes evaluation rules also needs the affected protocol identity |
| Client-only change | client version in all three plugin manifests and the CLI together |

Service and client releases stay independent: a client change never requires a
service release, and the manifest records both.

## Automated release PRs

The repository follows [OpenEvo's release-PR workflow](https://github.com/CARI-DAAL/open-evo/blob/main/.github/workflows/release-pr.yml):

```text
merge changes to main → maintain automation/release PR
                     → maintainer review + CPU CI
                     → merge release PR → version tag + GitHub Release
```

[`release-pr.yml`](../../.github/workflows/release-pr.yml) runs on pushes to
`main` and manual dispatch. It keeps one PR open, updating only `pyproject.toml`
and `CHANGELOG.md`. Each update temporarily makes the PR a draft, explicitly
dispatches the existing CPU CI matrix against its head, and checks that both
main and the prepared PR head are still current before making it ready.

Version suggestions use commit messages since the current `vX.Y.Z` tag for
changes to service sources, scripts, deployment assets, adapters, project
metadata or `.env.example`. Client-only changes under `plugins/` (including
their tests/docs) do not bump the service; mixed service/client changes do.
`feat` requests a minor release; `fix`, `perf` and `revert` request a patch.
`!` or a `BREAKING CHANGE:`/`BREAKING-CHANGE:` footer requests a major release,
except before 1.0, when it requests a minor release. Documentation, tests,
ordinary refactors and chores alone do not request a release. Existing client
manifest/CLI versions are checked and recorded, never automatically changed.

When there are no version tags, the first PR proposes the current declared
service version (initially `0.1.0`) and initializes its changelog entry. A merged
entry whose tag is still absent waits for publication instead of creating
another release PR. Backup tags under `backup/`, `archive/` and `recovery/`
never count as release baselines or trigger publication.

Merging the release PR is the maintainer's publication authorization.
[`release.yml`](../../.github/workflows/release.yml) accepts only a merged,
same-repository `automation/release` PR. It waits for successful dispatched CPU
CI on that exact reviewed head, then checks out the event's merge commit. The
merged tree must equal the tested tree, and the release change may touch only
the two release files. If main advanced and the merge contains additional code,
publication fails; refresh the release PR before merging to avoid this condition.

Publication creates an annotated `vX.Y.Z` tag and a GitHub Release with the
reviewed changelog notes and `release-manifest.json`, bound to `git:<merge SHA>`.
It does not push main, deploy a service, build an Ascend image, or publish to
PyPI. CPU CI does not establish NPU correctness or hardware counter availability;
maintainers must assess the required device evidence before approving a release.

The workflows use `GITHUB_TOKEN`; no PAT or registry secret is required. The
repository's Actions settings must allow GitHub Actions to create pull requests.
The preparation workflow requests `contents`, `pull-requests` and `actions`
write permissions; publication needs contents/actions write and checks read.
CI is explicitly dispatched because token-generated PR events are not a reliable
way to start unattended validation. See [GitHub's trigger rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

To refresh the pending release after merging these workflows:

```bash
gh workflow run release-pr.yml --ref main
```

Retry a failed publication using **Re-run failed jobs** on its original Actions
run. An existing tag must point to the same merge commit; an existing manifest
must have identical bytes. Retries can repair a missing Release or manifest
upload, but never move tags or overwrite an existing artifact. Publication
dispatches another preparation run to pick up changes merged while it ran.

The implementation is [`scripts/release_pr.py`](../../scripts/release_pr.py).
Its preparation can be inspected in a disposable clean checkout without
creating a tag, pushing or publishing:

```bash
python -m scripts.release_pr prepare \
  --body-file /tmp/release-body.md --output-file /tmp/release-outputs.txt
git diff -- pyproject.toml CHANGELOG.md
```
