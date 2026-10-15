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
