# machine2-retirement-20261008 service payload provenance

Captured 2026-10-08 by task m2-retire-service-code-20261008. Read-only byte copies;
source payloads under /home/e84381970/open-evo-work/runtime-artifacts/worktree-detach-20261006/
were NOT modified. Files included because their exact bytes exist in no commit of
CARI-DAAL/triton-op-evaluation (verified by blob-hash scan over all 42 live origin refs).

| file (relative under this dir) | sha256 | source role (per worktree-detach-20261006/MAPPING.md) |
|---|---|---|
| a-v3-cache-cutover-prep/staged-triton/backends/ascend/driver.py | 3942188b279b331b82abece2a59b4837cf8fb5d5ab894840876533c7478e297b | LIVE: manifest readonly mount -> venv triton pkg; policy v5 fingerprint. Patched triton-ascend backend (third-party code patch that materially implements the service engine). |
| a-v3-cache-cutover-prep/staged-triton/backends/ascend/npu_utils.cpp | 6bd908f6e53c8ae243a44def9af8a3706034fcbab7cc627658b6742a3f609140 | LIVE: manifest readonly mount; policy v5 fingerprint. |
| b-nb7-optional-cache-fallback/validated-timing-reuse-policy-v5.json | 1b128616f87c1dd257a32559dbd1158667508c9096d9cf4c9b23cc896683488e | LIVE: overlay TRITON_EVAL_REFERENCE_TIMING_REUSE_POLICY (sha-pinned). |
| b-nb7-optional-cache-fallback/validated-timing-reuse-policy-v5-rebound-20261007.json | 428ea024c10afbe8b40ed042bb23ffd2f9ff81988086257aebf0ffc61f050f97 | policy v5 rebound update 2026-10-07 (present in source payload dir and its SHA256SUMS-payloads). |
| c-v3-service-reference8/timing-reuse-delivery/validated-timing-reuse-policy-v2.json | 3d9198dfcaa68f81442f09a43c8aac88dd77de62c601eb561b7731b44dfe9d02 | evidence-only policy v2 (not in live chain). |

All sha256 values match worktree-detach-20261006/SHA256SUMS-payloads for the listed files.

Exclusions (deliberate):
- sealed-cache-index-reconciled-v3.json (1a3143b8...) — evidence/result data, not service code (owned elsewhere).
- credential/template files, e.g. /home/e84381970/open-evo-work/svc-validation-p0/repo/service-p0.env — withheld.
