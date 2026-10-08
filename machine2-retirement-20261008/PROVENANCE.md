# machine2 retirement backup — generic service result evidence (triton-op-evaluation)

Task id: m2-retire-results-20261008. Branch: `backup/machine2-retirement-20261008/results-service-evaluation` in CARI-DAAL/triton-op-evaluation (private).
Scope: NB3/NB7 real 51/51 service evidence + deployment provenance. No source code changes, no service restart, no computation. All files listed in `SHA256SUMS`.

## Contents and exact sources

### nb3-nb7-cache-integrate-20261006-pkg/ — deployed package + admission/canary evidence (real 51/51)
Source: `/home/e84381970/open-evo-work/runtime-artifacts/nb3-nb7-cache-integrate-20261006/pkg/` (stable dir; checkout deployed via branch nb3-nb7-cache-integrate = e74b1f5c56e03eaf4dc1fe25fc918afe1005ae69, live BUILD sha256:089a0261, manifest 2e2daf23, overlay 2f5808d1, index v2prime d1d59a1a6a4c6bbc773dec39c41556fb13cb990af2d0f9564a3d1c1d09d275d1, policy v7 42fe8d0fc6ead4ae4879fc204f417530f6ac80835e9339ddd6d1219bce2a97bf, approval 49d1d147ccc9c01e8499f5198d408cedd6ba14c607158242e186a5afcf2a5590).
- `model-bound-cache-index-v2prime.json` — 1046-entry cache binding index (sealed source deployment 84154dfb…, target deployment 5dce1c3f…). Binds the NB0 original-1046 latency values backed up on the open-evo branch.
- `validated-timing-reuse-policy-v7.json` — timing reuse policy (8-sample median, cross-same-model-card reuse, 3% sentinel, npu-strict).
- `deploy-sandbox-manifest.json`, `deploy-runtime-overlay.json`, `deploy-package-receipt-final.json`, `build-nb3nb7-receipt.json`, `build_*.py`, `verify_prep_artifacts.py` — package provenance (byte-pinned).
- `evidence/records/gates/` — admission gates: canary ev_0ee1d9f2, compile ev_6e7906fe, l3 ev_c57f115d (50 case), bmm ev_068471d5 (51/51 correctness+perf).
- `evidence/records/fleet-089a0261/` — 16/16 cards; `fleet-failed-run1-unmounted-python/` retained failure fact; `runtime-conditions.json`, `launch-binding.json`.
- `evidence/canaryA-stdout.txt` (+stderr) — official request #1 via real TaskAgent container → native /verify: EID ev_63688c7d600d4c55a12d40b8f18ceeac, attempt at_ec4f47a8, card 14, 183.7s, correct 51/51, reference cache hit 51/51, live input hash == sealed 51/51.
- `evidence/canaryB-stdout.txt` (+stderr) — official request #2 via same container → native /benchmark: EID ev_8de4c2f5a2854c1abe10fe0a81c86aff, attempt at_74dc90c9, card 9, 484.7s, correctness 51/51 (cache 51/51), performance succeeded measured 51/51 coverage 1, timing source cached_manifest 51/51 (8 samples/case), sentinel case1 0.98305ms vs cached 0.9865ms → drift +0.3497% < ±3%, deployment hit 5dce1c3f, overall_speedup 0.509 (informational).
- `evidence/cpuproof-preserved/` — interrupted old-BMM CPUproof receipt (NOT a CPU-equivalence proof; recorded as interrupted/unfinished).
- `evidence/workers.json`, `gate-process.log`, `fleet-process.log`.

### Reports (provenance documents)
- `nb3-nb7-cache-integrate-20261006-REPORT.md` — deployment/admission/canary chain with all identifiers.
- `nb7-cache-validity-review-20261006-REPORT.md` — NB7 cache validity review.
- `nb5-uni-grade-failure-fix-20261008-REPORT.md` — uni candidate crashes (kernel 5a3bb8be deterministic AI Core EZ9999; fix c49f53a deployed on branch nb5-uni-grade-failure-fix-20261008, workers run c49f53a while reported build remains 089a0261; verification EID ev_5b5a1160ddb8419c9ea1cfd6afb44354).

## Exclusions
Tensor blobs/cache binaries (42G, `/dev/shm/…/cache/blobs`), installers/packages (backups/cann-9.1.1-installers), full quarantine trees (835M pi workspace), repetitive session/debug logs, raw secret-bearing ATIF/session histories, Redis/database dumps. No credentials/tokens/private keys present (content scanned; only dummy unit-test fixtures matched).

## Unknowns kept unknown
- NB8/NB9 not run; evaluator_build for any future round2 build (e.g. 71fb6c42) never pinned live — workers at backup time run c49f53a while heartbeat reports 089a0261 (recorded honestly in the uni report).

## Restore recipe
`git clone -b backup/machine2-retirement-20261008/results-service-evaluation --single-branch https://github.com/CARI-DAAL/triton-op-evaluation.git && cd triton-op-evaluation/machine2-retirement-20261008 && sha256sum -c SHA256SUMS`
