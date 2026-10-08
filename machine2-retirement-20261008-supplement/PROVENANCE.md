# Machine2 retirement supplement (task m2-retire-open-evo-service-gaps-20261008)

Additive supplement to the already-verified machine2-retirement backups. Preserves ONLY unique
Open Evo / eval-service changed code and produced results that were missing from:
- CARI-DAAL/open-evo `backup/machine2-retirement-20261008/results-experiments` @ 16913b64f1683aa58d08af5fa5d2d02c5b53a6a8 (3417 files)
- CARI-DAAL/triton-op-evaluation `backup/machine2-retirement-20261008/results-service-evaluation` @ db29c3e5db13c60a13bda260c3d46e4def931d2b (151 files)
- CARI-DAAL/triton-op-evaluation `backup/machine2-retirement-20261008/service-*` @ c49f53ab / ddb54326 / 58cdd65e / 99d79c13 (all service repo code + 11 outside-Git payloads)

## Payload tree -> absolute source mapping (machine2)

| payload path | source |
|---|---|
| open-evo-bridge/* | /home/e84381970/open-evo-bridge/ (no-repo bridge daemon; live until 2026-10-01) |
| triton-eval-machine2/* | /home/e84381970/triton-eval-machine2/ (first machine2 eval-service deployment, 2026-09-14..22) |
| worktree-detach-20261006/b-nb7-optional-cache-fallback/sealed-cache-index-reconciled-v3.json | /home/e84381970/open-evo-work/runtime-artifacts/worktree-detach-20261006/b-nb7-optional-cache-fallback/ |
| redis-and-ramdisk-state-summary.json | read-only classification evidence (counts only; no raw state) |

## Classification (evidence per item)

UNIQUE IN-SCOPE, preserved here:
- open-evo-bridge/bridge.py (34038B, 2026-09-27) — live Open Evo bridge daemon ("interaction adapter + TLS
  front for the Triton eval service", docstring). Remote `feat/kernel-eval-legacy-bridge` @ fda60ce has only
  an older 18801B version (adapters/kernel-eval-legacy-bridge/bridge.py); live file is the deployed evolution.
- open-evo-bridge/legacy_contract.py (20910B) — differs from the 20700B remote branch copy (blob-checked).
- open-evo-bridge/{legacy_metrics.py, local_eval.py, native_exec.py, claim_and_run.sh, tasks.json,
  tasks-native.json(+.bak-cutover-20260927), bridge.py.bak-mw-prediagnostics} — absent from ALL 1213 blob
  objects of every remote triton-op-evaluation ref, open-evo objects and kernel-agent repo objects.
- triton-eval-machine2/{README.machine2.md, deployment.json, current-baseline.json, deployed-checksums.json,
  baseline-smoke-* , queue-smoke-*, smoke*, qwen-* preflights, install/redis-pull/launcher logs, logs/,
  manage-service.sh, smoke_agent.py, smoke_baseline.py, queue_smoke.py, legacy_backup_path.txt,
  catalog/catalog.json} — deployment state + smoke/baseline run outcomes; sha256/blob-absent from all
  preserved inventories (results staging 9070 files, service staging 399 files, toe remote blobs).
- worktree-detach-20261006/.../sealed-cache-index-reconciled-v3.json (496471B, sha256
  1a3143b853314b0cf0356f72d80115a37406208029dd659acff5693b6050c7ac) — reconciled sealed cache index:
  algorithm-file digests + per-device conditions (metadata only). Explicitly excluded by service-code owner
  (left to results lane) and not present in results-service-evaluation (which has v2prime index only).
  Cache-index gap retained per criteria.

ALREADY COVERED (dropped from staging; verify via refs above):
- open-evo-bridge/test_legacy_alignment.py — byte-identical to adapters/kernel-eval-legacy-bridge/ on
  feat/kernel-eval-legacy-bridge @ fda60ce (triton-op-evaluation).
- open-evo-bridge/reference/kernel-agent-52affcb9/{evaluation_observation.py, error_classify.py,
  PROVENANCE.md, LICENSE} — byte-identical copies exist in the same remote branch. (Upstream kernel-agent
  commit 52affcb9 is NOT on CARI-DAAL/triton-op-agent-evo; the remote branch copy is the preserved instance.)
- triton-eval-machine2/logs/worker_cpu_0.log — empty file, dropped.

OUT OF SCOPE / other owners (not preserved here):
- /mnt/bridge-ram: cannbot/KernelGYM-lane local-toolchain workspaces (242 dirs) and tools; no Open Evo/
  service result-or-deployed-code relationship established (see redis-and-ramdisk-state-summary.json);
  human scope: "kernel gym和我们无关 只有open evo 和 eval servcie". claim_and_run.sh there is byte-identical
  to the preserved open-evo-bridge copy.
- Redis AOF ~271MB (docker volume triton-eval-machine2-redis): queue/session infrastructure only
  (26156 inflight session markers + counters, no terminal result payloads); raw dump excluded per boundary.
- legacy-before-20260914T033756/ and legacy-adapter-before-recovery-fix/ (uni_agent framework snapshots) and
  recovery-fix.tgz (eval_service_client.py patch): framework/client patches belong to active
  m2-retire-evo-code-20261008; not uploaded here.
- catalog/references/*.py (20 NPUKernelBench reference kernel sources, 1012K): regenerable via
  open-evo/scripts/evo/import-npukernelbench.py; digest bindings preserved in catalog/catalog.json.
- smoke-host/, baseline-smoke-host/ (agent scratch workdirs): tooling scratch, outcomes captured in the
  preserved reports. venv/, pycaches, pid files: reproducible/stateless.
- Secrets withheld, originals untouched, no derivatives needed: bridge.env{,.bak-*} (0600),
  server-key.pem/server.pem/ca-bundle.pem, triton-eval-machine2/{capabilities.json, client.json,
  service.env, service.env.bak-full16, smoke-auth} (contain live tokens/keys). bridge.log + bridge-recovery*
  logs (11MB session/ops history) excluded as session history.

## Redaction
None. No secret-bearing file was copied, so no redacted derivative was created (original/derivative hashes N/A).

## Verification
SHA256SUMS covers every payload file. Read-back: fresh single-branch clone from GitHub, `sha256sum -c`.

## Restore recipe
git clone -b backup/machine2-retirement-20261008/supplement-service-gaps-20261008 --single-branch \
  https://github.com/CARI-DAAL/triton-op-evaluation.git && \
  cd triton-op-evaluation/machine2-retirement-20261008-supplement && sha256sum -c SHA256SUMS
