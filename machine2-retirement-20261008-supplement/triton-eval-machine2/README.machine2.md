# Machine2 eval service migration

The stopped local-model baseline remains stopped. No LLM requests are needed by
this service, and deploying it does not launch a baseline.

## Operational note: orphaned workers after an API crash

If the API dies (e.g. Redis outage) the per-card workers survive it. The stale
`logs/triton-eval.pids` then lists dead API pids but LIVE worker pids; deleting
the file and restarting orphans the old worker generation — it keeps consuming
the same Redis queue and holding device memory, and `npu-smi` shows two
`TritonEval` processes per chip. Recovery (2026-09-15): kill every `TritonEval`
process and `scripts/start.sh` wrapper whose pid is not in the current pid
file, then verify `npu-smi` shows exactly one `TritonEval` per chip (16 total).
`manage-service.sh stop` before any manual pid-file surgery.

## Open Evo integration (2026-09-15)

The service now also serves the Open Evo / Harbor platform on machine2:

- **Grade admission** is enabled: `capabilities.json` carries the
  `grader-machine2` trusted caller (`grade:submit`, `grade:read`). Its token is
  only in that file (mode 0600) and in the Open Evo launch environment; never
  print or commit it.
- **Release catalog**: `catalog/catalog.json` binds the 20 NPUKernelBench
  baseline tasks (manifest `npukernelbench-machine2-v1`, lv3×10 + lv4×10) to
  their reference digests; artifacts live under `catalog/references/`. The
  catalog, the Harbor benchmark tasks and the bridge task catalog are generated
  from one source by `open-evo/scripts/evo/import-npukernelbench.py`; regenerate
  all three together, never edit one in isolation.
- **Local fix 2** (beyond the canary `linecache` fix below):
  `triton_evaluation/execution_environment.py` now resolves the Triton version
  from the `triton_ascend` distribution as a fallback when the canonical
  `triton` distribution is absent (this deployment runs triton_ascend 3.2.0).
  Without it `triton_version` attests as null and every grade lands with
  `grade_eligible=false`. Not pushed upstream.
- The platform verifier reaches the service through the Open Evo bridge's TLS
  listener (`https://127.0.0.1:11909` on this host), which forwards bytes
  unchanged to `http://192.169.0.227:11907`; the verifier pins
  `127.0.0.0/8` as the allowed origin CIDR and trusts the bridge certificate
  via `SSL_CERT_FILE` (bundle at `open-evo-bridge/ca-bundle.pem`), not the
  system trust store.
- Grade traffic uses the trusted admission bucket (defaults: 60/min, 4
  in-flight). Raise `TRITON_EVAL_TRUSTED_*` in `service.env` before running
  wide concurrent experiments.

## Deployment

- Service repository: `/home/e84381970/triton-op-evaluation`
- Upstream main: `bcdd3d9524c4a175b8ee81e630f0768bc8f98b81` (API v2.1).
- Local branch `machine2/eval-service`, commit
  `661bde58fb908f42090a3835ab5cd678ce737fc7`: register the canary source in
  `linecache` so Triton JIT can retrieve it with `inspect.getsource`.
  Without this compatibility fix, all cards drain during startup. Health checks
  remain enabled. This fix has not been pushed upstream.
- Endpoint: `http://192.169.0.227:11907`; readiness: `GET /readyz`.
- API, one CPU worker, and one serial NPU worker per physical device 0–7 run in
  the existing `cc` runtime, using the private `venv` under this directory.
- Redis is the dedicated `triton-eval-redis-machine2` container on
  `127.0.0.1:16379`, with AOF and named volume `triton-eval-machine2-redis`.
  Preserve this volume for queued jobs and idempotency recovery.
- Agent sandbox: `triton-agent-cpu`, explicit Docker runtime **runc**,
  `ASCEND_VISIBLE_DEVICES=void`, no `/dev/davinci*`, no host `/home` mount.
  Explicit runc matters: machine2's default Docker runtime is Ascend and would
  inject all devices even without an explicit `--device` option.
- The sandbox receives only the trial client capability via a read-only mount
  at `/run/triton-eval-client.json`. Do not print or commit capabilities.

Each active NPU evaluation has its own child process. `npu-smi` may show both a
resident `TritonEval` worker and its `TritonEvalWork` child on one card; this does
not mean two evaluations are running there. The resident worker holds 64 MB
plus runtime context while idle. On this machine, npu-smi's board/chip pair maps
to physical device ID as `board * 2 + chip`.

```bash
bash /home/e84381970/triton-eval-machine2/manage-service.sh status
# Explicit operator actions when needed:
bash /home/e84381970/triton-eval-machine2/manage-service.sh stop
bash /home/e84381970/triton-eval-machine2/manage-service.sh start
```

Processes currently run inside `cc`; service startup is not registered as a host
boot service. Redis and the CPU sandbox have Docker restart policies. Logs are
`launcher.log` and `logs/{api,worker_cpu_0,worker_0,...,worker_7}.log` here.

## Baseline integration

Modified legacy repository:
`/home/l00515014/op_gen/self-evo/Uniagent-triton-ascend-evo`.

`run_verify` retains the CPU AST and frozen-reference integrity checks, packages
reference source plus unchanged JSON/JSONL sidecars, and submits correctness and
performance together to `/evaluations`. Only the evaluator executes the source
on NPU. It polls the durable evaluation ID, writes legacy verify/perf artifacts,
and retains the raw v2.1 resource with evaluator provenance. The existing best
snapshot and reward paths consume those artifacts. A service failure cannot
fall back to sandbox NPU execution or reuse a stale latest performance result.

`eval/run_compact.sh` now defaults to this service and the CPU sandbox.
`eval/run_baseline_eval_service.sh` preserves the previous baseline preset:
20 tasks, 8 samples, 12 agent workers, 150 turns, 128K context, compaction at 0.5,
and the previous reward weights. Correctness and performance are scheduled by
the service across 8 devices. Benchmark method remains profiler, warmup 5,
repeats 20, with geometric-mean speedup. New results record
`profiler_kernel_sum_v1` and the evaluator build; use a new output run rather than
mixing results across the legacy evaluator and this build. The v2.1 performance
response does not expose the old private kernel-metrics/bottleneck-hint fields.

After the replacement model is ready, launch explicitly from machine2:

```bash
cd /home/l00515014/op_gen/self-evo/Uniagent-triton-ascend-evo
export LLM_API_BASE_URL='http://<replacement-model-host>:<port>/v1'
export LLM_MODEL_NAME='<replacement-served-model-name>'
# Set LLM_API_KEY if required; it remains in the generator container.
bash eval/run_baseline_eval_service.sh
```

Both model URL and served name are required to avoid accidentally contacting the
old Flash model. The launcher allocates ports 22001–22012 and separate workspaces
and tool directories. The previous baseline outputs are preserved.

## Verification and recovery

- Adapter tests: 8 passed in the actual generator container.
- Canary, card health, worker tests: 20 passed in the service venv.
- `smoke-report.json`: actual CPU AgentEnv → run_verify → HTTP service → two
  JSONL-defined Add cases → performance, best snapshot, and reward.
- `queue-smoke-report.json`: 12 real concurrent requests, atomic read-only broker
  observations of state and card bindings, and full final service responses.
  All 12 completed with correctness and performance passing. Observed 8 running
  and 4 queued, all cards 0–7 used, maximum one running evaluation per card.
- `baseline-smoke-report.json`: archived BatchMatmul kernel evaluated through the
  CPU AgentEnv, without any model calls: 51/51 correctness cases passed and 51/51
  performance cases measured. Speedup 0.856153x; best snapshot and reward saved.
  This validates evaluator compatibility; it is not a new model baseline run.
- `queue-npu-smi.txt`: physical device process snapshot during the concurrency test.

The adapter stores request/idempotency receipts under each agent workspace's
`output/eval_service`. Retrying an interrupted call resumes the accepted ID, including collecting an
already completed result after the original client waiting deadline.
Only a new completed invocation or changed request creates a new evaluation.

Original legacy file backups and the deployment manifest are in the directory
named by `legacy_backup_path.txt`. Restore only files in that manifest if rolling
back this integration; there were unrelated existing local edits. The main local
development checkout and its uni-agent submodule were not modified by migration.
