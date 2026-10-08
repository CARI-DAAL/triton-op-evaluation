# Supported commands

Run commands from the repository root in the configured Python environment.
Python tools expose `--help`; preparation outputs belong outside the checkout.

| Task | Commands | Guide |
| --- | --- | --- |
| Compose deployment | `compose.sh`, `deployment_preflight.py` | [Deployment](../docs/operations/deployment.md) |
| Host workers and API | `start.sh`, `stop.sh`, `dl_wheels.sh` | [Deployment](../docs/operations/deployment.md) |
| Smoke and NPU gates | `test_eval_service.py`, `npu_release_gate.py` | [Tests](../tests/README.md) |
| Official task catalog | `register_official_tasks.py` | [Task registration](../docs/operations/official-task-registration.md) |
| Profiler and sandbox preparation | `prepare_profiler_runtime.py`, `prepare_engine_runtime_overlay.py` | [Profiler runtime](../docs/operations/profiler-runtime.md) |
| Profiler diagnostics | `probe_profiler_latency.py`, `replay_profiler_export.py` | [Profiler runtime](../docs/operations/profiler-runtime.md) |
| Reference-cache index | `reference_cache_index.py` | [Reference cache](../docs/operations/reference-cache.md) |

Retired compilation experiments are recorded in [Git history](../docs/history.md).
