# Supported commands

Run commands from the repository root in the configured Python environment.
Worker and preparation CLIs expose `--help`; preparation outputs belong outside
the checkout. The API entrypoint starts the configured listeners.

| Task | Commands | Guide |
| --- | --- | --- |
| Compose deployment | `compose.sh`, `deployment_preflight.py` | [Deployment](../docs/operations/deployment.md) |
| Manual API/worker startup | `python -m scripts.run_api`, `python -m scripts.run_worker --cpu`, `python -m scripts.run_worker --card 0` | [Deployment](../docs/operations/deployment.md) |
| Host workers and API | `start.sh`, `stop.sh`, `dl_wheels.sh` | [Deployment](../docs/operations/deployment.md) |
| Smoke and NPU gates | `test_eval_service.py`, `npu_release_gate.py` | [Tests](../tests/README.md) |
| Offline official-task import | `register_official_tasks.py` | [Task registration](../docs/operations/official-task-registration.md) |
| Profiler and sandbox preparation | `prepare_profiler_runtime.py`, `prepare_engine_runtime_overlay.py` | [Profiler runtime](../docs/operations/profiler-runtime.md) |
| Profiler diagnostics | `probe_profiler_latency.py`, `replay_profiler_export.py` | [Profiler runtime](../docs/operations/profiler-runtime.md) |
| Reference-cache index | `reference_cache_index.py` | [Reference cache](../docs/operations/reference-cache.md) |

The optional adapter in [`adapters/official_task.py`](adapters/official_task.py)
converts the official dataset format for `register_official_tasks.py`. Its B1
selection rules and provenance checks apply to that offline import; the serving
catalog accepts ordinary reference files without the adapter or sidecars.

Retired compilation experiments are recorded in [Git history](../docs/history.md).
