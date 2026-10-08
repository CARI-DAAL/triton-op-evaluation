# Triton evaluation plugin

This repository contains a client plugin for an independently deployed
triton-evaluation HTTP service. It follows the
[standard plugin layout](https://developers.openai.com/plugins/build/plugins):
`plugin.json` at the plugin root and a discoverable skill under `skills/`.

The [skill](skills/triton-eval/SKILL.md) bundles a Python 3.10+ standard-library
[CLI](skills/triton-eval/scripts/triton_eval.py). No service package, Torch or
Ascend installation is required on the client machine.

This example connects to a local service listening on `127.0.0.1:11907`.
For remote or containerized clients, set the URL to the address reachable from
the client. Compose uses the service's configured private host address; the
CLI itself has no default URL.

Configure the service URL and a caller capability-file path, then run directly:

```bash
export EVALUATION_URL=http://127.0.0.1:11907
export EVALUATION_CAPABILITY_FILE=/path/to/caller-token
python3 skills/triton-eval/scripts/triton_eval.py verify \
  --torch /path/to/torch_impl.py --triton /path/to/triton_impl.py \
  --cases /path/to/cases.json
```

Run this example from the plugin root. The three operation commands are
`verify`, `benchmark` and `profile`; `get EVALUATION_ID` recovers existing work.
Each command's `--help` describes its options. This is a repository package;
it does not install a global command or deploy the HTTP service.

`benchmark` reuses cached Torch baseline timing by default while measuring the
Triton candidate each time. Use `--no-baseline-cache` to measure Torch timing
fresh without reading or writing the cache, or `--use-baseline-cache` to select
the default explicitly. Timing results report `baseline_cache_hit` per case.
