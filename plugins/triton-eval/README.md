# Triton evaluation plugin

This repository contains a client plugin for an independently deployed
triton-evaluation HTTP service. `plugins/triton-eval/` is its location within this
repository; `triton-eval/` is the plugin root. Metadata and skill content have
different roles. Keep this package separate from the service source and Docker
deployment files; distribute the contents of `plugins/triton-eval/` as the plugin:

```text
plugins/triton-eval/
├── plugin.json                  # Portable identity and metadata (canonical)
├── .codex-plugin/plugin.json    # Codex compatibility manifest
├── .claude-plugin/plugin.json   # Claude Code manifest
└── skills/triton-eval/
    ├── SKILL.md
    └── scripts/triton_eval.py
```

The root manifest follows [OpenAI's portable plugin format](https://developers.openai.com/plugins/build/plugins).
The hidden manifests provide the documented client compatibility locations;
their `skills: "./skills/"` paths resolve from the plugin root. Keep all three
manifests' shared metadata consistent. Skills and their scripts stay under the
root `skills/` directory, as specified by the
[Claude Code manifest reference](https://code.claude.com/docs/en/plugins-reference).

The structure also follows actual open-source examples:
[OpenAI's Boltz CLI plugin](https://github.com/openai/plugins/tree/main/plugins/boltz-api-cli)
uses `.codex-plugin/plugin.json` beside its skills, and
[Anthropic's example plugin](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/example-plugin)
uses `.claude-plugin/plugin.json` beside its components. No generic `.plugin/`
directory or additional wrapper is needed.

The [skill](skills/triton-eval/SKILL.md) bundles a Python 3.10+ standard-library
[CLI](skills/triton-eval/scripts/triton_eval.py). No service package, Torch or
Ascend installation is required on the client machine.

For pipe-utilization feedback use `benchmark --method profiler`. The service
returns per-case `kernel_metrics` from the same timing capture; the CLI preserves
those fields in its JSON output. Event timing remains the default. See
[the response semantics](../../docs/usage/evaluations.md#profiler-benchmark-feedback).

This example connects to a local service listening on `127.0.0.1:11907`.
For remote or containerized clients, set the URL to the address reachable from
the client. Compose uses the service's configured private host address; the
CLI itself has no default URL.

Configure the service URL and caller capability-file path:

```bash
export EVALUATION_URL=http://127.0.0.1:11907
export EVALUATION_CAPABILITY_FILE=/path/to/capability
python3 plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py verify \
  --definition /path/to/definition.json --workloads /path/to/workloads.jsonl \
  --triton /path/to/triton_impl.py --save-handle evaluation-handle.json
```

Version 0.3.0 accepts the official definition and workload files directly.
Workloads can be JSONL or a JSON array. The definition contains `axes`, ordered
named `inputs`, `outputs`, optional `name`, and optional Python `reference`.
A workload has `uuid`, `axes`, and named input specifications, for example:

```json
{"uuid":"add-1024","axes":null,"inputs":{"x":{"type":"random","shape":[1024],"dtype":"float32"},"y":{"type":"random","shape":[1024],"dtype":"float32"}}}
```

The request seed defaults to 200 (`--seed` overrides it). A candidate is supplied
with `--triton`; custom input generator source uses `--custom-inputs`.
`verify` requires both the definition reference and candidate. `benchmark` and
`profile` also support a single implementation. For Torch alone, omit `--triton`.
For Triton alone, pass `--triton-only` alongside `--triton` to remove the reference
from the submitted definition. Files are read as data and never executed locally.

For custom workloads, declare `custom_inputs_entrypoint` in the definition.
With `--triton-only`, the client preserves the original reference source as
`custom_inputs_code` for that factory; the submitted definition omits its
mathematical reference. An explicit `--custom-inputs` file takes precedence.

The client prints one JSON resource to stdout and progress to stderr. It retains
HTTP admission, polling and recovery behavior: `--save-handle` records the URL,
evaluation ID, idempotency key and session ID; use `get EVALUATION_ID` against the
original URL to recover. Retry an ambiguous submission with unchanged files,
options, session and idempotency key. Client timeout does not cancel accepted work.

Comparison benchmarks cache Torch baseline timing by default. Pass
`--no-baseline-cache` to measure a fresh reference baseline. Single implementation
benchmarks measure fresh. Cache switches apply only to `benchmark`.

A successful single implementation measurement reports `correctness_verified:
false`; preserve that limitation when interpreting its timing or profile result.
See the [skill](skills/triton-eval/SKILL.md) for recovery and result interpretation.
