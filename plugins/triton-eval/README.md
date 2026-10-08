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

Run this example from the plugin root. `verify` requires both implementations.
For a single implementation, measure or profile with only its source flag:

```bash
python3 skills/triton-eval/scripts/triton_eval.py benchmark \
  --torch /path/to/torch_impl.py --cases /path/to/cases.json
python3 skills/triton-eval/scripts/triton_eval.py benchmark \
  --triton /path/to/triton_impl.py --cases /path/to/cases.json
python3 skills/triton-eval/scripts/triton_eval.py profile \
  --torch /path/to/torch_impl.py --cases /path/to/cases.json
```

`cases.json` is a JSON array of cases. Each provided source exports `run(*args)`.
Supplying both sources to `benchmark` retains correctness and speedup comparison.
One source returns measurement data with `implementation`,
`correctness_verified: false` and per-case latency for performance; it supplies
no reference speedup. The client prints `correctness not verified` to stderr and
preserves the service resource on stdout. Exit zero means the requested
measurement succeeded, not that a single implementation's correctness was proved.

`--save-handle` writes the original URL, evaluation ID, idempotency key and
session ID without the credential. Known IDs recover through `get` on that URL;
ambiguous submissions retry the unchanged source/cases with the same key/session.

The three operation commands are
`verify`, `benchmark` and `profile`; `get EVALUATION_ID` recovers existing work.
Each command's `--help` describes its options. This is a repository package;
it does not install a global command or deploy the HTTP service.

The repository does not register a marketplace or install this plugin. Codex's
`plugin add` command expects a configured marketplace selector, not a direct
directory path. Run the bundled script directly as above when using this checkout.

With both sources, `benchmark` reuses cached Torch baseline timing by default
while measuring the Triton candidate each time. Use `--no-baseline-cache` to
measure the comparison baseline fresh without reading or writing the cache, or
`--use-baseline-cache` to select the default explicitly. Comparison timing results
report `baseline_cache_hit` per case. Single-source benchmarks always measure
fresh, regardless of this flag, and neither read nor write the baseline cache.
