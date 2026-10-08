---
name: triton-eval
description: Measure or profile PyTorch or Triton operators, compare Triton against PyTorch for correctness and speedup, or recover an evaluation from a deployed triton-evaluation service.
---

# Triton evaluation

Use the bundled [client](scripts/triton_eval.py) with Python 3.10+; it needs only
the standard library. The HTTP service is deployed separately. Obtain its URL
and the caller capability-file path from the user's existing configuration:
`EVALUATION_URL` and `EVALUATION_CAPABILITY_FILE`, or the equivalent `--url` and
`--capability-file` flags. Resolve the script relative to this `SKILL.md`, then
invoke its absolute path from the directory containing the user's inputs.

## Submit an operator

Use a JSON **array** of cases and at least one UTF-8 source file (PyTorch or
Triton). Each supplied implementation defines `run(*args)`. Each case has positional
`inputs` and an optional `seed` (default `0`). An input is a scalar or a tensor
descriptor with `shape`, optional `dtype` (default `float32`) and optional
`range: [low, high]` (upper bound exclusive). For example, `cases.json` can be:

```json
[{"inputs": [{"shape": [1024], "dtype": "float32"}], "seed": 0}]
```

Choose `verify` with both sources for correctness. `benchmark` measures one
implementation, or checks correctness and speedup when both are supplied.
`profile` accepts either one or both sources. Supply only the implementations
the user wants measured; duplicating one source into two fields creates a
misleading comparison. Put options after the subcommand. With
`TRITON_EVAL_CLI` set to the resolved client path:

```bash
python3 "$TRITON_EVAL_CLI" verify \
  --torch torch_impl.py --triton triton_impl.py --cases cases.json \
  --save-handle evaluation-handle.json
```

For one implementation, omit the other flag:

```bash
python3 "$TRITON_EVAL_CLI" benchmark --torch torch_impl.py --cases cases.json
python3 "$TRITON_EVAL_CLI" profile --triton triton_impl.py --cases cases.json
```

The client reads the capability from its file and sends it only as the
Authorization credential. Keep credentials out of source files, cases, output
artifacts and pasted logs. The supplied source files and cases are sent to the
configured evaluation service; importing them locally is unnecessary.

Use subcommand `--help` for the available options. The client normally
polls for up to 960 seconds with a 1-second interval; `--timeout`,
`--poll-interval` and `--no-wait` control that client behavior. An elapsed client
wait does not cancel accepted work.

## Recover and interpret results

Once an `evaluation_id` is known, use `get` on its original service URL:

```bash
python3 "$TRITON_EVAL_CLI" get EVALUATION_ID --url ORIGINAL_SERVICE_URL
```

`--save-handle` records `url`, `evaluation_id`, `idempotency_key` and `session_id`.
Retain these values for recovery; a missing session is recorded as null. If a POST gets
no response, retry only with the same source bytes, cases, options, session and
`--idempotency-key`. A different key creates new work. After bounded retries or
a service outage, report the recoverable handle or original request identity;
resume recovery when the service is available.

Stdout is one JSON result; stderr contains progress and request identity. Read
the returned state and per-check evidence. A pending resource remains pending,
and `completed` alone does not mean the candidate passed. Report a failed
correctness verdict as a candidate outcome; report transport or infrastructure
failure separately, without inventing a correctness score. Preserve available
measurement method and profile evidence when summarizing results.

Single-implementation results identify `implementation: torch` or `triton` and
`correctness_verified: false`. Performance reports per-case `latency_ms` without
reference speedup; profiling reports its own evidence. A zero CLI exit status
for those operations means measurement success, not verified correctness. Keep
that limitation in the user-facing result even when timing or profiling succeeds.
Syntax/import prechecks also do not establish successful device JIT compilation.

Preserve partial-profile warnings. PyTorch profiling can include process
initialization and has bounded launch coverage; its observations do not establish
complete profiling of only the supplied `run` function.
