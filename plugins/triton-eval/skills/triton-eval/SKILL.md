---
name: triton-eval
description: Verify, benchmark or profile Triton operators against a PyTorch reference using a deployed triton-evaluation service, or recover an existing evaluation by ID.
---

# Triton evaluation

Use the bundled [client](scripts/triton_eval.py) with Python 3.10+; it needs only
the standard library. The HTTP service is deployed separately. Obtain its URL
and the caller capability-file path from the user's existing configuration:
`EVALUATION_URL` and `EVALUATION_CAPABILITY_FILE`, or the equivalent `--url` and
`--capability-file` flags. Resolve the script relative to this `SKILL.md`, then
invoke its absolute path from the directory containing the user's inputs.

## Submit an operator

Use three UTF-8 files: the PyTorch source, the Triton source and a JSON **array**
of cases. Both implementations define `run(*args)`. Each case has positional
`inputs` and an optional `seed` (default `0`). An input is a scalar or a tensor
descriptor with `shape`, optional `dtype` (default `float32`) and optional
`range: [low, high]` (upper bound exclusive). For example, `cases.json` can be:

```json
[{"inputs": [{"shape": [1024], "dtype": "float32"}], "seed": 0}]
```

Choose `verify` for correctness, `benchmark` for correctness plus timing, or
`profile` for profiling. Put options after the subcommand. With
`TRITON_EVAL_CLI` set to the resolved client path:

```bash
python3 "$TRITON_EVAL_CLI" verify \
  --torch torch_impl.py --triton triton_impl.py --cases cases.json \
  --save-handle evaluation-handle.json
```

The client reads the capability from its file and sends it only as the
Authorization credential. Keep credentials out of source files, cases, output
artifacts and pasted logs. The two source files and cases are sent to the
configured evaluation service; importing them locally is unnecessary.

For `benchmark`, Torch baseline timing is cached by default. Use
`--no-baseline-cache` when the user requests fresh Torch timing; this bypasses
cache reads and writes. The candidate is measured each time. Preserve per-case
`baseline_cache_hit` when reporting whether baseline timing was reused.

Use subcommand `--help` for the available options. The client normally
polls for up to 960 seconds with a 1-second interval; `--timeout`,
`--poll-interval` and `--no-wait` control that client behavior. An elapsed client
wait does not cancel accepted work.

## Recover and interpret results

Once an `evaluation_id` is known, use `get` on its original service URL:

```bash
python3 "$TRITON_EVAL_CLI" get EVALUATION_ID --url ORIGINAL_SERVICE_URL
```

Retain the ID, original URL, session and printed idempotency key. If a POST gets
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
