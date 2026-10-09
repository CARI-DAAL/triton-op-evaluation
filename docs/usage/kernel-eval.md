# Verify, benchmark and profile an operator

The shortcuts use the same definition/workloads request as
[`POST /evaluations`](evaluations.md). `workloads` and at least one source are
required; correctness additionally needs both sources. Each uses the same
`Authorization: Bearer` capability and returns the same durable evaluation
resource with HTTP 200 or 202.

| Endpoint | Selected checks |
| --- | --- |
| `POST /verify` | Correctness; requires both PyTorch and Triton |
| `POST /benchmark` | Performance for one source; correctness and comparison for both |
| `POST /profile` | Profiling; accepts one or both sources |

Create `request.json` using the [comparison example](evaluations.md#submit), then:

```bash
curl -sS -i "$EVALUATION_URL/verify?wait_seconds=0" \
  -H "Authorization: Bearer $(cat "$EVALUATION_CAPABILITY_FILE")" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: verify-example-1' \
  --data-binary @request.json
```

Change the endpoint and key to benchmark or profile. The endpoint selects checks
after request validation; if `checks` is supplied in the body, it must be valid
but the shortcut overrides it. Use `/evaluations` to choose checks explicitly.
Performance and profile options use the same request model shown in `/docs`.
`/benchmark` supports [Torch timing reuse](evaluations.md#reuse-torch-baseline-timings)
through `performance.use_baseline_cache`; the client exposes
`--use-baseline-cache` and `--no-baseline-cache`. These control the PyTorch
baseline in a two-source comparison. Single-source benchmarks always measure
fresh and neither read nor write the baseline cache.

`Idempotency-Key` and `X-Session-ID` are optional. With a key, identical retries
on the same instance and session recover the same evaluation. Without a key,
each call submits new work. Poll `/evaluations/{evaluation_id}` to recover the
resource after the HTTP wait ends. There is no task catalog, separate tool token,
trial-turn budget or observation-text envelope on these endpoints.

## PyTorch profiling scope

PyTorch-only profiling observes the profiling process, including initialization;
it cannot isolate every reported launch to the supplied `run` function. The
service limits op mode to 100 launches using supported native-name prefixes, so its case
reports are partial and carry coverage warnings. Deep mode selects the longest
observed kernel across the process and is also partial. Preserve these warnings
when interpreting results; neither mode promises complete operator coverage.
For the underlying profiler's filtering and launch-count options, see
[msprof op parameters](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html).

## Bundled client

The [triton-eval plugin](../../plugins/triton-eval/README.md) provides `verify`,
`benchmark`, `profile` and `get` commands through its skill's standalone Python
script. It reads source files and a JSON array of cases, sends the same public
request, and polls the durable resource. For example, with the client path in
`TRITON_EVAL_CLI` and URL/capability configured:

```bash
python3 "$TRITON_EVAL_CLI" benchmark --definition definition.json --workloads workload.jsonl
python3 "$TRITON_EVAL_CLI" profile --definition definition.json --workloads workload.jsonl \
  --triton triton_impl.py --triton-only
```

These measure one implementation with `correctness_verified: false`; a zero
exit status means the requested measurement succeeded. Its recovery workflow keeps
the original instance and idempotency key after transport failures.

## Native HTTP and HTTPS

Use `python -m scripts.run_api` to serve the configured HTTP listener and an
optional HTTPS listener from one process. For example:

```dotenv
TRITON_EVAL_API_HOST=10.0.0.11
TRITON_EVAL_API_PORT=11907
TRITON_EVAL_API_TLS_HOST=127.0.0.1
TRITON_EVAL_API_TLS_PORT=11909
TRITON_EVAL_API_TLS_CERTFILE=/absolute/service-tls/server.crt
TRITON_EVAL_API_TLS_KEYFILE=/absolute/service-tls/server.key
```

Set a service unit's working directory to the checkout and its `ExecStart` to
the configured interpreter followed by `-m scripts.run_api`. The
`uvicorn scripts.run_api:app` CLI serves one listener and does not activate the
paired-listener mode. Keep private keys outside the checkout with restricted
permissions; clients must trust the certificate authority.

Both listeners share authentication, one evaluation instance, Redis and one
ASGI lifespan. TLS settings and sockets are checked before serving, and failure
to bind the HTTPS listener does not leave an HTTP-only instance. Shutdown joins
both servers before maintenance cleanup. Local HTTP/TLS tests establish listener
behavior; each deployment must validate its own certificates and network path.
