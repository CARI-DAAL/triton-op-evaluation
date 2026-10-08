# Verify, benchmark and profile an operator

The three shortcuts accept the same `torch_code`, `triton_code` and `cases`
request as [`POST /evaluations`](evaluations.md). Each uses the same
`Authorization: Bearer` capability and returns the same durable evaluation
resource with HTTP 200 or 202.

| Endpoint | Selected checks |
| --- | --- |
| `POST /verify` | Correctness |
| `POST /benchmark` | Correctness and performance |
| `POST /profile` | Profiling |

Create `request.json` using the [three-field example](evaluations.md#submit), then:

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

`Idempotency-Key` and `X-Session-ID` are optional. With a key, identical retries
on the same instance and session recover the same evaluation. Without a key,
each call submits new work. Poll `/evaluations/{evaluation_id}` to recover the
resource after the HTTP wait ends. There is no task catalog, separate tool token,
trial-turn budget or observation-text envelope on these endpoints.

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
