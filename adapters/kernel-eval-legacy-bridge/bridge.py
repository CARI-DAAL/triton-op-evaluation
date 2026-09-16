"""Open Evo bridge: interaction adapter + TLS front for the Triton eval service.

Two listeners, one process:

- ``POST /verify|benchmark|profile`` (plain HTTP, reached from the task agent
  container): the in-container ``kernel_eval`` tool protocol. Each call is
  translated into one ``purpose="trial"`` evaluation on the upstream service,
  polled to a terminal state, and projected into the legacy observation the
  baseline model was trained against (kernel_agent@52affcb9): the
  ``[kernel_eval]`` envelope (action / artifact_sha256 / eval_budget /
  request_status) plus the verdict line, with legacy error classes, folded
  per-case failure text, legacy benchmark keys, and per-trial eval-budget
  accounting. See legacy_contract.py for the verbatim ported renderer.
- Transparent HTTPS reverse proxy: the platform verifier only dials pinned
  HTTPS origins, so it talks to this listener which forwards bytes unchanged
  to the plain-HTTP upstream.

Environment:
  BRIDGE_UPSTREAM            upstream base URL (default http://192.169.0.227:11907)
  BRIDGE_TOKEN               bearer token the kernel_eval tool presents
  BRIDGE_UPSTREAM_TOKEN      bearer token for upstream trial submissions
  BRIDGE_TASKS               path to the task catalog JSON
  BRIDGE_INTERACTION_PORT    default 11908
  BRIDGE_TLS_PORT            default 11909 (0 disables the TLS listener)
  BRIDGE_TLS_CERT/KEY        PEM paths for the TLS listener
  BRIDGE_MAX_WAIT_SECONDS    wall-clock cap per tool call (default 840)
  BRIDGE_PERF_PARAMS         JSON for performance params (profiler baseline)
  BRIDGE_MAX_EVAL_TURNS      per-trial eval budget (default 18, the NPU
                             baseline's max_eval_turns; negative disables)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import ssl
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web
from legacy_contract import (
    EvalBudgetRegistry,
    envelope,
    profile_verdict,
    verify_verdict,
)

logger = logging.getLogger("open-evo-bridge")

TRITON_IMPL_NAME = "triton_ascend_impl"
POLL_SECONDS = 2.0
TERMINAL_STATES = {"completed", "failed", "timed_out"}

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class Settings:
    def __init__(self) -> None:
        self.upstream = os.environ.get("BRIDGE_UPSTREAM", "http://192.169.0.227:11907").rstrip("/")
        self.token = os.environ.get("BRIDGE_TOKEN", "")
        self.upstream_token = os.environ.get("BRIDGE_UPSTREAM_TOKEN", "")
        self.tasks_path = Path(
            os.environ.get("BRIDGE_TASKS", "/home/e84381970/open-evo-bridge/tasks.json")
        )
        self.interaction_port = _env_int("BRIDGE_INTERACTION_PORT", 11908)
        self.tls_port = _env_int("BRIDGE_TLS_PORT", 11909)
        self.tls_cert = os.environ.get("BRIDGE_TLS_CERT")
        self.tls_key = os.environ.get("BRIDGE_TLS_KEY")
        self.max_wait = _env_int("BRIDGE_MAX_WAIT_SECONDS", 840)
        self.max_eval_turns = _env_int("BRIDGE_MAX_EVAL_TURNS", 18)
        self.perf_params = json.loads(
            os.environ.get(
                "BRIDGE_PERF_PARAMS",
                '{"warmup": 5, "repeats": 20, "method": "profiler"}',
            )
        )
        if not self.token or not self.upstream_token:
            raise SystemExit("BRIDGE_TOKEN and BRIDGE_UPSTREAM_TOKEN are required")


class TaskCatalog:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.tasks = {
            task_id: {
                "op_name": entry["op_name"],
                "reference": Path(entry["reference_path"]).read_text(encoding="utf-8"),
            }
            for task_id, entry in raw.items()
        }
        logger.info("task catalog loaded: %d tasks", len(self.tasks))


# ---------------------------------------------------------------------------
# interaction adapter (legacy kernel_eval session-facade semantics)
# ---------------------------------------------------------------------------


def _artifact_sha256(kernel_code: str) -> str:
    return hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()


class Adapter:
    def __init__(self, settings: Settings, catalog: TaskCatalog) -> None:
        self.settings = settings
        self.catalog = catalog
        self.budget = EvalBudgetRegistry(settings.max_eval_turns)

    async def handle_action(self, request: web.Request) -> web.Response:
        # Ordering mirrors the legacy SessionServer.do_POST: auth, then path,
        # then body shape, then task identity, then profile options — each
        # rejection uses the legacy HTTP status and {"error": ...} body the
        # kernel_eval CLI renders as "[kernel_eval] ERROR contacting evaluator".
        action = request.match_info["action"]
        if action not in {"verify", "benchmark", "profile"}:
            return web.json_response({"error": "not found"}, status=404)
        expected = f"Bearer {self.settings.token}"
        if request.headers.get("Authorization") != expected:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "invalid json"}, status=400)

        allowed = {"task_id", "kernel_code"}
        if action == "profile":
            allowed.update({"profile_mode", "case_idx", "profile_launch_count"})
        if any("reference" in str(key).lower() for key in body) or set(body) - allowed:
            return web.json_response(
                {"error": "reference fields are forbidden"}, status=400
            )
        task_id = body.get("task_id")
        kernel_code = body.get("kernel_code")
        if not isinstance(task_id, str) or not isinstance(kernel_code, str):
            return web.json_response(
                {"error": "task_id and kernel_code are required"}, status=400
            )
        task = self.catalog.tasks.get(task_id)
        if task is None:
            return web.json_response({"error": "task_id mismatch"}, status=403)

        checks = ["correctness"]
        performance = None
        profile = None
        if action == "benchmark":
            checks = ["correctness", "performance"]
            performance = dict(self.settings.perf_params)
        elif action == "profile":
            mode = body.get("profile_mode", "op")
            case_idx = body.get("case_idx")
            launch_count = body.get("profile_launch_count")
            valid_case = case_idx is None or (
                isinstance(case_idx, int)
                and not isinstance(case_idx, bool)
                and case_idx >= 0
            )
            valid_launch_count = launch_count is None or (
                isinstance(launch_count, int)
                and not isinstance(launch_count, bool)
                and 1 <= launch_count <= 100
            )
            if (
                mode not in ("op", "deep")
                or not valid_case
                or not valid_launch_count
                or (mode == "deep" and launch_count is not None)
            ):
                return web.json_response(
                    {"error": "invalid profile options"}, status=400
                )
            profile = {"mode": mode}
            if case_idx is not None:
                profile["case_indices"] = [case_idx]

        # One trial = one sandbox container = one client IP on the shared
        # docker network; budget it exactly like the legacy EvaluationSession.
        budget_key = str(request.remote)
        turn, blocked = self.budget.reserve(budget_key)
        cap = self.settings.max_eval_turns
        eval_budget = (
            f"{self.budget.used(budget_key)}/{cap}" if cap >= 0 else "unlimited"
        )
        if blocked == "budget_exhausted":
            result = envelope(
                {},
                action=action,
                kernel_code=kernel_code,
                eval_turn=None,
                eval_budget=eval_budget,
                request_status="budget_exhausted",
            )
            return web.json_response(result)

        return await self._evaluate(
            request,
            action,
            task,
            kernel_code,
            checks,
            performance,
            profile,
            turn,
            eval_budget,
        )

    async def _evaluate(
        self,
        request: web.Request,
        action: str,
        task: dict,
        kernel_code: str,
        checks: list[str],
        performance: dict | None,
        profile: dict | None,
        turn: int | None,
        eval_budget: str,
    ) -> web.Response:
        settings = self.settings
        payload = {
            "schema_version": "2.1",
            "purpose": "trial",
            "op_name": task["op_name"],
            "reference_code": task["reference"],
            "kernel_code": kernel_code,
            "triton_impl_name": TRITON_IMPL_NAME,
            "checks": checks,
            "non_compute": False,
            "deadline_seconds": min(900, settings.max_wait),
            "performance": performance,
            "profile": profile,
            "grade_context": None,
        }
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        idempotency = hashlib.sha256(body_bytes + action.encode()).hexdigest()
        # Session derives from the same digest as the idempotency key: retries
        # replay, while parallel samples with different kernels never share a
        # session (the service caps per-session in-flight submissions).
        session_id = f"bridge-{idempotency[:24]}"
        deadline = time.monotonic() + settings.max_wait
        session: ClientSession = request.app["upstream"]

        submit_headers = {
            "Authorization": f"Bearer {settings.upstream_token}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency,
            "X-Session-ID": session_id,
        }
        # Admission contention (429 rate-limited / 413 inflight cap) is
        # transient: resubmit with backoff inside the same tool call instead of
        # surfacing infra feedback that burns an agent step. The deterministic
        # idempotency key makes resubmission a replay, never a duplicate.
        resource = None
        for attempt in range(8):
            if time.monotonic() >= deadline:
                return self._infra(
                    action, kernel_code, turn, eval_budget,
                    f"{action} request failed: evaluation admission did not "
                    f"succeed within {settings.max_wait}s",
                )
            try:
                async with session.post(
                    f"{settings.upstream}/evaluations",
                    data=body_bytes,
                    headers=submit_headers,
                ) as response:
                    resource = await response.json()
                    if response.status in (429, 413):
                        code = ((resource.get("error") or {}).get("code")) or ""
                        if code in ("RATE_LIMITED", "LIMIT_EXCEEDED"):
                            await asyncio.sleep(min(15.0, 2.0 * (attempt + 1)))
                            continue
                    if response.status >= 400:
                        return self._infra(
                            action, kernel_code, turn, eval_budget,
                            f"{action} request failed: upstream rejected the "
                            f"evaluation: HTTP {response.status} "
                            f"{json.dumps(resource, ensure_ascii=False)[:300]}",
                        )
                    break
            except Exception as error:
                return self._infra(
                    action, kernel_code, turn, eval_budget,
                    f"{action} request failed: cannot reach the evaluation "
                    f"service: {error}",
                )
        else:
            return self._infra(
                action, kernel_code, turn, eval_budget,
                f"{action} request failed: evaluation admission kept being "
                f"rate limited",
            )

        evaluation_id = resource.get("evaluation_id")
        while resource.get("state") not in TERMINAL_STATES:
            if time.monotonic() >= deadline:
                return self._infra(
                    action, kernel_code, turn, eval_budget,
                    f"{action} request failed: evaluation {evaluation_id} did "
                    f"not finish within {settings.max_wait}s; it may complete "
                    f"later",
                )
            await asyncio.sleep(POLL_SECONDS)
            try:
                async with session.get(
                    f"{settings.upstream}/evaluations/{evaluation_id}",
                    headers={"Authorization": f"Bearer {settings.upstream_token}"},
                ) as response:
                    resource = await response.json()
            except Exception as error:
                return self._infra(
                    action, kernel_code, turn, eval_budget,
                    f"{action} request failed: polling failed for "
                    f"{evaluation_id}: {error}",
                )

        state = resource.get("state")
        result = resource.get("result") or {}
        if state == "failed":
            first = self._first_error(result)
            return self._infra(
                action, kernel_code, turn, eval_budget,
                f"evaluation {evaluation_id} failed: {first}",
            )
        if action == "profile":
            verdict = profile_verdict(result.get("profile"))
        else:
            verdict = verify_verdict(result, action)
        response = envelope(
            verdict,
            action=action,
            kernel_code=kernel_code,
            eval_turn=turn,
            eval_budget=eval_budget,
            request_status="infra_error" if verdict.get("infra_error") else "completed",
        )
        return web.json_response(response)

    @staticmethod
    def _first_error(result: dict) -> str:
        for check in ("correctness", "performance", "profile", "compile"):
            outcome = result.get(check) or {}
            error = outcome.get("error")
            if error:
                return f"{error.get('code', 'UNKNOWN')}: {error.get('message', '')}"
        return "unknown failure"

    @staticmethod
    def _infra(
        action: str,
        kernel_code: str,
        turn: int | None,
        eval_budget: str,
        message: str,
    ) -> web.Response:
        result = envelope(
            {"infra_error": True, "error": message},
            action=action,
            kernel_code=kernel_code,
            eval_turn=turn,
            eval_budget=eval_budget,
            request_status="infra_error",
        )
        return web.json_response(result)


# ---------------------------------------------------------------------------
# TLS reverse proxy
# ---------------------------------------------------------------------------


async def proxy(request: web.Request) -> web.StreamResponse:
    settings: Settings = request.app["settings"]
    session: ClientSession = request.app["upstream"]
    headers = {
        name: value for name, value in request.headers.items() if name.lower() not in _HOP_HEADERS
    }
    body = await request.read()
    url = f"{settings.upstream}{request.rel_url}"
    async with session.request(
        request.method, url, headers=headers, data=body, allow_redirects=False
    ) as response:
        proxy_response = web.StreamResponse(
            status=response.status,
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _HOP_HEADERS
            },
        )
        await proxy_response.prepare(request)
        async for chunk in response.content.iter_any():
            await proxy_response.write(chunk)
        await proxy_response.write_eof()
        return proxy_response


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ready": True})


def build_interaction_app(settings: Settings, catalog: TaskCatalog) -> web.Application:
    app = web.Application()

    async def _upstream(app: web.Application):
        app["upstream"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=15))
        yield
        await app["upstream"].close()

    app.cleanup_ctx.append(_upstream)
    app["settings"] = settings
    adapter = Adapter(settings, catalog)
    app.router.add_post("/{action}", adapter.handle_action)
    app.router.add_get("/health", health)
    return app


def build_proxy_app(settings: Settings) -> web.Application:
    app = web.Application()

    async def _upstream(app: web.Application):
        app["upstream"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=15))
        yield
        await app["upstream"].close()

    app.cleanup_ctx.append(_upstream)
    app["settings"] = settings
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = Settings()
    catalog = TaskCatalog(settings.tasks_path)
    asyncio.run(_serve(settings, catalog))


async def _serve(settings: Settings, catalog: TaskCatalog) -> None:
    interaction_runner = web.AppRunner(build_interaction_app(settings, catalog))
    await interaction_runner.setup()
    interaction_site = web.TCPSite(interaction_runner, "0.0.0.0", settings.interaction_port)
    await interaction_site.start()
    logger.info("interaction adapter listening on :%d", settings.interaction_port)
    if settings.tls_port and settings.tls_cert and settings.tls_key:
        proxy_runner = web.AppRunner(build_proxy_app(settings))
        await proxy_runner.setup()
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(settings.tls_cert, settings.tls_key)
        proxy_site = web.TCPSite(proxy_runner, "0.0.0.0", settings.tls_port, ssl_context=tls)
        await proxy_site.start()
        logger.info("tls proxy listening on :%d", settings.tls_port)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    main()
