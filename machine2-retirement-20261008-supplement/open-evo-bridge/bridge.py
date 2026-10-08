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
import shlex
import shutil
import ssl
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web
from legacy_metrics import kernel_metrics_from_profile, render_diagnostics
from legacy_contract import (
    EvalBudgetRegistry,
    envelope,
    profile_verdict,
    verify_verdict,
)

logger = logging.getLogger("open-evo-bridge")

TRITON_IMPL_NAME = "triton_ascend_impl"


def action_label(checks: list[str]) -> str:
    """Mirror the tool action for error text (checks -> action wording)."""
    if checks == ["profile"]:
        return "profile"
    if "performance" in checks:
        return "benchmark"
    return "verify"


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
        self.upstream = os.environ.get(
            "BRIDGE_UPSTREAM", "http://192.169.0.227:11907"
        ).rstrip("/")
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
        # Auto-diagnostics lever: on every successful verify/benchmark the
        # bridge composes kernel-metrics lines from a single-case (case 0)
        # internal profile evaluation. Set BRIDGE_AUTO_DIAGNOSTICS=0 to skip
        # that second evaluation entirely (observation loses the 3-line
        # diagnostics block, ~26s faster per call).
        self.auto_diagnostics = os.environ.get("BRIDGE_AUTO_DIAGNOSTICS", "1") != "0"

        # Execution site for trial evaluations. "remote" forwards to the
        # evaluation service; "local" runs the SAME orchestrator on a device
        # claimed from a shared flock pool (the sandbox-stage model), which
        # removes the queue that otherwise dominates a trial's wall clock.
        self.exec_mode = os.environ.get("BRIDGE_EXEC_MODE", "remote")
        # The sandbox-native execution adapter (original toolchain through
        # the same runner the platform's final position uses).
        self.native_catalog = os.environ.get(
            "BRIDGE_NATIVE_CATALOG",
            "/home/e84381970/open-evo-bridge/tasks.json",
        )
        self.local_container = os.environ.get("BRIDGE_LOCAL_CONTAINER", "cc")
        self.local_device_pool = os.environ.get("BRIDGE_LOCAL_DEVICE_POOL", "")
        self.local_lock_dir = os.environ.get(
            "BRIDGE_LOCAL_LOCK_DIR", "/tmp/device-locks"
        )
        self.local_work_root = Path(
            os.environ.get(
                "BRIDGE_LOCAL_WORK_ROOT", "/home/e84381970/open-evo-bridge/work"
            )
        )
        self.local_python = os.environ.get(
            "BRIDGE_LOCAL_PYTHON",
            "/home/e84381970/triton-eval-machine2/venv/bin/python",
        )
        self.local_env_file = os.environ.get(
            "BRIDGE_LOCAL_ENV_FILE", "/home/e84381970/triton-eval-machine2/service.env"
        )
        self.local_claim = os.environ.get(
            "BRIDGE_LOCAL_CLAIM", "/home/e84381970/open-evo-bridge/claim_and_run.sh"
        )
        self.local_eval = os.environ.get(
            "BRIDGE_LOCAL_EVAL", "/home/e84381970/open-evo-bridge/local_eval.py"
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
        self._native = None

    def native_executor(self):
        if self._native is None:
            from native_exec import NativeExecution

            self._native = NativeExecution(self.settings.native_catalog)
        return self._native

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

        allowed = {"task_id", "kernel_code", "session"}
        if action == "profile":
            allowed.update({"profile_mode", "case_idx", "profile_launch_count"})
        if any("reference" in str(key).lower() for key in body) or set(body) - allowed:
            return web.json_response(
                {"error": "reference fields are forbidden"}, status=400
            )
        task_id = body.get("task_id")
        kernel_code = body.get("kernel_code")
        session = body.get("session") or ""
        if not isinstance(task_id, str) or not isinstance(kernel_code, str):
            return web.json_response(
                {"error": "task_id and kernel_code are required"}, status=400
            )
        if not isinstance(session, str) or len(session) > 128:
            session = ""
        request["kernel_eval_session"] = session
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
            # The profile action maps to the profile CHECK — leaving checks at
            # ["correctness"] while carrying profile params violates the
            # contract rule "profile params require the profile check" and the
            # service rejects the request with 400.
            checks = ["profile"]
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
        if settings.exec_mode == "local":
            return await self._evaluate_local(
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
                    action,
                    kernel_code,
                    turn,
                    eval_budget,
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
                            action,
                            kernel_code,
                            turn,
                            eval_budget,
                            f"{action} request failed: upstream rejected the "
                            f"evaluation: HTTP {response.status} "
                            f"{json.dumps(resource, ensure_ascii=False)[:300]}",
                        )
                    break
            except Exception as error:
                return self._infra(
                    action,
                    kernel_code,
                    turn,
                    eval_budget,
                    f"{action} request failed: cannot reach the evaluation "
                    f"service: {error}",
                )
        else:
            return self._infra(
                action,
                kernel_code,
                turn,
                eval_budget,
                f"{action} request failed: evaluation admission kept being "
                f"rate limited",
            )

        evaluation_id = resource.get("evaluation_id")
        while resource.get("state") not in TERMINAL_STATES:
            if time.monotonic() >= deadline:
                return self._infra(
                    action,
                    kernel_code,
                    turn,
                    eval_budget,
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
                    action,
                    kernel_code,
                    turn,
                    eval_budget,
                    f"{action} request failed: polling failed for "
                    f"{evaluation_id}: {error}",
                )

        state = resource.get("state")
        result = resource.get("result") or {}
        if state == "failed":
            first = self._first_error(result)
            return self._infra(
                action,
                kernel_code,
                turn,
                eval_budget,
                f"evaluation {evaluation_id} failed: {first}",
            )
        if action == "profile":
            verdict = profile_verdict(result.get("profile"))
        else:
            verdict = verify_verdict(result, action)
        diagnostics = None
        correctness = (result.get("correctness") or {}).get("data") or {}
        if (
            action in ("verify", "benchmark")
            and correctness.get("outcome") == "correct"
        ):
            # Legacy run_verify reported kernel metrics and a bottleneck hint on
            # every successful verification. Compose them from an internal
            # profile evaluation (never budget-counted for the agent) so the
            # service-backed reply is field-for-field what the sandbox CLI gave.
            if self.settings.auto_diagnostics:
                diagnostics = await self._compose_diagnostics(
                    request, task, kernel_code
                )
        response = envelope(
            verdict,
            action=action,
            kernel_code=kernel_code,
            eval_turn=turn,
            eval_budget=eval_budget,
            request_status="infra_error" if verdict.get("infra_error") else "completed",
            diagnostics=diagnostics,
        )
        return web.json_response(response)

    async def _evaluate_local(
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
        """Run the evaluation on a device claimed from the local flock pool.

        Same orchestrator, same engine, same observation renderer as the remote
        path — only the execution site differs, which is the split the
        measurement design allows (see local_eval.py). The claim waits inside
        this tool call, so pool contention surfaces to the agent as a slower
        call rather than an admission error.
        """
        settings = self.settings
        session = request.get("kernel_eval_session", "")
        document, error = await self._run_local_evaluation(
            task, kernel_code, checks, performance, profile, session
        )
        if document is None:
            return self._infra(action, kernel_code, turn, eval_budget, error)
        verdict = (
            profile_verdict(document.get("profile"))
            if action == "profile"
            else verify_verdict(document, action)
        )
        diagnostics = None
        correctness = (document.get("correctness") or {}).get("data") or {}
        if (
            action in ("verify", "benchmark")
            and correctness.get("outcome") == "correct"
        ):
            # One case is enough: the legacy summary reports metrics for the
            # dominant kernel plus a bottleneck hint, both derived from a single
            # representative case. Profiling the whole case set here would add
            # minutes per tool call for data the observation never shows.
            diagnostics_document, _ = await self._run_local_evaluation(
                task,
                kernel_code,
                ["profile"],
                None,
                {"mode": "op", "case_indices": [0]},
                session,
            )
            if diagnostics_document is not None:
                metrics = kernel_metrics_from_profile(
                    diagnostics_document.get("profile")
                )
                diagnostics = render_diagnostics(metrics) or None
        # Session-best record (additive): rendered by the renderer's known
        # fields only, so the observation text stays byte-identical; the
        # in-container kernel_eval tool persists the files it carries.
        best = document.get("best")
        return web.json_response(
            envelope(
                {**verdict, **({"best": best} if best is not None else {})},
                action=action,
                kernel_code=kernel_code,
                eval_turn=turn,
                eval_budget=eval_budget,
                request_status="infra_error"
                if verdict.get("infra_error")
                else "completed",
                diagnostics=diagnostics,
            )
        )

    async def _run_local_evaluation(
        self,
        task: dict,
        kernel_code: str,
        checks: list[str],
        performance: dict | None,
        profile: dict | None,
        session: str = "",
    ) -> tuple[dict | None, str]:
        """Materialize, claim a device, run the orchestrator, return its document."""
        settings = self.settings
        if not settings.local_device_pool:
            return None, "local execution requires BRIDGE_LOCAL_DEVICE_POOL"
        action = (
            "benchmark"
            if "performance" in checks
            else ("profile" if checks == ["profile"] else "verify")
        )
        try:
            executor = self.native_executor()
            document = await asyncio.wait_for(
                executor.evaluate(task["op_name"], kernel_code, action, session=session),
                timeout=settings.max_wait,
            )
            return document, ""
        except TimeoutError:
            return None, (
                f"{action} request failed: local evaluation exceeded"
                f" {settings.max_wait}s"
            )
        except Exception as error:  # noqa: BLE001 — surface, never crash the tool
            return None, f"local evaluation failed: {error}"
        # Legacy local_eval path retained below for BRIDGE_EXEC_MODE=service.
        work.mkdir(parents=True, exist_ok=True)
        op_name = task["op_name"]
        reference_path = work / f"{op_name}_torch.py"
        kernel_path = work / f"{op_name}_impl.py"
        output_path = work / "result.json"
        reference_path.write_text(task["reference"], encoding="utf-8")
        kernel_path.write_text(kernel_code, encoding="utf-8")

        budget = min(settings.max_wait, 900)
        argv = [
            settings.local_python,
            settings.local_eval,
            "--op-name",
            op_name,
            "--reference",
            str(reference_path),
            "--kernel",
            str(kernel_path),
            "--work-dir",
            str(work),
            "--checks",
            ",".join(checks),
            "--timeout",
            str(budget),
            "--output",
            str(output_path),
        ]
        if performance is not None:
            argv += ["--performance", json.dumps(performance)]
        if profile is not None:
            argv += ["--profile", json.dumps(profile)]
        command = " ".join(shlex.quote(part) for part in argv)
        container_command = f"{shlex.quote(settings.local_claim)} {command}"
        docker_argv = [
            "docker",
            "exec",
            "-e",
            f"TRITON_EVAL_ENV_FILE={settings.local_env_file}",
            "-e",
            f"EVAL_DEVICE_IDS={settings.local_device_pool}",
            "-e",
            f"EVAL_LOCK_DIR={settings.local_lock_dir}",
            "-e",
            f"EVAL_TIMEOUT={budget}",
            "-w",
            str(work),
            settings.local_container,
            "bash",
            "-c",
            container_command,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=budget + 120
                )
            except asyncio.TimeoutError:
                process.kill()
                return None, (
                    f"{action_label(checks)} request failed: local evaluation "
                    f"exceeded {budget + 120}s"
                )
        except Exception as error:  # noqa: BLE001 — surface, never crash the tool
            return None, f"local evaluation could not start: {error}"

        if not output_path.exists():
            detail = stderr.decode("utf-8", "replace").strip()[-300:]
            return None, (
                f"local evaluation produced no result (exit {process.returncode}): "
                f"{detail}"
            )
        try:
            document = json.loads(output_path.read_text(encoding="utf-8"))
        except ValueError as error:
            return None, f"local evaluation produced invalid JSON: {error}"
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return document, ""

    async def _compose_diagnostics(
        self, request: web.Request, task: dict, kernel_code: str
    ) -> str | None:
        """Internal op-profile evaluation for legacy kernel_metrics/hints.

        Never budget-counted (the agent issued one action; this composition is
        the bridge reproducing the legacy run_verify summary), and any failure
        just omits the diagnostics rather than failing the agent's call.
        """
        payload = {
            "schema_version": "2.1",
            "purpose": "trial",
            "op_name": task["op_name"],
            "reference_code": task["reference"],
            "kernel_code": kernel_code,
            "triton_impl_name": TRITON_IMPL_NAME,
            "checks": ["profile"],
            "non_compute": False,
            "deadline_seconds": min(900, self.settings.max_wait),
            "performance": None,
            # One case is enough: the renderer consumes only the first
            # complete case (legacy_metrics.py). Pinning case 0 makes the
            # diagnostics leg cost ~26s instead of ~870s (whole-case-set
            # profiling) while producing the identical observation shape —
            # parity with the local path (see _evaluate_local).
            "profile": {"mode": "op", "case_indices": [0]},
            "grade_context": None,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        key = hashlib.sha256(body + b"profile-diagnostics").hexdigest()
        deadline = time.monotonic() + self.settings.max_wait
        try:
            session: ClientSession = request.app["upstream"]
            async with session.post(
                f"{self.settings.upstream}/evaluations",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.settings.upstream_token}",
                    "Content-Type": "application/json",
                    "Idempotency-Key": key,
                    "X-Session-ID": f"bridge-{key[:24]}",
                },
            ) as response:
                resource = await response.json()
                if response.status >= 400:
                    return None
            while resource.get("state") not in TERMINAL_STATES:
                if time.monotonic() >= deadline:
                    return None
                await asyncio.sleep(POLL_SECONDS)
                async with session.get(
                    f"{self.settings.upstream}/evaluations/{resource.get('evaluation_id')}",
                    headers={"Authorization": f"Bearer {self.settings.upstream_token}"},
                ) as response:
                    resource = await response.json()
            metrics = kernel_metrics_from_profile(
                (resource.get("result") or {}).get("profile")
            )
            return render_diagnostics(metrics) or None
        except Exception:
            logger.debug("diagnostics composition failed", exc_info=True)
            return None

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
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _HOP_HEADERS
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
        app["upstream"] = ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=15)
        )
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
        app["upstream"] = ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=15)
        )
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
    interaction_site = web.TCPSite(
        interaction_runner, "0.0.0.0", settings.interaction_port
    )
    await interaction_site.start()
    logger.info("interaction adapter listening on :%d", settings.interaction_port)
    if settings.tls_port and settings.tls_cert and settings.tls_key:
        proxy_runner = web.AppRunner(build_proxy_app(settings))
        await proxy_runner.setup()
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(settings.tls_cert, settings.tls_key)
        proxy_site = web.TCPSite(
            proxy_runner, "0.0.0.0", settings.tls_port, ssl_context=tls
        )
        await proxy_site.start()
        logger.info("tls proxy listening on :%d", settings.tls_port)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    main()
