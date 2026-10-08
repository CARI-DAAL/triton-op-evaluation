"""Kernel evaluation caller intent, independent of HTTP transport.

Owns catalog-task translation, durable trial turns, operation identity,
submission/recovery and optional diagnostics. Candidate execution remains in
fenced workers owned by the injected evaluation instance.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import ValidationError

from . import kernel_eval_renderer as renderer
from .api.contract import EvaluationRequest, ProfileParams, dump_contract
from .kernel_eval_budget import EvalBudgetRegistry
from .service.admission import ForbiddenError
from .service.broker import StorageError
from .service.catalog import ManifestBindingError
from .service.instance import EvaluationInstance, InstanceNotReadyError, QueueUnavailableError
from .service.security import Caller

logger = logging.getLogger(__name__)
TOOL_CALLER = Caller("kernel-eval", "trial", frozenset({"trial:submit", "trial:read"}))
_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TERMINAL_STATES = ("completed", "failed", "timed_out")


class ToolInputError(ValueError):
    """A rejected operation: no transport status or response object required."""

    def __init__(self, kind: Literal["invalid", "forbidden", "conflict"], message: str):
        super().__init__(message)
        self.kind = kind


class _EvaluationUnavailable(Exception):
    """A bounded, redacted failure rendered inside the tool envelope."""


def _profile_options(body: dict) -> dict | None:
    mode = body.get("profile_mode", "op")
    case = body.get("case_idx")
    launches = body.get("profile_launch_count")
    if (
        mode not in ("op", "deep")
        or (case is not None and (type(case) is not int or case < 0))
        or (launches is not None and (type(launches) is not int or not 1 <= launches <= 100))
        or (mode == "deep" and launches is not None)
    ):
        return None
    profile = {"mode": mode}
    if case is not None:
        profile["case_indices"] = [case]
    # The legacy contract validates launch_count but never forwards it.
    return profile


class KernelEvaluation:
    """Execute one tool operation with a durable budget and an owning instance.

    Authentication belongs to the caller's transport adapter. Reusing an
    explicit operation ID reserves the same turn and recovers the same work.
    Without one, each call costs a turn but identical evaluation input replays.
    A wait timeout never cancels already accepted work.
    """

    def __init__(
        self,
        instance: EvaluationInstance,
        budget: EvalBudgetRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.instance = instance
        self.budget = budget
        self._clock = clock
        self._sleep = sleep

    async def evaluate(
        self,
        action: str,
        body: dict,
        *,
        session_id: str | None,
        operation_id: str | None = None,
    ) -> dict:
        """Return a tool envelope or raise ToolInputError before accepting invalid work."""
        if action not in {"verify", "benchmark", "profile"}:
            raise ToolInputError("invalid", "invalid action")
        if not isinstance(body, dict):
            raise ToolInputError("invalid", "invalid json")
        instance = self.instance
        settings = instance.settings
        allowed = {"task_id", "kernel_code", "session"}
        if action == "profile":
            allowed.update({"profile_mode", "case_idx", "profile_launch_count"})
        if any("reference" in str(key).lower() for key in body) or set(body) - allowed:
            raise ToolInputError("invalid", "reference fields are forbidden")
        task_id, kernel_code = body.get("task_id"), body.get("kernel_code")
        if not isinstance(task_id, str) or not isinstance(kernel_code, str):
            raise ToolInputError("invalid", "task_id and kernel_code are required")
        try:
            task = instance.catalog.resolve_task(
                task_id, manifest_id=settings.kernel_eval_manifest_id
            )
        except ManifestBindingError:
            raise ToolInputError("forbidden", "task_id mismatch") from None
        profile = _profile_options(body) if action == "profile" else None
        if action == "profile" and profile is None:
            raise ToolInputError("invalid", "invalid profile options")
        if (
            not isinstance(session_id, str)
            or _SESSION_ID.fullmatch(session_id) is None
            or body.get("session") not in (None, session_id)
        ):
            raise ToolInputError("invalid", "invalid session")
        try:
            evaluation_request = EvaluationRequest.model_validate(
                {
                    "schema_version": "2.1",
                    "purpose": "trial",
                    "op_name": task.problem_id,
                    "reference_code": task.reference_code,
                    "kernel_code": kernel_code,
                    "triton_impl_name": "triton_ascend_impl",
                    "checks": (
                        ["profile"]
                        if profile
                        else ["correctness", "performance"]
                        if action == "benchmark"
                        else ["correctness"]
                    ),
                    "deadline_seconds": min(900, int(settings.kernel_eval_max_wait_seconds)),
                    "performance": {
                        "warmup": settings.kernel_eval_performance_warmup,
                        "repeats": settings.kernel_eval_performance_repeats,
                        "method": "profiler",
                    }
                    if action == "benchmark"
                    else None,
                    "profile": profile,
                }
            )
        except ValidationError:
            raise ToolInputError("invalid", "invalid evaluation request") from None
        payload = evaluation_request.model_dump_json(exclude_none=False).encode()
        digest = hashlib.sha256(payload).hexdigest()
        if operation_id is None:
            # Legacy callers consume a turn on each HTTP tool call, even when
            # identical source reuses a durable evaluation. Modern clients may
            # supply a stable key so a transport retry reuses its reserved turn.
            operation_id = uuid.uuid4().hex
            evaluation_identity = hashlib.sha256(payload + action.encode()).hexdigest()
        else:
            if (
                not isinstance(operation_id, str)
                or not 1 <= len(operation_id) <= 128
                or any(not 0x21 <= ord(char) <= 0x7E for char in operation_id)
            ):
                raise ToolInputError("invalid", "invalid Idempotency-Key")
            evaluation_identity = operation_id
        try:
            reservation = await asyncio.to_thread(
                self.budget.reserve, TOOL_CALLER.caller_id, session_id, operation_id, digest
            )
        except StorageError:
            return renderer.render_infra_error(
                action=action,
                kernel_code=kernel_code,
                eval_turn=None,
                eval_budget="unknown",
                message=f"{action} request failed: storage unavailable",
            )
        eval_budget = (
            f"{reservation.used}/{self.budget.maximum}" if self.budget.maximum >= 0 else "unlimited"
        )
        if reservation.decision == "budget_exhausted":
            return renderer.render_budget_exhausted(
                action=action,
                kernel_code=kernel_code,
                eval_budget=eval_budget,
            )
        if reservation.decision == "conflict":
            raise ToolInputError("conflict", "idempotency conflict")
        idempotency = hashlib.sha256(
            f"kernel-eval\0{session_id}\0{evaluation_identity}".encode()
        ).hexdigest()
        deadline = self._clock() + settings.kernel_eval_max_wait_seconds
        try:
            resource = await self._evaluate(evaluation_request, session_id, idempotency, deadline)
        except _EvaluationUnavailable as error:
            return renderer.render_infra_error(
                action=action,
                kernel_code=kernel_code,
                eval_turn=reservation.turn,
                eval_budget=eval_budget,
                message=f"{action} request failed: {error}",
            )
        diagnostics = None
        verdict = renderer.verify_verdict(resource.get("result") or {}, action)
        if (
            action in ("verify", "benchmark")
            and verdict.get("correct")
            and settings.kernel_eval_auto_diagnostics
        ):
            diagnostic_request = evaluation_request.model_copy(
                update={
                    "checks": ["profile"],
                    "performance": None,
                    "profile": ProfileParams(mode="op", case_indices=[0]),
                }
            )
            try:
                diagnostic = await self._evaluate(
                    diagnostic_request,
                    session_id,
                    hashlib.sha256((idempotency + ":diagnostics:case0").encode()).hexdigest(),
                    deadline,
                )
                diagnostics = renderer.compose_diagnostics(
                    (diagnostic.get("result") or {}).get("profile")
                )
            except _EvaluationUnavailable as error:
                logger.debug("optional kernel metrics unavailable (%s)", error)
        return renderer.render_envelope(
            resource,
            action=action,
            kernel_code=kernel_code,
            eval_turn=reservation.turn,
            eval_budget=eval_budget,
            diagnostics=diagnostics,
        )

    async def _evaluate(self, evaluation_request, session_id, idempotency, deadline) -> dict:
        instance = self.instance
        for attempt in range(8):
            if self._clock() >= deadline:
                raise _EvaluationUnavailable(
                    "evaluation admission did not succeed within the wait budget"
                )
            try:
                outcome = await asyncio.to_thread(
                    instance.submit,
                    evaluation_request,
                    TOOL_CALLER,
                    session_id=session_id,
                    idempotency_key=idempotency,
                )
            except (InstanceNotReadyError, QueueUnavailableError, StorageError):
                raise _EvaluationUnavailable("evaluation service unavailable") from None
            except (ForbiddenError, ManifestBindingError):
                raise _EvaluationUnavailable("evaluation admission rejected") from None
            if outcome.decision in ("rate_limited", "limit_exceeded"):
                remaining = deadline - self._clock()
                if remaining > 0:
                    await self._sleep(min(15.0, 2.0 * (attempt + 1), remaining))
                continue
            if outcome.decision == "conflict":
                raise _EvaluationUnavailable("evaluation idempotency conflict")
            resource = outcome.resource
            if resource is None:
                raise _EvaluationUnavailable("evaluation admission returned no resource")
            break
        else:
            raise _EvaluationUnavailable("evaluation admission kept being rate limited")
        while resource.state not in _TERMINAL_STATES:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise _EvaluationUnavailable(
                    f"evaluation {resource.evaluation_id} did not finish within the wait budget; "
                    "it may complete later"
                )
            await self._sleep(min(instance.settings.api_poll_interval_seconds, remaining))
            try:
                recovered = await asyncio.to_thread(
                    instance.get, resource.evaluation_id, TOOL_CALLER
                )
            except StorageError:
                raise _EvaluationUnavailable(
                    f"polling failed for {resource.evaluation_id}"
                ) from None
            if recovered is None:
                raise _EvaluationUnavailable(f"evaluation {resource.evaluation_id} is not visible")
            resource = recovered
        # Serve the JCS-restabilized contract document (as routes.py does): the
        # raw model_dump normalizes int->float and reorders embedded keys, which
        # would leak into the rendered observation text.
        return json.loads(dump_contract(resource))
