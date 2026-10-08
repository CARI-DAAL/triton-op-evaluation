#!/usr/bin/env python3
"""The sandbox-native execution adapter for the interaction bridge.

Reproduces the sandbox session facade on the ORIGINAL toolchain: the AST
decoy gate (``validate_triton_impl.py``, the session server's Step 0) plus
``verify``/``benchmark``/``profile`` through :class:`NativeToolchainRunner`
— the same runner the platform's final verification position uses, so the
agent-facing feedback and the final judgment share one engine.

The output document keeps the shape the bridge's observation renderers
already consume (``legacy_contract.verify_verdict`` /
``profile_verdict`` / ``_legacy_benchmark``), so observations stay
byte-identical to the sandbox-era contract while the execution site is
the flock-claimed device inside the NPU container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPEN_EVO_SRC = os.environ.get(
    "NATIVE_BRIDGE_OPEN_EVO_SRC", "/home/e84381970/open-evo/src"
)
if OPEN_EVO_SRC not in sys.path:
    sys.path.insert(0, OPEN_EVO_SRC)

from harbor.integrations.triton.kernel_measurement import (  # noqa: E402
    DeviceOperation,
)
from harbor.integrations.triton.native_binding import (  # noqa: E402
    NativeToolchainConfig,
)

#: Domain-failure kind -> (outcome literal, legacy error_type label).
_KIND_TO_OUTCOME = {
    "compile_error": "compile_error",
    "runtime_error": "runtime_error",
    "execution_timeout": "execution_timeout",
}
_KIND_TO_TYPE = {
    "compile_error": "CompilationError",
    "runtime_error": "RuntimeError",
    "execution_timeout": "TimeoutError",
    "wrong": "AccuracyError",
}


def _positive_latency(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


class BestSnapshotLayer:
    """The legacy best-snapshot bookkeeping on the bridge interaction path.

    Reproduces ``run_npu_command.sh``'s snapshot tail after every verify and
    benchmark: ``summarize_verify_result.py`` then ``snapshot_verify_best.py``
    (the ORIGINAL scripts, unmodified) over a persistent per-task workspace
    laid out like the sandbox's agent workdir — ``output/verify`` staging,
    ``metrics_best.json``, ``src/{op}_triton_ascend_impl_best.py`` and
    ``output/perf_result_best.json`` under the workspace root — including the
    mtime-gated perf binding that keeps one impl's speedup out of another
    impl's metrics. The ephemeral per-operation work dirs the runner builds
    are deleted after each run, so the persistent workspace here is where
    the session-best record accumulates.

    Bookkeeping only, never measurement: every failure is logged and
    swallowed so the eval call itself cannot fail through this layer.
    """

    IMPL_SUFFIX = "triton_ascend_impl"

    def __init__(self, scripts_dir: str, work_root: str, python: str | None = None):
        self.scripts_dir = Path(scripts_dir)
        self.work_root = Path(work_root)
        self.python = python or sys.executable
        self._locks: dict[str, asyncio.Lock] = {}

    # -- workspace --------------------------------------------------------

    def workspace(self, task_id: str, session: str = "") -> Path:
        # Colon-free slug: the work root's dirs must stay PYTHONPATH-safe.
        # The session key (the trial's container hostname) isolates one
        # trial's session-best record from every other trial of the same
        # task — without it the record silently becomes experiment-best.
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", task_id)
        if session:
            slug += "_" + re.sub(r"[^A-Za-z0-9._-]", "_", session)
        return self.work_root / f"bestws_{slug}"

    def lock(self, task_id: str, session: str = "") -> asyncio.Lock:
        return self._locks.setdefault(f"{task_id}:{session}", asyncio.Lock())

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @classmethod
    def _write_json(cls, path: Path, payload: dict[str, Any]) -> None:
        cls._write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    # -- legacy artifact reconstruction ------------------------------------

    def _verify_result(
        self, op_name: str, result: Any, expected_total: int
    ) -> dict[str, Any]:
        """The verifier JSON verify.py would have written (1-based case_idx)."""
        raw = result.raw or {}
        total = raw.get("total_cases")
        if not isinstance(total, int):
            total = expected_total
        passed = raw.get("passed_cases")
        if not isinstance(passed, int):
            passed = sum(1 for case in result.cases if case.passed is True)
        failures = [
            {
                "case_idx": int(item.get("case_idx") or 0) + 1,
                "error_type": item.get("error_type"),
                "error_msg": item.get("error_msg"),
            }
            for item in raw.get("failures") or []
            if isinstance(item, dict)
        ]
        return {
            "op_name": op_name,
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": max(total - passed, 0),
            "failures": failures,
        }

    def _perf_result(
        self, op_name: str, result: Any, expected_total: int
    ) -> dict[str, Any]:
        """The benchmark JSON benchmark.py would have written to output/."""
        per_shape: list[dict[str, Any]] = []
        speedups: list[float] = []
        for case in sorted(result.cases, key=lambda item: item.case_idx):
            reference = case.reference_latency_ms
            implementation = case.implementation_latency_ms
            measured = (
                case.passed is True
                and _positive_latency(reference)
                and _positive_latency(implementation)
            )
            speedup = (
                round(float(reference) / float(implementation), 4) if measured else None
            )
            if measured:
                speedups.append(speedup)
            per_shape.append(
                {
                    "case_idx": case.case_idx + 1,
                    "status": "pass" if case.passed is True else "fail",
                    "framework": (
                        {"avg_latency_ms": reference}
                        if _positive_latency(reference)
                        else None
                    ),
                    "implementation": (
                        {"avg_latency_ms": implementation}
                        if _positive_latency(implementation)
                        else None
                    ),
                    "speedup_vs_torch": speedup,
                }
            )
        overall = None
        if speedups:
            overall = round(
                math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
                4,
            )
        total = len(per_shape) or expected_total
        passed = sum(1 for item in per_shape if item["status"] == "pass")
        return {
            "op_name": op_name,
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": max(total - passed, 0),
            "speedup_vs_torch": overall,
            "per_shape_results": per_shape,
        }

    # -- script invocation (run_npu_command.sh's tail semantics) -----------

    async def _run_script(self, argv: list[str]) -> None:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 60)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"snapshot script timed out: {argv[1]}") from None
        text = stdout.decode(errors="replace").strip()
        if text:
            logger.info("%s", text.replace("\n", " | "))
        if proc.returncode:
            tail = stderr.decode(errors="replace").strip()[-300:]
            logger.warning(
                "snapshot script %s exited %s: %s", argv[1], proc.returncode, tail
            )

    async def _summarize(self, verify_path: Path, exit_code: int) -> None:
        summary_path = verify_path.with_name(verify_path.stem + "_summary.json")
        await self._run_script(
            [
                self.python,
                str(self.scripts_dir / "summarize_verify_result.py"),
                str(verify_path),
                "--exit-code",
                str(exit_code),
                "--write-json",
                str(summary_path),
            ]
        )

    async def _snapshot(self, op_name: str, ws: Path) -> None:
        await self._run_script(
            [
                self.python,
                str(self.scripts_dir / "snapshot_verify_best.py"),
                "--op-name",
                op_name,
                "--workspace-dir",
                str(ws),
                "--verify-dir",
                "output/verify",
                "--verify-result",
                "output/verify/verify_result.json",
                "--summary",
                "output/verify/verify_result_summary.json",
                "--triton-impl-name",
                self.IMPL_SUFFIX,
            ]
        )

    # -- recording ---------------------------------------------------------

    async def record_verify(
        self,
        task_id: str,
        session: str,
        op_name: str,
        kernel: str,
        result: Any,
        total: int,
    ) -> None:
        """Stage the impl + verify JSON, then summarize -> snapshot."""
        ws = self.workspace(task_id, session)
        verify_dir = ws / "output" / "verify"
        self._write_text(verify_dir / f"{op_name}_{self.IMPL_SUFFIX}.py", kernel)
        verify = self._verify_result(op_name, result, total)
        verify_path = verify_dir / "verify_result.json"
        self._write_json(verify_path, verify)
        exit_code = (
            0
            if verify["total_cases"] > 0
            and verify["passed_cases"] == verify["total_cases"]
            else 1
        )
        await self._summarize(verify_path, exit_code)
        await self._snapshot(op_name, ws)

    async def record_benchmark(
        self,
        task_id: str,
        session: str,
        op_name: str,
        kernel: str,
        result: Any,
        total: int,
    ) -> None:
        """Re-run the snapshot so a better same-impl speedup can promote.

        The perf file is only recorded when the benchmarked kernel IS the
        staged verified impl — binding another kernel's speedup into the
        staged impl's metrics is exactly the cross-contamination the
        original mtime gate exists to prevent.
        """
        ws = self.workspace(task_id, session)
        verify_dir = ws / "output" / "verify"
        verify_path = verify_dir / "verify_result.json"
        staged = verify_dir / f"{op_name}_{self.IMPL_SUFFIX}.py"
        if not verify_path.is_file() or not staged.is_file():
            logger.info(
                "[best-snapshot] benchmark skipped: no verified impl staged for %s",
                op_name,
            )
            return
        try:
            staged_code = staged.read_text(encoding="utf-8")
        except OSError:
            staged_code = None
        if staged_code != kernel:
            logger.info(
                "[best-snapshot] benchmark skipped: kernel differs from the"
                " staged verified impl for %s",
                op_name,
            )
            return
        self._write_json(
            ws / "output" / "perf_result.json",
            self._perf_result(op_name, result, total),
        )
        await self._snapshot(op_name, ws)

    def best_block(
        self, task_id: str, session: str, op_name: str
    ) -> dict[str, Any] | None:
        """The compact session-best record for the observation (additive)."""
        ws = self.workspace(task_id, session)
        metrics = self._read_json(ws / "metrics_best.json")
        if metrics is None:
            return None
        impl_path = ws / "src" / f"{op_name}_triton_ascend_impl_best.py"
        impl_code = None
        if impl_path.is_file():
            try:
                impl_code = impl_path.read_text(encoding="utf-8")
            except OSError:
                impl_code = None
        return {
            "op_name": op_name,
            "impl_name": impl_path.name,
            "passed_cases": metrics.get("passed_cases"),
            "total_cases": metrics.get("total_cases"),
            "correctness_ok": bool(metrics.get("correctness_ok")),
            "compile_ok": bool(metrics.get("compile_ok")),
            "speedup_vs_torch": metrics.get("speedup_vs_torch"),
            "reward": metrics.get("reward"),
            "metrics": metrics,
            "impl_code": impl_code,
        }


def _hash_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class NativeExecution:
    """One bridge-process executor over per-task cached runners."""

    def __init__(self, catalog_path: str, *, sweep: bool = True):
        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self._runners: dict[str, Any] = {}
        self._config = NativeToolchainConfig()
        self._best = BestSnapshotLayer(
            scripts_dir=os.environ.get(
                "NATIVE_SNAPSHOT_SCRIPTS_DIR", "/mnt/bridge-ram/toolchain-scripts"
            ),
            work_root=self._config.work_root,
        )
        if sweep:
            self._sweep_orphans()

    def _sweep_orphans(self) -> None:
        """Kill toolchain processes orphaned by a previous bridge lifetime.

        docker exec kills only the exec'd shell; claimed toolchain children
        survive a bridge restart, keep their flock FDs (the pool then never
        frees) and their NPU memory. Match strictly on OUR scripts dir so
        the grade service's processes (named TritonEval, different tree)
        are never touched, and clear stale work directories with it.
        """
        import subprocess

        scripts = self._config.scripts_dir
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self._config.container,
                    "bash",
                    "-lc",
                    f"pkill -9 -f '{scripts}' 2>/dev/null; "
                    f"rm -rf {self._config.work_root}/op_* 2>/dev/null; true",
                ],
                capture_output=True,
                timeout=120,
            )
        except Exception:  # noqa: BLE001 — sweep is best-effort at startup
            pass

    def _runner(self, task_id: str):
        if task_id not in self._runners:
            entry = self.catalog.get(task_id)
            if entry is None:
                raise KeyError(f"unknown task_id {task_id}")
            metadata = {
                "triton": {
                    "op_name": entry["op_name"],
                    "manifest_id": "npukernelbench-machine2-v1",
                    "problem_id": task_id,
                    "reference_file": entry["reference_path"],
                    "sidecar_file": entry.get("sidecar_path", ""),
                }
            }
            # The catalog carries absolute task-dir paths: resolve through
            # runner_for_task with the task root implied by those paths.
            reference = Path(entry["reference_path"])
            sidecar = Path(entry.get("sidecar_path") or "")
            if not reference.is_file():
                raise FileNotFoundError(f"reference missing: {reference}")
            if not sidecar.is_file():
                raise FileNotFoundError(f"sidecar missing: {sidecar}")
            config = self._config
            from harbor.integrations.triton.native_runner import NativeToolchainRunner

            self._runners[task_id] = NativeToolchainRunner(
                op_name=entry["op_name"],
                reference_path=reference,
                sidecar_path=sidecar,
                scripts_dir=config.scripts_dir,
                claim_script=config.claim_script,
                container=config.container,
                python=config.python,
                device_pool=config.device_pool,
                lock_dir=config.lock_dir,
                work_root=config.work_root,
                env_file=config.env_file,
                profile_script_path=config.profile_script,
                toolchain_timeout_sec=config.timeout_sec,
            )
        return self._runners[task_id]

    def case_count(self, task_id: str) -> int:
        entry = self.catalog.get(task_id)
        if entry is None:
            raise KeyError(f"unknown task_id {task_id}")
        return int(entry.get("cases") or 0)

    async def decoy_gate(self, task_id: str, kernel: str) -> bool | None:
        """The session server's Step 0: the AST regression gate.

        Returns True when the kernel is PROVEN a regression (decoy); None
        when the gate itself could not run (fall through to the engine,
        which surfaces a real error). Runs without a device claim — the
        check is pure AST analysis.
        """
        config = self._config
        import base64

        payload = base64.b64encode(kernel.encode()).decode()
        script = (
            f"printf %s '{payload}' | base64 -d > /tmp/decoy_gate_kernel.py && "
            f"{config.python} {config.scripts_dir}/validate_triton_impl.py"
            f" /tmp/decoy_gate_kernel.py --json"
        )
        argv = ["docker", "exec", config.container, "bash", "-lc", script]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), 120)
        except TimeoutError:
            return None
        text = stdout.decode(errors="replace")
        start = text.find("{")
        if start < 0:
            return None
        try:
            verdict = json.loads(text[start:])
        except ValueError:
            return None
        if verdict.get("valid") is False:
            return True
        if verdict.get("valid") is True:
            return False
        return None

    async def evaluate(
        self, task_id: str, kernel: str, action: str, session: str = ""
    ) -> dict[str, Any]:
        """One tool call -> the document the renderers consume."""
        total = self.case_count(task_id)
        if action in ("verify", "benchmark"):
            decoy = await self.decoy_gate(task_id, kernel)
            if decoy:
                return {
                    "correctness": {
                        "status": "succeeded",
                        "data": {
                            "decoy": True,
                            "outcome": "wrong",
                            "passed_cases": total,
                            "total_cases": total,
                            "failures": [],
                        },
                    }
                }
        runner = self._runner(task_id)
        kernel_bytes = kernel.encode("utf-8")
        operation = DeviceOperation(
            operation_id="bridge-"
            + _hash_hex(f"{task_id}:{action}:{kernel}".encode())[:24],
            session=f"bridge-{task_id}",
            action=action,
            purpose="interactive",
            kernel_sha256=_hash_hex(kernel_bytes),
            conditions_sha256=_hash_hex(b"bridge-interactive"),
            cases=tuple(range(total)),
        )
        result = await runner.run(operation, kernel_bytes)
        if action == "profile":
            return {"profile": {"status": "succeeded", "data": result.raw}}
        document = self._correctness_document(result, total)
        if action == "benchmark":
            document["performance"] = self._performance_document(result)
        best = await self._record_best(
            task_id, kernel, result, action, total, session=session
        )
        if best is not None:
            document["best"] = best
        return document

    async def _record_best(
        self,
        task_id: str,
        kernel: str,
        result: Any,
        action: str,
        total: int,
        *,
        session: str = "",
    ) -> dict[str, Any] | None:
        """Run the legacy best-snapshot layer; never fail the eval call.

        A decoy verdict reproduces the sandbox flow where the AST gate
        refuses BEFORE verify.py runs, so no verify artifact exists and no
        snapshot is taken.
        """
        entry = self.catalog.get(task_id) or {}
        op_name = str(entry.get("op_name") or task_id)
        layer = self._best
        try:
            async with layer.lock(task_id, session):
                if action == "verify":
                    await layer.record_verify(
                        task_id, session, op_name, kernel, result, total
                    )
                elif action == "benchmark":
                    await layer.record_benchmark(
                        task_id, session, op_name, kernel, result, total
                    )
                return layer.best_block(task_id, session, op_name)
        except Exception:  # noqa: BLE001 — bookkeeping must not fail measurement
            logger.warning("best-snapshot layer failed for %s", task_id, exc_info=True)
            return None

    def _correctness_document(self, result, total: int) -> dict:
        cases = list(result.cases)
        passed = sum(1 for case in cases if case.passed is True)
        raw_failures = list((result.raw or {}).get("failures") or [])
        kinds = sorted({case.failure for case in cases if case.failure})
        if passed == total and total > 0:
            outcome = "correct"
        elif len(kinds) == 1:
            outcome = _KIND_TO_OUTCOME.get(kinds[0], "wrong")
        else:
            outcome = "wrong"
        failures = [
            {
                "case_idx": item.get("case_idx"),
                "error_type": item.get("error_type")
                or _KIND_TO_TYPE.get(
                    next(
                        (
                            case.failure
                            for case in cases
                            if case.case_idx == item.get("case_idx")
                        ),
                        None,
                    ),
                    "RuntimeError",
                ),
                "error_msg": item.get("error_msg"),
            }
            for item in raw_failures
        ]
        return {
            "correctness": {
                "status": "succeeded",
                "data": {
                    "outcome": outcome,
                    "passed_cases": passed,
                    "total_cases": total,
                    "failures": failures,
                },
            }
        }

    def _performance_document(self, result) -> dict:
        speedups = []
        per_case = []
        for case in result.cases:
            speedup = case.speedup_vs_reference
            per_case.append(
                {
                    "case_idx": case.case_idx,
                    "reference_latency_ms": case.reference_latency_ms,
                    "implementation_latency_ms": case.implementation_latency_ms,
                }
            )
            if speedup and math.isfinite(speedup) and speedup > 0:
                speedups.append(speedup)
        overall = None
        if speedups:
            overall = math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            )
        return {
            "status": "succeeded",
            "data": {
                "overall_speedup_vs_reference": overall,
                "per_case": per_case,
                "selected_cases": len(per_case),
                "measured_cases": len(speedups),
            },
        }
