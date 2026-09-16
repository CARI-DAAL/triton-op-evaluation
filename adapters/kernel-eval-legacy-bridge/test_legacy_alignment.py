"""Alignment tests: bridge observation == original kernel_agent@52affcb9 output.

Run: python test_legacy_alignment.py

1. The ported renderer/classifier are byte-identical to the originals
   (vendored verbatim under reference/kernel-agent-52affcb9/).
2. v2.1 resources map onto verdicts with the original semantics.
3. End-to-end through the aiohttp adapter against a fake v2.1 upstream.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# The ORIGINAL modules, vendored verbatim (Apache-2.0, see reference/
# kernel-agent-52affcb9/LICENSE) and loaded by file path.
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

import legacy_contract as lc  # noqa: E402

_AGENT_ROOT = HERE / "reference" / "kernel-agent-52affcb9"


def _load(name: str, filename: str):
    spec = spec_from_file_location(name, _AGENT_ROOT / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_render = _load(
    "original_evaluation_observation", "evaluation_observation.py"
).render_evaluation_observation
original_classify = _load(
    "original_error_classify", "error_classify.py"
).classify_error

def check(name: str, actual: object, expected: object) -> None:
    assert actual == expected, f"{name}\n  actual:   {actual!r}\n  expected: {expected!r}"
    print(f"ok   {name}")


# ---------------------------------------------------------------------------
# 1. Ported renderer/classifier == originals, byte for byte
# ---------------------------------------------------------------------------


def test_renderer_port_is_verbatim() -> None:
    cases = [
        # (verdict fields..., action) — all envelope paths
        dict(action="verify", artifact_sha256="a" * 64, eval_budget="1/18",
             request_status="completed", infra_error=True, error="transport gone"),
        dict(action="benchmark", artifact_sha256="b" * 64, eval_budget="2/18",
             request_status="infra_error", error="upstream 500"),
        dict(action="verify", artifact_sha256="c" * 64, eval_budget="3/18",
             request_status="budget_exhausted"),
        dict(action="verify", artifact_sha256="d" * 64, eval_budget="4/18",
             request_status="completed", compiled=False,
             error_class="jit_compile_error", error="CompilationError: boom"),
        dict(action="benchmark", artifact_sha256="e" * 64, eval_budget="5/18",
             request_status="completed", compiled=True, decoy=True,
             error="decoy: torch passthrough detected"),
        dict(action="verify", artifact_sha256="f" * 64, eval_budget="6/18",
             request_status="completed", compiled=True, correct=False,
             error_class="numerical_mismatch", passed_cases=49, total_cases=51,
             error="[case 3] AccuracyError: Kernel produced incorrect results"),
        dict(action="verify", artifact_sha256="0" * 64, eval_budget="7/18",
             request_status="completed", compiled=True, correct=True,
             passed_cases=51, total_cases=51),
        dict(action="benchmark", artifact_sha256="1" * 64, eval_budget="8/18",
             request_status="completed", compiled=True, correct=True,
             passed_cases=51, total_cases=51, benchmark_available=False),
        dict(action="benchmark", artifact_sha256="2" * 64, eval_budget="9/18",
             request_status="completed", compiled=True, correct=True,
             passed_cases=51, total_cases=51, benchmark_available=True,
             speedup=1.2345678,
             benchmark={"speedup_vs_torch": 1.2345678, "per_shape_results": []}),
        dict(action="profile", artifact_sha256="3" * 64, eval_budget="10/18",
             request_status="completed", profile_state="partial",
             task_status="SUCCEEDED",
             result={"selected_cases": 3, "partial_cases": 1},
             error="profile partial"),
        dict(action="profile", artifact_sha256="4" * 64, eval_budget="11/18",
             request_status="completed", profile_state="failed",
             task_status="FAILED", result=None, error="profile failed"),
        dict(action="verify", artifact_sha256="5" * 64, eval_budget="12/18",
             request_status="completed", compiled=True, correct=False,
             error_class=None, passed_cases=0, total_cases=1,
             error="x" * 4000),  # exercise truncation
    ]
    for index, case in enumerate(cases):
        check(
            f"render_verbatim[{index}]",
            lc.render_evaluation_observation(case),
            original_render(case),
        )


def test_classifier_port_is_verbatim() -> None:
    texts = [
        None,
        "",
        "CompilationError: at triton/compiler/_compile.py",
        "min_dot_size assert failed",
        "Kernel execution failed: Traceback (most recent call last):",
        "Shape mismatch: (2, 3) vs (3, 2)",
        "Kernel produced incorrect results max_abs_err=0.5",
        "NaN position mismatch",
        "some totally unknown failure",
        "PtxasError: register pressure",
        "cuda error 700",
    ]
    for text in texts:
        check(f"classify[{text!r}]", lc.classify_error(text), original_classify(text))


# ---------------------------------------------------------------------------
# 2. v2.1 -> legacy verdict mapping
# ---------------------------------------------------------------------------

def render_mapped(result: dict, action: str = "verify") -> str:
    verdict = lc.verify_verdict(result, action)
    env = lc.envelope(verdict, request_status="completed", **ENVELOPE_WITH(action))
    return env["observation"]


def ENVELOPE_WITH(action: str) -> dict:  # noqa: N802 - small helper
    return dict(
        action=action,
        kernel_code="def kernel(): pass\n",
        eval_turn=1,
        eval_budget="1/18",
    )


SHA = hashlib.sha256(b"def kernel(): pass\n").hexdigest()


def test_v21_mapping() -> None:
    # correct verify
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "correct", "passed": True,
            "passed_cases": 51, "total_cases": 51, "failures": [],
        }},
    })
    check(
        "verify_correct",
        obs.splitlines()[-1],
        "[kernel_eval][verify][correct] cases=51/51",
    )
    check(
        "verify_correct_envelope",
        obs.splitlines()[:5],
        ["[kernel_eval]", "action=verify", f"artifact_sha256={SHA}",
         "eval_budget=1/18", "request_status=completed"],
    )

    # correct benchmark, measured
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "correct", "passed": True,
            "passed_cases": 51, "total_cases": 51, "failures": [],
        }},
        "performance": {"status": "succeeded", "data": {
            "aggregation": "geometric_mean_v1",
            "overall_speedup_vs_reference": 1.2345678,
            "selected_cases": 51, "measured_cases": 51, "coverage": 1.0,
            "per_case": [{
                "case_idx": 0,
                "input_shapes": [
                    {"input_index": 0, "dimensions": [128, 128], "dtype": "float16"},
                ],
                "implementation_latency_ms": 0.5,
                "reference_latency_ms": 0.9,
                "speedup_vs_reference": 1.8,
            }],
            "unmeasured_cases": [],
        }},
    }, action="benchmark")
    check(
        "benchmark_measured_line",
        obs.splitlines()[-2],
        "[kernel_eval][benchmark][measured] verify_cases=51/51 "
        "speedup_geomean=1.235x",
    )
    bench = json.loads(obs.splitlines()[-1].removeprefix("benchmark="))
    check(
        "benchmark_legacy_keys",
        sorted(bench),
        ["coverage", "passed_cases", "per_shape_results", "speedup_vs_torch",
         "total_cases"],
    )
    check("benchmark_speedup", bench["speedup_vs_torch"], 1.2345678)
    check(
        "benchmark_per_shape_legacy",
        {k: bench["per_shape_results"][0][k] for k in
         ("case_idx", "status", "speedup_vs_torch")},
        {"case_idx": 0, "status": "pass", "speedup_vs_torch": 1.8},
    )

    # correct benchmark, performance check skipped -> unavailable
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "correct", "passed": True,
            "passed_cases": 51, "total_cases": 51, "failures": [],
        }},
        "performance": {"status": "skipped", "reason": {
            "code": "DEPENDENCY_NOT_PASSED", "message": "x"}},
    }, action="benchmark")
    check(
        "benchmark_unavailable",
        obs.splitlines()[-1],
        "[kernel_eval][benchmark][unavailable] verify_cases=51/51",
    )

    # wrong with failures: folded text + legacy class
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "wrong", "passed": False,
            "passed_cases": 49, "total_cases": 51,
            "failures": [
                {"case_idx": 2, "kind": "value_mismatch",
                 "metrics": {"max_abs_diff": 0.5, "matched_ratio": 0.9},
                 "sample": []},
                {"case_idx": 7, "kind": "shape_mismatch", "metrics": {}},
            ],
        }},
    })
    lines = obs.splitlines()
    # Legacy classifier precedence on mixed failures: shape_mismatch is
    # checked before numerical_mismatch, so the class is shape_mismatch.
    check(
        "wrong_label_cases",
        lines[5],
        "[kernel_eval][verify][shape_mismatch] cases=49/51",
    )
    check(
        "wrong_folded_error",
        "\n".join([lines[6].removeprefix("error=")] + lines[7:]),
        "[case 2] AccuracyError: Kernel produced incorrect results; "
        '{"max_abs_diff": 0.5, "matched_ratio": 0.9}\n'
        "[case 7] Shape mismatch: Output shape differs from the reference shape",
    )

    # compile_error via compile check excerpt -> jit_compile_error label
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "compile_error", "passed": False,
            "passed_cases": 0, "total_cases": 51, "failures": [],
        }},
        "compile": {"status": "succeeded", "data": {
            "outcome": "compile_error",
            "error_excerpt": "triton/compiler/errors.py CompilationError: bad",
        }},
    })
    check(
        "compile_error_label",
        obs.splitlines()[5],
        "[kernel_eval][verify][jit_compile_error]",
    )

    # runtime_error domain failure with traceback -> runtime_error label
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "runtime_error", "passed": False,
            "passed_cases": 0, "total_cases": 51, "failures": [],
            "domain_failure": {
                "code": "SUBJECT_RUNTIME_ERROR", "phase": "execute",
                "message": "Traceback (most recent call last):\n  ValueError: x",
                "case_idx": 0,
            },
        }},
    })
    check(
        "runtime_error_label",
        obs.splitlines()[5],
        "[kernel_eval][verify][runtime_error] cases=0/51",
    )

    # execution_timeout -> runtime_error (legacy TIMEOUT parity)
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "execution_timeout", "passed": False,
            "passed_cases": 20, "total_cases": 51, "failures": [],
        }},
    })
    check(
        "execution_timeout_label",
        obs.splitlines()[5],
        "[kernel_eval][verify][runtime_error] cases=20/51",
    )

    # decoy: legacy fallback text
    obs = render_mapped({
        "correctness": {"status": "succeeded", "data": {
            "outcome": "wrong", "passed": False,
            "passed_cases": 0, "total_cases": 51, "failures": [],
            "decoy": True,
        }},
    })
    check(
        "decoy_lines",
        obs.splitlines()[5:],
        ["[kernel_eval][verify][decoy]",
         "error=decoy: torch passthrough detected"],
    )

    # correctness check failed -> infra envelope
    verdict = lc.verify_verdict({
        "correctness": {"status": "failed", "error": {
            "code": "EVALUATION_FAILED", "message": "worker died",
            "phase": "execution", "check": "correctness", "retryable": False}},
    }, "verify")
    env = lc.envelope(verdict, request_status="infra_error", **ENVELOPE_WITH("verify"))
    check(
        "infra_envelope",
        env["observation"].splitlines(),
        ["[kernel_eval]", "action=verify", f"artifact_sha256={SHA}",
         "eval_budget=1/18", "request_status=infra_error",
         "error=EVALUATION_FAILED: worker died"],
    )

    # correctness check timeout -> runtime_error
    verdict = lc.verify_verdict({
        "correctness": {"status": "timeout", "error": {
            "code": "CHECK_TIMEOUT", "message": "budget exceeded",
            "phase": "execution", "check": "correctness", "retryable": False}},
    }, "verify")
    check("check_timeout_class", verdict["error_class"], "runtime_error")
    check("check_timeout_compiled", verdict["compiled"], True)

    # profile: complete / partial / all-failed / timeout / failed check
    complete = lc.profile_verdict({"status": "succeeded", "data": {
        "mode": "op",
        "counts": {"selected": 3, "complete": 3, "partial": 0, "failed": 0},
        "cases": [{"case_idx": 0, "status": "complete",
                   "kernels": [{"op_name": "k", "time_ms": 1.0, "percentage": 90.0}]}],
    }})
    check("profile_complete_state", complete["profile_state"], "complete")
    env = lc.envelope({**complete}, request_status="completed",
                      **ENVELOPE_WITH("profile"))
    check(
        "profile_complete_render",
        env["observation"].splitlines()[5],
        "[kernel_eval][profile][complete] task_status=SUCCEEDED",
    )
    check(
        "profile_result_line",
        env["observation"].splitlines()[6].startswith("result="),
        True,
    )

    partial = lc.profile_verdict({"status": "succeeded", "data": {
        "mode": "op",
        "counts": {"selected": 3, "complete": 1, "partial": 1, "failed": 1},
        "cases": [],
    }})
    check("profile_partial_state", partial["profile_state"], "partial")

    failed = lc.profile_verdict({"status": "succeeded", "data": {
        "mode": "op",
        "counts": {"selected": 3, "complete": 0, "partial": 0, "failed": 3},
        "cases": [],
    }})
    check("profile_all_failed_state", failed["profile_state"], "failed")

    timeout = lc.profile_verdict({"status": "timeout", "error": {
        "code": "CHECK_TIMEOUT", "message": "too slow",
        "phase": "execution", "check": "profile", "retryable": False}})
    check("profile_timeout", (timeout["profile_state"], timeout["task_status"]),
          ("timeout", "TIMEOUT"))

    rejected = lc.profile_verdict({"status": "skipped", "reason": {
        "code": "RESOURCE_DEADLINE", "message": "no room"}})
    check("profile_rejected", (rejected["profile_state"], rejected["infra_error"]),
          ("infra_error", True))

    # cross-check: mapped verdicts render identically through the ORIGINAL
    # renderer (proves the envelope we build is exactly what the legacy
    # session server would have emitted for the same verdict).
    for name, verdict, action, status in [
        ("correct", lc.verify_verdict({
            "correctness": {"status": "succeeded", "data": {
                "outcome": "correct", "passed": True,
                "passed_cases": 51, "total_cases": 51, "failures": []}},
        }, "benchmark"), "benchmark", "completed"),
        ("wrong", lc.verify_verdict({
            "correctness": {"status": "succeeded", "data": {
                "outcome": "wrong", "passed": False,
                "passed_cases": 1, "total_cases": 2,
                "failures": [{"case_idx": 1, "kind": "value_mismatch",
                              "metrics": {"max_abs_diff": 1.0}}]}},
        }, "verify"), "verify", "completed"),
    ]:
        ours = lc.envelope(verdict, request_status=status, **ENVELOPE_WITH(action))
        theirs = original_render({**verdict, **{
            k: ENVELOPE_WITH(action)[k] for k in
            ("action", "eval_turn", "eval_budget")
        }, "artifact_sha256": SHA, "request_status": status})
        check(f"cross_render[{name}]", ours["observation"], theirs)


# ---------------------------------------------------------------------------
# 3. End-to-end through the adapter with a fake upstream
# ---------------------------------------------------------------------------


def resource(result: dict, state: str = "completed") -> dict:
    return {
        "schema_version": "2.1",
        "evaluation_id": "eval-e2e",
        "revision": 1,
        "state": state,
        "created_at": "2026-09-16T00:00:00Z",
        "updated_at": "2026-09-16T00:00:00Z",
        "started_at": "2026-09-16T00:00:00Z",
        "finished_at": "2026-09-16T00:00:00Z",
        "deadline_at": "2026-09-16T00:00:00Z",
        "request_digest": "0" * 64,
        "reference_digest": "0" * 64,
        "result": result,
    }


async def run_adapter_e2e() -> None:
    import bridge as bridge_mod
    from aiohttp import ClientSession, web

    correctness = {"status": "succeeded", "data": {
        "outcome": "correct", "passed": True,
        "passed_cases": 2, "total_cases": 2, "failures": []}}
    evaluation_result = {
        "correctness": correctness,
        "performance": None,
        "profile": None,
        "provenance": {"schema_version": "2.1"},
    }
    submitted = {}

    async def fake_upstream(request: web.Request) -> web.Response:
        if request.path == "/evaluations" and request.method == "POST":
            submitted["body"] = await request.json()
            submitted["key"] = request.headers.get("Idempotency-Key")
            return web.json_response(resource(evaluation_result), status=201)
        raise AssertionError(request.path)

    upstream_app = web.Application()
    upstream_app.router.add_post("/evaluations", fake_upstream)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, "127.0.0.1", 0)
    await upstream_site.start()
    upstream_port = upstream_site._server.sockets[0].getsockname()[1]

    settings = bridge_mod.Settings.__new__(bridge_mod.Settings)
    settings.upstream = f"http://127.0.0.1:{upstream_port}"
    settings.token = "bridge-tok"
    settings.upstream_token = "up-tok"
    settings.max_wait = 30
    settings.max_eval_turns = 2
    settings.perf_params = {"warmup": 5, "repeats": 20, "method": "profiler"}
    catalog = bridge_mod.TaskCatalog.__new__(bridge_mod.TaskCatalog)
    catalog.tasks = {"t1": {"op_name": "op1", "reference": "REF"}}

    adapter_app = bridge_mod.build_interaction_app(settings, catalog)
    adapter_runner = web.AppRunner(adapter_app)
    await adapter_runner.setup()
    adapter_site = web.TCPSite(adapter_runner, "127.0.0.1", 0)
    await adapter_site.start()
    port = adapter_site._server.sockets[0].getsockname()[1]

    async with ClientSession() as client:
        headers = {
            "Authorization": "Bearer bridge-tok",
            "Content-Type": "application/json",
        }
        body = {"task_id": "t1", "kernel_code": "def kernel(): pass\n"}

        # 401 without token
        async with client.post(f"http://127.0.0.1:{port}/verify", json=body) as r:
            check("e2e_401", (r.status, await r.json()), (401, {"error": "unauthorized"}))

        # 404 unknown action
        async with client.post(
            f"http://127.0.0.1:{port}/explode", json=body, headers=headers
        ) as r:
            check("e2e_404", (r.status, await r.json()), (404, {"error": "not found"}))

        # 403 unknown task
        async with client.post(
            f"http://127.0.0.1:{port}/verify",
            json={"task_id": "nope", "kernel_code": "x"}, headers=headers,
        ) as r:
            check("e2e_403", (r.status, await r.json()),
                  (403, {"error": "task_id mismatch"}))

        # 400 reference fields forbidden
        async with client.post(
            f"http://127.0.0.1:{port}/verify",
            json={"task_id": "t1", "kernel_code": "x", "reference_code": "R"},
            headers=headers,
        ) as r:
            check("e2e_400_reference", (r.status, await r.json()),
                  (400, {"error": "reference fields are forbidden"}))

        # 400 invalid profile options
        async with client.post(
            f"http://127.0.0.1:{port}/profile",
            json={"task_id": "t1", "kernel_code": "x", "profile_mode": "weird"},
            headers=headers,
        ) as r:
            check("e2e_400_profile", (r.status, await r.json()),
                  (400, {"error": "invalid profile options"}))

        # verify: full envelope + verdict line
        async with client.post(
            f"http://127.0.0.1:{port}/verify", json=body, headers=headers
        ) as r:
            payload = await r.json()
        expected_sha = hashlib.sha256(b"def kernel(): pass\n").hexdigest()
        check(
            "e2e_verify_observation",
            payload["observation"],
            "\n".join([
                "[kernel_eval]",
                "action=verify",
                f"artifact_sha256={expected_sha}",
                "eval_budget=1/2",
                "request_status=completed",
                "[kernel_eval][verify][correct] cases=2/2",
            ]),
        )
        check("e2e_request_status", payload["request_status"], "completed")
        check("e2e_verdict_fields",
              (payload["correct"], payload["compiled"], payload["speedup"]),
              (True, True, 0.0))
        check(
            "e2e_upstream_payload",
            (submitted["body"]["checks"], submitted["body"]["op_name"],
             submitted["body"]["kernel_code"]),
            (["correctness"], "op1", "def kernel(): pass\n"),
        )
        check("e2e_idempotency_key", len(submitted["key"]), 64)

        # budget exhaustion on the third call (max 2)
        async with client.post(
            f"http://127.0.0.1:{port}/verify", json=body, headers=headers
        ) as r:
            second = await r.json()
        check("e2e_second_budget", second["eval_budget"], "2/2")
        async with client.post(
            f"http://127.0.0.1:{port}/verify", json=body, headers=headers
        ) as r:
            third = await r.json()
        check("e2e_budget_exhausted_status", third["request_status"],
              "budget_exhausted")
        check(
            "e2e_budget_exhausted_observation",
            third["observation"],
            "\n".join([
                "[kernel_eval]",
                "action=verify",
                f"artifact_sha256={expected_sha}",
                "eval_budget=2/2",
                "request_status=budget_exhausted",
                "feedback=interactive evaluation budget exhausted; "
                "no evaluator request started",
            ]),
        )

    await adapter_runner.cleanup()
    await upstream_site.stop()
    await upstream_runner.cleanup()


def main() -> None:
    test_renderer_port_is_verbatim()
    test_classifier_port_is_verbatim()
    test_v21_mapping()
    asyncio.run(run_adapter_e2e())
    print("\nALL ALIGNMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
