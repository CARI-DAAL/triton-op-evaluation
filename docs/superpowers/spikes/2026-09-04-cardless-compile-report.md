# Cardless Compile Spike Report — 2026-09-04

Executed on the 910B1 NPU host (docker `triton-eval:1.1`, triton-ascend 3.2.1 /
triton 3.5.0 / torch_npu 2.9.0, CANN 9.0.0.0430). Raw data: `spike-report/cardless.json`,
`spike-report/oncard_cache.json`, `spike-report/oncard_classify.json` (host-side,
gitignored; full contents reproduced in the appendices of this memo).

## Verdict: compile_stage = on_card

| Criterion (spec §8) | Result | Evidence |
|---|---|---|
| 1. cardless compile works (≥3 modules, no /dev/davinci*) | ✅ (with discovered workarounds) | cardless.json env_facts (`davinci_nodes: []`, `npu_available: false`) + 3 good modules `ok: true` via `ast_source` |
| 2. on-card cache hit rate (>50% target, >80% stretch) | **0%** (production-shaped warmup path); **100%** with matched `ast_source` path | oncard_cache.json `hit_rate: 0.0`; follow-up `ast_hit_check` on-card: `new_files: 0` for all fns, `add_kernel` 3.91 ms vs 2 464 ms cold |
| 3. failure classification consistency | 3/4 consistent; 1 corpus defect | oncard_classify.json table below |
| 4. compile wall-time distribution | p50 ≈ 2.17 s / p95 ≈ 2.46 s (cold cardless compiles: 2464.84 / 2173.4 / 739.86 ms) | cardless.json `compile_wall_ms` |

## Working compile strategy: ast_source (+ arch shim)

`warmup` is **dead cardless**: `JITFunction.warmup()` → `driver.active.get_current_device()`
→ `aclInit` → driver error 507899 (no device). The `ast_source` strategy works after three
version-3.2.1 discoveries (fixes now in `scripts/spike/cardless_compile.py`):

1. `triton.backends.backends["ascend"].driver` is the **class** — instantiate before
   `get_current_target()` (was `TypeError: missing 'self'`).
2. triton 3.5 renamed the ASTSource kwarg: `constexprs` → `constants`.
3. The compile pipeline fetches the SoC arch from the **device**
   (`compiler.py:519` → `NPUUtils().get_arch()` → `rtGetSocVersion`) even for pure
   compilation, and `NPUUtils.__init__` reads `TRITON_ASCEND_ARCH` but never uses it
   (dead code). A shim (`NPUUtils.get_arch = env`) makes compilation cardless; the
   bisheng compile subprocesses themselves never touch a device.

## Classification table

| kernel | cardless stage | on-card outcome | Plan D action |
|---|---|---|---|
| fail_syntax | syntax | wrong | none |
| fail_import_dep | import | wrong | none |
| fail_frontend | compile_failed | runtime_error | attribution widening: cardless says compile, service says execute-phase runtime_error (predicted gap) |
| fail_backend | compile_failed | **correct** | corpus defect: `num_warps=999` lives at the call site, so the ast_source path never applies it — and triton-ascend 3.2.1 accepts it on-card anyway. Replace with a genuinely backend-invalid kernel before relying on this row |

## Decision rationale

Compile itself is fully cardless-capable (criterion 1 ✅ after the three discovered fixes),
and the artifacts are byte-identical enough to be reused on-card **when both sides use the
same `ast_source` call shape** (100% reuse, ms-scale hits). But criterion 2 fails as
measured through the production-shaped path: a warmup/JIT compile on-card writes entirely
new cache entries (`hit_rate: 0.0`, warm==cold wall-time), because JIT-path cache keys
carry launch specialization/attrs that the ASTSource path does not. Until the cache keys
converge (upstream work), cardless pre-compilation saves the service nothing, so the gate
lands on `on_card`. Revisit `cpu_lane` if: (a) upstream wires `TRITON_ASCEND_ARCH` into
`NPUUtils.get_arch` (2-line fix), and (b) the JIT/ast_source cache-key divergence is closed
or Plan D adopts ast_source-shaped compilation on-card.

## Appendix: execution environment deltas from the plan

- No docker compose on this host (docker 18.09): both stacks translated 1:1 to `docker run`
  (`--runtime=runc` for the cardless leg — the host's ascend default runtime must not
  inject devices into the cardless container; card 4 substituted for the plan's davinci0,
  which is occupied).
- Spike-report/cache files are root-owned after container runs (`sudo chown` to read).
- The classify leg ran against the production service at `127.0.0.1:10907` (cards 4-7),
  not the plan's npu0 smoke stack at :11908.

---

## Appendix A: spike-report/cardless.json
*(attempt.error fields truncated to the decisive lines; raw file on host has full text)*

```json
{
 "env_facts": {
  "davinci_nodes": [],
  "torch": "2.9.0",
  "torch-npu": "2.9.0",
  "triton": "3.5.0",
  "triton-ascend": "3.2.1",
  "npu_available": false,
  "triton_backends": [
   "ascend"
  ]
 },
 "modules": {
  "good_add": {
   "expect": "ok",
   "stage": "compiled",
   "ok": true,
   "attempts": [
    {
     "fn": "add_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...2769 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "add_kernel",
     "strategy": "ast_source",
     "wall_ms": 2464.84,
     "ok": true
    }
   ],
   "compile_wall_ms": 2464.84,
   "cache_files_delta": 6
  },
  "good_multi": {
   "expect": "ok",
   "stage": "compiled",
   "ok": true,
   "attempts": [
    {
     "fn": "add_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...3978 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "add_kernel",
     "strategy": "ast_source",
     "wall_ms": 2.5,
     "ok": true
    },
    {
     "fn": "scale_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...5187 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "scale_kernel",
     "strategy": "ast_source",
     "wall_ms": 2173.44,
     "ok": true
    }
   ],
   "compile_wall_ms": 2175.94,
   "cache_files_delta": 5
  },
  "good_autotune": {
   "expect": "ok",
   "stage": "compiled",
   "ok": true,
   "attempts": [
    {
     "fn": "matmul_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...6396 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "matmul_kernel",
     "strategy": "ast_source",
     "wall_ms": 739.86,
     "ok": true
    }
   ],
   "compile_wall_ms": 739.86,
   "cache_files_delta": 5
  },
  "fail_syntax": {
   "expect": "syntax",
   "stage": "syntax",
   "ok": true,
   "attempts": []
  },
  "fail_import_dep": {
   "expect": "import",
   "stage": "import",
   "ok": true,
   "attempts": [],
   "error": "Traceback (most recent call last):\n  File \"/spike/cardless_compile.py\", line 143, in run_module\n    module = import_module_from(name, src_path)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/spike/cardless_compile.py\", line 121, in import_module_from\n    spec.loader.exec_module(module)\n  File \"<frozen importlib._bootstrap_external>\", 940, in exec_module\nModuleNotFoundError: No module named 'definitely_not_a_real_pkg_xyz'\n"
  },
  "fail_frontend": {
   "expect": "frontend",
   "stage": "compile_failed",
   "ok": true,
   "attempts": [
    {
     "fn": "bad_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...7605 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "bad_kernel",
     "strategy": "ast_source",
     "ok": false,
     "error": "triton.compiler.errors.CompilationError: at 2:14:\nNameError('undefined_name is not defined')\n<...1398 chars total — full text in spike-report/cardless.json>"
    }
   ],
   "compile_wall_ms": 0.0,
   "cache_files_delta": 0
  },
  "fail_backend": {
   "expect": "any_compile",
   "stage": "compile_failed",
   "ok": false,
   "attempts": [
    {
     "fn": "warp999_kernel",
     "strategy": "warmup",
     "ok": false,
     "error": "        Get device cnt failed, retCode=0x7020010[FUNC:CheckDeviceIdIsValid][FILE:api_error.cc][LINE:4791]\n        drv devId is invalid, drv devId=0, retCode=0x7020010[FUNC:SetDefaultDeviceId][FILE:api_error.cc][LINE:5436]\n        rtSetDefaultDeviceId execution failed, reason=driver error:internal error[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        set default device id failed, ret:507899[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n        [Process][DefaultDevice]process HandleDefaultDevice failed[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:132]\n        The argument is invalid.Reason: rtGetDevMsg execution failed, the feature is not supported.\n<...8814 chars total — full text in spike-report/cardless.json>"
    },
    {
     "fn": "warp999_kernel",
     "strategy": "ast_source",
     "wall_ms": 2393.37,
     "ok": true
    }
   ],
   "compile_wall_ms": 2393.37,
   "cache_files_delta": 5
  }
 },
 "all_ok": false
}
```

## Appendix B: spike-report/oncard_cache.json

```json
{
 "modules": {
  "good_add": {
   "warm_ms": 28698.34,
   "warm_new_files": 9,
   "cold_ms": 30028.68,
   "cold_new_files": 10,
   "hit": false
  },
  "good_multi": {
   "warm_ms": 6473.3,
   "warm_new_files": 6,
   "cold_ms": 6228.15,
   "cold_new_files": 6,
   "hit": false
  },
  "good_autotune": {
   "warm_ms": 5688.21,
   "warm_new_files": 11,
   "cold_ms": 5705.32,
   "cold_new_files": 11,
   "hit": false
  }
 },
 "hit_rate": 0.0
}
```

## Appendix C: spike-report/oncard_classify.json

```json
{
 "modules": {
  "fail_syntax": {
   "state": "completed",
   "oncard_outcome": "wrong",
   "evaluation_id": "ev_ff9d38a08b6e4ce4b6cac5f4b7c28382",
   "cardless_stage": "syntax",
   "consistent": true
  },
  "fail_import_dep": {
   "state": "completed",
   "oncard_outcome": "wrong",
   "evaluation_id": "ev_0162c28ec49a4e23a47046db38dcb2c6",
   "cardless_stage": "import",
   "consistent": true
  },
  "fail_frontend": {
   "state": "completed",
   "oncard_outcome": "runtime_error",
   "evaluation_id": "ev_254e0342a3af4d7c90ca1412ac16fffa",
   "cardless_stage": "compile_failed",
   "consistent": true
  },
  "fail_backend": {
   "state": "completed",
   "oncard_outcome": "correct",
   "evaluation_id": "ev_272fe35dafed4c6ebacc90ef5e95a8a7",
   "cardless_stage": "compile_failed",
   "consistent": false
  }
 }
}
```
