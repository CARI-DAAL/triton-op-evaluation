# Cardless triton-ascend Compile Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically answer "can triton-ascend compile kernels without any NPU device, and can the on-card execution reuse those cached artifacts?" — the hard gate (spec §8) that decides `compile_stage=cpu_lane` vs `on_card` for the compile-prefilter plan.

**Architecture:** Throwaway spike code under `scripts/spike/` (stdlib-only, never imported by the service). A cardless Docker stack (no `/dev/davinci*`) runs a compile probe against an embedded kernel corpus; a second stack with one device measures cache-hit on card; a classification runner posts failure kernels to the existing npu0 smoke service. Outputs are JSON reports + a human gate-decision memo.

**Tech Stack:** Python 3.11 (kernelgym conda env on NPU host), triton 3.5.0 / triton-ascend 3.2.0, docker compose, stdlib only (no new deps).

**Spec:** `docs/superpowers/specs/2026-09-03-kernel-eval-ascend-optimization-design.md` (§8 spike acceptance criteria are the checklist this plan operationalizes)

**Plan map (spec decomposition):** this is **Plan A**. Plan B (rollout quota class + strict-priority dispatch), Plan C (card health loop + poison quarantine), Plan D (compile lane + caches — **shape depends on this spike's verdict**), Plan E (event timing + v2.1 contract) get their own plan files when started.

## Global Constraints

- Spike code is THROWAWAY: every file under `scripts/spike/` carries a `THROWAWAY spike` header comment; nothing in `triton_evaluation/`, `contracts/`, `tests/`, or prod `docker-compose*.yml` is modified.
- Stdlib only in spike scripts (no pyyaml, no requests; HTTP via `urllib` like `scripts/test_eval_service.py`).
- Ruff applies to `scripts/` (line-length 100, `target-version = "py310"`); engine-style Chinese comments are fine, matching repo tone.
- NPU-host runs use the digest-pinned image via `TRITON_EVAL_IMAGE` (same env var as prod compose) and bypass `scripts/compose.sh` (its preflight enforces prod invariants that don't apply to the spike stack).
- Spike output dirs `spike-cache/` and `spike-report/` are gitignored.
- Exit-code conventions follow repo scripts: 0 success / 1 expectation-mismatch / 2 usage-or-environment violation.
- Dev machine (no torch_npu) can only run `--self-test` and `py_compile` checks; hardware steps run on the NPU host.

---

### Task 1: Spike corpus + self-test

**Files:**
- Create: `scripts/spike/corpus.py`
- Create: `scripts/spike/selftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by Tasks 2 & 4):
  - `MODULES: dict[str, dict]` — keys `source: str`, `expect: "ok"|"syntax"|"import"|"frontend"|"any_compile"`.
  - `SIGNATURES: dict[str, dict[str, list[tuple[str, str]]]]` — module → jit-fn → `[(arg_name, kind)]`, kind ∈ `tensor:fp32 | tensor:fp16 | scalar:i32 | scalar:fp32 | constexpr:<int>`.
  - `EXTRA_OPTIONS: dict[str, dict]` — module → extra compile kwargs (e.g. `{"num_warps": 999}`).

- [ ] **Step 1: Write the failing self-test**

`scripts/spike/selftest.py`:

```python
#!/usr/bin/env python3
"""THROWAWAY spike: corpus integrity self-test. Stdlib only; runs on any machine."""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    try:
        from corpus import EXTRA_OPTIONS, MODULES, SIGNATURES
    except ImportError as exc:
        print(f"FAIL: cannot import corpus: {exc}")
        return 1

    failures = []
    for name, entry in MODULES.items():
        src, expect = entry["source"], entry["expect"]
        try:
            tree = ast.parse(src)
        except SyntaxError:
            if expect != "syntax":
                failures.append(f"{name}: unexpected SyntaxError (expect={expect})")
            continue
        if expect == "syntax":
            failures.append(f"{name}: expected SyntaxError but parsed cleanly")
            continue
        if expect == "ok":
            if "ModelNew" not in src:
                failures.append(f"{name}: ok module missing ModelNew")
            if "@triton.jit" not in src:
                failures.append(f"{name}: ok module missing @triton.jit")
        fn_args = {
            n.name: {a.arg for a in n.args.args}
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        }
        for fn_name, spec in SIGNATURES.get(name, {}).items():
            if fn_name not in fn_args:
                failures.append(f"{name}: signature spec for unknown fn {fn_name}")
                continue
            for arg_name, kind in spec:
                if arg_name not in fn_args[fn_name]:
                    failures.append(f"{name}.{fn_name}: spec arg {arg_name} not in def")
                if kind.startswith("constexpr:") and not kind.split(":")[1].isdigit():
                    failures.append(f"{name}.{fn_name}: bad constexpr kind {kind}")
    for name in EXTRA_OPTIONS:
        if name not in MODULES:
            failures.append(f"EXTRA_OPTIONS references unknown module {name}")

    for msg in failures:
        print(f"FAIL: {msg}")
    print(f"selftest: {len(MODULES)} modules, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it — verify it fails**

Run: `python3 scripts/spike/selftest.py`
Expected: exit 1, `FAIL: cannot import corpus: No module named 'corpus'`

- [ ] **Step 3: Write the corpus**

`scripts/spike/corpus.py` (THROWAWAY header; four sections below are the full file content):

Header + good modules:

```python
"""THROWAWAY spike: kernel corpus for the cardless-compile probe.

Modules follow the service's KernelBench-style contract: ModelNew(nn.Module)
+ get_input_groups(). SIGNATURES describe each @triton.jit function's launch
signature for device-free compilation; scalar values are divisible by 16 on
purpose (canonical specialization, spec §8).
"""

GOOD_ADD = '''import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_input_groups():
    return [
        [torch.randn(4096), torch.randn(4096)],
        [torch.randn(8192), torch.randn(8192)],
    ]
'''

GOOD_MULTI = '''import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

@triton.jit
def scale_kernel(x_ptr, output_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * scale, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        tmp = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, tmp, n, BLOCK_SIZE=1024)
        out = torch.empty_like(x)
        scale_kernel[grid](tmp, out, n, 2.0, BLOCK_SIZE=1024)
        return out

def get_input_groups():
    return [[torch.randn(4096), torch.randn(4096)]]
'''

GOOD_AUTOTUNE = '''import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc.to(tl.float16))

class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                             triton.cdiv(N, meta["BLOCK_N"]))
        matmul_kernel[grid](a, b, c, M, N, K)
        return c

def get_input_groups():
    return [[torch.randn(256, 256, dtype=torch.float16),
             torch.randn(256, 256, dtype=torch.float16)]]
'''
```

Failure modules + tables:

```python
FAIL_SYNTAX = '''import torch

def broken(:
    return 1
'''

FAIL_IMPORT_DEP = '''import definitely_not_a_real_pkg_xyz  # noqa: F401
import torch

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x
'''

FAIL_FRONTEND = '''import torch
import triton
import triton.language as tl

@triton.jit
def bad_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = undefined_name + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, tl.load(x_ptr + offsets))

class ModelNew(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        bad_kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
'''

FAIL_BACKEND = GOOD_ADD.replace(
    "BLOCK_SIZE=1024)", "BLOCK_SIZE=1024, num_warps=999)", 1
).replace("add_kernel", "warp999_kernel")

MODULES = {
    "good_add": {"source": GOOD_ADD, "expect": "ok"},
    "good_multi": {"source": GOOD_MULTI, "expect": "ok"},
    "good_autotune": {"source": GOOD_AUTOTUNE, "expect": "ok"},
    "fail_syntax": {"source": FAIL_SYNTAX, "expect": "syntax"},
    "fail_import_dep": {"source": FAIL_IMPORT_DEP, "expect": "import"},
    "fail_frontend": {"source": FAIL_FRONTEND, "expect": "frontend"},
    "fail_backend": {"source": FAIL_BACKEND, "expect": "any_compile"},
}

SIGNATURES = {
    "good_add": {"add_kernel": [
        ("x_ptr", "tensor:fp32"), ("y_ptr", "tensor:fp32"),
        ("output_ptr", "tensor:fp32"), ("n_elements", "scalar:i32"),
        ("BLOCK_SIZE", "constexpr:1024"),
    ]},
    "good_multi": {
        "add_kernel": [
            ("x_ptr", "tensor:fp32"), ("y_ptr", "tensor:fp32"),
            ("output_ptr", "tensor:fp32"), ("n_elements", "scalar:i32"),
            ("BLOCK_SIZE", "constexpr:1024"),
        ],
        "scale_kernel": [
            ("x_ptr", "tensor:fp32"), ("output_ptr", "tensor:fp32"),
            ("n_elements", "scalar:i32"), ("scale", "scalar:fp32"),
            ("BLOCK_SIZE", "constexpr:1024"),
        ],
    },
    "good_autotune": {"matmul_kernel": [
        ("a_ptr", "tensor:fp16"), ("b_ptr", "tensor:fp16"),
        ("c_ptr", "tensor:fp16"), ("M", "scalar:i32"), ("N", "scalar:i32"),
        ("K", "scalar:i32"), ("BLOCK_M", "constexpr:64"),
        ("BLOCK_N", "constexpr:64"), ("BLOCK_K", "constexpr:32"),
    ]},
    "fail_frontend": {"bad_kernel": [
        ("x_ptr", "tensor:fp32"), ("output_ptr", "tensor:fp32"),
        ("n_elements", "scalar:i32"), ("BLOCK_SIZE", "constexpr:1024"),
    ]},
    "fail_backend": {"warp999_kernel": [
        ("x_ptr", "tensor:fp32"), ("y_ptr", "tensor:fp32"),
        ("output_ptr", "tensor:fp32"), ("n_elements", "scalar:i32"),
        ("BLOCK_SIZE", "constexpr:1024"),
    ]},
}

EXTRA_OPTIONS = {"fail_backend": {"num_warps": 999}}
```

- [ ] **Step 4: Run self-test — verify it passes**

Run: `python3 scripts/spike/selftest.py`
Expected: exit 0, `selftest: 7 modules, 0 failures`

- [ ] **Step 5: Lint + commit**

```bash
ruff check scripts/spike/ && ruff format --check scripts/spike/ || true
git add scripts/spike/corpus.py scripts/spike/selftest.py
git commit -m "spike: cardless-compile corpus + integrity self-test"
```

---

### Task 2: Cardless compile probe

**Files:**
- Create: `scripts/spike/cardless_compile.py`

**Interfaces:**
- Consumes: `MODULES`, `SIGNATURES`, `EXTRA_OPTIONS` from `scripts/spike/corpus.py`.
- Produces: JSON report at `--report` path with keys `env_facts`, `modules{name → {stage, ok, attempts[{strategy, ok, wall_ms, error}], cache_files_delta}}`, `compile_wall_ms` per module. Task 5's gate memo reads this report.

- [ ] **Step 1: Write the probe**

`scripts/spike/cardless_compile.py`:

```python
#!/usr/bin/env python3
"""THROWAWAY spike: compile triton-ascend kernels with NO NPU device.

Strategies tried per kernel, in order (first success wins, all recorded):
  S1 warmup      — JITFunction.warmup(*dtypes, grid=(1,))  (triton's own
                   device-free compile path; MockTensor wrap, data_ptr()==0)
  S2 ast_source  — triton.compile(ASTSource(...), target=<explicit backend target>)

Exit 0: every module matched its corpus expectation AND every "ok" module
compiled via >=1 strategy. Exit 1: expectation mismatch. Exit 2: environment
violation (an NPU device is visible — this run proves nothing).
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import EXTRA_OPTIONS, MODULES, SIGNATURES  # noqa: E402


def env_facts() -> dict:
    facts = {"davinci_nodes": sorted(p.name for p in Path("/dev").glob("davinci*"))}
    for dist in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            facts[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            facts[dist] = "not-installed"
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        facts["npu_available"] = bool(torch.npu.is_available())
    except Exception as exc:  # import itself must not kill the probe
        facts["npu_available"] = f"import-error: {exc}"
    try:
        import triton.backends
        facts["triton_backends"] = sorted(triton.backends.backends.keys())
    except Exception as exc:
        facts["triton_backends"] = f"error: {exc}"
    return facts


def count_cache_files(cache_dir: Path) -> int:
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file()) if cache_dir.exists() else 0


def build_args(spec):
    """Map signature spec to warmup-style args (torch dtypes for tensors)."""
    import torch

    args, kwargs = [], {}
    for name, kind in spec:
        family, _, value = kind.partition(":")
        if family == "tensor":
            args.append(getattr(torch, {"fp32": "float32", "fp16": "float16"}[value]))
        elif family == "scalar":
            args.append(4096 if value == "i32" else 2.0)  # 4096: divisible by 16
        elif family == "constexpr":
            kwargs[name] = int(value)
    return args, kwargs


def strategy_warmup(jit_fn, spec, extra):
    from triton.runtime.autotuner import Autotuner

    if isinstance(jit_fn, Autotuner):
        jit_fn = jit_fn.fn  # default config only; no sweep on CPU (spec §8)
    args, kwargs = build_args(spec)
    kwargs.update(extra)
    t0 = time.perf_counter()
    jit_fn.warmup(*args, grid=(1,), **kwargs)
    return time.perf_counter() - t0


def strategy_ast_source(jit_fn, spec, extra):
    import triton
    import triton.backends
    from triton.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner

    if isinstance(jit_fn, Autotuner):
        jit_fn = jit_fn.fn
    candidates = [n for n in triton.backends.backends
                  if n.lower() not in ("cuda", "nvidia", "hip", "amd", "cpu")]
    if not candidates:
        raise RuntimeError("no non-CUDA triton backend registered")
    target = triton.backends.backends[candidates[0]].driver.get_current_target()
    signature, constexprs = {}, {}
    for i, (name, kind) in enumerate(spec):
        family, _, value = kind.partition(":")
        if family == "tensor":
            signature[name] = {"fp32": "*fp32", "fp16": "*fp16"}[value]
        elif family == "scalar":
            signature[name] = "i32" if value == "i32" else "fp32"
        else:
            signature[name] = "constexpr"
            constexprs[name] = int(value)
    src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
    t0 = time.perf_counter()
    triton.compile(src, target=target, options=dict(extra or {}))
    return time.perf_counter() - t0


STRATEGIES = [("warmup", strategy_warmup), ("ast_source", strategy_ast_source)]


def import_module_from(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_jit_fns(module) -> dict:
    out = {}
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        return out
    for attr, obj in vars(module).items():
        if isinstance(obj, (JITFunction, Autotuner)):
            out[attr] = obj
    return out


def run_module(name, entry, workdir: Path, cache_dir: Path) -> dict:
    result = {"expect": entry["expect"], "stage": None, "ok": False, "attempts": []}
    src_path = workdir / f"{name}.py"
    src_path.write_text(entry["source"])
    try:
        module = import_module_from(name, src_path)
    except SyntaxError:
        result.update(stage="syntax", ok=entry["expect"] == "syntax")
        return result
    except Exception:
        result.update(stage="import", ok=entry["expect"] == "import",
                      error=traceback.format_exc(limit=3))
        return result

    jit_fns = find_jit_fns(module)
    if not jit_fns:
        result.update(stage="no_jit_fn", ok=False)
        return result

    all_ok = True
    for fn_name, jit_fn in jit_fns.items():
        spec = SIGNATURES.get(name, {}).get(fn_name)
        if spec is None:
            result["attempts"].append({"fn": fn_name, "strategy": None,
                                       "ok": False, "error": "no signature spec"})
            all_ok = False
            continue
        before = count_cache_files(cache_dir)
        compiled = False
        for strat_name, strat in STRATEGIES:
            attempt = {"fn": fn_name, "strategy": strat_name}
            try:
                attempt["wall_ms"] = round(strat(jit_fn, spec,
                                                 EXTRA_OPTIONS.get(name, {})) * 1e3, 2)
                attempt["ok"] = True
                compiled = True
            except Exception:
                attempt["ok"] = False
                attempt["error"] = traceback.format_exc(limit=5)
            result["attempts"].append(attempt)
        result.setdefault("compile_wall_ms", 0.0)
        result["compile_wall_ms"] += sum(
            a.get("wall_ms", 0.0) for a in result["attempts"] if a.get("ok"))
        result["cache_files_delta"] = count_cache_files(cache_dir) - before
        if entry["expect"] == "ok" and not compiled:
            all_ok = False
        if entry["expect"] in ("frontend", "any_compile") and compiled:
            all_ok = False  # failure corpus must NOT compile cleanly
    if entry["expect"] == "ok":
        result.update(stage="compiled", ok=all_ok)
    elif entry["expect"] in ("frontend", "any_compile"):
        result.update(stage="compile_failed", ok=all_ok)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/spike-cache")
    ap.add_argument("--report", default="/spike-report/cardless.json")
    ap.add_argument("--self-test", action="store_true",
                    help="corpus integrity only; no torch/triton imports")
    args = ap.parse_args()

    if args.self_test:
        import selftest
        return selftest.main()

    facts = env_facts()
    print(json.dumps({"env_facts": facts}, indent=2))
    if facts["davinci_nodes"] or facts["npu_available"] is True:
        print("ENV VIOLATION: an NPU device is visible; cardless result would be void")
        return 2

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)  # before triton compile paths run

    report = {"env_facts": facts, "modules": {}}
    with tempfile.TemporaryDirectory(prefix="spike_") as work:
        for name, entry in MODULES.items():
            report["modules"][name] = run_module(name, entry, Path(work), cache_dir)

    report["all_ok"] = all(m["ok"] for m in report["modules"].values())
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, out)
    print(json.dumps({n: {"stage": m["stage"], "ok": m["ok"],
                          "wall_ms": m.get("compile_wall_ms")}
                      for n, m in report["modules"].items()}, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dev-machine checks**

Run: `python3 -m py_compile scripts/spike/cardless_compile.py && python3 scripts/spike/cardless_compile.py --self-test`
Expected: exit 0, `selftest: 7 modules, 0 failures` (self-test path imports no torch/triton)

- [ ] **Step 3: Lint + commit**

```bash
ruff check scripts/spike/cardless_compile.py
git add scripts/spike/cardless_compile.py
git commit -m "spike: cardless triton-ascend compile probe (warmup + ast_source strategies)"
```

---

### Task 3: Spike docker stacks + runbook README

**Files:**
- Create: `docker-compose.spike-cardless.yml`
- Create: `docker-compose.spike-oncard.yml`
- Create: `scripts/spike/README.md`
- Modify: `.gitignore` (append two lines; create the file if absent)

**Interfaces:**
- Consumes: `scripts/spike/cardless_compile.py` (Task 2), `scripts/spike/oncard_cache_check.py` + `oncard_classify.py` (Task 4 — compose references land now, scripts verified in Task 4).
- Produces: runnable commands used by Task 5's runbook.

- [ ] **Step 1: Write the cardless compose file**

`docker-compose.spike-cardless.yml` — mirrors `docker-compose.npu0-test.yml` mounts/env but **zero devices**, no redis, no privileged, escape-hatch command. Note `TRITON_EVAL_SKIP_READINESS=1` (entrypoint runs readiness before `exec "$@"`):

```yaml
# THROWAWAY spike stack: cardless compile probe. NOT via scripts/compose.sh —
# prod preflight invariants (gateway, catalog, secret) do not apply here.
#   TRITON_EVAL_IMAGE=<digest> docker compose -f docker-compose.spike-cardless.yml up
services:
  spike-cardless:
    image: ${TRITON_EVAL_IMAGE:?TRITON_EVAL_IMAGE must be a sha256 digest reference}
    network_mode: host
    shm_size: "4gb"
    environment:
      TRITON_EVAL_SKIP_READINESS: "1"
      HOME: /tmp
      CANN_SET_ENV_PATH: /home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh
      ASCEND_DRIVER_PATH: /usr/local/Ascend/driver
    command:
      - /home/n84449292/miniconda3/envs/kernelgym/bin/python3
      - /spike/cardless_compile.py
      - --cache-dir
      - /spike-cache
      - --report
      - /spike-report/cardless.json
    volumes:
      - /home/n84449292/miniconda3:/home/n84449292/miniconda3:ro
      - /home/n84449292/.local:/home/n84449292/.local:ro
      - /home/canada_group_folder/CANN/9.0.0.0430:/home/canada_group_folder/CANN/9.0.0.0430
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
      - ./scripts/spike:/spike:ro
      - ./spike-cache:/spike-cache
      - ./spike-report:/spike-report
    # NO devices: section — that is the whole point of this stack.
```

- [ ] **Step 2: Write the on-card compose file**

`docker-compose.spike-oncard.yml` — same but mounts `/dev/davinci0` (+ manager/svm/hdc like npu0-test) and runs the cache-hit check:

```yaml
# THROWAWAY spike stack: on-card cache-hit measurement against the shared
# TRITON_CACHE_DIR populated by the cardless probe.
services:
  spike-oncard:
    image: ${TRITON_EVAL_IMAGE:?TRITON_EVAL_IMAGE must be a sha256 digest reference}
    network_mode: host
    privileged: true
    shm_size: "8gb"
    environment:
      TRITON_EVAL_SKIP_READINESS: "1"
      HOME: /tmp
      CANN_SET_ENV_PATH: /home/canada_group_folder/CANN/9.0.0.0430/ascend-toolkit/set_env.sh
      ASCEND_DRIVER_PATH: /usr/local/Ascend/driver
      ASCEND_RT_VISIBLE_DEVICES: "0"
    command:
      - /home/n84449292/miniconda3/envs/kernelgym/bin/python3
      - /spike/oncard_cache_check.py
      - --cache-dir
      - /spike-cache
      - --cold-cache-dir
      - /spike-cold-cache
      - --report
      - /spike-report/oncard_cache.json
    devices:
      - /dev/davinci_manager
      - /dev/devmm_svm
      - /dev/hisi_hdc
      - /dev/davinci0
    volumes:
      - /home/n84449292/miniconda3:/home/n84449292/miniconda3:ro
      - /home/n84449292/.local:/home/n84449292/.local:ro
      - /home/canada_group_folder/CANN/9.0.0.0430:/home/canada_group_folder/CANN/9.0.0.0430
      - /usr/local/dcmi:/usr/local/dcmi
      - /usr/local/Ascend/driver:/usr/local/Ascend/driver
      - /etc/ascend_install.info:/etc/ascend_install.info
      - ./scripts/spike:/spike:ro
      - ./spike-cache:/spike-cache
      - ./spike-cold-cache:/spike-cold-cache
      - ./spike-report:/spike-report
```

- [ ] **Step 3: Write the spike README (runbook) and gitignore entries**

`scripts/spike/README.md`:

```markdown
# THROWAWAY spike: cardless triton-ascend compile

Gate for `compile_stage=cpu_lane` (spec §8). Nothing here is production code.

## Run (on the NPU host, repo checkout)

```bash
mkdir -p spike-cache spike-cold-cache spike-report
export TRITON_EVAL_IMAGE=<pinned sha256 digest>

# 1. cardless compile probe (criteria 1, 4; half of 3)
docker compose -f docker-compose.spike-cardless.yml up --abort-on-container-exit
echo "exit=$?"   # 0 = all corpus expectations met cardless

# 2. on-card cache reuse (criterion 2)
docker compose -f docker-compose.spike-oncard.yml up --abort-on-container-exit

# 3. on-card failure classification via the npu0 smoke service (other half of 3)
python3 scripts/spike/oncard_classify.py \
  --url http://127.0.0.1:11908 --token <trial-token>

# 4. fill docs/superpowers/spikes/<date>-cardless-compile-report.md from
#    spike-report/*.json; record the compile_stage gate decision.
```
```

Append to `.gitignore` (create with just these lines if the file doesn't exist):

```
/spike-cache/
/spike-cold-cache/
/spike-report/
```

- [ ] **Step 4: Render check**

Run: `docker compose -f docker-compose.spike-cardless.yml config -q` (skip with a note if docker isn't on the dev machine)
Expected: exit 0 (YAML renders; `${TRITON_EVAL_IMAGE:?}` guard means set a dummy: `TRITON_EVAL_IMAGE=x docker compose ... config -q`)

- [ ] **Step 5: Commit**

```bash
git add docker-compose.spike-cardless.yml docker-compose.spike-oncard.yml \
        scripts/spike/README.md .gitignore
git commit -m "spike: cardless/oncard docker stacks + runbook"
```

---

### Task 4: On-card cache-hit + failure-classification runners

**Files:**
- Create: `scripts/spike/oncard_cache_check.py`
- Create: `scripts/spike/oncard_classify.py`

**Interfaces:**
- Consumes: `MODULES`, `SIGNATURES` from `scripts/spike/corpus.py`; the running npu0 smoke service at `--url` (classification runner only).
- Produces: `/spike-report/oncard_cache.json` with `modules{name → {warm_ms, cold_ms, warm_new_files, cold_new_files, hit}}`, `hit_rate`; `/spike-report/oncard_classify.json` with `modules{name → {cardless_stage, oncard_outcome, consistent}}`.

- [ ] **Step 1: Write the cache-hit checker**

`scripts/spike/oncard_cache_check.py`:

```python
#!/usr/bin/env python3
"""THROWAWAY spike: on-card cache-reuse measurement.

Parent re-execs one child per (module, cache-dir) so TRITON_CACHE_DIR is set
before triton initializes. hit := warm run adds ~no cache files AND is much
faster than the cold run (recompile avoided on card).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HIT_MAX_NEW_FILES = 2      # metadata churn tolerance
HIT_MAX_TIME_RATIO = 0.6   # warm first-forward < 60% of cold


def count_files(d: Path) -> int:
    return sum(1 for _ in d.rglob("*") if _.is_file()) if d.exists() else 0


def child(module_name: str, workdir: Path) -> dict:
    """Runs with TRITON_CACHE_DIR already in env (set by parent pre-exec)."""
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    from corpus import MODULES

    cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
    src = workdir / f"{module_name}.py"
    src.write_text(MODULES[module_name]["source"])
    spec = importlib.util.spec_from_file_location(module_name, src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.ModelNew().npu()
    inputs = [[x.npu() if isinstance(x, torch.Tensor) else x
               for x in group] for group in module.get_input_groups()][0]
    before = count_files(cache_dir)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(*inputs)
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {"first_forward_ms": round(wall_ms, 2),
            "new_cache_files": count_files(cache_dir) - before}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/spike-cache")
    ap.add_argument("--cold-cache-dir", default="/spike-cold-cache")
    ap.add_argument("--report", default="/spike-report/oncard_cache.json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--module")
    args = ap.parse_args()

    if args.child:
        with tempfile.TemporaryDirectory(prefix="spike_oncard_") as work:
            print("RESULT " + json.dumps(child(args.module, Path(work))))
        return 0

    from corpus import MODULES

    ok_modules = [n for n, e in MODULES.items() if e["expect"] == "ok"]
    report = {"modules": {}}
    for name in ok_modules:
        row = {}
        for label, cdir in (("warm", args.cache_dir), ("cold", args.cold_cache_dir)):
            env = dict(os.environ, TRITON_CACHE_DIR=cdir)
            Path(cdir).mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", "--module", name],
                env=env, capture_output=True, text=True, timeout=900)
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if proc.returncode != 0 or line is None:
                row[label] = {"error": (proc.stderr or proc.stdout)[-500:]}
                continue
            data = json.loads(line[len("RESULT "):])
            row[f"{label}_ms"] = data["first_forward_ms"]
            row[f"{label}_new_files"] = data["new_cache_files"]
        if "warm_ms" in row and "cold_ms" in row:
            row["hit"] = (row["warm_new_files"] <= HIT_MAX_NEW_FILES
                          and row["warm_ms"] < HIT_MAX_TIME_RATIO * row["cold_ms"])
        report["modules"][name] = row

    hits = [m.get("hit") for m in report["modules"].values() if "hit" in m]
    report["hit_rate"] = round(sum(hits) / len(hits), 3) if hits else 0.0
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd_tmp = out.with_suffix(".tmp")
    fd_tmp.write_text(json.dumps(report, indent=2))
    fd_tmp.replace(out)
    print(json.dumps(report, indent=2))
    return 0  # report-only; the gate decision is human (Task 5)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the classification runner**

`scripts/spike/oncard_classify.py`:

```python
#!/usr/bin/env python3
"""THROWAWAY spike: on-card failure classification via the npu0 smoke service.

POSTs each failure-corpus kernel with checks=["correctness"] and records the
terminal outcome, to be compared against the cardless stage (criterion 3).
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import MODULES  # noqa: E402

REFERENCE = """import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_input_groups():
    return [[torch.randn(4096), torch.randn(4096)]]
"""


def http_json(method, url, body, headers, timeout=15.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def classify_one(base, headers, name, source, deadline=600) -> dict:
    body = {
        "schema_version": "2.0",
        "purpose": "trial",
        "op_name": f"spike_{name}",
        "reference_code": REFERENCE,
        "kernel_code": source,
        "checks": ["correctness"],
        "deadline_seconds": deadline,
    }
    post_headers = {**headers, "Idempotency-Key": f"spike-{name}-{uuid.uuid4()}"}
    resource = http_json("POST", f"{base}/evaluations?wait_seconds=0",
                         body, post_headers)
    eval_id = resource["evaluation_id"]
    end = time.monotonic() + deadline + 60
    while resource.get("state") not in ("completed", "failed", "timed_out"):
        if time.monotonic() > end:
            return {"oncard_outcome": "client_gave_up"}
        time.sleep(2.0)
        resource = http_json("GET", f"{base}/evaluations/{eval_id}", None, headers)
    data = ((resource.get("result") or {}).get("correctness") or {}).get("data") or {}
    return {"state": resource.get("state"), "oncard_outcome": data.get("outcome"),
            "evaluation_id": eval_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11908")
    ap.add_argument("--token", required=True, help="trial capability Bearer token")
    ap.add_argument("--report", default="spike-report/oncard_classify.json")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.token}",
               "X-Session-ID": "spike-classify"}
    prior = {}
    prior_path = Path("spike-report/cardless.json")
    if prior_path.exists():
        prior = {n: m.get("stage")
                 for n, m in json.loads(prior_path.read_text())["modules"].items()}

    report = {"modules": {}}
    for name, entry in MODULES.items():
        if entry["expect"] == "ok":
            continue
        row = classify_one(args.url, headers, name, entry["source"])
        row["cardless_stage"] = prior.get(name, "unknown")
        # consistent := service agrees the kernel is not runnable; the exact
        # outcome→stage mapping is documented in the spike report (spec §8
        # criterion 3 is a documentation requirement, not a forced match).
        row["consistent"] = row.get("oncard_outcome") in (
            "compile_error", "runtime_error", "wrong")
        report["modules"][name] = row

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Dev-machine checks**

Run: `python3 -m py_compile scripts/spike/oncard_cache_check.py scripts/spike/oncard_classify.py && python3 scripts/spike/selftest.py`
Expected: exit 0 both; corpus self-test still green (FAIL_BACKEND is derived from GOOD_ADD — confirm `warp999_kernel` rename kept the signature spec aligned: the selftest validates this)

- [ ] **Step 4: Lint + commit**

```bash
ruff check scripts/spike/
git add scripts/spike/oncard_cache_check.py scripts/spike/oncard_classify.py
git commit -m "spike: on-card cache-hit and failure-classification runners"
```

---

### Task 5: Execute on NPU host + gate decision memo

**Files:**
- Create: `docs/superpowers/spikes/2026-09-XX-cardless-compile-report.md` (actual date)

**Interfaces:**
- Consumes: everything above; the pinned image digest; a trial capability token for the npu0 stack.
- Produces: the **`compile_stage` gate decision** (`cpu_lane` vs `on_card`) + measured parameters (working compile strategy, hit rate, compile wall-time p50/p95, classification table) that Plan D consumes verbatim.

- [ ] **Step 1: Run the cardless probe on the NPU host**

```bash
cd <repo checkout on NPU host>
mkdir -p spike-cache spike-cold-cache spike-report
export TRITON_EVAL_IMAGE=<pinned sha256 digest of the current prod image>
docker compose -f docker-compose.spike-cardless.yml up --abort-on-container-exit
```
Expected: container exit 0; `spike-report/cardless.json` has `env_facts.davinci_nodes == []`, `npu_available == false`, every module `ok: true`, `cache_files_delta > 0` for good modules. If a strategy errors, the report's `attempts[].error` says why — that error text drives whether Plan D uses `warmup` or `ast_source`.

- [ ] **Step 2: Run the on-card cache measurement**

```bash
docker compose -f docker-compose.spike-oncard.yml up --abort-on-container-exit
```
Expected: `spike-report/oncard_cache.json` `hit_rate` ≥ 0.5 target / ≥ 0.8 stretch (spec §8 criterion 2). Record actuals.

- [ ] **Step 3: Run on-card failure classification**

```bash
python3 scripts/spike/oncard_classify.py --url http://127.0.0.1:11908 --token <trial-token>
```
(Assumes the npu0 smoke stack is up per README; get the token from the stack's capability secret.)
Expected: `spike-report/oncard_classify.json` records each failure kernel's service-side `outcome`. Known expected gap (found during plan research): on-card, backend JIT compile failures surface at first launch → `runtime_error`(phase execute), while cardless they are compile failures by construction. Document actuals per kernel; this table defines Plan D's attribution-widening work.

- [ ] **Step 4: Write the gate decision memo**

`docs/superpowers/spikes/2026-09-XX-cardless-compile-report.md`, exactly this shape:

```markdown
# Cardless Compile Spike Report — <date>

## Verdict: compile_stage = <cpu_lane | on_card>

| Criterion (spec §8) | Result | Evidence |
|---|---|---|
| 1. cardless compile works (≥3 modules, no /dev/davinci*) | ✅/❌ | cardless.json env_facts + stages |
| 2. on-card cache hit rate (>50% target, >80% stretch) | __% | oncard_cache.json hit_rate |
| 3. failure classification consistency | table below | oncard_classify.json |
| 4. compile wall-time distribution | p50 __s / p95 __s | cardless.json compile_wall_ms |

## Working compile strategy: <warmup | ast_source> (per-kernel detail in cardless.json)

## Classification table
| kernel | cardless stage | on-card outcome | Plan D action |
|---|---|---|---|
| fail_syntax | syntax | compile_error | none |
| ... | | | |

## Decision rationale
<2-3 sentences: why cpu_lane or on_card; if on_card, which criterion failed>
```

- [ ] **Step 5: Commit the memo**

```bash
git add docs/superpowers/spikes/
git commit -m "spike: cardless-compile report — compile_stage gate decision"
```

---

## Self-Review Notes (author check, completed at plan-write time)

- **Spec coverage:** Plan A operationalizes spec §8's five spike criteria and nothing more (by design; B–E are separate plans per the plan map). Criterion 5's degradation path is the memo's `on_card` verdict — no code needed in this plan.
- **No placeholders:** every code step contains the full file content; the only `<...>` values are environment facts the executor genuinely must supply (date, digest, token, host checkout path).
- **Type consistency:** `MODULES`/`SIGNATURES`/`EXTRA_OPTIONS` shapes are identical across corpus, selftest, probe, cache checker, classifier; report JSON keys referenced in Task 5 match what Tasks 2/4 write. `FAIL_BACKEND` derives from `GOOD_ADD` via `.replace("add_kernel", "warp999_kernel")` — its `SIGNATURES` entry names `warp999_kernel`, and selftest validates fn/arg alignment so a rename drift fails fast on the dev machine.
- **Known soft spot (accepted):** triton-ascend 3.2.0's exact backend registry name and `ASTSource` signature are version-sensitive; the probe records per-strategy errors precisely so the spike *discovers* the working call — that discovery is the spike's purpose, and both strategies are real code, not stubs.
