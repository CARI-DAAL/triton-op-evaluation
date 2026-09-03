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
