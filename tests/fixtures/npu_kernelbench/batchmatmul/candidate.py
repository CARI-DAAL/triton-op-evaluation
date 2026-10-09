"""Batched matrix multiplication (torch.bmm) as a Triton kernel for Ascend NPU.

out[b] = A[b] @ B[b]   with A:(BATCH, M, K), B:(BATCH, K, N), out:(BATCH, M, N)

A single 2-D grid folds the batch axis (axis 1) and the (m, n) tile axes
(axis 0) into one launch.  fp32 accumulation is used for every input dtype and
the boundary masks are compile-time specialised so the hot loop issues plain,
unmasked loads whenever the shape allows it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    num_pid_m, num_pid_n,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_mn = num_pid_m * num_pid_n
    pid_b = pid // num_pid_mn
    pid_mn = pid % num_pid_mn

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_mn % num_pid_in_group) % group_size_m
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    a_row = (offs_m[:, None] < M)
    b_col = (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=a_row, other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=b_col, other=0.0)
        else:
            k_rem = K - _k * BLOCK_K
            a_k = (offs_k[None, :] < k_rem)
            b_k = (offs_k[:, None] < k_rem)
            if EVEN_M:
                a = tl.load(a_ptrs, mask=a_k, other=0.0)
            else:
                a = tl.load(a_ptrs, mask=a_row & a_k, other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs, mask=b_k, other=0.0)
            else:
                b = tl.load(b_ptrs, mask=b_k & b_col, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=a_row & b_col)


_BLOCK_M = 128
_BLOCK_N = 256
_BLOCK_K = 64
_GROUP_M = 8


def _bmm(A, B):
    batch = A.size(0)
    M = A.size(1)
    K = A.size(2)
    N = B.size(2)

    a_strides = A.stride()
    b_strides = B.stride()

    out = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    if batch == 0 or M == 0 or N == 0:
        return out

    # 16-bit operands can afford a deeper K tile: wider contiguous DMA rows.
    if A.dtype == torch.float32:
        block_k = 128
    else:
        block_k = 128

    num_pid_m = (M + _BLOCK_M - 1) // _BLOCK_M
    num_pid_n = (N + _BLOCK_N - 1) // _BLOCK_N
    grid = (batch * num_pid_m * num_pid_n,)

    _bmm_kernel[grid](
        A, B, out,
        M, N, K,
        a_strides[0], a_strides[1], a_strides[2],
        b_strides[0], b_strides[1], b_strides[2],
        out.stride(0), out.stride(1), out.stride(2),
        num_pid_m, num_pid_n,
        EVEN_M=(M % _BLOCK_M == 0),
        EVEN_N=(N % _BLOCK_N == 0),
        EVEN_K=(K % block_k == 0),
        GROUP_M=_GROUP_M,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return _bmm(A, B)
