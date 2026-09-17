#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""verify.py / benchmark.py 共用的工具函数，避免重复实现。"""
import ctypes
from typing import Any, Dict, List

# prctl(PR_SET_NAME) option code，见 set_process_title()。
_PR_SET_NAME = 15


def enforce_conv_fp32() -> None:
    """关闭 HF32 卷积并回读校验 (torch_npu.npu.conv.allow_hf32=False)。"""
    import logging

    log = logging.getLogger("triton_op_verifier")
    try:
        import torch_npu

        conv = torch_npu.npu.conv
        # 写入后立即回读，防止设置被静默忽略。
        conv.allow_hf32 = False
        observed = conv.allow_hf32
    except AttributeError as exc:
        log.warning(
            "torch_npu.npu.conv.allow_hf32 不可用，跳过关闭 HF32 卷积: %s", exc
        )
        return
    if observed is not False:
        raise RuntimeError(
            "无法关闭 HF32 卷积: torch_npu.npu.conv.allow_hf32 "
            f"回读值为 {observed!r}，预期为 False"
        )
    log.info(
        "已关闭 HF32 卷积: torch_npu.npu.conv.allow_hf32=False（回读校验通过）"
    )


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    try:
        import torch
    except Exception:
        torch = None

    descs: List[Dict[str, Any]] = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def set_process_title(title: str) -> None:
    """设置内核进程名 (/proc/<pid>/comm)，让 `npu-smi info` 显示直观名称而非 "python3"。

    npu-smi 读取 /proc/<pid>/comm（TASK_COMM_LEN=16，可用 15 字节），通过
    prctl(PR_SET_NAME) 改名。零依赖、失败静默——重命名不得影响评测运行。
    """
    # comm 上限 15 字节，自行截断以保证命名可预期。
    title = title[:15]

    # prctl(PR_SET_NAME) 设 /proc/<pid>/comm —— npu-smi 实际读取的字段。
    try:
        libc = ctypes.CDLL(None)
    except OSError:
        try:
            libc = ctypes.CDLL("libc.so.6")
        except OSError:
            return
    try:
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_long
        libc.prctl(
            _PR_SET_NAME,
            ctypes.c_char_p(title.encode("utf-8", "replace")),
            0,
            0,
            0,
        )
    except Exception:
        pass


def build_input_groups(torch_module: Any) -> List[Any]:
    """构造 reference 的输入组；CPU 缺 kernel 时切到 NPU 上下文重试。

    部分 reference 用「NPU 有 kernel、CPU 无 kernel」的算子/精度组合声明输入
    （例如 torch.randn(shape, dtype=torch.int32)：torch_npu 在 NPU 上实现了整型
    normal，CPU 侧抛 NotImplementedError("normal_kernel_cpu")）。沙箱阶段的验证
    以 NPU 为环境设备运行，这类 reference 可正常构造输入；因此构造抛
    NotImplementedError 时改用 NPU 默认设备重试，而不是让整个评测失败。

    只对 NotImplementedError 重试：其它异常（reference 本身写错等）保持原样
    上抛，重试也不会掩盖真错误。
    """
    try:
        return torch_module.get_input_groups()
    except NotImplementedError:
        import logging

        logging.getLogger("triton_op_verifier").info(
            "输入构造在 CPU 上下文不可用，切换到 NPU 默认设备重试"
        )
        import torch

        torch.set_default_device("npu")
        return torch_module.get_input_groups()
