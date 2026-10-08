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
import os
import time
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


class _OfficialInputGroups:
    """Seed-zero random access without retaining all official workload tensors."""
    def __init__(self, module):
        self.module = module
        self.total = module.get_case_count()
        if type(self.total) is not int or self.total < 1:
            raise ValueError("official task must declare a positive case count")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if type(index) is not int or index < 0 or index >= self.total:
            raise IndexError("official case index is out of range")
        import torch
        torch.manual_seed(0)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(0)
        return self.module.get_input_group(index)

    def __iter__(self):
        """One ordered pass; each case starts from the exact RNG state the
        random-access path would have (seed zero, preceding cases consumed).

        Iterating via __getitem__ re-materializes cases 0..i for every i
        (O(n^2) tensor generation: 51 BMM51 cases = 1326 materializations of
        28 GB). Advancing the adapter's own generator once and replaying the
        RNG state around each pull keeps every pull's inputs bit-identical to
        the random-access path while materializing each case exactly once,
        retaining only one case's tensors at a time. The per-pull state
        restore also preserves the old isolation property: work the engine
        interleaves between pulls (reference/candidate execution) cannot
        consume RNG that would otherwise shift later cases' inputs.
        """
        import torch
        iterator = getattr(self.module, "_iter_input_groups", None)
        if not callable(iterator):
            for index in range(self.total):
                yield self[index]
            return
        npu = getattr(torch, "npu", None)

        def snapshot():
            npu_state = None
            # The npu generator only participates when the ambient device is
            # NPU (the engine sets it before materialization). CPU contexts
            # must not touch the device: capture is skipped, not attempted.
            # torch.get_default_device() carries an index once the device is
            # established ("npu:0", not "npu"): compare the type, never the
            # exact string — the string form silently skipped capture and let
            # between-pull reseeds shift every later case's inputs.
            default_device = getattr(torch, "get_default_device", lambda: "cpu")()
            if npu is not None and str(default_device).partition(":")[0] == "npu":
                try:
                    npu_state = npu.get_rng_state()
                except Exception:
                    if npu.is_available():
                        return None
            try:
                return torch.random.get_rng_state(), npu_state
            except Exception:
                return None

        def restore(state):
            torch.random.set_rng_state(state[0])
            if state[1] is not None:
                npu.set_rng_state(state[1])

        torch.manual_seed(0)
        if npu is not None:
            npu.manual_seed(0)
        state = snapshot()
        stream = iterator()
        for index in range(self.total):
            if state is None:
                # State capture unavailable (device present but not
                # snapshot-able): keep random-access semantics for this and
                # every remaining case; identical inputs, O(n^2) cost.
                yield self[index]
                continue
            restore(state)
            case = next(stream)
            state = snapshot()
            yield case


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
    if (hasattr(torch_module, "_ADAPTER_VERSION")
            and hasattr(torch_module, "get_input_group")
            and hasattr(torch_module, "get_case_count")):
        return _OfficialInputGroups(torch_module)
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


# 环境设备探测的重试间隔：宿主驱动短暂拒绝设备枚举（drvGetDevNum rc!=0，
# 2026-10-06T18:45–18:48Z 生产窗口实测）时，新进程通常在秒到分钟级恢复。
# 测试 monkeypatch 本常量到 0 以去掉 sleep。
AMBIENT_POLL_SECONDS = 5.0
# 未显式传 wait_seconds 的调用点（benchmark / profile runner）在部署策略为
# npu-strict 时的有界等待上限：环境设备是硬性要求，等它短暂恢复优于立刻
# 失败或静默退 CPU；cpu-first 部署保持单次探测不变。verify.py 显式传预算。
AMBIENT_WAIT_SECONDS = 240.0
_REFERENCE_POLICY_ENV = "TRITON_EVAL_REFERENCE_DEVICE_POLICY"


def ensure_npu_ambient_device(torch, *, wait_seconds: float | None = None) -> bool:
    """把 NPU 设为进程的环境默认设备 —— 沙箱阶段验证的执行环境不变量。

    沙箱阶段的验证以 NPU 为环境设备运行：reference 无论在
    get_input_groups() 还是 forward() 内部新建的张量（权重、掩码等）
    都默认落在 NPU 上。所有执行 reference/模型代码的进程入口都必须在
    执行前调用本函数，保持与沙箱相同的执行环境 —— 这是单一不变量，
    不要在调用点复制实现（复制体分叉曾让 profile runner 崩了一个 PR 周期）。

    入参是调用方已 import 的 torch 模块（本模块保持 stdlib-only，不直接
    import torch）。进程可能尚未 import torch_npu —— torch.npu 属性不
    存在时表达式求值本身会抛 AttributeError，必须用 getattr 解引用。

    wait_seconds 允许有界等待设备枚举恢复：宿主驱动层存在秒级到分钟级的
    瞬态窗口（drvGetDevNum 直接失败），期间 torch.npu.is_available() 为
    False 而设备实际存在；重试只影响建立环境设备的时机，等待耗尽仍返回
    False —— 调用方（npu-strict）依旧立刻显式失败，本函数绝不选择 CPU。
    未显式传值时由部署策略推导：npu-strict 等 AMBIENT_WAIT_SECONDS，
    其余（含未配置）为 0，即历史单次探测行为。

    返回是否已切换。计数片段（decoy 快速通道）刻意不调用本函数而用
    CPU-first + NotImplementedError 回退：decoy 是高频路径，每次都付
    CANN 初始化的秒级代价不划算；引擎路径无此顾虑，一律切换。
    """
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    available = getattr(npu, "is_available", None)
    if available is None:
        return False
    if wait_seconds is None:
        wait_seconds = (
            AMBIENT_WAIT_SECONDS
            if os.environ.get(_REFERENCE_POLICY_ENV) == "npu-strict"
            else 0.0
        )
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        if available():
            torch.set_default_device("npu")
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(AMBIENT_POLL_SECONDS, remaining))


def prepare_candidate_inputs(reference_model, inputs, *, trace=None):
    """Bind omitted official parameters to the defaults of the pinned run().

    The raw workload retains its adapter-owned sentinel for reference execution
    and cache binding. Only the candidate receives this derived positional list.
    Neither candidate defaults nor a copied sentinel define official semantics.
    A reference class suffices on cache hits; it is never instantiated here.
    """
    import inspect

    namespace = getattr(getattr(reference_model, "forward", None), "__globals__", {})
    if namespace.get("_ADAPTER_VERSION") != "official-task-v1":
        return list(inputs)
    order = namespace.get("_PARAM_ORDER")
    run = namespace.get("run")
    if not isinstance(order, (list, tuple)) or not callable(run) or "_ABSENT" not in namespace:
        raise ValueError("official candidate inputs require the bound adapter contract")
    if len(inputs) != len(order):
        raise ValueError("official candidate input count differs from parameter order")
    parameters = inspect.signature(run).parameters
    absent = namespace["_ABSENT"]
    candidate_inputs = []
    defaulted = []
    for name, value in zip(order, inputs):
        if name not in parameters:
            raise ValueError("official candidate parameter is absent from run signature")
        if value is absent:
            value = parameters[name].default
            if value is inspect.Parameter.empty:
                raise ValueError("official workload omitted a required parameter: " + name)
            defaulted.append(name)
        candidate_inputs.append(value)
    if trace is not None:
        from triton_evaluation.runtime.reference_cache import value_sha256
        trace.update(
            candidate_input_semantics="official-run-defaults-v1",
            candidate_defaulted_parameters=defaulted,
            raw_input_sha256=value_sha256(inputs, input_value=True),
            candidate_input_sha256=value_sha256(candidate_inputs, input_value=True),
        )
    return candidate_inputs


def invoke_model(model, inputs):
    """按模型声明的签名调用：位置槽传位置参数，溢出按参数名传 keyword-only。

    沙箱阶段的某些 reference 用 keyword-only 参数声明可选输入
    （``def forward(self, x1, x2, *, x1_scale=None, x2_scale=None)``），
    而用例输入是一个有序列表。按签名契约，位置槽（POSITIONAL_*）接受前
    N 个输入，其余输入应按声明顺序绑定到 keyword-only 参数名 —— 直接
    ``model(*inputs)`` 会抛 "takes N positional arguments" TypeError，
    该错误落在参考侧、无法归因，整题退化为 unscored。

    适配条件严格受限：仅在签名确实声明了 keyword-only 参数、没有
    ``*args``、且输入数量恰好等于位置槽 + keyword-only 槽数时改写调用；
    其余一切情况原样调用，让 Python 的自然错误语义保持不变。
    """
    import inspect

    try:
        signature = inspect.signature(type(model).forward)
    except (TypeError, ValueError):
        return model(*inputs)

    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    kinds = {parameter.kind for parameter in parameters}
    if inspect.Parameter.VAR_POSITIONAL in kinds:
        return model(*inputs)

    positional_slots = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_slots = [
        parameter for parameter in parameters if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if not keyword_slots or len(inputs) != len(positional_slots) + len(keyword_slots):
        return model(*inputs)

    keywords = {
        parameter.name: value
        for parameter, value in zip(keyword_slots, inputs[len(positional_slots) :])
    }
    return model(*inputs[: len(positional_slots)], **keywords)
