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

"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

多 shape 模式下：每个 shape 独立 try/except，全部跑完后落盘 verify_result.json。
策略 A：passed < total 即整体判失败（exit 1），同时失败清单记录在 JSON 的 `failures` 字段。

精度判定按"`--non-compute` 开关 + 输入 dtype + 输出 dtype"分流到 5 类路径，
详见同目录 SKILL.md 的"精度判定规则"小节（唯一权威说明）。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]
"""
import argparse
import gc
import importlib
import json
import logging
import math
import os
import signal
import sys
import subprocess
import traceback
from dataclasses import dataclass


ERROR_MSG_LIMIT = 2000

# 主进程模式外层 kill 相对 --timeout 的宽限秒数：让子进程内的 per-subject
# 定时器有机会先把"候选执行超时"权威归因写入 verify_result.json。
OUTER_GRACE_SECONDS = 60

REQUIRED_MATCHED_RATIO = 0.9

# allclose 判定阈值 (atol, rtol)：|actual - golden| <= atol + rtol * |golden|

# golden（参考输出）优先落在 CPU：fp32 精确计算、与 NPU 负载解耦；参考实现
# 调 npu:: 专有算子、或 CPU 不支持该 dtype/格式时回退到 NPU 重试。只有回退
# 后仍失败才判该用例失败；精度判定本身不受 golden 落点影响（沙箱语义）。
GOLDEN_DEVICE_CPU = "cpu"
ALLCLOSE_TOLS_STR = {
    "float32": (1e-3, 2**(-13)),  # 2**(-13)=1.220703125e-4
    "float16": (9e-2, 2**(-10)),  # 2**(-10)=9.765625e-4
    "bfloat16": (1e-1, 2**(-7)),  # 2**(-7)=7.8125e-3
}
ALLCLOSE_DEFAULT_TOLS = ALLCLOSE_TOLS_STR["float32"]


class AccuracyError(AssertionError):
    """精度判定失败异常，附带结构化 metrics 便于下游统计。"""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


class _CandidateImportError(Exception):
    """候选代码在加载模块契约时失败。"""

    def __init__(self, torch_module, error_msg, cause_name):
        super().__init__("candidate module import failed")
        self.torch_module = torch_module
        self.error_msg = error_msg
        self.cause_name = cause_name


class _CandidateInitializeError(Exception):
    """候选模型初始化失败（engine 可证明归因于 candidate）。"""


class _CandidateCompileError(Exception):
    """Candidate JIT compilation failed at first launch (phase compile)."""


class _CandidateExecuteError(Exception):
    """候选 forward 执行抛错（engine 可证明归因于 candidate）。"""


class _CandidateExecutionTimeout(Exception):
    """候选单次执行超过 per-subject 预算（engine 可证明归因于 candidate）。"""


def _is_compile_error(exc: BaseException) -> bool:
    """First-launch JIT compile failures surface as triton's CompilationError,
    raised lazily inside the first kernel launch. Duck-typed by name/module so
    the engine stays importable without triton (engine runs on CPU workers too)."""
    cls = type(exc)
    return cls.__name__ == "CompilationError" and cls.__module__.startswith("triton.")


# engine 私有归因对象允许的 phase 取值（仅 candidate compile / initialize /
# execute / candidate execution timeout 四种）。
CANDIDATE_PHASES = ("compile", "initialize", "execute", "execution_timeout")


def _candidate_error(phase, message):
    """engine 私有归因对象：仅在 harness 能证明 candidate 起源时附加。

    message 必须单行且脱敏（不含源码、输入值、绝对路径或 traceback）。
    reference 侧失败、外层 subprocess 超时、输出缺失或退出码含义不明时，
    绝不附加该对象。
    """
    assert phase in CANDIDATE_PHASES
    return {"phase": phase, "message": message}


@dataclass
class CaseContext:
    """单个测试用例在整体序列中的定位（1-based）。"""
    case_idx: int
    total_cases: int
    # 单次候选执行预算秒数（correctness_algorithm 的一部分）；None 表示不启用。
    subject_budget_seconds: object = None


@dataclass
class _CompareCtx:
    """compare() 分流到各类判定时共享的输出/输入元信息（仅用于日志与 metrics）。"""
    data_type: object
    input_type: object
    input_dtype: object
    finite_count: int
    size: int


@dataclass
class ModelPair:
    """framework / impl 模型成对，并记录 golden 实际所在 device。"""
    framework: object
    impl: object
    golden_device: object = None


@dataclass
class InputSpec:
    """compare() 的输入侧描述：输入类型与最高精度 dtype（由 _infer_input_type 推断）。"""
    input_type: object = None
    input_dtype: object = None


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402	 
from _common_utils import build_input_groups as _build_input_groups_shared  # noqa: E402
from _common_utils import ensure_npu_ambient_device as _ensure_npu_ambient_shared  # noqa: E402
from _common_utils import describe_input as _describe_input_shared  # noqa: E402
from _common_utils import enforce_conv_fp32 as _enforce_conv_fp32  # noqa: E402
from _common_utils import invoke_model as _invoke_model_shared  # noqa: E402
from _common_utils import set_process_title as _set_process_title  # noqa: E402
from _divergence import classify_divergence as _classify_divergence  # noqa: E402

logger = logging.getLogger("triton_op_verifier.verify")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


class _ShapeMismatchError(AssertionError):
    """输出形状不一致 — 携带 framework/impl 两端 shape，供失败归因使用。"""

    def __init__(self, message, fw_shape, impl_shape):
        super().__init__(message)
        self.fw_shape = fw_shape
        self.impl_shape = impl_shape


def _shape_mismatch_failure(case_idx, fw_shape, impl_shape):
    """构造形状不一致的 failure dict（携带 kind=shape_mismatch，sample 为空）。"""
    return {
        "case_idx": case_idx,
        "kind": "shape_mismatch",
        "sample": [],
        "error_type": "_ShapeMismatchError",
        "error_msg": f"输出形状不一致 (shape mismatch): framework={tuple(fw_shape)}, impl={tuple(impl_shape)}",
    }


def _failure_divergence(reference, candidate, shape, *, atol, rtol,
                        small_value_threshold, small_value_error, rel_threshold,
                        small_count, normal_count, sample_k=8):
    """从扁平 Python list 计算 {kind, sample}（engine 侧先把 tensor 转成 list）。"""
    return _classify_divergence(
        reference, candidate, shape,
        atol=atol, rtol=rtol,
        small_value_threshold=small_value_threshold, small_value_error=small_value_error,
        rel_threshold=rel_threshold, small_count=small_count, normal_count=normal_count,
        sample_k=sample_k,
    )


def _diagnostic_value(value):
    if isinstance(value, complex):
        return float(abs(value))
    return float(value)


def _attach_failure_divergence(error, reference, candidate, shape, data_type):
    """Attach diagnostics without changing the existing accuracy verdict."""
    if error.metrics.get("divergence") is not None:
        return
    reference = [_diagnostic_value(value) for value in reference]
    candidate = [_diagnostic_value(value) for value in candidate]
    small_value_threshold, small_value_error, rel_threshold = get_limits(data_type)
    atol, rtol = get_allclose_tols(data_type)
    finite_reference = [value for value in reference if math.isfinite(value)]
    small_count = error.metrics.get("small_count")
    normal_count = error.metrics.get("normal_count")
    if not isinstance(small_count, int) or not isinstance(normal_count, int):
        small_count = sum(abs(value) < small_value_threshold for value in finite_reference)
        normal_count = len(finite_reference) - small_count
    try:
        divergence = _failure_divergence(
            reference,
            candidate,
            shape,
            atol=atol,
            rtol=rtol,
            small_value_threshold=small_value_threshold,
            small_value_error=small_value_error,
            rel_threshold=rel_threshold,
            small_count=small_count,
            normal_count=normal_count,
        )
    except Exception:
        divergence = {"kind": "scattered", "sample": []}
    error.metrics["divergence"] = divergence


def describe_input(inputs):
    """输入列表的结构化描述（用于 JSON）。"""
    return _describe_input_shared(inputs)


def cleanup_npu_memory():
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception as e:
        # 例外：非 NPU 环境或 torch_npu 不可用时清理无意义，仅记录调试信息，不影响主流程
        logger.debug("跳过 NPU 显存清理（环境不支持 torch_npu）: %s: %s", type(e).__name__, e)
    gc.collect()


def get_limits(data_type):
    """根据数据类型返回精度判定的三元组 (small_value_threshold, small_value_error, rel_threshold)。

    参考 NPU Benchmark 精度对比方法：
    - small_value_threshold：判定元素是否落在"小值域"的阈值
    - small_value_error：小值域元素的绝对误差上限
    - rel_threshold：正常值域元素的相对误差上限，同时也是 MERE 的判定阈值

    阈值表：
    | 数据类型      | small_value_threshold | small_value_error | rel_threshold |
    |--------------|-----------------------|-------------------|---------------|
    | FLOAT16      | 2^{-11}               | 2^{-16}           | 2^{-10}       |
    | BFLOAT16     | 2^{-8}                | 2^{-16}           | 2^{-7}        |
    | FLOAT32      | 2^{-14}               | 2^{-30}           | 2^{-13}       |
    | HiFloat32    | 2^{-12}               | 2^{-28}           | 2^{-11}       |
    | FLOAT8 E4M3  | 2^{-4}                | 2^{-6}            | 2^{-3}        |
    | FLOAT8 E5M2  | 2^{-3}                | 2^{-5}            | 2^{-2}        |

    由于 torch.dtype 中没有直接定义 HiFloat32，可通过字符串传入 "hifloat32" 获取对应阈值。
    """  # noqa: E501
    import torch

    # 字符串映射（用于 HiFloat32 或其他自定义类型）
    str_to_limits = {
        "float16": (2**(-11), 2**(-16), 2**(-10)),
        "bfloat16": (2**(-8), 2**(-16), 2**(-7)),
        "float32": (2**(-14), 2**(-30), 2**(-13)),
        "hifloat32": (2**(-12), 2**(-28), 2**(-11)),
        "float8_e4m3": (2**(-4), 2**(-6), 2**(-3)),
        "float8_e5m2": (2**(-3), 2**(-5), 2**(-2)),
        "fp8_e4m3": (2**(-4), 2**(-6), 2**(-3)),
        "fp8_e5m2": (2**(-3), 2**(-5), 2**(-2)),
    }
    if isinstance(data_type, str):
        return str_to_limits.get(data_type.lower(), (2**(-14), 2**(-30), 2**(-13)))

    # torch.dtype 映射
    dtype_limits_map = {
        torch.float16: (2**(-11), 2**(-16), 2**(-10)),
        torch.bfloat16: (2**(-8), 2**(-16), 2**(-7)),
        torch.float32: (2**(-14), 2**(-30), 2**(-13)),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_limits_map[float8_e4m3] = (2**(-4), 2**(-6), 2**(-3))

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_limits_map[float8_e5m2] = (2**(-3), 2**(-5), 2**(-2))

    return dtype_limits_map.get(data_type, (2**(-14), 2**(-30), 2**(-13)))


def get_allclose_tols(data_type):
    """根据数据类型返回 allclose 判定的 (atol, rtol)。

    判定公式：|actual - golden| <= atol + rtol * |golden|

    阈值表（实际取值见 ALLCLOSE_TOLS_STR）：
    | 数据类型  | atol  | rtol     |
    |----------|-------|----------|
    | FLOAT32  | 1e-3  | 2**(-13) |
    | FLOAT16  | 9e-2  | 2**(-10) |
    | BFLOAT16 | 1e-1  | 2**(-7)  |

    未识别 dtype 走 fp32 默认。
    """
    import torch

    if isinstance(data_type, str):
        return ALLCLOSE_TOLS_STR.get(data_type.lower(), ALLCLOSE_DEFAULT_TOLS)

    dtype_map = {
        torch.float16: ALLCLOSE_TOLS_STR["float16"],
        torch.bfloat16: ALLCLOSE_TOLS_STR["bfloat16"],
        torch.float32: ALLCLOSE_TOLS_STR["float32"],
    }
    return dtype_map.get(data_type, ALLCLOSE_DEFAULT_TOLS)


def _is_integer_dtype(dtype):
    """判断 torch.dtype 是否为整数类型（不含 bool / 不含浮点 / 不含复数）。"""
    import torch
    if dtype == torch.bool:
        return False
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _build_dtype_rank():
    """dtype 精度优先级表：值越大精度越高。

    顺序：fp64 > fp32 > fp16 > bf16 > fp8 > int64 > int32 > int16 > int8 > bool
    """
    import torch
    rank = {
        torch.float64: 100,
        torch.float32: 90,
        torch.float16: 80,
        torch.bfloat16: 70,
        torch.int64: 50,
        torch.int32: 40,
        torch.int16: 30,
        torch.int8: 20,
        torch.uint8: 20,
        torch.bool: 10,
    }
    for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        if dt is not None:
            rank[dt] = 60
    return rank


_DTYPE_RANK = None


def _dtype_rank(dtype):
    global _DTYPE_RANK
    if _DTYPE_RANK is None:
        _DTYPE_RANK = _build_dtype_rank()
    return _DTYPE_RANK.get(dtype, 0)


def _is_int_like_dtype(dtype):
    """判断 dtype 属于"整型类"输入（含 bool；不含浮点/复数）。"""
    import torch
    if dtype is None:
        return False
    if dtype == torch.bool:
        return True
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _is_tensor_list(x):
    """判断 x 是否为非空、且元素全为 torch.Tensor 的 list/tuple。"""
    import torch
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    return all(isinstance(e, torch.Tensor) for e in x)


def _infer_input_type(inputs):
    """从 inputs 推断输入类型，返回 ("float" | "int" | "no_tensor", input_dtype | None)。

    判定优先级（KernelBench / NPUKernelBench 统一处理）：
    1. 若存在 torch.Tensor 输入：取所有 tensor 中最高精度 dtype 作为输入类型
    2. 若不存在 tensor，但存在 list/tuple of Tensor（tensor_list）：取第一个 tensor_list 的首元素 dtype
    3. 其他情况（全为标量 attr / 无输入）：返回 ("no_tensor", None)

    bool 输入归到 "int" 类（按规则：bool 输出单独处理；bool 输入与 int 同等对待）。
    """
    import torch
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    source = None
    candidate_dtypes = []
    if tensors:
        candidate_dtypes = [t.dtype for t in tensors]
        top_dtype = max(candidate_dtypes, key=_dtype_rank)
        source = "tensor"
    else:
        tensor_lists = [x for x in inputs if _is_tensor_list(x)]
        if tensor_lists:
            top_dtype = tensor_lists[0][0].dtype
            candidate_dtypes = [top_dtype]
            source = "tensor_list"
        else:
            logger.info(
                "  [输入类型判定] 来源=无 tensor 输入（全 attr 或空），input_type=no_tensor"
            )
            return "no_tensor", None

    input_type = "int" if _is_int_like_dtype(top_dtype) else "float"
    logger.info(
        f"  [输入类型判定] 来源={source}，候选 dtypes={[str(dt) for dt in candidate_dtypes]}，"
        f"最高精度={top_dtype}，input_type={input_type}"
    )
    return input_type, top_dtype


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。"""
    if hasattr(torch_module, "get_input_groups"):
        groups = _build_input_groups_shared(torch_module)
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            "模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def _view_as_int_dtype(dt):
    """把浮点 / fp8 dtype 映射到等宽整型 dtype，用于 view-as-int 比特比较；不支持返回 None。"""
    import torch
    if dt in (torch.float64, torch.complex64):
        return torch.int64
    if dt in (torch.float32,):
        return torch.int32
    if dt in (torch.float16, torch.bfloat16):
        return torch.int16
    for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
        fp8 = getattr(torch, name, None)
        if fp8 is not None and dt == fp8:
            return torch.int8
    return None


def _binary_exact_is_equal(fw, impl):
    """非计算类底层比特相等判定（复数 / 浮点 view-as-int，整型直接比较）。"""
    import torch
    if fw.dtype.is_complex:
        fw_real_bits = torch.view_as_real(fw)
        impl_real_bits = torch.view_as_real(impl)
        view_dt = _view_as_int_dtype(torch.float32) if fw.dtype == torch.complex64 else torch.int64
        return torch.equal(fw_real_bits.view(view_dt), impl_real_bits.view(view_dt))
    if fw.dtype.is_floating_point:
        view_dt = _view_as_int_dtype(fw.dtype)
        if view_dt is None:
            raise AssertionError(f"非计算类不支持的浮点 dtype: {fw.dtype}")
        return torch.equal(fw.view(view_dt), impl.view(view_dt))
    return torch.equal(fw, impl)


def _binary_exact_detail(fw, impl):
    """构造非计算类比对失败的违例计数与前 N 个位置详情，返回 (violation_count, detail)。"""
    import torch
    if fw.dtype.is_floating_point and not fw.dtype.is_complex:
        view_dt = _view_as_int_dtype(fw.dtype)
        fw_bits = fw.view(view_dt).flatten()
        impl_bits = impl.view(view_dt).flatten()
        diff_mask = fw_bits != impl_bits
        violation_count = int(diff_mask.sum().item())
        violation_idx = torch.where(diff_mask)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f"前 {num_to_show} 个 bit 不一致位置:\n"
        fw_flat = fw.flatten()
        impl_flat = impl.flatten()
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"  位置[{idx}]: framework={fw_flat[idx].item()} "
                f"(bits=0x{fw_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x}), "
                f"impl={impl_flat[idx].item()} "
                f"(bits=0x{impl_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x})\n"
            )
        return violation_count, detail

    fw_flat = fw.flatten()
    impl_flat = impl.flatten()
    diff_mask = fw_flat != impl_flat
    violation_count = int(diff_mask.sum().item())
    violation_idx = torch.where(diff_mask)[0]
    num_to_show = min(10, len(violation_idx))
    detail = f"前 {num_to_show} 个不一致位置:\n"
    for i in range(num_to_show):
        idx = violation_idx[i].item()
        detail += (
            f"  位置[{idx}]: framework={fw_flat[idx].item()}, "
            f"impl={impl_flat[idx].item()}\n"
        )
    return violation_count, detail


def _compare_binary_exact(fw_out, impl_out, data_type):
    """非计算类：二进制完全一致比对。

    - 浮点 dtype：通过 view-as-int 比较底层 bit pattern，可识别 NaN payload 差异
    - 整型 / bool：直接 torch.equal
    - 复数：实部/虚部分别 view-as-int 比较
    """
    import torch

    fw = fw_out.contiguous().detach().cpu()
    impl = impl_out.contiguous()
    if isinstance(impl, torch.Tensor):
        impl = impl.detach().cpu()
    else:
        raise AssertionError(f"非计算类实现输出必须是 Tensor，实际为 {type(impl).__name__}")

    if fw.shape != impl.shape:
        raise AssertionError(
            f"非计算类验证失败，输出形状不一致: framework={fw.shape}, impl={impl.shape}"
        )
    if fw.dtype != impl.dtype:
        raise AssertionError(
            f"非计算类验证失败，输出 dtype 不一致: framework={fw.dtype}, impl={impl.dtype}"
        )

    if _binary_exact_is_equal(fw, impl):
        return

    violation_count, detail = _binary_exact_detail(fw, impl)
    metrics = {
        "category": "non_compute",
        "dtype": str(data_type),
        "violation_count": violation_count,
        "total_elements": int(fw.numel()),
    }
    raise AccuracyError(
        f"验证失败 dtype={data_type} (非计算类，要求二进制完全一致): "
        f"{violation_count}/{fw.numel()} 元素不一致\n{detail}",
        metrics,
    )


def _check_nan_inf(fw_flat, impl_flat, size):
    """校验 NaN / Inf 的位置与符号一致；不一致抛 AccuracyError。"""
    import torch
    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AccuracyError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}",
            {
                "category": "nan_position_mismatch",
                "framework_nan_count": fw_nan_count,
                "implementation_nan_count": impl_nan_count,
                "total_elements": size,
            },
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AccuracyError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}",
            {
                "category": "inf_position_mismatch",
                "framework_inf_count": fw_inf_count,
                "implementation_inf_count": impl_inf_count,
                "total_elements": size,
            },
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AccuracyError(
                "验证失败，Inf 符号不匹配",
                {"category": "inf_sign_mismatch", "total_elements": size},
            )


def _compare_bool_output(fw_finite, impl_finite, ctx):
    """bool 输出：torch.equal 严格相等。"""
    import torch
    logger.info(
        f"  [评测模式] 模式=bool_output（bool 输出），"
        f"输入 dtype={ctx.input_dtype}（{ctx.input_type}），输出 dtype={ctx.data_type}；"
        f"误差要求=torch.equal 严格相等（finite={ctx.finite_count}/{ctx.size}）"
    )
    if torch.equal(fw_finite, impl_finite):
        return
    diff_idx = torch.where(fw_finite != impl_finite)[0]
    violation_count = int(diff_idx.numel())
    num_to_show = min(10, violation_count)
    detail = f"前 {num_to_show} 个不一致位置:\n"
    for i in range(num_to_show):
        idx = diff_idx[i].item()
        detail += (
            f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
            f"impl={impl_finite[idx].item()}\n"
        )
    metrics = {
        "category": "bool_output",
        "dtype": str(ctx.data_type),
        "violation_count": violation_count,
        "total_finite": int(fw_finite.numel()),
    }
    raise AccuracyError(
        f"验证失败 dtype={ctx.data_type} (bool 输出，要求严格相等): "
        f"{violation_count}/{fw_finite.numel()} 元素不一致\n{detail}",
        metrics,
    )


def _compare_quant_fp_to_int(fw_finite, impl_finite, ctx):
    """量化类（fp 输入 → int 输出）：要求 |actual - golden| <= 1。"""
    import torch
    logger.info(
        f"  [评测模式] 模式=quant_fp_to_int（量化类 fp→int），"
        f"输入 dtype={ctx.input_dtype}（{ctx.input_type}），输出 dtype={ctx.data_type}；"
        f"误差要求=|actual - golden| <= 1（finite={ctx.finite_count}/{ctx.size}）"
    )
    diff = (fw_finite.to(torch.int64) - impl_finite.to(torch.int64)).abs()
    violation_count = int((diff > 1).sum().item())
    if violation_count == 0:
        return
    # Same double-domain rule as _compare_integer_compute: int64 quant
    # differences exceed the RFC 8785 safe integer domain and the metric
    # enters the response, where canonical serialization would raise
    # IntegerDomainError and null out provenance.
    max_diff = float(diff.max().item())
    violation_idx = torch.where(diff > 1)[0]
    num_to_show = min(10, len(violation_idx))
    detail = f"前 {num_to_show} 个量化误差超限位置:\n"
    for i in range(num_to_show):
        idx = violation_idx[i].item()
        detail += (
            f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
            f"impl={impl_finite[idx].item()}, "
            f"|diff|={diff[idx].item()} (允许<=1)\n"
        )
    metrics = {
        "category": "quant_fp_to_int",
        "dtype": str(ctx.data_type),
        "input_type": ctx.input_type,
        "max_abs_diff": max_diff,
        "violation_count": violation_count,
        "total_finite": int(diff.numel()),
        "tolerance": 1,
    }
    raise AccuracyError(
        f"验证失败 dtype={ctx.data_type} (量化类 fp->int，要求|diff|<=1): "
        f"{violation_count}/{diff.numel()} 元素超限，max_abs_diff={max_diff}\n"
        f"{detail}",
        metrics,
    )


def _compare_integer_compute(fw_finite, impl_finite, ctx):
    """整数计算类：要求严格相等（|actual - golden| == 0）。"""
    import torch
    logger.info(
        f"  [评测模式] 模式=integer_compute（整数计算类），"
        f"输入 dtype={ctx.input_dtype}（{ctx.input_type}），输出 dtype={ctx.data_type}；"
        f"误差要求=|actual - golden| == 0（严格相等，finite={ctx.finite_count}/{ctx.size}）"
    )
    if torch.equal(fw_finite, impl_finite):
        return
    diff = (fw_finite.to(torch.int64) - impl_finite.to(torch.int64)).abs()
    violation_count = int((diff > 0).sum().item())
    # int64 differences reach 9.2e18: far outside the RFC 8785 safe integer
    # domain. Emit the metric as a double so canonical serialization cannot
    # raise IntegerDomainError and fail the whole evaluation with null
    # provenance (observed as hardware-fingerprint mismatches downstream).
    max_diff = float(diff.max().item())
    violation_idx = torch.where(diff > 0)[0]
    num_to_show = min(10, len(violation_idx))
    detail = f"前 {num_to_show} 个不一致位置:\n"
    for i in range(num_to_show):
        idx = violation_idx[i].item()
        detail += (
            f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
            f"impl={impl_finite[idx].item()}, "
            f"|diff|={diff[idx].item()}\n"
        )
    metrics = {
        "category": "integer_compute",
        "dtype": str(ctx.data_type),
        "input_type": ctx.input_type,
        "max_abs_diff": max_diff,
        "violation_count": violation_count,
        "total_finite": int(diff.numel()),
        "tolerance": 0,
    }
    raise AccuracyError(
        f"验证失败 dtype={ctx.data_type} (整数计算类，要求严格相等): "
        f"{violation_count}/{diff.numel()} 元素不一致，max_abs_diff={max_diff}\n"
        f"{detail}",
        metrics,
    )


def _compare_float_output(fw_finite, impl_finite, ctx):
    """浮点计算类：dtype-aware 三项判定（委托 _check_accuracy_npu_benchmark）。"""
    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)
    sv_thr_pre, sv_err_pre, rel_thr_pre = get_limits(ctx.data_type)
    atol_pre, rtol_pre = get_allclose_tols(ctx.data_type)
    logger.info(
        f"  [评测模式] 模式=float_compute（浮点计算类），"
        f"输入 dtype={ctx.input_dtype}（{ctx.input_type}），输出 dtype={ctx.data_type}；"
        f"误差要求=三项 AND："
        f"(1)max_error_cap |diff|<=atol+rtol*|golden| "
        f"[atol={atol_pre:.3e}, rtol={rtol_pre:.3e}]，"
        f"(2)matched_ratio>={REQUIRED_MATCHED_RATIO} "
        f"[小值域 sv_thr={sv_thr_pre:.3e}/sv_err={sv_err_pre:.3e}，"
        f"正常域 rel_thr={rel_thr_pre:.3e}]，"
        f"(3)MERE<{rel_thr_pre:.3e}（finite={ctx.finite_count}/{ctx.size}）"
    )
    _check_accuracy_npu_benchmark(fw_finite, impl_finite, ctx.data_type)


def compare(fw_out, impl_out, data_type, input_spec=None, non_compute=False):
    """对比框架输出和实现输出。

    Args:
        fw_out: 框架（金标准）输出 Tensor
        impl_out: 被测实现输出 Tensor
        data_type: 输出 dtype（与 fw_out.dtype 一致）
        input_spec: InputSpec(input_type, input_dtype)，描述输入侧（由 _infer_input_type() 推断）；
            input_type "float"/"int"/"no_tensor"/None 参与"输出整型时"的分流，
            input_dtype 输入最高精度 dtype 仅用于诊断打印。None 时按全 None 处理。
        non_compute: 若 True，强制走二进制完全一致路径（搬移 / Cast 等算子）

    决策矩阵（non_compute=False 时）：
        | 输出 dtype | 输入类型           | 类别           | 判定                |
        |-----------|------------------|---------------|--------------------|
        | bool      | 任意              | bool 输出      | torch.equal         |
        | int       | int               | 整数计算类      | |diff| == 0         |
        | int       | float             | 量化类         | |diff| <= 1         |
        | int       | no_tensor         | 整数计算类     | |diff| == 0（最严）   |
        | float     | 任意              | 浮点计算类     | 三项判定（按输出 dtype）|
    """
    import torch
    if input_spec is None:
        input_spec = InputSpec()
    input_type = input_spec.input_type
    input_dtype = input_spec.input_dtype
    if isinstance(impl_out, torch.Tensor) and tuple(fw_out.shape) != tuple(impl_out.shape):
        raise _ShapeMismatchError(
            f"验证失败，输出形状不一致: framework={fw_out.shape}, impl={impl_out.shape}",
            tuple(fw_out.shape), tuple(impl_out.shape),
        )
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise _ShapeMismatchError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}",
            tuple(fw_flat.shape), tuple(impl_flat.shape),
        )

    try:
        # 非计算类：二进制完全一致（先于其他判定，跳过 NaN/Inf/finite 过滤）
        if non_compute:
            logger.info(
                f"  [评测模式] 模式=non_compute（非计算类），"
                f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
                f"误差要求=二进制完全一致（view-as-int bit pattern 全等，含 NaN payload）"
            )
            _compare_binary_exact(fw_out, impl_out, data_type)
            return

        _check_nan_inf(fw_flat, impl_flat, size)

        finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
        finite_count = finite_mask.sum().item()
        if finite_count == 0:
            logger.warning("警告: 所有值都是非有限值，跳过精度检查")
            return

        fw_finite = fw_flat[finite_mask]
        impl_finite = impl_flat[finite_mask]
        ctx = _CompareCtx(data_type, input_type, input_dtype, finite_count, size)

        # bool 输出独立处理：严格相等
        if fw_finite.dtype == torch.bool:
            _compare_bool_output(fw_finite, impl_finite, ctx)
            return

        # 输出整型：按 input_type 分流
        # input_type == "float" → 量化类 (|diff|<=1)
        # input_type == "int" 或 "no_tensor" 或 None → 整数计算类 (|diff|==0，最严)
        if _is_integer_dtype(fw_finite.dtype):
            if input_type == "float":
                _compare_quant_fp_to_int(fw_finite, impl_finite, ctx)
            else:
                _compare_integer_compute(fw_finite, impl_finite, ctx)
            return

        # 输出浮点：按浮点精度标准执行（dtype-aware 三项判定）
        _compare_float_output(fw_finite, impl_finite, ctx)
    except AccuracyError as e:
        _attach_failure_divergence(
            e,
            fw_flat.tolist(),
            impl_flat.tolist(),
            tuple(fw_out.shape),
            data_type,
        )
        raise


def _compute_accuracy_state(golden, actual, data_type):
    """计算 NPU Benchmark 三项判定的指标与中间张量。

    Returns:
        (is_pass, metrics, arrays)。metrics 为落盘用结构化指标，arrays 保存供失败时
        构造违例详情的中间张量。
    """
    import torch

    # 统一升 float32，避免低精度 dtype 自身误差污染计算
    golden_f = golden.float()
    actual_f = actual.float()

    sv_thr, sv_err, rel_thr = get_limits(data_type)
    atol, rtol = get_allclose_tols(data_type)

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()

    # 分桶（用于 #2 matched_ratio）
    small_mask = abs_golden < sv_thr
    normal_mask = ~small_mask

    # 元素级 matched（#2 口径）
    small_ok = abs_diff <= sv_err
    rel_err = abs_diff / (abs_golden + 1e-7)
    normal_ok = rel_err <= rel_thr
    matched_mask = torch.where(small_mask, small_ok, normal_ok)

    total_finite = matched_mask.numel()
    matched_count = int(matched_mask.sum().item())
    matched_ratio = matched_count / total_finite if total_finite > 0 else 1.0
    max_abs_diff = abs_diff.max().item() if total_finite > 0 else 0.0

    # #1 allclose：逐元素判定，要求 100% 通过
    allclose_bound = atol + rtol * abs_golden
    allclose_mask = abs_diff <= allclose_bound
    allclose_violation_count = int((~allclose_mask).sum().item()) if total_finite > 0 else 0
    allclose_ok = allclose_violation_count == 0

    # MERE：对所有 finite 元素计算相对误差再取均值（分母统一 |golden| + 1e-7 防除零）
    normal_count = int(normal_mask.sum().item())
    if total_finite > 0:
        mere = rel_err.mean().item()
        mere_ok = mere < rel_thr
    else:
        mere = None
        mere_ok = True

    ratio_ok = matched_ratio >= REQUIRED_MATCHED_RATIO
    is_pass = allclose_ok and ratio_ok and mere_ok

    metrics = {
        "matched_ratio": matched_ratio, "max_abs_diff": max_abs_diff, "MERE": mere,
        "rel_threshold": rel_thr, "small_value_threshold": sv_thr, "small_value_error": sv_err,
        "atol": atol, "rtol": rtol, "max_error_cap_violation_count": allclose_violation_count,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO, "total_finite": total_finite,
        "matched_count": matched_count, "small_count": int(small_mask.sum().item()),
        "normal_count": normal_count,
        "checks": {
            "max_error_cap": allclose_ok, "required_matched_ratio": ratio_ok, "MERE": mere_ok,
        },
    }
    arrays = {
        "abs_diff": abs_diff, "rel_err": rel_err, "small_mask": small_mask,
        "matched_mask": matched_mask, "allclose_mask": allclose_mask,
        "allclose_bound": allclose_bound,
    }
    return is_pass, metrics, arrays


def _format_allclose_detail(golden, actual, arrays):
    """构造 #1 max_error_cap 违例的前 N 个位置详情。"""
    import torch
    abs_diff = arrays["abs_diff"]
    allclose_bound = arrays["allclose_bound"]
    indices = torch.where(~arrays["allclose_mask"])[0]
    num_to_show = min(10, len(indices))
    out = f"前 {num_to_show} 个 max_error_cap 违例位置:\n"
    for i in range(num_to_show):
        idx = indices[i].item()
        out += (
            f"  位置[{idx}]: framework={golden[idx]:.6e}, "
            f"impl={actual[idx]:.6e}, |diff|={abs_diff[idx]:.6e} "
            f"(允许<=atol+rtol*|golden|={allclose_bound[idx]:.6e})\n"
        )
    return out


def _format_matched_detail(golden, actual, arrays, metrics):
    """构造 #2 matched_ratio 未通过的前 N 个位置详情（区分小值域 / 正常域）。"""
    import torch
    abs_diff = arrays["abs_diff"]
    rel_err = arrays["rel_err"]
    small_mask = arrays["small_mask"]
    sv_err = metrics["small_value_error"]
    rel_thr = metrics["rel_threshold"]
    indices = torch.where(~arrays["matched_mask"])[0]
    num_to_show = min(10, len(indices))
    out = f"前 {num_to_show} 个 matched 未通过位置:\n"
    for i in range(num_to_show):
        idx = indices[i].item()
        if small_mask[idx].item():
            out += (
                f"  位置[{idx}] (小值域): framework={golden[idx]:.6e}, "
                f"impl={actual[idx]:.6e}, |diff|={abs_diff[idx]:.6e} "
                f"(允许<={sv_err:.6e})\n"
            )
        else:
            out += (
                f"  位置[{idx}] (正常域): framework={golden[idx]:.6e}, "
                f"impl={actual[idx]:.6e}, 相对误差={rel_err[idx]:.6e} "
                f"(允许<={rel_thr:.6e})\n"
            )
    return out


def _format_accuracy_error(golden, actual, data_type, metrics, arrays):
    """根据失败的检查项构造精度报错信息（含前 N 个违例位置）。"""
    checks = metrics["checks"]
    mere = metrics["MERE"]
    mere_str = f"{mere:.6e}" if mere is not None else "n/a"
    error_msg = (
        f"验证失败 dtype={data_type}: "
        f"max_error_cap_violations={metrics['max_error_cap_violation_count']}/{metrics['total_finite']} "
        f"(atol={metrics['atol']:.6e}, rtol={metrics['rtol']:.6e}, "
        f"max_abs_diff={metrics['max_abs_diff']:.6e}, ok={checks['max_error_cap']}), "
        f"matched_ratio={metrics['matched_ratio']:.6f} "
        f"(req>={REQUIRED_MATCHED_RATIO}, ok={checks['required_matched_ratio']}), "
        f"MERE={mere_str} (rel_thr={metrics['rel_threshold']:.6e}, ok={checks['MERE']}); "
        f"small_count={metrics['small_count']}, normal_count={metrics['normal_count']}\n"
    )

    # 仅在对应检查失败时打印各自的违例位置（前 N 个）
    if not checks["max_error_cap"]:
        error_msg += _format_allclose_detail(golden, actual, arrays)
    if not checks["required_matched_ratio"]:
        error_msg += _format_matched_detail(golden, actual, arrays, metrics)
    return error_msg


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证（三项判定）。

    元素级 matched 定义（用于 #2 matched_ratio）：
    - |golden| < small_value_threshold（小值域）：|diff| <= small_value_error
    - 否则（正常值域）：|diff| / (|golden| + 1e-7) <= rel_threshold

    通过条件（三项 AND）：
    1. allclose: 所有元素满足 |diff| <= atol + rtol * |golden|（dtype-aware）
    2. matched_ratio = sum(matched) / total_finite >= REQUIRED_MATCHED_RATIO（0.9）
    3. MERE < rel_threshold（对所有 finite 元素计算相对误差再取均值，
       分母统一用 |golden| + 1e-7 防除零）

    Args:
        golden: 参考输出（金标准）
        actual: 被测实现输出
        data_type: 数据类型，用于获取对应阈值

    Raises:
        AccuracyError: 当精度验证未通过时，异常的 metrics 属性携带结构化指标
    """
    is_pass, metrics, arrays = _compute_accuracy_state(golden, actual, data_type)
    if is_pass:
        return
    raise AccuracyError(
        _format_accuracy_error(golden, actual, data_type, metrics, arrays),
        metrics,
    )


def _subject_alarm_handler(signum, frame):
    raise _CandidateExecutionTimeout(
        "candidate execution exceeded the per-subject budget"
    )


def _run_candidate_forward(impl_model, inputs, subject_budget_seconds):
    """执行候选 forward；可选 per-subject 预算（SIGALRM）内未完成即归因超时。

    预算只包裹候选调用本身：harness 存活且候选未在预算内返回时，engine
    可以权威地把超时归因给 candidate execute 阶段。
    """
    if not subject_budget_seconds:
        try:
            return _invoke_model_shared(impl_model, inputs)
        except Exception as exc:
            if _is_compile_error(exc):
                raise _CandidateCompileError(
                    f"candidate compilation raised {type(exc).__name__}"
                ) from exc
            raise _CandidateExecuteError(
                f"candidate execution raised {type(exc).__name__}"
            ) from exc
    previous_handler = signal.signal(signal.SIGALRM, _subject_alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, subject_budget_seconds)
    try:
        return _invoke_model_shared(impl_model, inputs)
    except _CandidateExecutionTimeout:
        raise
    except Exception as exc:
        if _is_compile_error(exc):
            raise _CandidateCompileError(
                f"candidate compilation raised {type(exc).__name__}"
            ) from exc
        raise _CandidateExecuteError(
            f"candidate execution raised {type(exc).__name__}"
        ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_single_case(models: ModelPair, inputs, device, case_ctx: CaseContext, non_compute=False):
    """验证单组输入。失败时抛出 AssertionError / AccuracyError。"""
    import torch

    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    logger.info("  测试第 %d/%d 组输入...", case_idx, total_cases)

    # 推断输入类型（"float" / "int" / "no_tensor"）→ 决定输出整型时走整数计算 vs 量化
    input_type, input_dtype = _infer_input_type(inputs)

    golden_device = models.golden_device or device
    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(golden_device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        impl_output = _run_candidate_forward(
            models.impl, inputs_for_impl, case_ctx.subject_budget_seconds
        )
        framework_output = _invoke_model_shared(models.framework, inputs_for_framework)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    logger.info("  [输出概览] 共 %d 个输出，non_compute=%s", len(framework_output), non_compute)

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                logger.info("  [输出 %d] shape=%s, dtype=%s", i, list(fw_out.shape), data_type)
                compare(
                    fw_out, impl_out, data_type,
                    InputSpec(input_type, input_dtype),
                    non_compute=non_compute,
                )
            except _ShapeMismatchError:
                # 裸 raise 保留原始异常类型与 fw_shape/impl_shape 属性，使 _try_run_case
                # 的 except _ShapeMismatchError 分支能命中（否则会被下方 AssertionError 吞掉）。
                raise
            except AccuracyError as e:
                raise AccuracyError(f"[用例 {case_idx}/{total_cases}] {str(e)}", e.metrics) from e
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def _load_verify_modules(op_name, verify_dir, triton_impl_name):
    """导入 framework / impl 模块并解析关键符号。"""
    sys.path.insert(0, verify_dir)
    torch_module = importlib.import_module(f"{op_name}_torch")
    framework_cls = torch_module.Model
    get_init_inputs = torch_module.get_init_inputs
    try:
        impl_module = importlib.import_module(f"{op_name}_{triton_impl_name}")
        impl_cls = impl_module.ModelNew
    except Exception as exc:
        raise _CandidateImportError(
            torch_module,
            truncate_error(traceback.format_exc()),
            type(exc).__name__,
        ) from exc
    return {
        "torch_module": torch_module,
        "framework_cls": framework_cls,
        "impl_cls": impl_cls,
        "get_init_inputs": get_init_inputs,
    }


def _instantiate_models(framework_cls, impl_cls, get_init_inputs, device,
                        golden_device=None):
    """同种子分别实例化 framework 与 impl 模型。

    golden_device 给定时 framework 落在该设备（GOLDEN_DEVICE_CPU 策略），
    否则与 impl 同 device。两者的构造顺序与消耗的随机数保持一致。
    """
    import torch
    init_params = get_init_inputs()
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    framework_model = framework_cls(*init_params).to(golden_device or device)
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    try:
        impl_model = impl_cls(*init_params).to(device)
    except Exception as exc:
        raise _CandidateInitializeError(
            f"candidate initialization failed: {type(exc).__name__}"
        ) from exc
    return framework_model, impl_model


def _candidate_failure(case_ctx, inputs, phase, error):
    """构造可归因 candidate 失败的 failure dict（单行脱敏 message）。"""
    return {
        "case_idx": case_ctx.case_idx,
        "input_desc": describe_input(inputs),
        "error_type": type(error).__name__,
        "error_msg": str(error),
        "kind": None,
        "sample": [],
        "candidate_error": _candidate_error(phase, str(error)),
    }


_CANDIDATE_ERROR_PHASES = (
    (_CandidateCompileError, "compile"),
    (_CandidateInitializeError, "initialize"),
    (_CandidateExecuteError, "execute"),
    (_CandidateExecutionTimeout, "execution_timeout"),
)
_CANDIDATE_ERROR_TYPES = tuple(cls for cls, _phase in _CANDIDATE_ERROR_PHASES)


def _attempt_case(modules, case_ctx: CaseContext, inputs, device,
                   golden_device, non_compute):
    """在指定 golden_device 上跑一遍；成功返回 None，失败把异常抛给调用方。"""
    framework_model = None
    impl_model = None
    try:
        framework_model, impl_model = _instantiate_models(
            modules["framework_cls"],
            modules["impl_cls"],
            modules["get_init_inputs"],
            device,
            golden_device=golden_device,
        )
        run_single_case(
            ModelPair(framework_model, impl_model, golden_device), inputs, device,
            case_ctx, non_compute=non_compute,
        )
    finally:
        del framework_model
        del impl_model
        cleanup_npu_memory()


def _try_run_case(modules, case_ctx: CaseContext, inputs, device, non_compute=False):
    """执行单个 case：成功返回 (True, None)；失败返回 (False, failure_dict)。

    golden 优先落 CPU；CPU 跑不起来（参考实现调 npu:: 专有算子、CPU 不支持
    该 dtype/格式等）时回退 golden 到 NPU 重试，只有回退后仍失败才判失败。
    AccuracyError / AssertionError / _ShapeMismatchError 是比对结论（golden
    已成功运行），候选归因错误与 golden 落点无关 —— 三者都不重试。
    """
    try:
        try:
            _attempt_case(
                modules, case_ctx, inputs, device, GOLDEN_DEVICE_CPU, non_compute
            )
            return True, None
        except (AccuracyError, AssertionError) + _CANDIDATE_ERROR_TYPES:
            raise
        except Exception as cpu_error:
            logger.info(
                "  [用例 %d/%d] golden 落 CPU 时执行失败，回退 golden 到 NPU 重试: %s: %s",
                case_ctx.case_idx, case_ctx.total_cases,
                type(cpu_error).__name__, str(cpu_error)[:160],
            )
        _attempt_case(modules, case_ctx, inputs, device, device, non_compute)
        return True, None
    except _ShapeMismatchError as e:
        # 依赖 run_single_case 中的 except _ShapeMismatchError: raise 透传（否则会被展平为普通 AssertionError）。
        logger.error("  [用例 %d/%d] 形状不一致", case_ctx.case_idx, case_ctx.total_cases)
        return False, _shape_mismatch_failure(
            case_ctx.case_idx, e.fw_shape, e.impl_shape,
        )
    except _CANDIDATE_ERROR_TYPES as e:
        # harness 可证明归因于 candidate：import 与首 launch 两阶段的 compile 失败
        # 及 initialize / execute / execution_timeout 均在此落盘私有归因对象。
        phase = dict(_CANDIDATE_ERROR_PHASES)[type(e)]
        logger.error(
            "  [用例 %d/%d] 候选 %s 失败: %s",
            case_ctx.case_idx, case_ctx.total_cases, phase, e,
        )
        logger.error("%s", traceback.format_exc())
        return False, _candidate_failure(case_ctx, inputs, phase, e)
    except Exception as e:
        err_detail = traceback.format_exc()
        logger.error(
            "  [用例 %d/%d] 失败: %s: %s",
            case_ctx.case_idx, case_ctx.total_cases, type(e).__name__, e,
        )
        failure = {
            "case_idx": case_ctx.case_idx,
            "input_desc": describe_input(inputs),
            "error_type": type(e).__name__,
            "error_msg": truncate_error(err_detail),
            "kind": None,
            "sample": [],
        }
        if isinstance(e, AccuracyError):
            failure["metrics"] = e.metrics
            div = e.metrics.get("divergence")
            if div is not None:
                failure["kind"] = div.get("kind")
                failure["sample"] = div.get("sample", [])
        return False, failure


def _write_verify_result(output_path, result):
    """落盘 verify_result.json；失败仅警告。"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("验证结果已保存到: %s", output_path)
    except Exception as e:
        logger.warning("警告: 无法写入 verify_result.json: %s", e)


def verify_implementations(
    op_name, verify_dir, triton_impl_name="triton_ascend_impl",
    output_path=None, non_compute=False, subject_budget_seconds=None,
):
    """验证框架实现和生成实现的结果一致性。

    每个 shape 独立 try/except，全部跑完后写 verify_result.json。

    Args:
        non_compute: 若 True，所有 case 走"非计算类"二进制完全一致判定（搬移/Cast 等算子）
        subject_budget_seconds: 单次候选执行预算（秒）；候选 forward 超过预算时
            以 candidate execution timeout 归因并停止后续 case。None 表示不启用。

    Returns:
        (passed_cases, total_cases)
    """
    import torch
    import torch_npu  # noqa: F401

    candidate_import_error = None
    try:
        modules = _load_verify_modules(op_name, verify_dir, triton_impl_name)
    except _CandidateImportError as exc:
        candidate_import_error = exc
        modules = {"torch_module": exc.torch_module}

    # 在获取输入之前设置种子，确保随机生成的输入可复现
    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    # 执行环境不变量：与沙箱一致以 NPU 为环境设备（见 _common_utils）。
    _ensure_npu_ambient_shared(torch)

    input_groups, total_cases = resolve_input_provider(modules["torch_module"])
    device = None if candidate_import_error is not None else torch.device("npu")

    failures = []
    passed_cases = 0
    for case_idx, inputs in enumerate(input_groups, start=1):
        if candidate_import_error is not None:
            failures.append({
                "case_idx": case_idx,
                "input_desc": describe_input(inputs),
                "error_type": "CompilationError",
                "error_msg": candidate_import_error.error_msg,
                "candidate_error": _candidate_error(
                    "compile",
                    f"candidate module import failed: {candidate_import_error.cause_name}",
                ),
            })
            continue
        ok, failure = _try_run_case(
            modules,
            CaseContext(
                case_idx=case_idx,
                total_cases=total_cases,
                subject_budget_seconds=subject_budget_seconds,
            ),
            inputs, device, non_compute=non_compute,
        )
        if ok:
            passed_cases += 1
        else:
            failures.append(failure)
            if failure.get("candidate_error") is not None:
                # 候选归因失败后继续跑剩余 case 不会改变 subject outcome。
                break

    failed_cases = total_cases - passed_cases

    if output_path is None:
        output_path = os.path.join(verify_dir, "verify_result.json")
    result = {
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failures": failures,
    }
    _write_verify_result(output_path, result)

    if failed_cases == 0:
        logger.info("验证成功：共 %d 组测试用例全部通过", total_cases)
    else:
        logger.error(
            "验证失败：%d/%d 组通过，%d 组失败（详见 %s）",
            passed_cases,
            total_cases,
            failed_cases,
            output_path,
        )

    return passed_cases, total_cases


def _terminate_expired_subprocess(proc, waited_seconds: int) -> None:
    """Whole-run watchdog exit: kill the tree and die as exit code 3.

    Exit 3 is a deliberate, identifiable death — distinct from 0 (normal) and
    1 (generic engine error) — so the evaluator can classify an over-budget
    but otherwise healthy run as a deadline/budget problem instead of a
    retryable crash.
    """
    proc.kill()
    proc.wait()
    logger.error("验证超时（%d秒），已终止子进程", waited_seconds)
    sys.exit(3)


if __name__ == "__main__":
    _set_process_title("TritonEvalWork")
    _setup_logger()
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="单次候选执行预算秒数（per-subject budget，默认 900）；主进程外层超时另加宽限",
    )
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--output", default=None,
        help="验证结果 JSON 输出路径（默认 {verify_dir}/verify_result.json）",
    )
    parser.add_argument(
        "--non-compute", action="store_true",
        help="非计算类算子（搬移 / Cast 等），所有 case 走二进制完全一致判定",
    )
    parser.add_argument(
        "--subprocess", action="store_true",
        help=argparse.SUPPRESS,  # 内部参数：子进程模式，直接执行验证
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        logger.error("错误: 验证目录不存在: %s", verify_dir)
        sys.exit(1)

    if args.subprocess:
        # 子进程模式：直接执行验证逻辑；--timeout 作为单次候选执行预算，
        # 超预算的候选 forward 被子进程内定时器权威归因后落盘。
        # 先关闭 HF32 卷积并回读校验，避免低精度卷积污染精度比对
        _enforce_conv_fp32()
        try:
            passed, total = verify_implementations(
                args.op_name, verify_dir, args.triton_impl_name, args.output,
                non_compute=args.non_compute,
                subject_budget_seconds=args.timeout,
            )
        except Exception as e:
            logger.error("%s", e)
            logger.error("%s", traceback.format_exc())
            sys.exit(1)
        # 策略 A：passed < total → exit 1
        sys.exit(0 if passed == total and total > 0 else 1)
    else:
        # 主进程模式：启动子进程执行验证，超时后 kill 整个进程树
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--op_name", args.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", args.triton_impl_name,
            "--timeout", str(args.timeout),
            "--subprocess",
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        if args.non_compute:
            cmd.append("--non-compute")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # 外层 kill 预留宽限：让子进程的 per-subject 定时器先完成归因落盘；
            # 只有宽限也用尽（归因不明）才杀死进程树。
            stdout, stderr = proc.communicate(timeout=args.timeout + OUTER_GRACE_SECONDS)

            sys.stdout.buffer.write(stdout)
            sys.stdout.buffer.flush()
            sys.stderr.buffer.write(stderr)
            sys.stderr.buffer.flush()
            sys.exit(proc.returncode)

        except subprocess.TimeoutExpired:
            _terminate_expired_subprocess(proc, args.timeout + OUTER_GRACE_SECONDS)
