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

# 性能测试脚本 — 使用 torch_npu.profiler 测试生成算子的性能表现

import argparse
from collections.abc import Sequence
import gc
import importlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, replace


# ============================================================================
# 日志配置
# ============================================================================

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402	 
from _common_utils import build_input_groups as _build_input_groups_shared  # noqa: E402
from _common_utils import ensure_npu_ambient_device as _ensure_npu_ambient_shared  # noqa: E402
from _common_utils import describe_input as _describe_input_shared  # noqa: E402
from _common_utils import enforce_conv_fp32 as _enforce_conv_fp32  # noqa: E402
from _common_utils import prepare_candidate_inputs as _prepare_candidate_inputs_shared  # noqa: E402
from _common_utils import invoke_model as _invoke_model_shared  # noqa: E402
from _common_utils import set_process_title as _set_process_title  # noqa: E402

logger = logging.getLogger("triton_op_verifier.benchmark")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
# npu-strict 环境设备建立的有界等待上限（秒），与 verify.py 的守护封顶一致；
# 仅用于任何用户 kernel/reference 执行前的设备初始化探测。
_AMBIENT_WAIT_CAP_SECONDS = 240.0
REPEATS_DEFAULT = 20
TRITON_IMPL_NAME_DEFAULT = "triton_ascend_impl"
ERROR_MSG_LIMIT = 2000
MEASUREMENT_METHOD_DEFAULT = "profiler"
CV_THRESHOLD_DEFAULT = 0.3


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BenchmarkConfig:
    """性能测试配置"""
    op_name: str
    verify_dir: str
    triton_impl_name: str = TRITON_IMPL_NAME_DEFAULT
    warmup: int = WARMUP_DEFAULT
    repeats: int = REPEATS_DEFAULT
    skip_framework: bool = False
    framework_latency_ms: float = 0.0
    clear_l2_cache: bool = False  # 是否清除 L2 cache
    keep_res: bool = False        # 是否保留 profiling 结果目录
    max_retries: int = 3          # 采集失败时的最大重试次数
    measurement_method: str = MEASUREMENT_METHOD_DEFAULT  # event | profiler
    cv_threshold: float = CV_THRESHOLD_DEFAULT  # event 时延稳定性门限（变异系数）
    # 1-based original case indices to benchmark; None means the full task.
    case_indices: Optional[List[int]] = None
    # Populated only from an operator-pinned cache, never public request fields.
    reference_latencies_ms: object = None
    reference_measure_eight: bool = False
    # {case_idx: eight historical samples} from the pinned cache manifests.
    reference_cache_samples: object = None
    # Honest label for the sample evidence served with reference_latencies_ms:
    # cached_manifest (same-deployment cache hit) or historical_validated
    # (owner-approved cross-runtime timing-reuse policy).
    reference_evidence_source: Optional[str] = None
    # Policy digest proving the historical_validated authorization.
    reference_evidence_proof: Optional[str] = None
    reference_cache_trace: object = None
    baseline_timings: object = None
    disable_reference_cache: bool = False


@dataclass
class CaseContext:
    """单个测试用例在整体序列中的定位（1-based）。"""
    case_idx: int
    total_cases: int


@dataclass
class MeasureContext:
    """单次 profiling 测量所需的全部参数。"""
    model: Any
    inputs: List[Any]
    warmup: int
    repeats: int
    profile_name: str
    device: Any
    method: str = MEASUREMENT_METHOD_DEFAULT  # event | profiler
    cv_threshold: float = CV_THRESHOLD_DEFAULT


@dataclass
class Measurement:
    """单次 profiling 测量结果三元组：算子分项耗时 / 平均时延 / 峰值显存。"""
    operators: Dict[str, float]
    latency_ms: Optional[float]
    peak_memory: float


@dataclass
class ModelPair:
    """framework / impl 模型成对。"""
    framework: Any
    impl: Any
    reference_input_model: Any = None


@dataclass
class BenchmarkModelSpec:
    """单 shape benchmark 所需的模型工厂三元组。"""
    framework_cls: Any
    impl_cls: Any
    get_init_inputs: Any


@dataclass
class SpeedupBuckets:
    """按 classify_speedup 把通过的 shape 分桶后的结果。"""
    valid_speedups: List[float] = field(default_factory=list)
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)


@dataclass
class PerfAggregate:
    """跨 shape 的 latency / memory / 算子分项耗时聚合结果。"""
    avg_fw: float
    avg_impl: float
    avg_fw_mem: float
    avg_impl_mem: float
    fw_ops: Dict[str, float]
    impl_ops: Dict[str, float]
    n: int


@dataclass
class PerformanceResult:
    """单次性能测试结果"""
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, float]
    # Reference-side evidence only: the eight per-measurement latencies behind
    # avg_latency_ms, their source (fresh_measurements | cached_manifest |
    # historical_validated) and, for historical reuse, the policy digest that
    # authorized it. None when no sample evidence exists (single scalar).
    latency_samples_ms: Optional[List[float]] = None
    latency_samples_source: Optional[str] = None
    latency_samples_proof_sha256: Optional[str] = None
    # Which reference path served this case: cached_manifest |
    # historical_validated (accelerator hits), fresh_measurements (the
    # cache's own miss/drift remeasure path) or original_measurements
    # (the production e4a36f7 single-measurement path). None on the
    # implementation side and for preset skip-framework reference values.
    reference_path: Optional[str] = None


@dataclass
class SingleShapeResult:
    """单个 shape 的性能测试结果。

    失败用例时 framework / implementation / speedup_vs_torch 为 None，
    status="fail" 且附带 error_type / error_msg。
    通过用例但 speedup 异常（NaN/Inf/0/负数）时 status="pass"，
    speedup_vs_torch 落盘为 null，case_idx 收集到 BenchmarkResult 的对应分类列表。
    """
    case_idx: int
    input_desc: List[Dict[str, Any]]
    status: str = "pass"           # "pass" | "fail"
    framework: Optional[PerformanceResult] = None
    implementation: Optional[PerformanceResult] = None
    speedup_vs_torch: Optional[float] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


@dataclass
class BenchmarkResult:
    """完整性能测试结果。

    speedup_vs_torch: 各通过 shape 加速比的几何平均(不含异常 shape)。
    *_indices: 五类异常 shape 的 case_idx 列表(从 1 开始),与 passed_cases 不冲突,
               异常 shape 仍计入 passed_cases,只是 s_i 不进入几何平均。
    """
    op_name: str
    warmup: int
    repeats: int
    framework: Optional[PerformanceResult]
    implementation: Optional[PerformanceResult]
    speedup_vs_torch: Optional[float]
    measurement_method: str = MEASUREMENT_METHOD_DEFAULT  # event | profiler
    total_cases: int = 1
    passed_cases: int = 0
    failed_cases: int = 0
    max_retries: int = 3
    # 1-based original case identities actually benchmarked; None means the
    # full contiguous task (legacy shape). With a selection the identities are
    # sparse but never reindexed.
    case_indices: Optional[List[int]] = None
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)
    per_shape_results: List[SingleShapeResult] = field(default_factory=list)


@dataclass
class OverallAggregate:
    """compute_overall 的聚合结果：跨 shape 的整体均值与异常索引分类。"""
    framework: Optional[PerformanceResult]
    implementation: Optional[PerformanceResult]
    speedup_vs_torch: Optional[float]
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)


# ============================================================================
# 通用辅助函数
# ============================================================================

def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    """截断过长错误信息：保留头尾各 limit/2 字符。"""
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    return _describe_input_shared(inputs)


def cleanup_npu_memory() -> None:
    """清理 NPU 显存，避免单个 shape 失败后连锁 OOM。"""
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception as e:
        # 例外：非 NPU 环境或 torch_npu 不可用时清理无意义，仅记录调试信息，不影响主流程
        logger.debug("跳过 NPU 显存清理（环境不支持 torch_npu）: %s: %s", type(e).__name__, e)
    gc.collect()


# ============================================================================
# 输入解析
# ============================================================================

def resolve_inputs(op_name: str, verify_dir: str):
    """解析任务文件的输入提供方式。

    支持两种格式：
        - get_inputs(): 旧格式，返回单组输入
        - get_input_groups(): 新格式，返回多组输入列表

    Returns:
        输入组列表 (List[List[Any]])
    """
    import torch  # noqa: F401
    sys.path.insert(0, verify_dir)
    torch_module = importlib.import_module(f"{op_name}_torch")

    if hasattr(torch_module, "get_input_groups"):
        return _build_input_groups_shared(torch_module)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()]
    else:
        raise AttributeError(
            "模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def prepare_model_fn(model: Any, inputs: List[Any], device: Any) -> callable:
    """准备模型用于性能测试，返回测试函数"""
    import torch
    import torch_npu  # noqa: F401

    with torch.no_grad():
        _ = _invoke_model_shared(model, inputs)
    torch.npu.synchronize()

    def test_fn():
        with torch.no_grad():
            _ = _invoke_model_shared(model, inputs)
        torch.npu.synchronize()

    return test_fn


def find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def cleanup_profile_path(profile_path: str) -> None:
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


# ============================================================================
# 性能分析逻辑
# ============================================================================

def parse_operator_latency(
    profile_path: str, warmup: int, active_count: int, keep_res: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """从 kernel_details.csv 提取算子时延数据，计算平均执行时间（毫秒）。

    兼容多个 Triton kernel 场景：
    - 每次 test_fn() 调用可能启动多个 kernel，kernel_details.csv 中每行对应一个 kernel 的一次执行
    - 通过 Name groupby 统计每个 kernel 的耗时，再汇总得到总时延
    """
    import pandas as pd

    # 在 profile_path 目录下递归查找 kernel_details.csv
    kernel_details_file = None
    for root, _, files in os.walk(profile_path):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break

    if kernel_details_file is None or not os.path.exists(kernel_details_file):
        if not keep_res:
            cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(kernel_details_file)
    except Exception:
        if not keep_res:
            cleanup_profile_path(profile_path)
        return None, None

    # 按 Name groupby，统计每个 kernel 在 active 阶段的平均耗时
    operator_avg_times = {}
    grouped = df.groupby("Name")
    for name, group in grouped:
        # 每个 kernel 的总执行次数 = total = warmup + active_count
        # 前 warmup 次为 warmup 数据，后 active_count 次为有效数据
        # 按行顺序，取后 active_count 行（active 阶段）
        active_rows = group.iloc[-active_count:]
        total_us = active_rows["Duration(us)"].sum()
        avg_us = total_us / active_count
        operator_avg_times[name] = avg_us

    # 总时延 = 所有 kernel 在单次 test_fn() 调用中的耗时之和
    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1e3

    # 清理 profile_path（由 keep_res 控制）
    if not keep_res:
        cleanup_profile_path(profile_path)

    return operator_avg_times, round(total_avg_ms, 4)


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str,
                             clear_l2_cache: bool = False) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。"""
    import torch
    import torch_npu
    import triton.runtime as runtime
    from triton_evaluation.runtime.profiler_runtime import (
        load_runtime as _load_profiler_runtime,
        check_synchronous_handler as _check_profiler_sync,
        profiler_export_runtime as _profiler_export_runtime,
    )

    # Runtime compatibility is operator-pinned and scoped to this engine
    # process. Keep the original synchronous torch_npu callback and formula.
    if _load_profiler_runtime() is not None:
        _check_profiler_sync(torch_npu.profiler.tensorboard_trace_handler)

    # 手动 warmup（执行一次）
    test_fn()
    torch.npu.synchronize()

    # experimental_config 按 testing.py 方式构造
    # _ExperimentalConfig 为 torch_npu.profiler 暴露的实验配置入口，经 getattr 取用
    experimental_config_cls = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental_config = experimental_config_cls(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    # 总迭代次数：warmup 次预热 + repeats 次有效采集
    total = warmup + repeats

    # profile_path 保持 benchmark.py 当前方式
    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")

    # clear_l2_cache 支持
    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()
        buffer.sum()
        torch.npu.synchronize()

    # 无 schedule，用 with 上下文 + for 循环
    with _profiler_export_runtime() as runtime_evidence, torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if clear_l2_cache:
                buffer.sum()
                torch.npu.synchronize()
            test_fn()
            torch.npu.synchronize()

    if runtime_evidence.get("enabled"):
        with open(os.path.join(profile_path, "profiler-runtime.json"), "w") as runtime_record:
            json.dump(runtime_evidence, runtime_record, sort_keys=True)

    if clear_l2_cache:
        del buffer

    return profile_path


def _median(values: list) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _cv(values: list) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return float("inf")
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5 / mean


def _stable_latency(samples_ms: list, cv_threshold: float):
    """Median of the iteration samples; None when the spread is unstable
    (surfaces as NO_STABLE_MEASUREMENT through the existing None-latency path)."""
    if len(samples_ms) < 3 or _cv(samples_ms) > cv_threshold:
        return None
    return round(_median(samples_ms), 4)


def _measure_event(test_fn, warmup: int, repeats: int, cv_threshold: float):
    """torch.npu.Event per-iteration timing (spec §9). Events are stream-ordered
    device timestamps; the test_fn's internal syncs do not pollute the window."""
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    for _ in range(warmup):
        test_fn()
    torch.npu.synchronize()
    samples: List[float] = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        test_fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return _stable_latency(samples, cv_threshold)


def measure_single(
        ctx: MeasureContext,
        clear_l2_cache: bool = False,
        keep_res: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.reset_peak_memory_stats()
    test_fn = prepare_model_fn(ctx.model, ctx.inputs, ctx.device)

    if ctx.method == "event":
        latency_ms = _measure_event(test_fn, ctx.warmup, ctx.repeats, ctx.cv_threshold)
        operators = None
    else:
        profile_path = run_profiler_with_config(
            test_fn, ctx.warmup, ctx.repeats, ctx.profile_name,
            clear_l2_cache=clear_l2_cache,
        )
        operators, latency_ms = parse_operator_latency(profile_path, ctx.warmup, ctx.repeats, keep_res=keep_res)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def _move_inputs_to_device(inputs: List[Any], device: Any) -> List[Any]:
    """把张量类输入搬到目标 device，标量原样透传。"""
    import torch
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            result.append(x.to(device))
        elif isinstance(x, list):
            result.append(_move_inputs_to_device(x, device))
        else:
            result.append(x)
    return result


def _measure_framework(
    framework_model: Any,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, float], Optional[float], float, Optional[Tuple[List[float], str, Optional[str]]]]:
    """测量 framework 端时延 / 显存 / 算子分项；返回值末位为参考样本证据。

    样本证据是 (八个时延样本, 来源标签, 授权策略摘要)；无样本证据（单次测量或
    预设参考值）时为 None。
    """
    if config.reference_latencies_ms is not None:
        samples = None
        if config.reference_cache_samples is not None:
            samples = config.reference_cache_samples.get(case_idx)
        source = config.reference_evidence_source or "cached_manifest"
        evidence = (samples, source, config.reference_evidence_proof) if samples else None
        return {}, config.reference_latencies_ms[case_idx], 0.0, evidence
    if config.reference_measure_eight:
        live = replace(config, reference_measure_eight=False)
        measurements = [_measure_framework(framework_model, inputs, live, device, case_idx)
                        for _ in range(8)]
        latencies = [measurement[1] for measurement in measurements]
        if any(value is None or not math.isfinite(value) or value <= 0 for value in latencies):
            raise RuntimeError("reference remeasurement requires eight valid positive samples")
        operators = {}
        for measurement in measurements:
            for name, latency in measurement[0].items():
                operators[name] = operators.get(name, 0.0) + latency / 8
        return (operators, _median(latencies), max(measurement[2] for measurement in measurements),
                (latencies, "fresh_measurements", None))
    if config.skip_framework:
        logger.info("    跳过 Framework 测试，使用参考延迟: %.4f ms", config.framework_latency_ms)
        return {}, config.framework_latency_ms, 0.0, None

    inputs_framework = _move_inputs_to_device(inputs, device)
    logger.info("    测试 Framework (warmup=%d, active=%d)...", config.warmup, config.repeats)
    operators, latency_ms, peak_memory = measure_single(
        MeasureContext(
            model=framework_model,
            inputs=inputs_framework,
            warmup=config.warmup,
            repeats=config.repeats,
            profile_name=f"framework_profile_case{case_idx}",
            device=device,
            method=config.measurement_method,
            cv_threshold=config.cv_threshold,
        ),
        clear_l2_cache=config.clear_l2_cache,
        keep_res=config.keep_res,
    )
    return operators or {}, latency_ms, peak_memory, None


def _measure_impl(
    impl_model: Any,
    inputs_impl: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
) -> Tuple[Dict[str, float], Optional[float], float]:
    """测量 impl 端时延 / 显存 / 算子分项。"""
    logger.info("    测试 Implementation (warmup=%d, active=%d)...", config.warmup, config.repeats)
    operators, latency_ms, peak_memory = measure_single(
        MeasureContext(
            model=impl_model,
            inputs=inputs_impl,
            warmup=config.warmup,
            repeats=config.repeats,
            profile_name=f"impl_profile_case{case_idx}",
            device=device,
            method=config.measurement_method,
            cv_threshold=config.cv_threshold,
        ),
        clear_l2_cache=config.clear_l2_cache,
        keep_res=config.keep_res,
    )
    return operators or {}, latency_ms, peak_memory


def _compute_speedup(framework_latency_ms: float, impl_latency_ms: float) -> float:
    """framework / impl 都为正时按比值算 speedup，否则 0。"""
    if impl_latency_ms > 0 and framework_latency_ms > 0:
        return framework_latency_ms / impl_latency_ms
    return 0


def _reference_path_label(config: BenchmarkConfig) -> Optional[str]:
    """本Run参考路径的逐用例标签（同一Run内所有用例同路径）。"""
    if config.reference_latencies_ms is not None:
        return config.reference_evidence_source or "cached_manifest"
    if config.reference_measure_eight:
        return "fresh_measurements"
    if config.skip_framework:
        return None
    return "original_measurements"


def _build_perf_pair(
    framework: Measurement,
    impl: Measurement,
    reference_evidence: Optional[Tuple[List[float], str, Optional[str]]] = None,
    reference_path: Optional[str] = None,
) -> Tuple[PerformanceResult, PerformanceResult]:
    """把 framework / impl 的三元测量结果打包成两份 PerformanceResult。"""
    samples, source, proof = reference_evidence if reference_evidence else (None, None, None)
    return (
        PerformanceResult(
            avg_latency_ms=round(framework.latency_ms, 4),
            peak_memory_mb=round(framework.peak_memory, 2),
            operators=framework.operators,
            latency_samples_ms=samples,
            latency_samples_source=source,
            latency_samples_proof_sha256=proof,
            reference_path=reference_path,
        ),
        PerformanceResult(
            avg_latency_ms=round(impl.latency_ms, 4),
            peak_memory_mb=round(impl.peak_memory, 2),
            operators=impl.operators,
        ),
    )


def run_single_benchmark(
    models: ModelPair,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_ctx: CaseContext,
) -> Tuple[PerformanceResult, PerformanceResult, float]:
    """对单组输入进行性能测试（支持跳过 framework 测试）。

    Returns:
        (framework_result, implementation_result, speedup)
    """
    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    logger.info("  测试第 %d/%d 组输入...", case_idx, total_cases)

    candidate_inputs = _prepare_candidate_inputs_shared(
        models.reference_input_model or models.framework, inputs,
    )
    inputs_impl = _move_inputs_to_device(candidate_inputs, device)

    framework_operators, framework_latency_ms, framework_peak_memory, reference_evidence = _measure_framework(
        models.framework, inputs, config, device, case_idx,
    )
    impl_operators, impl_latency_ms, impl_peak_memory = _measure_impl(
        models.impl, inputs_impl, config, device, case_idx,
    )

    if (not config.skip_framework and framework_latency_ms is None) or impl_latency_ms is None:
        raise RuntimeError(
            f"[用例 {case_idx}/{total_cases}] 无法从 profiler 提取有效时延数据"
        )

    speedup = _compute_speedup(framework_latency_ms, impl_latency_ms)
    fw_perf, impl_perf = _build_perf_pair(
        Measurement(framework_operators, framework_latency_ms, framework_peak_memory),
        Measurement(impl_operators, impl_latency_ms, impl_peak_memory),
        reference_evidence,
        reference_path=_reference_path_label(config),
    )
    return fw_perf, impl_perf, round(speedup, 4)


def classify_speedup(s: Any) -> str:
    """对单个 shape 的 speedup 值分类。

    判定优先级：none → nan → inf → negative → zero → valid

    Returns:
        "none" | "nan" | "inf" | "negative" | "zero" | "valid"
    """
    if s is None:
        return "none"
    if not isinstance(s, (int, float)):
        return "none"
    if math.isnan(s):
        return "nan"
    if math.isinf(s):
        return "inf"
    if s < 0:
        return "negative"
    if s == 0:
        return "zero"
    return "valid"


def _classify_passed_results(passed) -> SpeedupBuckets:
    """按 classify_speedup 把通过的 shape 分桶。"""
    buckets = SpeedupBuckets()
    for r in passed:
        category = classify_speedup(r.speedup_vs_torch)
        if category == "valid":
            buckets.valid_speedups.append(r.speedup_vs_torch)
        elif category == "nan":
            buckets.nan_indices.append(r.case_idx)
        elif category == "inf":
            buckets.inf_indices.append(r.case_idx)
        elif category == "negative":
            buckets.negative_indices.append(r.case_idx)
        elif category == "zero":
            buckets.zero_indices.append(r.case_idx)
        else:  # "none"
            buckets.none_indices.append(r.case_idx)
    return buckets


def _aggregate_perf(passed) -> PerfAggregate:
    """聚合通过的 shape 的平均延时 / 显存 / 算子分项耗时。"""
    n = len(passed)
    avg_fw = sum(r.framework.avg_latency_ms for r in passed) / n
    avg_impl = sum(r.implementation.avg_latency_ms for r in passed) / n
    avg_fw_mem = sum(r.framework.peak_memory_mb for r in passed) / n
    avg_impl_mem = sum(r.implementation.peak_memory_mb for r in passed) / n

    fw_ops: Dict[str, float] = {}
    impl_ops: Dict[str, float] = {}
    for r in passed:
        for op, t in r.framework.operators.items():
            fw_ops[op] = fw_ops.get(op, 0) + t
        for op, t in r.implementation.operators.items():
            impl_ops[op] = impl_ops.get(op, 0) + t
    return PerfAggregate(avg_fw, avg_impl, avg_fw_mem, avg_impl_mem, fw_ops, impl_ops, n)


def _geomean_speedup(valid_speedups):
    """对数域几何平均，空列表返回 None。"""
    if not valid_speedups:
        return None
    return round(
        math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups)),
        4,
    )


def compute_overall(results: List[SingleShapeResult]) -> OverallAggregate:
    """基于通过的 shape 做几何平均聚合。

    异常 shape（speedup 为 None/NaN/Inf/负数/0）不进入几何平均，
    但其 case_idx 收集到对应类别列表中，供报告展示。
    异常 shape 仍计入 passed_cases（算子功能正常，只是测不准）。

    Returns:
        OverallAggregate；全部 shape 均失败时，framework/implementation/speedup 为 None，
        索引列表为空。
    """
    passed = [r for r in results if r.status == "pass" and r.framework and r.implementation]
    buckets = _classify_passed_results(passed)

    if not passed:
        return OverallAggregate(
            framework=None,
            implementation=None,
            speedup_vs_torch=None,
            nan_indices=buckets.nan_indices,
            inf_indices=buckets.inf_indices,
            zero_indices=buckets.zero_indices,
            negative_indices=buckets.negative_indices,
            none_indices=buckets.none_indices,
        )

    agg = _aggregate_perf(passed)
    overall_speedup = _geomean_speedup(buckets.valid_speedups)

    return OverallAggregate(
        framework=PerformanceResult(
            avg_latency_ms=round(agg.avg_fw, 4),
            peak_memory_mb=round(agg.avg_fw_mem, 2),
            operators={k: round(v / agg.n, 4) for k, v in agg.fw_ops.items()},
        ),
        implementation=PerformanceResult(
            avg_latency_ms=round(agg.avg_impl, 4),
            peak_memory_mb=round(agg.avg_impl_mem, 2),
            operators={k: round(v / agg.n, 4) for k, v in agg.impl_ops.items()},
        ),
        speedup_vs_torch=overall_speedup,
        nan_indices=buckets.nan_indices,
        inf_indices=buckets.inf_indices,
        zero_indices=buckets.zero_indices,
        negative_indices=buckets.negative_indices,
        none_indices=buckets.none_indices,
    )


def _load_benchmark_modules(config: BenchmarkConfig):
    """导入 framework / impl 模块；返回三元组 (FrameworkModel, ModelNew, get_init_inputs)。"""
    sys.path.insert(0, config.verify_dir)
    torch_module = importlib.import_module(f"{config.op_name}_torch")
    impl_module = importlib.import_module(f"{config.op_name}_{config.triton_impl_name}")
    return torch_module.Model, impl_module.ModelNew, torch_module.get_init_inputs


def _instantiate_bench_models(framework_cls, impl_cls, get_init_inputs, device, skip_framework=False):
    """同种子分别实例化 framework 与 impl 模型。"""
    import torch
    init_params = get_init_inputs()
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    framework_model = None if skip_framework else framework_cls(*init_params).to(device)
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    impl_model = impl_cls(*init_params).to(device)
    return framework_model, impl_model


def _safe_del_model(name, model_ref):
    """记录模型是否未创建；仅作为调试提示。"""
    if model_ref is None:
        # 例外：try 体内早期失败时变量未定义/为 None，删除无意义；仅记录调试信息
        logger.debug("%s 未创建，无需删除", name)


def _run_shape_case(config, model_spec: BenchmarkModelSpec,
                    inputs, device, case_ctx: CaseContext) -> SingleShapeResult:
    """执行单个 shape 的 benchmark；失败时重试，最终仍失败则返回 status=fail 的结果。"""
    case_idx = case_ctx.case_idx
    total_cases = case_ctx.total_cases
    input_desc = describe_input(inputs)

    last_error_type = None
    last_error_msg = None

    for attempt in range(config.max_retries):
        framework_model = None
        impl_model = None
        try:
            if config.reference_latencies_ms is None:
                framework_model, impl_model = _instantiate_bench_models(
                    model_spec.framework_cls, model_spec.impl_cls, model_spec.get_init_inputs, device,
                )
            else:
                framework_model, impl_model = _instantiate_bench_models(
                    model_spec.framework_cls, model_spec.impl_cls, model_spec.get_init_inputs, device,
                    skip_framework=True,
                )
            fw_perf, impl_perf, speedup = run_single_benchmark(
                ModelPair(framework_model, impl_model, model_spec.framework_cls), inputs, config, device, case_ctx,
            )
            return SingleShapeResult(
                case_idx=case_idx,
                input_desc=input_desc,
                status="pass",
                framework=fw_perf,
                implementation=impl_perf,
                speedup_vs_torch=speedup,
            )
        except Exception as e:
            last_error_type = type(e).__name__
            last_error_msg = traceback.format_exc()
            if attempt < config.max_retries - 1:
                logger.warning(
                    "  [用例 %d/%d] 第 %d/%d 次尝试失败: %s: %s，将重试...",
                    case_idx, total_cases, attempt + 1, config.max_retries,
                    last_error_type, e,
                )
        finally:
            _safe_del_model("framework_model", framework_model)
            _safe_del_model("impl_model", impl_model)
            framework_model = None
            impl_model = None
            cleanup_npu_memory()

    logger.error(
        "  [用例 %d/%d] 失败（已重试 %d 次）: %s",
        case_idx, total_cases, config.max_retries, last_error_type,
    )
    return SingleShapeResult(
        case_idx=case_idx,
        input_desc=input_desc,
        status="fail",
        error_type=last_error_type,
        error_msg=truncate_error(last_error_msg),
    )


def _resolve_selected_indices(case_indices, total_cases):
    """把 1-based 选择解析成有序去重列表；None 表示全部用例。"""
    if case_indices is None:
        return list(range(1, total_cases + 1))
    indices = sorted(set(int(index) for index in case_indices))
    if not indices:
        raise RuntimeError("performance case selection is empty")
    if indices[0] < 1 or indices[-1] > total_cases:
        raise RuntimeError(
            f"performance case selection out of range 1..{total_cases}")
    return indices


def _try_validated_timing_reuse(config, model_spec, input_groups, case_indices, device):
    """Apply the owner-pinned historical timing-reuse policy when eligible.

    Timing evidence only: an exact per-case binding match (same reference,
    inputs, case, physical device and validated 8x warmup/repeats profiler
    contract) serves the REAL old eight samples and their median, labeled
    historical_validated with the policy digest as proof. The existing
    sentinel contract is retained: a fresh eight-sample measurement of the
    first selected case is compared with the historical median, and any drift
    beyond 3% (monotone for the source generation and card) forces the
    original fresh measurement path. Any miss leaves the engine on its
    original execution path.
    """
    if config.skip_framework:
        return False
    from pathlib import Path

    from triton_evaluation.runtime.reference_timing_reuse import (
        current_device_id,
        load_validated_timing_reuse,
    )
    reuse = load_validated_timing_reuse()
    if reuse is None:
        return False
    device_id = current_device_id()
    policy = os.environ.get("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    source = Path(config.verify_dir, f"{config.op_name}_torch.py").read_text(encoding="utf-8")
    cache = reuse.source_cache_for(device_id)
    manifests = reuse.reuse_timings(
        config=config,
        reference_source=source,
        reference_device_policy=policy,
        device_id=device_id,
        case_indices=list(case_indices),
        input_groups=input_groups,
    )
    if manifests is None:
        return False
    from triton_evaluation.runtime.reference_cache import canonical_bytes, sha256
    first_idx = list(case_indices)[0]
    cached_ms = manifests[first_idx].reference_latency_median_ms
    # Fresh sentinel: the same live eight-sample protocol the same-generation
    # cache uses, recorded separately from the historical per-case samples.
    import torch
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    init_params = model_spec.get_init_inputs()
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    reference_model = model_spec.framework_cls(*init_params).to(device)
    try:
        live = replace(config, reference_latencies_ms=None, reference_measure_eight=True)
        _, measured_ms, _, _ = _measure_framework(
            reference_model, input_groups[0], live, device, first_idx
        )
    finally:
        reference_model = None
        cleanup_npu_memory()
    remeasure = cache.needs_remeasurement(cached_ms, measured_ms)
    manifest_shas = [
        sha256(canonical_bytes(manifest.model_dump(mode="json")))
        for manifest in manifests.values()
    ]
    if remeasure:
        # Drift or a prior marker: never claim a historical hit; the caller
        # keeps this trace and measures every case fresh.
        config.reference_cache_trace = {
            "deployment_sha256": reuse.policy.source_deployment_sha256,
            "device_id": device_id,
            "status": "timing_reuse_drift",
            "manifests": manifest_shas,
            "reference_memory_measured": True,
            "timing_reuse_policy_sha256": reuse.policy_sha256,
            "timing_reuse_scope_statement": reuse.policy.scope_statement,
            "sentinel_case_idx": first_idx,
            "sentinel_cached_ms": cached_ms,
            "sentinel_measured_ms": measured_ms,
            "sentinel_relative_drift": abs(measured_ms / cached_ms - 1.0),
        }
        return False
    config.reference_measure_eight = False
    config.reference_latencies_ms = {
        case_idx: manifest.reference_latency_median_ms
        for case_idx, manifest in manifests.items()
    }
    config.reference_cache_samples = {
        case_idx: list(manifest.latency_samples_ms)
        for case_idx, manifest in manifests.items()
    }
    config.reference_evidence_source = "historical_validated"
    config.reference_evidence_proof = reuse.policy_sha256
    config.reference_cache_trace = {
        "deployment_sha256": reuse.policy.source_deployment_sha256,
        "device_id": device_id,
        "status": "timing_reuse",
        "manifests": manifest_shas,
        "reference_memory_measured": False,
        "timing_reuse_policy_sha256": reuse.policy_sha256,
        "timing_reuse_scope_statement": reuse.policy.scope_statement,
        "sentinel_case_idx": first_idx,
        "sentinel_cached_ms": cached_ms,
        "sentinel_measured_ms": measured_ms,
        "sentinel_relative_drift": abs(measured_ms / cached_ms - 1.0),
    }
    return True


def _configure_reference_cache(config, model_spec, input_groups, device, case_indices=None):
    """Measure a real reference sentinel before candidate timing, then choose denominators.

    Drift invalidation is monotone for this card and deployment generation. Missing
    entries execute the complete reference; they are never called cache hits.
    ``input_groups`` are the groups actually selected for this benchmark run and
    ``case_indices`` their original 1-based identities.
    """
    from pathlib import Path
    from triton_evaluation.runtime.reference_cache import (
        ReferenceCache, ReferenceCacheMiss, official_source_metadata,
        canonical_bytes, sha256,
    )
    config.reference_latencies_ms = None
    config.reference_measure_eight = False
    config.reference_cache_samples = None
    config.reference_evidence_source = None
    config.reference_evidence_proof = None
    config.reference_cache_trace = None
    cache = ReferenceCache.from_environment()
    if cache is None:
        # Uncached grade, exactly as production e4a36f7: the ORIGINAL
        # reference path measures each case once. The owner-pinned
        # timing-reuse policy stays a pure optional accelerator — a
        # validated hit serves it, every miss keeps the original path.
        _try_validated_timing_reuse(config, model_spec, input_groups, case_indices, device)
        return
    source = Path(config.verify_dir, f"{config.op_name}_torch.py").read_text(encoding="utf-8")
    if official_source_metadata(source) is None:
        # Not a pinned official reference: no cached evidence may serve it;
        # the ORIGINAL e4a36f7 path measures it exactly as before.
        _try_validated_timing_reuse(config, model_spec, input_groups, case_indices, device)
        return
    if config.skip_framework:
        raise RuntimeError("reference cache cannot be combined with an unbound skip-framework value")
    cache.check_measurement(config)
    policy = os.environ.get("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    trace = {"deployment_sha256": cache.deployment.digest, "device_id": cache.device_id,
             "status": "miss", "manifests": [], "reference_memory_measured": False}
    indices = (list(case_indices) if case_indices is not None
               else list(range(1, len(input_groups) + 1)))
    manifests = []
    case_iterator = iter(input_groups)
    for case_idx in indices:
        inputs = next(case_iterator)
        try:
            manifest = cache.load_case(source, case_idx, inputs,
                                       reference_device_policy=policy)
        except ReferenceCacheMiss:
            manifest = None
        # Drop this workload (and the miss traceback's frames holding it)
        # before the iterator or the fallback allocates another one.
        del inputs
        if manifest is None:
            # Preserve complete execution when an audited task/card pair is
            # absent — unless the validated historical policy covers it.
            if _try_validated_timing_reuse(
                config, model_spec, input_groups, case_indices, device
            ):
                return
            config.reference_measure_eight = True
            if config.reference_cache_trace is None:
                config.reference_cache_trace = trace
            return
        manifests.append(manifest)
    if not manifests:
        raise RuntimeError("reference cache cannot benchmark an empty task")
    trace["manifests"] = [sha256(canonical_bytes(manifest.model_dump(mode="json")))
                          for manifest in manifests]
    # The first selected case is the frozen sentinel. It runs on EVERY evaluation,
    # including a card already invalidated earlier in this generation.
    import torch
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    init_params = model_spec.get_init_inputs()
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    reference_model = model_spec.framework_cls(*init_params).to(device)
    try:
        live = replace(config, reference_latencies_ms=None, reference_measure_eight=True)
        _, measured_ms, _, _ = _measure_framework(reference_model, input_groups[0], live, device, indices[0])
    finally:
        reference_model = None
        cleanup_npu_memory()
    cached_ms = manifests[0].reference_latency_median_ms
    remeasure = cache.needs_remeasurement(cached_ms, measured_ms)
    trace.update(status="remeasured" if remeasure else "hit", sentinel_case_idx=indices[0],
                 sentinel_cached_ms=cached_ms, sentinel_measured_ms=measured_ms,
                 sentinel_relative_drift=abs(measured_ms / cached_ms - 1.0))
    if remeasure:
        config.reference_measure_eight = True
        trace["reference_memory_measured"] = True
    else:
        config.reference_latencies_ms = {manifest.binding.case_idx: manifest.reference_latency_median_ms
                                        for manifest in manifests}
        # Return the real historical eight samples as labeled cached evidence;
        # never fabricate samples around the cached scalar.
        config.reference_cache_samples = {manifest.binding.case_idx: list(manifest.latency_samples_ms)
                                          for manifest in manifests}
        config.reference_evidence_source = "cached_manifest"
    config.reference_cache_trace = trace


def _write_reference_cache_trace(config):
    if config.reference_cache_trace is None:
        return
    from pathlib import Path
    path = Path(config.verify_dir) / "verify_reference.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["performance_cache"] = config.reference_cache_trace
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def _validate_baseline_timings(values, case_indices):
    """Validate host-supplied denominators before importing the candidate."""
    if not isinstance(values, dict) or set(values) != {str(index) for index in case_indices}:
        raise ValueError("baseline timings must exactly cover selected cases")
    result = {}
    for index in case_indices:
        value = values[str(index)]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("baseline timings must be finite positive numbers")
        result[index] = float(value)
    return result


def benchmark_reference_only(config):
    """Measure the reference without importing or executing any candidate."""
    import torch
    import torch_npu  # noqa: F401

    device = torch.device("npu")
    policy = os.environ.get("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    wait = _AMBIENT_WAIT_CAP_SECONDS if policy == "npu-strict" else 0.0
    if not _ensure_npu_ambient_shared(torch, wait_seconds=wait) and policy == "npu-strict":
        raise RuntimeError("npu-strict benchmark requires the ambient NPU device")
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    groups = resolve_inputs(config.op_name, config.verify_dir)
    indices = _resolve_selected_indices(config.case_indices, len(groups))
    module = importlib.import_module(f"{config.op_name}_torch")
    live = replace(config, reference_latencies_ms=None, reference_measure_eight=False,
                   skip_framework=False, reference_cache_samples=None)
    values = {}
    for index in indices:
        model = None
        try:
            init_params = module.get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            model = module.Model(*init_params).to(device)
            _, latency, _, _ = _measure_framework(model, groups[index - 1], live, device, index)
            values[str(index)] = latency
        finally:
            model = None
            cleanup_npu_memory()
    validated = _validate_baseline_timings(values, indices)
    return {"total_cases": len(groups), "case_indices": indices,
            "reference_latencies_ms": {str(index): validated[index] for index in indices},
            "measurement_method": config.measurement_method,
            "warmup": config.warmup, "repeats": config.repeats}


class _SelectedInputGroups(Sequence):
    """Reiterable case selection without retaining materialized device inputs."""

    def __init__(self, groups, indices):
        self.groups = groups
        self.indices = tuple(index - 1 for index in indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _SelectedInputGroups(self.groups,
                                        [value + 1 for value in self.indices[index]])
        return self.groups[self.indices[index]]

    def __iter__(self):
        # Preserve the historical official adapter's O(n) full-pass iterator.
        if self.indices == tuple(range(len(self.groups))):
            yield from self.groups
        else:
            for index in self.indices:
                yield self.groups[index]


def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试，支持多组输入。每个 shape 独立 try/except。"""
    import torch
    import torch_npu  # noqa: F401

    device = torch.device("npu")

    # 执行环境不变量：与沙箱一致以 NPU 为环境设备（见 _common_utils）。
    # npu-strict 下设备初始化探测允许有界等待驱动瞬态恢复（仅在此处、
    # 任何用户 kernel/reference 执行之前，受同一整次运行预算约束）；
    # 等待耗尽仍不可用则立刻大声失败 —— 绝不静默退回 CPU 物化路径。
    # 注：benchmark.py 字节属于 timing-reuse policy 的目标指纹，本变更
    # 使该 policy 在本运行时不再授权复用（回退 fresh 测量，miss 非失败）。
    _reference_policy = os.environ.get("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first")
    _ambient_wait = _AMBIENT_WAIT_CAP_SECONDS if _reference_policy == "npu-strict" else 0.0
    if not _ensure_npu_ambient_shared(torch, wait_seconds=_ambient_wait):
        if _reference_policy == "npu-strict":
            raise RuntimeError(
                "npu-strict benchmark requires the ambient NPU device; "
                "torch.npu is unavailable in this engine context"
            )

    # Correctness and performance use identical seed-zero input bytes.
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    input_groups = resolve_inputs(config.op_name, config.verify_dir)
    total_cases = len(input_groups)
    # Selection preserves original 1-based case identities; it only chooses
    # which groups this performance run measures.
    selected_indices = _resolve_selected_indices(config.case_indices, total_cases)
    selected_groups = _SelectedInputGroups(input_groups, selected_indices)
    selected_cases = len(selected_indices)

    if config.baseline_timings is not None:
        config.reference_latencies_ms = _validate_baseline_timings(config.baseline_timings, selected_indices)
        config.reference_evidence_source = "read_through_baseline"

    framework_cls, impl_cls, get_init_inputs = _load_benchmark_modules(config)
    model_spec = BenchmarkModelSpec(framework_cls, impl_cls, get_init_inputs)
    if config.baseline_timings is None and not config.disable_reference_cache:
        _configure_reference_cache(config, model_spec, selected_groups, device,
                                   case_indices=selected_indices)
    _write_reference_cache_trace(config)

    per_shape_results: List[SingleShapeResult] = []
    selected_iterator = iter(selected_groups)
    for case_idx in selected_indices:
        inputs = next(selected_iterator)
        try:
            per_shape_results.append(_run_shape_case(
                config, model_spec, inputs, device,
                CaseContext(case_idx=case_idx, total_cases=selected_cases),
            ))
        finally:
            # Drop this workload before the iterator allocates the next one.
            del inputs

    passed_cases = sum(1 for r in per_shape_results if r.status == "pass")
    failed_cases = selected_cases - passed_cases

    overall = compute_overall(per_shape_results)

    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        max_retries=config.max_retries,
        measurement_method=config.measurement_method,
        framework=overall.framework,
        implementation=overall.implementation,
        speedup_vs_torch=overall.speedup_vs_torch,
        total_cases=selected_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        case_indices=(None if config.case_indices is None else selected_indices),
        nan_indices=overall.nan_indices,
        inf_indices=overall.inf_indices,
        zero_indices=overall.zero_indices,
        negative_indices=overall.negative_indices,
        none_indices=overall.none_indices,
        per_shape_results=per_shape_results,
    )


def _perf_to_dict(p: Optional[PerformanceResult]) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    return {
        "avg_latency_ms": p.avg_latency_ms,
        "peak_memory_mb": p.peak_memory_mb,
        "operators": {name: round(avg_us, 4) for name, avg_us in p.operators.items()},
    }


def _normalize_shape_speedup(s: Optional[float]) -> Optional[float]:
    """落盘前规范单 shape speedup：异常值统一写为 None（JSON 中即 null）。"""
    return s if classify_speedup(s) == "valid" else None


def _measurement_method_tag(method: str, measured: bool) -> str:
    """落盘的测量方法标签：event 且该 shape 有时延数据时为 npu_event_v1。"""
    if method == "event" and measured:
        return "npu_event_v1"
    return "profiler_kernel_sum_v1"


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将 BenchmarkResult 转换为字典格式。"""
    base_dict: Dict[str, Any] = {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
        "max_retries": result.max_retries,
        "total_cases": result.total_cases,
        "case_indices": result.case_indices,
        "passed_cases": result.passed_cases,
        "failed_cases": result.failed_cases,
        "nan_indices": result.nan_indices,
        "inf_indices": result.inf_indices,
        "zero_indices": result.zero_indices,
        "negative_indices": result.negative_indices,
        "none_indices": result.none_indices,
        "measurement_method": _measurement_method_tag(
            result.measurement_method, result.framework is not None
        ),
        "framework": _perf_to_dict(result.framework),
        "implementation": _perf_to_dict(result.implementation),
        "speedup_vs_torch": result.speedup_vs_torch,
    }

    # per_shape_results 保留全量（含失败用例），带 status 列；
    # 异常 speedup（NaN/Inf/0/负数/None）落盘为 null
    base_dict["per_shape_results"] = [
        {
            "case_idx": r.case_idx,
            "input_desc": r.input_desc,
            "status": r.status,
            "framework": (
                {
                    "avg_latency_ms": r.framework.avg_latency_ms,
                    "peak_memory_mb": r.framework.peak_memory_mb,
                    "latency_samples_ms": r.framework.latency_samples_ms,
                    "latency_samples_source": r.framework.latency_samples_source,
                    "latency_samples_proof_sha256": r.framework.latency_samples_proof_sha256,
                    "reference_path": r.framework.reference_path,
                } if r.framework else None
            ),
            "implementation": (
                {
                    "avg_latency_ms": r.implementation.avg_latency_ms,
                    "peak_memory_mb": r.implementation.peak_memory_mb,
                } if r.implementation else None
            ),
            "speedup_vs_torch": _normalize_shape_speedup(r.speedup_vs_torch),
            "measurement_method": _measurement_method_tag(
                result.measurement_method, r.framework is not None
            ),
            "error_type": r.error_type,
            "error_msg": r.error_msg,
        }
        for r in result.per_shape_results
    ]

    return base_dict


# ============================================================================
# 命令行入口
# ============================================================================

VERIFY_GATE_FAILURES_TO_PRINT = 5
VERIFY_GATE_EXIT_CODE = 2


class VerifyGateError(Exception):
    """L1 闸门未通过时抛出，由 main() 统一捕获并退出，避免在内部函数中调用 sys.exit。"""

    def __init__(self, message: str = "", exit_code: int = VERIFY_GATE_EXIT_CODE):
        super().__init__(message)
        self.exit_code = exit_code


def resolve_verify_json_name(triton_impl_name: str) -> str:
    """按 impl_name 推导 verify_result json 文件名。

    - triton_ascend_impl（默认）→ verify_result.json（Phase 3）
    - triton_baseline / triton_optimized → verify_result_{suffix}.json（Phase 4）
    - 其他自定义名 → verify_result_{name 去掉 triton_ 前缀}.json
    """
    if triton_impl_name == TRITON_IMPL_NAME_DEFAULT:
        return "verify_result.json"
    suffix = triton_impl_name
    if suffix.startswith("triton_"):
        suffix = suffix[len("triton_"):]
    return f"verify_result_{suffix}.json"


def _load_verify_json(verify_json_path, triton_impl_name):
    """读取 verify_result.json；缺失或读取失败抛出 VerifyGateError 由 main 处理。"""
    if not os.path.isfile(verify_json_path):
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：未找到 verify_result 文件\n"
            "  expected: %s\n"
            "  triton_impl_name: %s\n"
            "  请先运行 verify.py 并确保精度校验全部通过",
            verify_json_path,
            triton_impl_name,
        )
        raise VerifyGateError("verify_result 文件不存在")

    try:
        with open(verify_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：verify_result 文件读取失败\n"
            "  path: %s\n"
            "  error: %s: %s",
            verify_json_path,
            type(e).__name__,
            e,
        )
        raise VerifyGateError("verify_result 文件读取失败") from e


def _emit_gate_failures(failures):
    """打印 verify failures 摘要。"""
    if not failures:
        return
    logger.error(
        "  前 %d 条 failures（共 %d 条）：",
        min(VERIFY_GATE_FAILURES_TO_PRINT, len(failures)),
        len(failures),
    )
    for f_item in failures[:VERIFY_GATE_FAILURES_TO_PRINT]:
        logger.error(
            "    - case_idx=%s error_type=%s input_desc=%s",
            f_item.get("case_idx"),
            f_item.get("error_type"),
            f_item.get("input_desc"),
        )


def check_verify_gate(verify_dir: str, triton_impl_name: str) -> None:
    """L1 闸门：benchmark 启动前必须确认对应 verify_result 全过。

    不通过时抛出 VerifyGateError（由 main 捕获并以非零码退出），
    stderr 打印路径 / 计数 / failures 摘要，便于上游 agent 把错误等价映射到
    verify 失败处理路径。
    """
    verify_json_name = resolve_verify_json_name(triton_impl_name)
    verify_json_path = os.path.join(verify_dir, verify_json_name)

    verify_data = _load_verify_json(verify_json_path, triton_impl_name)

    total = verify_data.get("total_cases", 0)
    passed = verify_data.get("passed_cases", 0)
    failures = verify_data.get("failures", []) or []

    if total == 0:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：verify_result 中 total_cases=0\n"
            "  path: %s\n"
            "  说明 verify.py 未实际跑任何 shape，benchmark 无意义",
            verify_json_path,
        )
        raise VerifyGateError("verify_result total_cases=0")

    if passed != total:
        logger.error(
            "[L1 闸门] 拒绝执行 benchmark：精度校验未全通过\n"
            "  path: %s\n"
            "  passed_cases: %d/%d\n"
            "  triton_impl_name: %s",
            verify_json_path,
            passed,
            total,
            triton_impl_name,
        )
        _emit_gate_failures(failures)
        raise VerifyGateError("verify_result 未全部通过")


def _build_argparser():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--triton_impl_name", default=TRITON_IMPL_NAME_DEFAULT,
                       help="Triton 实现模块名")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 20）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument("--skip_framework", action="store_true",
                       help="跳过 framework 性能测试（GPU Kernel 模式使用）")
    parser.add_argument("--framework_latency_ms", type=float, default=0.0,
                       help="预设的 framework 参考延迟（毫秒），用于计算 speedup")
    parser.add_argument("--clear_l2_cache", action="store_true",
                       help="每次迭代前清除 L2 cache")
    parser.add_argument("--keep_res", action="store_true",
                       help="保留 profiling 结果目录（默认清理）")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="采集失败时的最大重试次数（默认 3）")
    parser.add_argument("--measurement-method", choices=["event", "profiler"],
                        default=MEASUREMENT_METHOD_DEFAULT,
                        help="时延测量方法（默认 profiler；event 为 torch.npu.Event 逐迭代计时）")
    parser.add_argument("--cv-threshold", type=float, default=CV_THRESHOLD_DEFAULT,
                        help="event 方法的时延稳定性门限：变异系数超过该值判定为不稳定（默认 0.3）")
    parser.add_argument("--case-indices", default=None,
                       help="逗号分隔的 1-based 用例序号，只测量这些用例（默认全部）；"
                            "输出保留原始 case_idx，不重新编号")
    parser.add_argument("--reference-only", action="store_true",
                        help="Internal reference-only timing invocation")
    parser.add_argument("--baseline-timings-json", default=None,
                        help="Internal host-supplied per-case baseline timings")
    parser.add_argument("--disable-reference-cache", action="store_true",
                        help="Bypass legacy reference timing caches")
    return parser


def _emit_summary(result_dict):
    logger.info("\n性能测试结果:")
    logger.info("  通过率: %s/%s", result_dict["passed_cases"], result_dict["total_cases"])
    if result_dict["speedup_vs_torch"] is not None:
        logger.info("  框架实现 - 平均延迟: %.4f ms", result_dict["framework"]["avg_latency_ms"])
        logger.info("  生成实现 - 平均延迟: %.4f ms", result_dict["implementation"]["avg_latency_ms"])
        logger.info("  加速比 (几何平均): %.4fx", result_dict["speedup_vs_torch"])
    else:
        logger.info("  无可用加速比数据（全部 shape 失败或 speedup 异常）")

    excluded_total = (
        len(result_dict["nan_indices"]) + len(result_dict["inf_indices"])
        + len(result_dict["zero_indices"]) + len(result_dict["negative_indices"])
        + len(result_dict["none_indices"])
    )
    if excluded_total > 0:
        logger.info(
            "  异常 shape (不计入几何平均): "
            "nan=%s, inf=%s, zero=%s, neg=%s, none=%s",
            result_dict["nan_indices"],
            result_dict["inf_indices"],
            result_dict["zero_indices"],
            result_dict["negative_indices"],
            result_dict["none_indices"],
        )


def _save_or_print_result(result_dict, output_path):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info("\n结果已保存到: %s", output_path)
    else:
        logger.info("\n结果:")
        logger.info("%s", json.dumps(result_dict, indent=2, ensure_ascii=False))


def _parse_case_indices(raw: Optional[str]) -> Optional[List[int]]:
    """解析 --case-indices（逗号分隔的 1-based 序号）；None 表示全部用例。"""
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--case-indices must be a comma-separated list of positive integers")
    indices = [int(part) for part in parts]
    if any(index < 1 for index in indices):
        raise ValueError("--case-indices must be positive 1-based case numbers")
    return indices


def _build_config(args, verify_dir):
    return BenchmarkConfig(
        op_name=args.op_name,
        verify_dir=verify_dir,
        triton_impl_name=args.triton_impl_name,
        warmup=args.warmup,
        repeats=args.repeats,
        skip_framework=args.skip_framework,
        framework_latency_ms=args.framework_latency_ms,
        clear_l2_cache=args.clear_l2_cache,
        keep_res=args.keep_res,
        max_retries=args.max_retries,
        measurement_method=args.measurement_method,
        cv_threshold=args.cv_threshold,
        case_indices=_parse_case_indices(args.case_indices),
        baseline_timings=(None if args.baseline_timings_json is None
                          else json.loads(args.baseline_timings_json)),
        disable_reference_cache=args.disable_reference_cache,
    )


def main():
    _setup_logger()
    args = _build_argparser().parse_args()

    # 关闭 HF32 卷积并回读校验，确保性能/精度测试与 verify 一致
    _enforce_conv_fp32()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        logger.error("错误: 验证目录不存在: %s", verify_dir)
        sys.exit(1)

    if not args.reference_only:
        try:
            check_verify_gate(verify_dir, args.triton_impl_name)
        except VerifyGateError as e:
            sys.exit(e.exit_code)

    try:
        config = _build_config(args, verify_dir)
        if args.reference_only:
            result_dict = benchmark_reference_only(config)
            _save_or_print_result(result_dict, args.output)
            sys.exit(0)
        result = benchmark_implementations(config)
        result_dict = result_to_dict(result)
        _emit_summary(result_dict)
        _save_or_print_result(result_dict, args.output)
        # 只要脚本正常跑完就 exit 0（由 Agent 读 JSON 判断）
        sys.exit(0)
    except Exception as e:
        logger.error("性能测试失败: %s", e)
        logger.error("%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    _set_process_title("EvalEngine")
    main()
