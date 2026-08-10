# evaluation API v2（稳定契约 + 可组合 checks）设计文档

- 日期：2026-08-06
- 审阅确认：2026-08-10
- 状态：已确认，进入实现
- 位置：`triton_evaluation/api/`（同步路由 + 正式 response model）、
  `triton_evaluation/evaluators/`（engine JSON → 正式 model）
- 取代：v1 `/verify`、`/profile` 和隐式 benchmark → 唯一入口 `POST /evaluations`
- 关联消费方：kernel-harness 的 `te` CLI（agent 经 `execute_bash` 调用）

## 1. 目标与非目标

eval service v2 面向 kernel-harness agent、批量跑分和后续消费方，提供稳定、显式、
可组合的评测契约。它解决 v1 的三个问题：

- `/verify?measure_speed=true` 混合正确性与性能职责；
- `TaskResponse.result: dict` 直接透传 engine JSON，脚本输出即隐式 API；
- correctness 只有聚合指标，没有可供 agent 修复 close-but-wrong kernel 的发散结构。

v2 使用一个同步资源、可组合 `checks` 和正式 pydantic response model，并把
`kind` + `sample` 作为 correctness 的固有诊断。

非目标：

- 不做多机分布式；后端继续使用单机多卡、Redis queue 和 worker。
- 不改变 `verify.py` 的精度判定算法，也不改变 `benchmark.py` / `profile_runner.py`
  的测量算法。`verify.py` 只 additive 地增加 `kind` + `sample`。
- 不把服务变成可执行任意代码的通用 runner。
- v2 不实现后台任务取消。同步请求超时后，已提交任务仍可能继续运行。

## 2. 已确认决策

| 决策 | 选择 |
|---|---|
| 资源 | `POST /evaluations`，body 中组合 `checks` |
| 客户端语义 | 同步：一次 POST 返回最终 `EvaluationResult` |
| 后端语义 | 继续异步 queue + worker；API 内部 submit + poll Redis |
| timeout | 从 API 收到请求起计算的整体同步等待预算，包含排队和全部 checks |
| 超时后任务 | API 返回 `TIMEOUT`；Redis job 不取消，worker 可继续执行 |
| 契约 | pydantic `EvaluationResult`；不透传 engine dict |
| correctness 诊断 | 每个数值/形状失败固有携带 `kind` + `sample` |
| performance 依赖 | 请求中必须同时含 correctness，且只在 correctness 通过后运行 |
| profile 依赖 | 独立；correctness 不通过或 decoy 也不跳过 profile |
| v1 | 删除 `/verify`、`/profile` 和公开 `/tasks/{id}`，不留 alias |
| 版本 | 直接 `/evaluations`，不加 URL 前缀，也不加 `revision` / `judge_version` |
| 鉴权 | 无 token；依赖网络隔离和按 session 的在途限制 |
| 调度 | FIFO |

## 3. HTTP API

### 3.1 `POST /evaluations`

请求：

```json
{
  "op_name": "addmm",
  "reference_code": "import torch ... class Model ...",
  "kernel_code": "import torch ... class ModelNew ...",
  "triton_impl_name": "triton_ascend_impl",
  "checks": ["correctness", "performance"],
  "non_compute": false,
  "timeout": 900,
  "performance": {"warmup": 5, "repeats": 50},
  "profile": null
}
```

请求头：

```http
X-Session-ID: <kernel-harness session/container id>
```

规则：

- `Check = Literal["correctness", "performance", "profile"]`；`checks` 非空且不得重复。
- `performance` 出现在 `checks` 时，`correctness` 也必须出现；否则返回 HTTP 400。
- `performance` / `profile` 参数只在对应 check 出现时保留，否则规范化为 `None`。
- `timeout` 取值 10–3600 秒，默认 900 秒。
- `X-Session-ID` 必填，只允许 1–128 个 `[A-Za-z0-9._:-]` 字符；缺失或非法返回
  HTTP 400。
- 同一个 session 最多一个未完成的后端 evaluation。Redis lease 已存在时返回 HTTP 429，
  并带 `Retry-After: 1`。

成功接收请求后，API 创建一个 `evaluation` job，再直接轮询 Broker；不会把 `task_id`
作为一次中间响应返回。任务在 `timeout` 内进入终态时返回对应 `EvaluationResult`。
若整体等待预算耗尽，API 返回 HTTP 200、`status="TIMEOUT"`、
`error="evaluation request timed out"` 和空 payload；后台任务继续。

客户端 HTTP timeout 必须为请求 `timeout + 60s`。客户端只自动重试连接拒绝与明确的
5xx；HTTP read timeout 不自动重试，因为服务可能已经提交了任务。

### 3.2 `GET /healthz`

保持存活检查：Redis 可用时返回 `{"redis":"ok"}`，否则返回 `{"redis":"down"}`。

### 3.3 非公开后端状态

Redis task record 及 `task_id` 继续用于 API 内部轮询、日志和管理，但没有公开的
`GET /tasks/{id}` 或 `GET /evaluations/{id}`。`task_id` 只在最终 `EvaluationResult`
中用于追踪。

## 4. 正式 response model

```python
Check = Literal["correctness", "performance", "profile"]
TerminalStatus = Literal["SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"]


class EvaluationRequest(BaseModel):
    op_name: str
    reference_code: str
    kernel_code: str
    triton_impl_name: str = "triton_ascend_impl"
    checks: list[Check] = Field(min_length=1)
    non_compute: bool = False
    timeout: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None


class EvaluationResult(BaseModel):
    task_id: str
    status: TerminalStatus
    error: str | None = None
    result: EvaluationPayload


class EvaluationPayload(BaseModel):
    correctness: CorrectnessResult | None = None
    performance: PerformanceResult | None = None
    profile: ProfileResult | None = None
```

`EvaluationResult` 不含 `revision`。`QUEUED` / `RUNNING` 不出现在消费方契约，因为同步
POST 只返回终态。

### 4.1 Correctness

```python
FiniteValue = float | Literal["inf", "-inf", "nan"]

DivergenceKind = Literal[
    "constant_offset", "per_axis", "precision_loss", "sign_flip",
    "value_clamp", "shape_mismatch", "scattered",
]


class CaseMetrics(BaseModel):
    max_abs_diff: float | None = None
    mean_abs_err: float | None = None
    matched_ratio: float | None = None
    mere: float | None = None
    small_count: int | None = None
    normal_count: int | None = None
    atol: float | None = None
    rtol: float | None = None


class SampleElement(BaseModel):
    idx: list[int]
    candidate: FiniteValue
    reference: FiniteValue
    diff: FiniteValue


class CaseFailure(BaseModel):
    case_idx: int
    kind: DivergenceKind | None = None
    metrics: CaseMetrics = Field(default_factory=CaseMetrics)
    sample: list[SampleElement] = Field(default_factory=list)
    error: str | None = None


class CorrectnessResult(BaseModel):
    passed: bool
    compiled: bool
    decoy: bool = False
    passed_cases: int
    total_cases: int
    failures: list[CaseFailure] = Field(default_factory=list)
```

- 数值失败和 shape mismatch 的 `kind` 必填；compile/runtime 失败时为 `None`。
- 有 `kind` 时 `error=None`；compile/runtime 失败在 `error` 中放截断后的原因。
- `sample` 最多 8 个点，按 `abs(diff)` 降序；索引是行主序多维坐标。
- 非有限值编码为字符串；shape mismatch 的 sample 为空。
- engine 没有 mean absolute error，因此 `mean_abs_err=None`；MERE 映射到 `mere`。

### 4.2 Performance

`benchmark.py` 的 raw `per_shape_results` 含 input descriptors、framework/implementation
对象和失败项。v2 只投影通过且 speedup 有效的 shape，避免把 engine 私有结构变成契约：

```python
class PerShapeResult(BaseModel):
    shape: str
    latency_ms: float
    throughput: float | None = None
    speedup_vs_torch: float


class PerformanceResult(BaseModel):
    speedup_vs_torch: float
    per_shape: list[PerShapeResult] = Field(default_factory=list)
```

- `shape` 由该 case 的 tensor `input_desc[].shape` 生成紧凑 JSON；单 tensor 例如
  `"[20120,512]"`，多 tensor 例如 `"[[16,32],[32,64]]"`。
- `latency_ms` 使用 implementation 的 `avg_latency_ms`。
- engine 不提供可靠 op-count，因此当前 `throughput=None`。
- benchmark 有任意 failed case、无有效 overall speedup 或没有有效 per-shape 结果时，
  performance check 为 `FAILED`，而不是构造不完整的 `PerformanceResult`。

### 4.3 Profile

profile 不透传 raw dict。它把 `profile_runner.py` 的多 case op/deep 输出投影为以下稳定子集：

```python
ProfileState = Literal["complete", "partial"]
ProfileCaseState = Literal["complete", "partial", "failed"]
BoundKind = Literal[
    "MTE2 BOUND", "CUBE BOUND", "VEC BOUND", "FIXP BOUND",
    "MTE3 BOUND", "SCALAR BOUND", "NO BOUND",
]


class OpProfile(BaseModel):
    case_idx: int
    op_name: str
    time_ms: float
    percentage: float | None = None


class DeepProfileSummary(BaseModel):
    case_idx: int
    status: ProfileCaseState
    target_kernel: str | None = None
    bound: BoundKind | None = None
    rule: str | None = None
    mte2_ratio: float | None = None
    cube_ratio: float | None = None
    vec_ratio: float | None = None
    fixp_ratio: float | None = None
    mte3_ratio: float | None = None
    scalar_ratio: float | None = None
    per_core_spread_ratio: float | None = None
    per_core_verdict: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ProfileData(BaseModel):
    mode: Literal["op", "deep"]
    selected_cases: int
    completed_cases: int
    partial_cases: int
    failed_cases: int
    timed_out: bool
    ops: list[OpProfile] | None = None
    summaries: list[DeepProfileSummary] | None = None


class ProfileResult(BaseModel):
    profile_state: ProfileState
    data: ProfileData
```

- op 模式填 `ops`、`summaries=None`。每个 kernel 的 `duration_us` 转成 `time_ms`；
  `percentage` 是该 case 内有效 kernel duration 的占比。
- deep 模式填 `summaries`、`ops=None`。固定 ratio 字段来自 `bound.ratios`；recommendation
  只保留正式文本，不暴露 raw evidence。
- 任一 case partial/failed、job timed out 或完成数少于选择数时，`profile_state="partial"`。

## 5. divergence 分类

分类在 `verify.py` 内执行，因为只有它持有 candidate/reference tensor 和已计算的精度上下文。
decision tree 互斥，按以下顺序先命中先返回：

1. `shape_mismatch`：shape 不同；
2. `sign_flip`：强负相关或 diff 近似 `-2 * reference`；
3. `value_clamp`：candidate 大量饱和且范围显著收窄；
4. `constant_offset`：diff 方差小且均值显著非零；
5. `per_axis`：误差集中在单轴；
6. `precision_loss`：ULP band 小误差且 small-value case 主导；
7. `scattered`：无结构模式命中。

分类器必须是 pure stdlib，接收 flat Python list + 原 shape，使 torch-free CI 可测试。
真实 NPU 校准集覆盖各 kind 至少数十个失败 case；验收 precision > 90%、recall > 80%。
达不到标准的 kind 必须合并或删除，不能用低置信标签污染消费方。

## 6. engine → model 塑形与 check orchestration

- `verify.py` additive 地把 `kind` + `sample` 写入每个失败 record；已有字段和判定不变。
- `benchmark.py` / `profile_runner.py` 不改。
- `evaluators/shaping.py` 是唯一 raw JSON → stable payload 的映射层；不做 divergence 分类。
- `EvaluationEvaluator` 顺序执行 correctness → performance → profile。
- performance 只在请求 correctness 且 correctness `passed=True` 时执行。
- profile 不依赖 correctness；wrong/decoy kernel 仍执行 profile。
- correctness mismatch 是一次成功完成的 check，整体仍可为 `SUCCEEDED`；失败原因在
  `correctness.passed=False`。
- sub-evaluator 的基础设施/超时失败会反映为整体终态；优先级为
  `TIMEOUT > FAILED > REJECTED > SUCCEEDED`。profile 已产生的部分结果仍塑形返回。
- performance check 失败时整体 `FAILED`、`performance=None`，但保留 correctness/profile。

## 7. 同步等待、孤儿任务与 session lease

API 在创建 task 前获取 Redis session lease，lease value 为 `task_id`。只有 worker 在写入
最终 task record 后释放该 lease；API 自身的整体等待超时不会释放它，所以 orphan job
期间同一 session 不能再次排队。lease 还设置有限 TTL，覆盖 job 最大执行预算、reaper
grace 和一次请求 timeout，避免 worker 永久消失时死锁。

API poll 使用可配置的短间隔，测试中可设为 0。每次 poll 只读取 Redis task record；
不在 API 进程里执行 evaluator。同步长连接占用一个请求协程，但不阻塞 event loop。

## 8. 兼容与迁移

- 删除 `/verify`、`/profile`、`/tasks/{id}`；它们返回 404。
- 删除公开请求中的 `measure_speed` 语义；性能通过 `checks` 和 `performance` 子对象表达。
- 老消费方一次性迁移到同步 `POST /evaluations`，并把 HTTP timeout 配成 body timeout + 60s。
- response 绝不回传 `reference_code` 或测试输入原值。

## 9. 网络与部署

- 不引入 token 或 `EVALUATION_TOKEN`。
- 宿主进程默认绑定 `127.0.0.1`。容器部署可在私有 Docker network 内监听
  `0.0.0.0`，但只向宿主 loopback 发布端口；不得把服务发布到远程网卡。
- sandbox/container 通过私有 Docker network 访问 eval service；所有请求发送
  `X-Session-ID`。
- worker 数量至少覆盖预期并发长请求；HTTP/反向代理 read timeout 至少为 body timeout
  + 30 秒，官方客户端使用 +60 秒。
- 如果将来把服务暴露到远程网络或多租户 host，再引入认证、租户配额和更强 rate limit。

## 10. 测试与验收

- 契约：所有终态和 checks 组合 round-trip；无 revision；`kind`/`sample` 非有限值正确。
- 请求校验：空/重复 checks、performance-without-correctness、非法 session header。
- 同步 API：submit → poll → terminal 一次返回；排队/运行超时返回 synthetic TIMEOUT；
  后台 task 不被取消。
- session lease：同 session 并发返回 429；不同 session 可提交；worker 终态释放 lease。
- 塑形：fixture engine JSON 映射到 typed correctness/performance/op-profile/deep-profile。
- divergence：七种 synthetic pattern；非有限 sample 排序与 token 化。
- 组合规则：correctness fail 跳过 performance 但不跳过 profile；decoy 同样不跳过 profile。
- 旧路由：`/verify`、`/profile`、`/tasks/{id}` 均为 404。
- 回归：所有保留的 worker、broker、engine、profile、benchmark tests 通过。
- NPU 校准：真实失败集达到 precision > 90%、recall > 80%，否则收缩 kind 集合。

