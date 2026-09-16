# Evaluation API v2（durable resource + 可组合 checks）设计文档

> Historical design. The active interface and deployment decisions are specified
> in [the v2.1 cutover](../architecture/evaluations-v21-contract.md). Compatibility windows described below are retired.

- 日期：2026-08-06
- 最近修订：2026-08-13
- 状态：目标契约已批准；durable runner、调用方隔离和消费方迁移待实现；真实 NPU
  divergence/measurement 校准是发布门禁
- 规范来源：本文件是 Evaluation API v2 `schema_version="2.0"` 的唯一 canonical design；
  已被取代的同名副本和旧 implementation plan 已从 workspace 删除。消费方通过
  pinned JSON schema/golden fixture digest 校验 wire contract，不复制或重新解释第二套 schema。
- 位置：`triton_evaluation/api/`（HTTP resource + typed models）、
  `triton_evaluation/evaluators/`（engine JSON → stable models）
- 取代：v1 `/verify`、`/profile`、隐式 benchmark，以及 v2 早期的同步-only 草案
- 关联消费方：framework-free kernel harness、Eval Hub durable runner、trusted observer

## 1. 目标、边界与术语

Evaluation API v2 提供一个小而稳定的 JSON/HTTP 契约。评测是可恢复查询的
`EvaluationResource`，而不是绑定到一次 HTTP 长连接的返回值；correctness、performance 和
profile 是有依赖关系的可组合 checks；engine 的私有 JSON 只能经 typed shaping 后出现在
正式响应中。

目标：

- 让断连或 Hub 重启后可以用 `evaluation_id` 恢复，不重复消耗 NPU；
- 明确区分 resource 生命周期、check 执行结果和 kernel 领域结果；
- 保留 correctness → performance 顺序、decoy、结构化 divergence sample、非有限值安全编码；
- 给正式 grade 提供 manifest-bound reference、request/reference digest 和可复现 provenance；
- 保持内部研究部署简单：单机多卡、Redis queue、worker 和 caller-scoped opaque capability。

非目标：

- 不引入 Kafka、Celery、Kubernetes、工作流平台或复杂微服务拆分；
- 不把服务变成可执行任意代码的通用 runner；
- v2 不支持取消已经接受的 evaluation；
- 不设计 OAuth、证书签发或通用 TLS 平台。若服务越过当前受控 bridge/host 信任边界，必须
  单独进行安全评审；
- 不公开卡忙闲、队列深度、Redis task 或 worker 管理 API。

本文使用以下固定名称：

- `EvaluationResource`：持久化生命周期和可查询句柄；
- `EvaluationResult`：terminal resource 的逐 check 结果；
- `CheckOutcome`：每个 requested check 的 discriminated outcome；
- `ErrorInfo`：稳定、可重试判定且已脱敏的错误；
- `Provenance`：影响 correctness、measurement 或 reward 的版本与环境；
- `GradeRecord`：Hub/standalone trusted coordinator 持久化的不可变 scoring disposition；发生正式
  Evaluation 时引用其 ID、digests/result/provenance，正式提交前取消时保留明确的 unscored 处置。

## 2. 已批准的核心决策

| 主题 | 决策 |
|---|---|
| 资源 | `POST /evaluations` 创建或重放；`GET /evaluations/{evaluation_id}` 查询 |
| 幂等 | POST 必须带 `Idempotency-Key`；作用域是已认证 caller |
| 同步等待 | POST 可短暂等待；未终态返回 `202 + Location`，已终态返回 `200` |
| HTTP 等待 | 等待预算结束或客户端断连不改变任务状态，客户端随后 GET |
| 生命周期 | `queued → running → terminal`，状态只能单调前进，terminal 不可改写 |
| 执行语义 | 至少一次领取、fenced terminal commit；不声称 exactly-once execution |
| deadline | 接受时固定绝对 `deadline_at`；过期任务不启动，晚到 attempt 不得覆盖 terminal |
| 检查顺序 | correctness → performance；profile 独立且在 performance 后执行 |
| subject outcome | correct/wrong/compile error/runtime error/subject execution timeout 都是 correctness `succeeded`；仅 correct 通过 |
| 评分边界 | 非 correct 的 subject outcome 可评分且 reward=0；evaluator infra/check deadline/resource deadline 才 unscored |
| 身份 | `X-Session-ID` 只用于 correlation/次级配额；身份来自 caller-scoped capability |
| 正式成绩 | 仅 trusted Hub 或被授权 observer 可提交 `purpose="grade"`，且 reference 必须绑定 manifest |
| 版本 | URL 可保持无版本前缀；payload `schema_version` 和兼容策略是必需的 |
| 调度/部署 | FIFO、单机 Redis + workers、受控 bridge gateway、部署前 preflight |

## 3. HTTP API

### 3.1 通用 headers、身份与限额

生产请求使用：

```http
Authorization: Bearer <caller-scoped opaque capability>
X-Session-ID: <correlation-id>
Idempotency-Key: <required-on-POST>
Content-Type: application/json
```

- capability 在服务端静态配置为 `trial:submit/read`、`grade:submit/read` 或
  `observer:read` scope，并映射到稳定 `caller_id`；日志、资源可见性、幂等和主限额均按
  caller 计算。响应绝不回显 credential。
- interactive harness 使用 trial capability，只能提交 `purpose="trial"`。trusted Hub 使用
  grade capability；trusted observer 默认只读，只有显式授予 `grade:submit` 后才能发起正式
  grade。trial 与 trusted traffic 使用独立并发桶和速率桶，trusted 容量不被 trial 耗尽。
  具体阈值是部署配置，但必须有非零上限并通过测试。
- `X-Session-ID` 必填，只允许 1–128 个 `[A-Za-z0-9._:-]` 字符。它只做 correlation 和
  caller 内的次级在途配额，不能授权资源读取，不能证明用户、租户或 grade 身份。
- `Idempotency-Key` 在 POST 上必填，只允许 1–128 个可见 ASCII 字符（不含空白和控制字符）。
- 默认请求 body 上限 4 MiB；`reference_code` 和 `kernel_code` 分别最多 2 MiB UTF-8；
  `op_name` 1–128 字符；checks 最多 3 个；profile case selection 最多 128 个。
  超限在创建 resource 前拒绝。

内部部署可以用 mounted secret file 提供 opaque capabilities；本文不扩展为身份平台。开发模式
可以显式启用独立的 development credential，但不得作为生产默认值。

### 3.2 `POST /evaluations?wait_seconds=N`

`wait_seconds` 是可选 HTTP 等待预算，取值 0–60 秒，默认 0。它不进入规范化 request，也不
改变 evaluation deadline。请求示例：

```json
{
  "schema_version": "2.0",
  "purpose": "trial",
  "op_name": "addmm",
  "reference_code": "import torch ... class Model ...",
  "kernel_code": "import torch ... class ModelNew ...",
  "triton_impl_name": "triton_ascend_impl",
  "checks": ["correctness", "performance"],
  "non_compute": false,
  "deadline_seconds": 900,
  "performance": {"warmup": 5, "repeats": 50},
  "profile": null,
  "grade_context": null
}
```

规则：

- `checks` 非空、不得重复；服务固定按 correctness → performance → profile 执行，数组顺序
  不改变执行语义。performance 出现时 correctness 也必须出现。
- 对应 check 未请求时，其参数必须为 `null` 或省略；请求后规范化为显式默认值。
- `deadline_seconds` 取值 10–3600，默认 900。服务接受时保存
  `deadline_at = accepted_at + deadline_seconds`，后续重放不延长 deadline。
- `purpose="trial"` 时 `grade_context` 必须为空，结果永远 `grade_eligible=false`。
- `purpose="grade"` 只允许 `grade:submit` caller，且必须提供 `grade_context`。
  `manifest_id` 标识 grader 运维在 benchmark/task 发布时预注册的 immutable release manifest；
  它不是 Eval Hub 为某次 run 生成的 `RunManifest` / `EvalManifest`，也不能由调用方动态注册。
  一个已发布的 `manifest_id` 永远不能重绑到不同 problem/reference；任何内容变化都发布新 ID。
- 最小 catalog 是随 grader 部署、只读挂载的受信 release 目录。每个条目固定
  `(manifest_id, problem_id, reference_digest, reference artifact)`；catalog 更新是离线
  benchmark release + grader redeploy/preflight 操作，API 不提供 manifest 注册或修改端点。
- Eval Hub 或 standalone trusted coordinator 只按 `(manifest_id, problem_id)` 从自身配置的同一
  受信 release 目录解析 canonical `reference_code`，计算 SHA-256，并填入 request 与
  `grade_context`；不得采用 answer、用户上传内容或 per-run manifest 中的 reference。解析必须
  拒绝目录穿越和逃逸受信目录的 symlink。
- service 独立在其 catalog 中校验 `(manifest_id, problem_id, reference_digest)` 三元组，并校验
  body 中 `reference_code` UTF-8 bytes 的 SHA-256 与绑定 digest 一致。任一步失败都不创建
  resource。

返回：

- resource 在等待预算内 terminal：HTTP 200 + `EvaluationResource`；
- 等待预算结束时仍为 queued/running：HTTP 202 + 当前 `EvaluationResource`，并带
  `Location: /evaluations/{evaluation_id}`；
- `wait_seconds=0` 在创建或重放非终态 resource 时直接返回 202；
- 客户端断连、反向代理 read timeout 或 wait budget 用尽都不合成任务 `timed_out`，也不取消
  后台任务。客户端保留 `evaluation_id` 并查询 Location。

#### 幂等与 request digest

幂等作用域为 `(caller_id, Idempotency-Key)`。服务在同一事务中写入 idempotency record、
`EvaluationResource` 和 queue entry：

- 同 key + 同规范化 request：返回原 `evaluation_id`；当前终态返回 200，否则返回 202；
- 同 key + 不同规范化 request：HTTP 409 `IDEMPOTENCY_CONFLICT`；
- 并发的相同请求只能创建一个 resource 和一个初始 queue entry。

规范化 request 是 schema 校验并补齐默认值后的 canonical JSON：object key 排序、UTF-8、
checks 按固定执行顺序、case indices 排序；源码字符串的字节内容不做 trim 或换行改写。
`wait_seconds`、`X-Session-ID` 和 transport headers 不参与规范化。`request_digest` 是该字节串的
小写 hex SHA-256，`reference_digest` 是 `reference_code` UTF-8 bytes 的小写 hex SHA-256。
idempotency record 与 resource 至少保留 24 小时且使用同一公开配置；生产 Hub 的保留窗口必须
覆盖最长 runner 恢复窗口。过期策略不得在 resource 尚非终态时删除记录。

### 3.3 `GET /evaluations/{evaluation_id}`

caller 必须拥有该 resource 的 read scope；trusted observer 可按显式 observer policy 读取。
存在且可见时始终返回 HTTP 200 + 当前 `EvaluationResource`。不存在和不可见统一返回 404，
避免枚举其他 caller 的 ID。非终态响应可带 `Retry-After: 1`；客户端使用有上限的退避。
`deadline_at` 只约束 service execution，不是删除或停止查询的时间：到点后 client 仍以 GET 恢复
terminal resource，直到协调器冻结且不超过 service retention 的 `max_recovery_age`；不得盲目用新 key
重提 POST。

Eval Hub 必须在首次 POST 后立即持久化 `evaluation_id`。Hub 进程重启、HTTP 断连或等待超时后
通过 GET 恢复；只有在没有拿到 POST 响应时，才用原 `Idempotency-Key` 和原 request 安全重放
POST。Hub 持久化的是 service `evaluation_id`，不是 Redis task/attempt ID。

### 3.4 HTTP error envelope

所有非 2xx 响应，包括框架 validation error，都使用同一 envelope：

```python
class ErrorInfo(StrictModel):
    code: str                 # 稳定、大写 snake case，1..64 chars
    phase: Literal["request", "auth", "admission", "queue", "execution",
                   "shaping", "storage", "deadline"]
    check: Check | None
    retryable: bool
    message: str              # 已脱敏、单行、最多 512 UTF-8 bytes

class ErrorEnvelope(StrictModel):
    schema_version: Literal["2.0"]
    request_id: str
    error: ErrorInfo
```

message 不得包含源码、输入值、credential、绝对路径、内部地址或未截断 traceback。
稳定 HTTP 映射：malformed/非法组合为 400；无效 credential 为 401；scope 或 manifest binding
拒绝为 403；不可见 resource 为 404；幂等冲突为 409；body/字段过大为 413；schema 字段错误为
422；caller/session 限流为 429；未 ready 或 admission 暂不可用为 503；未预期服务错误为 500。
429/503 应带 `Retry-After`。resource 创建后的执行错误写入 resource 的 check outcomes，不再用
HTTP 5xx 隐藏已创建的 durable resource。

初始稳定 error codes 为 `INVALID_REQUEST`、`UNAUTHENTICATED`、`FORBIDDEN`、
`MANIFEST_BINDING_FAILED`、`NOT_FOUND`、`IDEMPOTENCY_CONFLICT`、`LIMIT_EXCEEDED`、
`RATE_LIMITED`、`NOT_READY`、`QUEUE_UNAVAILABLE`、`WORKER_LOST`、
`EVALUATOR_EXECUTION_FAILED`、`INVALID_ENGINE_OUTPUT`、`DEADLINE_EXCEEDED` 和 `INTERNAL`。
新增 code 是兼容变更；既有 code 的含义和 retryable 值不能静默改变。客户端只依据
`retryable` 和 HTTP status 决定 transport retry，未知 code 默认不可重试；重试 POST 仍必须使用
原 idempotency key/request。

`ErrorInfo` 只描述 transport、admission、worker、engine protocol、storage、orchestration 或
evaluator deadline 等基础设施错误；其 `phase="execution"` 也指 evaluator 执行阶段，而非被测
kernel 的领域结论。已经由健康 harness 明确归因到 candidate 的编译错误、运行错误和单次被测
执行超时使用第 4.2 节 `DomainFailure`，绝不映射为 `ErrorInfo`。反之，来源不明的进程消失、
worker/engine crash、无效 engine output 或 lease/deadline 丢失必须 fail closed 为 failed/timeout
`CheckOutcome`，绝不猜测为 candidate failure。

### 3.5 存活与就绪

- `GET /livez` 只证明 API 进程能响应，不访问 Redis/NPU；
- `GET /readyz` 检查 schema/storage 可写、Redis 可达以及至少一个未过期的兼容 worker
  heartbeat；失败返回 503 的 `ErrorEnvelope`；
- 两个端点都不公开队列深度、worker 数、卡 busy/idle、caller 信息或内部地址。

## 4. 正式 schema 与不变量

以下是契约级伪代码。服务端 request parser 和 response/shaping model 都继承 `StrictModel`，
其配置为 `extra="forbid"`，防止 engine 私有字段意外泄露；独立 consumer 为兼容同 major 的
新增 optional response 字段，应使用 tolerant response reader。request 不做字符串到数值或
bool 的宽松 coercion。所有普通 JSON float 字段必须为 finite；NaN/Inf 只允许出现在显式
sentinel union 中。

```python
Check = Literal["correctness", "performance", "profile"]
Purpose = Literal["trial", "grade"]
ResourceState = Literal["queued", "running", "completed", "failed", "timed_out"]

class PerformanceParams(StrictModel):
    warmup: int = Field(default=5, ge=0, le=100)
    repeats: int = Field(default=50, ge=1, le=1000)

class ProfileParams(StrictModel):
    mode: Literal["op", "deep"] = "op"
    case_indices: list[int] | None = Field(default=None, max_length=128)
    max_kernels_per_case: int = Field(default=200, ge=1, le=1000)
    include_recommendations: bool = True

class GradeContext(StrictModel):
    manifest_id: str = Field(min_length=1, max_length=128)  # immutable release manifest
    problem_id: str = Field(min_length=1, max_length=128)
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

class EvaluationRequest(StrictModel):
    schema_version: Literal["2.0"]
    purpose: Purpose = "trial"
    op_name: str = Field(min_length=1, max_length=128)
    reference_code: str
    kernel_code: str
    triton_impl_name: str = Field(default="triton_ascend_impl", min_length=1, max_length=128)
    checks: list[Check] = Field(min_length=1, max_length=3)
    non_compute: bool = False
    deadline_seconds: int = Field(default=900, ge=10, le=3600)
    performance: PerformanceParams | None = None
    profile: ProfileParams | None = None
    grade_context: GradeContext | None = None
```

`case_indices` 必须非负且唯一，规范化后升序。任何 request float 扩展字段也必须由统一 finite
validator 拒绝 NaN/Inf；JSON parser 必须拒绝非标准 `NaN`、`Infinity` token。

### 4.1 Resource、result 与 CheckOutcome 三层

```python
class EvaluationResource(StrictModel):
    schema_version: Literal["2.0"]
    evaluation_id: str
    revision: int
    state: ResourceState
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    deadline_at: datetime
    request_digest: str
    reference_digest: str
    result: EvaluationResult | None

class SkipInfo(StrictModel):
    code: Literal["DEPENDENCY_NOT_PASSED", "PRIOR_CHECK_FAILED", "RESOURCE_DEADLINE"]
    depends_on: Check | None = None
    message: str

class EvaluationResult(StrictModel):
    correctness: CheckOutcome[CorrectnessResult] | None
    performance: CheckOutcome[PerformanceResult] | None
    profile: CheckOutcome[ProfileResult] | None
    provenance: Provenance

class CheckSucceeded[T](StrictModel):
    status: Literal["succeeded"]
    data: T
    error: None = None
    reason: None = None

class CheckFailed(StrictModel):
    status: Literal["failed"]
    data: None = None
    error: ErrorInfo
    reason: None = None

class CheckSkipped(StrictModel):
    status: Literal["skipped"]
    data: None = None
    error: None = None
    reason: SkipInfo

class CheckTimedOut(StrictModel):
    status: Literal["timeout"]
    data: None = None
    error: ErrorInfo
    reason: None = None

type CheckOutcome[T] = Annotated[
    CheckSucceeded[T] | CheckFailed | CheckSkipped | CheckTimedOut,
    Field(discriminator="status"),
]
```

`SkipInfo.message` 是最多 256 bytes 的脱敏单行文本。新增 skip code 是兼容变更；客户端对未知
skip code 保持 skipped、不能推断为 succeeded。

不变量：

- queued/running 时 `result=null`、`finished_at=null`；terminal 时二者非空；started_at 只在
  至少一个 attempt 实际开始后非空。`revision >= 1` 且每次可见 mutation 严格递增，
  `updated_at` 同步更新。
- 每个 requested check 的 outcome 必须非空，未请求 check 必须为 `null`。即使 resource 因
  deadline 或 orchestration failure 终止，也要把 active check 填为 timeout/failed、未开始
  checks 填为 skipped。
- `succeeded` 当且仅当 data 非空且 error/reason 为空；`failed`/`timeout` 当且仅当 data 为空且
  error 非空；`skipped` 当且仅当 data/error 为空且 reason 非空。
- `state="completed"` 表示 orchestration 正常得到所有 requested outcomes，不表示 kernel 正确。
  `failed` 只表示 resource 级 orchestration/storage protocol 失败；`timed_out` 只表示任务绝对
  deadline 到达。check 级失败仍由 outcome 表达。
- `CheckOutcome.status` 判断 evaluator 是否得到可信的领域结论；`CorrectnessResult.outcome`
  判断 candidate 的领域结论。correct、wrong、compile_error、runtime_error 和
  execution_timeout 都是 `status="succeeded"`；后四者 `passed=false`、可评分且 reward=0。
- worker/engine/storage/orchestration failure、check deadline 或 resource deadline 没有可信的
  subject outcome，必须是 failed/timeout（或依赖 skipped）且 unscored，绝不能放进
  `CorrectnessResult`。取消不是 API 状态；Hub 取消 run 只会令该 run 的 grade unscored，不把
  service result 改写为任何 subject outcome。

### 4.2 Correctness 与非有限 sample

校准前稳定的 divergence taxonomy 刻意保持粗粒度：

```python
FiniteOrSentinel = float | Literal["inf", "-inf", "nan"]
DivergenceKind = Literal["shape_mismatch", "value_mismatch", "non_finite_mismatch"]
CorrectnessDomainOutcome = Literal[
    "correct", "wrong", "compile_error", "runtime_error", "execution_timeout"
]

class CaseMetrics(StrictModel):
    max_abs_diff: float | None = None
    matched_ratio: float | None = None
    mere: float | None = None
    small_count: int | None = None
    normal_count: int | None = None
    atol: float | None = None
    rtol: float | None = None

class SampleElement(StrictModel):
    idx: list[int]
    candidate: FiniteOrSentinel
    reference: FiniteOrSentinel
    diff: FiniteOrSentinel

class CaseFailure(StrictModel):
    case_idx: int
    kind: DivergenceKind
    metrics: CaseMetrics
    sample: list[SampleElement] = Field(default_factory=list, max_length=8)

class DomainFailure(StrictModel):
    code: Literal[
        "SUBJECT_COMPILE_ERROR", "SUBJECT_RUNTIME_ERROR", "SUBJECT_EXECUTION_TIMEOUT"
    ]
    phase: Literal["compile", "initialize", "execute"]
    case_idx: int | None = Field(default=None, ge=0)
    message: str  # redacted, single-line, 1..512 UTF-8 bytes

class CorrectnessResult(StrictModel):
    outcome: CorrectnessDomainOutcome
    passed: bool
    decoy: bool = False
    passed_cases: int
    total_cases: int
    failures: list[CaseFailure]
    domain_failure: DomainFailure | None = None
```

- `total_cases >= 1`、`0 <= passed_cases <= total_cases`；`passed` 当且仅当
  `outcome="correct"`。`failures` 的 `case_idx` 唯一且在范围内，并满足
  `passed_cases + len(failures) <= total_cases`。
- `correct` 要求 `passed_cases=total_cases`、`failures=[]`、`domain_failure=null`、
  `decoy=false`。`wrong` 要求所有 cases 已完成、`domain_failure=null`，且至少有一个 mismatch
  failure 或 `decoy=true`；两者均为可评分的 subject outcome。
- `wrong` 还要求 `passed_cases + len(failures) = total_cases`。`compile_error`、`runtime_error`、
  `execution_timeout` 要求 `domain_failure` 非空、code 与 outcome 一一对应、`decoy=false`，并允许
  保留 failure 之前已完成 cases 的计数/数值 mismatch。compile_error 固定
  `phase="compile", case_idx=null`；runtime_error 的 phase 为 initialize/execute；
  execution_timeout 可发生在任何受控 candidate phase。compile/initialize 的 case_idx 必须为空；
  execute 的 case_idx 必须非空且在范围内。
- `DomainFailure` 仅描述被测 candidate。message 必须单行、脱敏且最多 512 UTF-8 bytes，不得包含
  源码、输入值、credential、绝对路径或 traceback。reference/manifest 自身无法编译或运行属于
  evaluator/benchmark infrastructure failure，不得归因给 candidate。它刻意不含 `retryable`；
  subject outcome 的评分规则固定，transport retry 只能由 `ErrorInfo` 驱动。
- `runtime_error` 只接受健康 harness 对 candidate exception/exit/signal 的 typed attribution。
  `execution_timeout` 仅在健康 harness 仍持有有效 lease，且能权威确认 candidate phase 超过
  evaluator 的 per-subject execution budget 时产生；该 budget 是 versioned
  `correctness_algorithm` 的一部分。resource/check deadline、worker/engine crash、lease 丢失或
  无法判定归因的超时必须产生 timeout/failed `CheckOutcome`，并保持 unscored。
- 数值、shape 或非有限值 mismatch 必须有稳定 `kind`，并只用于 `wrong` 的 case diagnostics。
- sample 最多 8 点，按有限 `abs(diff)` 降序并用 index 打破平局；`idx` 是行主序多维坐标。
  shape mismatch 的 sample 为空。NaN/Inf 编码为字符串 sentinel，普通 float 字段严格 finite。
- `mere` 仅在 engine 有有效 finite 值时返回。删除 engine 从不提供的 `mean_abs_err`。
- 更细的 `constant_offset`、`per_axis`、`precision_loss`、`sign_flip`、`value_clamp`、
  `scattered` 仅可进入明确标注 experimental 的版本化诊断字段，不能用于 reward 或稳定消费。
  真实 NPU 校准完成前，该字段默认不发布。

以下两个 correctness outcome 片段说明评分边界。candidate 编译错误是成功得到的领域结论，可评分
且 reward=0：

```json
{
  "status": "succeeded",
  "data": {
    "outcome": "compile_error",
    "passed": false,
    "decoy": false,
    "passed_cases": 0,
    "total_cases": 8,
    "failures": [],
    "domain_failure": {
      "code": "SUBJECT_COMPILE_ERROR",
      "phase": "compile",
      "case_idx": null,
      "message": "candidate compilation failed"
    }
  },
  "error": null,
  "reason": null
}
```

worker 丢失没有可信 subject outcome，只能 unscored：

```json
{
  "status": "failed",
  "data": null,
  "error": {
    "code": "WORKER_LOST",
    "phase": "execution",
    "check": "correctness",
    "retryable": true,
    "message": "worker lease expired before a subject outcome was committed"
  },
  "reason": null
}
```

### 4.3 Performance、聚合与 coverage

```python
class InputShape(StrictModel):
    input_index: int
    dimensions: list[int]
    dtype: str | None = None

class PerformanceCase(StrictModel):
    case_idx: int
    input_shapes: list[InputShape]
    implementation_latency_ms: float
    reference_latency_ms: float
    speedup_vs_reference: float

class UnmeasuredPerformanceCase(StrictModel):
    case_idx: int
    input_shapes: list[InputShape]
    reason_code: Literal["UNSUPPORTED_CASE", "NO_STABLE_MEASUREMENT"]

class PerformanceResult(StrictModel):
    aggregation: Literal["geometric_mean_v1"]
    overall_speedup_vs_reference: float
    selected_cases: int
    measured_cases: int
    coverage: float
    per_case: list[PerformanceCase]
    unmeasured_cases: list[UnmeasuredPerformanceCase]
```

- `case_idx` 在两个列表间唯一；`input_shapes` 来自结构化 input descriptors，禁止压缩为字符串。
  dimensions 必须是非负整数；没有 tensor input 时允许空列表。
- latency 和 speedup 必须 finite 且 `> 0`。删除 engine 不提供可靠 op-count 时永远为空的
  `throughput`。
- `selected_cases = measured_cases + len(unmeasured_cases)`，
  `measured_cases = len(per_case)`，`coverage = measured_cases / selected_cases`，并要求
  `selected_cases > 0`、`measured_cases > 0`。coverage 必须在 `[0, 1]`。
- `overall_speedup_vs_reference` 是所有 measured case 的
  `exp(mean(log(speedup_vs_reference)))`，不按 shape 或 latency 加权；算法固定为
  `geometric_mean_v1` 并记录在 provenance。消费方必须同时展示 overall 和 coverage。
- 单个 case 不支持或无法获得稳定样本时进入 `unmeasured_cases`；benchmark runner 整体无法
  运行、整体结果不稳定或没有任何有效 measurement 时 outcome 是 failed + `ErrorInfo`。
  correctness check 只有在 `status="succeeded"` 且 `data.outcome="correct"` 时才能进入
  performance；其余四种 subject outcomes 均使 performance skipped
  `DEPENDENCY_NOT_PASSED`。correctness failed/timeout 时 performance 同样 skipped，但整个正式
  grade 保持 unscored。skipped 不是裸 `None`；只有未请求 performance 时字段才为 null。

### 4.4 Profile discriminated union 与计数

```python
ProfileCaseStatus = Literal["complete", "partial", "failed"]

class ProfileCounts(StrictModel):
    selected: int
    complete: int
    partial: int
    failed: int

class OpKernel(StrictModel):
    op_name: str
    time_ms: float
    percentage: float
    metrics: OpKernelMetrics = Field(default_factory=OpKernelMetrics)

class OpKernelMetrics(StrictModel):
    # msprof op 采集的逐 kernel 利用率,统一为 [0, 1] 分数:
    # pipe 比率原样(采集即为分数),Memory/L2 带宽与命中率列名带 (%),
    # 归一为分数;采集未产出的指标保持 None。
    aic_cube_ratio: float | None = None
    aic_mte2_ratio: float | None = None
    aic_fixpipe_ratio: float | None = None
    aiv_vec_ratio: float | None = None
    aiv_mte2_ratio: float | None = None
    aiv_mte3_ratio: float | None = None
    gm_to_l1_usage: float | None = None
    l0c_to_gm_usage: float | None = None
    gm_to_ub_usage: float | None = None
    ub_to_gm_usage: float | None = None
    aic_l2_hit: float | None = None
    aiv_l2_hit: float | None = None

class OpProfileCase(StrictModel):
    case_idx: int
    status: ProfileCaseStatus
    kernels: list[OpKernel]
    error: ErrorInfo | None = None

class DeepRatios(StrictModel):
    mte2: float | None = None
    cube: float | None = None
    vec: float | None = None
    fixp: float | None = None
    mte3: float | None = None
    scalar: float | None = None

class DeepProfileCase(StrictModel):
    case_idx: int
    status: ProfileCaseStatus
    target_kernel: str | None
    bound: Literal["MTE2", "CUBE", "VEC", "FIXP", "MTE3", "SCALAR", "NONE"] | None
    ratios: DeepRatios
    per_core_spread_ratio: float | None
    recommendations: list[str]
    missing_metrics: list[str]
    warnings: list[str]
    error: ErrorInfo | None = None

class OpProfileResult(StrictModel):
    mode: Literal["op"]
    counts: ProfileCounts
    cases: list[OpProfileCase]

class DeepProfileResult(StrictModel):
    mode: Literal["deep"]
    counts: ProfileCounts
    cases: list[DeepProfileCase]

ProfileResult = Annotated[OpProfileResult | DeepProfileResult, Field(discriminator="mode")]
```

- `selected == complete + partial + failed == len(cases)`；case_idx 唯一；每个 case 的 status 与
  相应计数一致。
- complete case 的 `error=null`；failed case 的 `error` 必填；partial case 只有在存在局部
  可用数据时允许，并必须通过 `missing_metrics`、warnings 或 error 解释缺口。
- `DeepRatios` 是 `mte2/cube/vec/fixp/mte3/scalar` 六个 optional finite float 的 strict
  model，所有非空值在 `[0, 1]`，不能出现其他 key。op case 内每个 time/percentage 必须
  finite、非负；非空 kernels 的 percentage 总和允许 measurement rounding 误差但必须在
  `[99.0, 101.0]`。
- profile 全局 deadline 到达时使用 timeout `CheckOutcome`，不在 data 中再放含混的
  `timed_out` bool。profile runner 正常结束但部分 case 缺指标时可以返回 succeeded outcome +
  discriminated partial cases。

### 4.5 Provenance 与 GradeRecord

```python
class ExecutionEnvironment(StrictModel):
    hardware_model: str
    hardware_count: int
    driver_version: str
    runtime_version: str
    torch_version: str | None
    torch_npu_version: str | None
    triton_version: str | None
    toolchain_version: str

class Provenance(StrictModel):
    schema_version: Literal["2.0"]
    evaluator_build: str             # admission 时固定的 image digest or git SHA
    correctness_algorithm: str
    decoy_algorithm: str
    benchmark_algorithm: str
    profiler_algorithm: str
    grade_eligible: bool
    manifest_id: str | None
    problem_id: str | None
    execution_environment: ExecutionEnvironment | None
```

resource 在 admission 时固定目标 `evaluator_build` 和 algorithm versions；environment 字段来自
实际 worker 而非 API 启动环境。没有 attempt 的 queued deadline 可以令
`execution_environment=null`；任何包含 correctness/profile data 或 performance measurement 的
result 都必须有完整 environment，缺失时相应 check 只能 failed。影响 correctness、decoy、
测量、聚合或 reward 的变化必须更新对应 algorithm version 或 `evaluator_build`。不得包含
hostname、设备序列号或 credential。trial 的 manifest/problem 必须为空且
`grade_eligible=false`；只有 manifest 校验通过、resource completed 且所有 reward 所需
provenance 完整的 trusted grade 才能为 true。

`GradeRecord` 总是持久化 coordinator 已冻结的 run/item/attempt、catalog/task、evaluator/reward-policy、
resource-usage 和 scoring-disposition 字段。只有从 terminal formal Evaluation result 派生时，
`evaluation_id`、request/reference digest、完整 `EvaluationResult`、result artifact digest 和
`Provenance` 才作为一组全部必填。正式 POST 前 coordinator cancellation 仍创建一条 GradeRecord：
这些 formal Evaluation 字段全部为 null，`outcome="canceled"`、correctness unknown、
`scoring_state="unscored"`、numeric reward 为 null、`unscored_reason="COORDINATOR_CANCELED"`、
`train_eligible=false`、`retryable=false`；若 final artifact 尚未冻结，其 digest 也为 null。若取消发生
在 formal request 或 `evaluation_id` 已持久化之后、可信 terminal result 之前，则保留所有已持久化的
request/ID/evidence 字段，尚不存在的 result/provenance 字段保持 null。取消 CAS 获胜后，任何晚到的
Evaluation 响应都不得改写 canceled GradeRecord 或 run/item 终态；其他未取消 consumer 可独立读取同一
service resource。`grade_eligible` 只证明 caller/reference/provenance 满足正式 grade 的信任前提，
不等于本次 result 已经 scored。Hub/standalone trusted coordinator 使用以下唯一映射：

- correctness succeeded 且 subject outcome 为 correct：按 manifest 固定 reward 规则评分；若该
  reward 要求 performance，则 performance 必须 succeeded 且满足 manifest 固定的 coverage policy，
  否则保留 correctness=correct，但 `scoring_state="unscored"`、numeric reward 为空，并记录稳定的
  `REQUIRED_PERFORMANCE_UNAVAILABLE` reason；是否以新 attempt 重试只由 coordinator 的冻结错误策略决定；
- correctness succeeded 且 outcome 为 wrong（包括 `decoy=true`）、compile_error、runtime_error 或
  execution_timeout：`scoring_state="scored"` 且 numeric reward 精确为 0；dependency-skipped
  performance 不改变它。decoy flag 必须保留，v1 consumer 不把 decoy 放入训练，且不得重试它；
- correctness failed/timeout/skipped、resource failed/timed_out、或所需 evaluator check
  failed/timeout：`scoring_state="unscored"` 且不得写入伪装成成绩的 numeric 0；
- coordinator 取消某次 run 不取消或改写 service resource；按上段三阶段字段规则创建不可变 unscored
  `COORDINATOR_CANCELED` GradeRecord，且 cancellation 不可重试。

Hub 不得从 trial result 生成正式 grade，也不得在缺 digest/provenance 时默认为可评分。正式
GradeRecord 还必须保存 scoring_state；unscored 时保存稳定 reason，而非从自由文本推断。

## 5. Check orchestration 与 shaping

- `verify.py` 可 additive 地输出稳定 coarse `kind` + sample；已有精度判定不变。
- `benchmark.py` / `profile_runner.py` 的 raw 输出不是 API。`evaluators/shaping.py` 是唯一
  engine JSON → stable domain model 的边界，必须拒绝额外字段、非有限普通 float、计数矛盾和
  缺失 case identity。
- evaluator 固定顺序执行 correctness → performance → profile。performance 只在
  correctness check succeeded 且 `data.outcome="correct"` 时运行，不能仅凭容易漂移的派生
  `passed` 字段判断。
- profile 不依赖 correctness：wrong answer 或 decoy 仍可 profile。若更早的基础设施错误使
  worker 环境不可信，profile 可 skipped `PRIOR_CHECK_FAILED`，不能伪装成未请求的 null。
- 一个 check 的 infrastructure failure 只产生 failed/timeout outcome；candidate
  compile/runtime/subject-timeout 则产生 succeeded correctness + typed domain data。是否继续
  独立 check 由上述依赖规则决定。已有 typed partial profile 可以保留在成功的 profile data，
  failed/timeout outcome 不携带 data，避免 `data + error` 双重语义。
- response 永不返回 `reference_code`、`kernel_code`、原测试输入、engine 私有对象或 traceback。

## 6. Durable runner、状态机、deadline 与 fencing

唯一状态机：

```text
queued ─────▶ running ─────▶ completed
  │              ├────────▶ failed
  │              └────────▶ timed_out
  ├───────────────────────▶ failed
  └───────────────────────▶ timed_out
```

terminal 集合为 `{completed, failed, timed_out}`。任何 terminal → 其他状态、running → queued、
或 terminal 内容改写都非法。持久化 record 带单调 `revision`，所有状态更新使用 CAS。

最小执行协议：

1. POST transaction 原子创建 resource/idempotency record 并入队；重复 delivery 可以发生。
2. worker 领取时检查 `now < deadline_at`，以 CAS 将 queued 变 running 或在已 running resource
   上取得新 attempt；每次领取获得唯一 `attempt_id` 和单调 `fencing_token`。
3. running attempt 定期续租并 heartbeat。只有当前 token 可以续租、写 check 临时状态和提交
   terminal result；每次写同时校验 resource revision 与非 terminal。
4. lease 过期时 reaper 在 deadline 前可签发更高 token 给 replacement attempt，resource 仍是
   running；旧 attempt 立即失去写权限。若不能安全重试则 CAS 为 failed。
5. queued resource 到达 deadline 时直接 CAS timed_out；running resource 到达 deadline 时由
   watchdog/worker CAS timed_out 并提高 fence。过 deadline 的 queue delivery 不启动 evaluator。
6. 晚到、重复或失去 lease 的 attempt 只能记录 `fenced_write_rejected` metric，绝不能覆盖
   terminal state/result。terminal write、resource result 和 idempotency mapping 在一个原子
   storage transition 中完成。

这提供 durable at-least-once attempt 和 single-winner terminal commit，不提供 exactly-once
execution。worker crash、API crash、Redis reconnect 和客户端断连不能丢失已接受的 resource。
取消仍是 non-goal；Hub 取消 run 时只停止新提交并可忽略晚到结果。

“durable”在本部署中至少覆盖 API/worker/Redis 进程重启和主机进程崩溃：resource、idempotency
record、queue 和 terminal result 必须落在 crash-persistent storage；当前最小 Redis 方案必须
启用持久化并在返回首次 2xx 前确认对应事务达到配置的 fsync durability，不能使用纯内存配置。
灾难级磁盘丢失和跨机容灾不在 v2 范围内。

## 7. 兼容与迁移

URL 不要求版本化，但每个顶层 request、resource、error envelope，以及每个独立持久化的
artifact wrapper 都带 `schema_version`；嵌套的 result、check outcome 和 `ErrorInfo` 继承其顶层
对象版本，不机械重复版本字段。版本采用 `MAJOR.MINOR`：

- 同 major 只允许增加 optional response 字段或新枚举能力协商；client 必须忽略未知 response
  字段，但不能把未知 discriminant 当成功。
- 删除/重命名字段、改变默认值/聚合、放宽正式 grade 来源或改变状态/outcome 语义必须升 major。
- request 始终 `extra="forbid"`；服务可在有期限的迁移窗口同时接受当前与前一 schema major，
  但按请求版本返回对应 schema，不能静默 reinterpret。
- `evaluator_build` 和 algorithm versions 与 schema version 独立；measurement/reward 语义变化
  即使不改变 JSON shape，也必须更新 provenance。

契约发布物由本仓 typed models 可重复生成，至少包含 request、resource、error envelope 的
JSON Schema，以及 queued/running/每种 terminal outcome 和错误边界的 golden JSON fixtures。
发布记录给出这些字节的 SHA-256；生成物必须在 CI 中重建后 byte-for-byte 一致。消费方 pin
已发布 digest 并在部署 preflight/contract test 中校验；在首个发布物生成前，本文件仍是唯一
规范来源，Hub/Harness 不得自行声明一套替代 schema。

迁移顺序：

1. 先部署 durable storage、GET、idempotency、fencing、auth scopes 和 v2 schema，保持旧消费方
   路由在有期限的兼容窗口可用；
2. 再升级 Eval Hub：生成并持久化 idempotency key/evaluation_id，以 GET 恢复，保存完整
   outcomes/digests/provenance；
3. 再升级 framework-free harness：trial capability + 纯 HTTP/JSON adapter，不依赖 Eval Hub、
   FastAPI/Pydantic 或 service Python 包；
4. 观察兼容指标和 stored artifact 可读性后，移除 `/verify`、`/profile`、`/tasks/{id}` 及早期
   同步-only adapter。移除日期和支持的 schema 窗口必须在部署文档中发布。

Eval Hub adapter 返回/持久化 `EvaluationResource`，而不是把 POST 封装成不可恢复的一次同步
函数。它分别解释 resource state、check status 与 correctness/decoy/speedup；performance
skipped/failed/timeout 不能归并成 `None`。Hub 只有在 trusted、manifest-bound、
`grade_eligible=true` 时生成 `GradeRecord`。

## 8. 网络、部署与 preflight

- eval API 和 Redis Compose service 保持 `network_mode: host`。Redis 只监听
  `127.0.0.1:6379`；bridge consumer 不直接访问 Redis。
- 生产 API 只绑定 `kernel-agent-sandbox-net` 启动时实际解析出的唯一 IPv4 gateway。明确拒绝
  `0.0.0.0`、`::`、loopback、物理网卡/公网地址、缺失 network 和 gateway 不匹配。
- sandbox harness、Eval Hub 和 host observer 访问同一受控 API 地址，但分别使用 trial、grade
  和 observer capability。不能因共享 `X-Session-ID` 获得彼此权限。
- host-side `scripts/compose.sh` 的 `up`、`start`、`restart` 必须先运行 pure-stdlib
  `scripts/deployment_preflight.py`；失败时不得启动或重启，也不把 Docker socket 挂进容器。
- preflight 读取 Docker network IPAM，校验唯一 gateway、API/PROXY/EVALUATION_URL host 一致、
  固定端口、proxy bypass、rendered Compose host network、Redis loopback；还必须校验生产
  capability secret file 存在、权限不宽于部署要求、scope 不重复，以及启用 grade 时 manifest
  catalog/引用 artifacts 只读可达、release manifest identity 唯一、每个三元组唯一且 digest 与
  artifact 实际内容一致。preflight 只报告配置项名称，不打印 secret 或 reference_code。
- HTTP/proxy timeout 只需覆盖允许的 `wait_seconds` 加小量传输余量，不再要求覆盖最长任务
  deadline。客户端对 202/断连使用 GET 或幂等 POST 恢复。
- 若需要 bridge 外远程调用或多租户 host，必须先评审 TLS、credential distribution、租户隔离
  和更强 rate limit，不能只放宽 bind 地址。

## 9. 可观测性

公开 live/readiness 只提供第 3.5 节信息。内部 metrics 至少包括：

- 按 caller class/purpose/result 的 POST accepted、idempotent replay/conflict、401/403/413/429；
- resource 当前状态数、terminal 数、queue wait、run duration、end-to-end duration和 deadline
  expiry；
- 各 check 的 succeeded/failed/skipped/timeout、skip/error code、correctness domain outcome、
  pass/decoy、GradeRecord scored/unscored reason、performance coverage 和 measurement rejection；
- worker heartbeat age、lease expiry、attempt retry、fencing token advance、late/fenced write reject、
  CAS conflict 和 reaper action；
- shaping/schema compatibility error、manifest binding failure、evaluator build 与 algorithm version。

metrics/logs 不能带源码、输入、credential 或完整 session ID；高基数 evaluation/session ID 只进
受控结构化日志并按策略采样。告警至少覆盖 ready 持续失败、无新鲜 worker、deadline 异常上升、
lease/fencing 异常和 trusted grade 失败率。

## 10. 测试、验收与发布门禁

- schema/契约：所有 resource states、requested/unrequested checks 和四种 CheckOutcome round-trip；
  `extra="forbid"`、严格类型、request/body/code/case limits、所有普通 float finite；所有非 2xx
  都是统一 `ErrorEnvelope`。
- 幂等：首次创建、相同 key+相同 canonical request 的串行/并发重放、同 key 不同 request 409、
  defaults/check order/case order 规范化、不同 caller key 隔离、记录保留。
- durable HTTP：POST 200/202 + Location、GET queued/running/terminal、wait 到期不改变 state、
  客户端断连后 GET 恢复、Hub 在无 POST 响应时用同 key 恢复。
- runner：worker crash、API crash、lease expiry、replacement attempt、重复 delivery、late result、
  stale fencing token、terminal CAS race、queued/running deadline；验证 terminal 永不改写且过期
  attempt 不启动。
- domain/评分：correct、wrong、compile_error、runtime_error、execution_timeout 五种
  CorrectnessResult round-trip；全部 correctness succeeded，仅 correct passed。后四种正式 grade
  精确计 0，correct 才运行 performance；wrong/decoy 不自动跳过 profile。
- infra 边界：worker/engine/storage/orchestration failure、check timeout、resource timeout、lease
  loss 和 dependency skip 清晰分离且 unscored；竞态或归因不明的 timeout fail closed，绝不塑形为
  `execution_timeout`，也不以 null 混淆。
- shaping：engine fixtures 映射 typed correctness/performance/op/deep profile；case_idx、结构化
  input_shapes、geometric mean、coverage、profile discriminant/count invariants；不存在
  `mean_abs_err`/`throughput`。
- diagnostics：shape/value/non-finite stable kinds；sample 上限、排序、多维索引、sentinel；
  DomainFailure outcome/code/phase 不变量、case index、512-byte 脱敏边界；candidate
  compile/runtime errors 不进入 ErrorInfo，evaluator errors 不进入 DomainFailure。
- 安全/限额：无/错 credential、trial 提交 grade、跨 caller GET、伪造 session、manifest/digest
  mismatch、trial/trusted 独立限流、日志与错误不泄露源码/secret/path。grade manifest fixture
  必须覆盖预注册 immutable release、拒绝 per-run RunManifest/EvalManifest、受信目录解析、目录
  穿越/symlink 逃逸和不存在的三元组；不存在动态 manifest 注册 API。
- 兼容：当前/前一支持 schema 的读写 fixture、未知 optional response 字段、未知 discriminant
  fail closed、stored artifact migration、旧路由按公布窗口退役。
- 契约发布物：从 typed models 重建 JSON Schema 与 golden fixtures 后 byte-for-byte 相同，发布
  SHA-256 可复算；Hub/Harness 的 pinned digest 正例通过，任一字节或 digest 变化均 fail closed。
- health/observability：live 不依赖 Redis、ready 覆盖 Redis/storage/worker heartbeat；公开响应不
  泄露卡忙闲/队列；必要 metrics 具有 caller class、state、outcome/error code 维度。
- 部署 preflight：fake Docker CLI 覆盖 gateway/network/bind/port/proxy/Redis/host network、
  capability file 和 manifest catalog 的成功与失败；失败不得调用 Compose，且不打印 secret。
- portable schema consumer：仅用 HTTP/JSON、固定 `jsonschema==4.26.0` 和 stdlib 验证 submit、三个
  exact `$defs` entrypoint、幂等恢复和 polling；测试环境不 import Eval Hub、FastAPI/Pydantic 或
  service package，也不手写第二套 wire validator。Hub 与 standalone trusted coordinator fixture
  还要验证从受信 release 目录解析 reference，以及 subject failure=zero、infra/cancel=unscored 的
  GradeRecord 映射；cancel fixture 必须覆盖 POST 前全-null formal fields、ID 已持久化后的 evidence
  retention，以及晚到 terminal Evaluation 不改写 cancellation winner。

无 NPU 的 CI 必须通过以上 deterministic tests。正式发布还必须在目标 NPU/driver/toolchain
组合上完成：correctness/decoy 回归、真实失败集的 coarse taxonomy 校准、benchmark 稳定性与
geometric mean/coverage 复算、profile count/ratio 校验、worker crash/lease fencing soak test。
任何未校准的细粒度 divergence kind 不得进入稳定 schema；NPU calibration 未通过时本版本
不得标记 production-ready。
