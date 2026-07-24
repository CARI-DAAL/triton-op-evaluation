# Profile Endpoint（msprof）设计文档

- 日期：2026-07-25
- 状态：设计已确认，待文档审阅
- 适用仓库：`triton-op-evaluation`
- 对应实施计划：`docs/plans/2026-07-25-profile-endpoint-msprof.md`（后续修订为 v2）

本文档在 profile 和公开 API 边界上取代
`docs/specs/2026-07-23-triton-evaluation-design.md` 中“独立
`/benchmark`”与“host simulator profile”的早期设想；当前实现中的 benchmark
是 `/verify` 的内部阶段，本次 profile 使用真实 NPU。

## 1. 目标与定位

为 Triton Op Eval Service 新增独立的异步 `POST /profile` 接口，在真实
Ascend NPU 上执行 `msprof op` 轻量分析或原生 `msprof` 深度分析，并返回逐
case 的结构化性能诊断。

整体使用方式参考 `cannbot-skills` 中 `triton-op-generator` 的算子生成与优化
流程；具体 msprof 采集、指标解释和 Bound 判定以
`ops/ops-profiling` 为语义基线。服务只提供执行能力，不复制上层
Phase 0–7 的生成、验证、优化编排。

公开评测入口只有：

- `POST /verify`：正确性验证；`measure_speed=true` 时在验证全通过后内部执行
  benchmark。
- `POST /profile`：独立性能诊断，不要求先完成 `/verify`。

`benchmark.py` 和 `BenchmarkEvaluator` 继续作为 `/verify` 的内部实现，不新增
`POST /benchmark`，也不新增 `JobType.BENCHMARK`。

## 2. 参考实现与权威来源

设计采用“复用语义、服务化适配”，不直接依赖相邻仓库的运行时路径。

| 来源 | 复用内容 |
|---|---|
| `cannbot-skills/plugins-official/triton-op-generator/AGENTS.md` | verify、测速、诊断之间的上层流程边界 |
| `cannbot-skills/ops/ops-profiling/SKILL.md` | msprof 工具选择和诊断流程 |
| `ops-profiling/scripts/msprof_profile_run.sh` | deep 模式 warmup、7 组 metrics、sample-based 采集顺序 |
| `ops-profiling/scripts/msprof_perf_summary.py` | CSV/SQLite 解析和结构化分析 |
| `ops-profiling/references/msprof-guide.md` | 主 Bound 的优先级、字段和阈值 |
| `ops-profiling/references/optimization_quickref.md` | Bound 到优化建议的映射 |
| `kernelgym-npu/.../msprof_profile.py` | 已验证的 `msprof op` runner、kernel 过滤、CLI 参数和 CSV 解析 |

移植的代码必须保留原许可证头，并在模块注释中标明来源。实现不得在部署时
import 或调用上述相邻仓库中的 Python 模块。

## 3. 范围

### 3.1 本次实现

- 新增 `JobType.PROFILE`、`ProfileRequest`、`ProfileEvaluator` 和
  `POST /profile`。
- 支持 `profile_mode="op"` 和 `profile_mode="deep"`。
- 默认遍历全部输入 case，也可用 `case_idx` 只分析一个 case。
- 每个 case 独立采集、解析、诊断和记录错误。
- profile runner 复用现有 HF32 fail-closed 约束。
- 支持总任务超时、单次采集超时、进程树终止、部分结果快照和临时目录清理。
- 返回直接位于任务 `result` 下的稳定 JSON，不增加 `result.profile` 包装层。

### 3.2 非目标

- 不在 `/verify` 中自动触发 `/profile`。
- 不新增公开 `/benchmark` 接口。
- 不实现实验管理、baseline 选择、对比页面或前端。
- 不持久化原始 msprof 目录；服务只持久化结构化结果。
- 不修改 verify 的容差、benchmark 的计时或现有结果口径。
- 不把多个 case 的原始指标合并成一个虚构的“平均 kernel”。
- 不支持 MC²/fork 多 rank 算子；该能力需要独立设计进程拓扑和卡分配。

## 4. 方案选择

### 4.1 采用：服务化 Python 适配

将采集和分析拆成专用 Python 模块，由 `ProfileEvaluator` 在绑定的 NPU 卡上
启动。这样可以统一请求校验、多 case 编排、超时、结构化结果和测试。

### 4.2 未采用：原样调用 shell 总入口

`msprof_profile_run.sh` 面向人工工作目录和独立可执行文件。原样调用会使
case 选择、异步状态、部分结果和进程清理难以与现有 worker 契约对齐。

### 4.3 未采用：嵌入 `/verify`

profile 慢且用于解释性能，不属于正确性门禁。将其嵌入 `/verify` 会延长常规
验证任务，并使上层无法独立决定何时诊断。

## 5. API 契约

### 5.1 提交

```http
POST /profile
Content-Type: application/json
```

```json
{
  "op_name": "softmax",
  "reference_code": "...",
  "kernel_code": "...",
  "triton_impl_name": "triton_ascend_impl",
  "profile_mode": "deep",
  "case_idx": null,
  "profile_launch_count": null,
  "timeout": 3600
}
```

字段定义：

| 字段 | 类型与默认值 | 语义 |
|---|---|---|
| `op_name` | `str`，必填 | 任务名和物化文件名前缀 |
| `reference_code` | `str`，必填 | 含 `Model`、输入提供函数和 `get_init_inputs` 的参考代码 |
| `kernel_code` | `str`，必填 | 含 `ModelNew` 和 Triton kernel 的实现代码 |
| `triton_impl_name` | `str = "triton_ascend_impl"` | 实现文件后缀 |
| `profile_mode` | `Literal["op", "deep"] = "op"` | 轻量或深度采集 |
| `case_idx` | `int \| None = None`，整数须 `>= 0` | `None` 表示全部 case；整数表示只分析该 case |
| `profile_launch_count` | `int \| None = None`，范围 `1..100` | 仅用于 `op` 模式，显式传给 `msprof op --launch-count` |
| `timeout` | `int = 3600`，范围 `10..86400` | 整个 profile job 的 wall-clock 预算，单位秒 |

`profile_mode="deep"` 与非空 `profile_launch_count` 组合由 API 返回 HTTP 422，
避免静默忽略参数。

成功提交返回：

```json
{"task_id": "<opaque-id>"}
```

结果仍通过现有接口查询：

```http
GET /tasks/{task_id}
```

### 5.2 case 选择

- stdlib-only 的 `engine/profile.py` 不直接 import 含 torch 的任务模块。它先
  启动生成 runner 的 `--describe-cases` 子命令，以 JSON 取得 case 总数和
  `input_desc`，再校验和选择 case。
- 存在 `get_input_groups()` 时，优先调用并完整迭代一次；接受 list、tuple 或
  generator。
- `get_input_groups()` 返回空迭代器属于输入错误，不回退到 `get_inputs()`。
- 不存在 `get_input_groups()` 时，调用 `get_inputs()` 并视为唯一的
  `case_idx=0`。
- 两个函数都不存在、函数抛错或 `case_idx` 越界时，任务为 `FAILED`，不执行
  msprof。
- `case_idx=None` 默认选择全部 case；`case_idx=N` 只选择索引 N。
- 顶层 `total_cases` 表示任务中可用的 case 总数，`selected_cases` 表示本次
  实际选择数。因此单 case 选择时二者可以不同。

## 6. 架构与模块边界

```text
POST /profile
  → Job(type="profile")
  → Redis queue
  → 单卡 Worker
  → ProfileEvaluator
  → engine/profile.py
      ├─ resolve cases
      ├─ case_idx=None：逐个 case
      ├─ case_idx=N：指定 case
      ├─ msprof_op.py
      └─ msprof_deep.py
  → profile_result.json
  → task.result
```

| 文件 | 职责 |
|---|---|
| `triton_evaluation/models.py` | Profile job/request 字段及校验 |
| `triton_evaluation/api/routes.py` | `POST /profile` 入队 |
| `triton_evaluation/evaluators/profile.py` | 物化文件、构造 CLI、映射状态、读取部分结果 |
| `triton_evaluation/evaluators/__init__.py` | 注册 `"profile"`；仍不注册 `"benchmark"` |
| `triton_evaluation/engine/profile.py` | CLI、多 case 编排、deadline、快照、汇总 |
| `triton_evaluation/engine/msprof_op.py` | `msprof op` 命令、runner 和 CSV 解析 |
| `triton_evaluation/engine/msprof_deep.py` | deep 采集、跨 metric 合并、Bound、per-core |

三个 engine profile 模块的顶层只依赖 Python 标准库。`torch` 和 `torch_npu`
只允许出现在生成的 runner 中，使 parser 和 orchestrator 可在无 NPU 的测试
环境导入。

### 6.1 排队与卡独占

- `/profile` 与 `/verify` 使用同一个 Redis job queue，不绕过 worker 直接运行。
- 每个常驻 worker 绑定一张物理 NPU，并串行处理任务；只有 worker 完成当前
  job 后才会 dequeue 下一个 job。
- profile job 不与任何先前 verify job 建立卡亲和性。哪张已配置 NPU 的
  worker 先空闲并 dequeue 到该 job，就使用哪张卡。
- profile job 一旦开始，其全部选中 case、warmup 和 7+1 轮 deep 采集都固定
  在领取该 job 的同一张卡上，中途不跨卡迁移。
- 因此服务管理范围内，同一张卡上的 verify、内部 benchmark 和 profile 不会
  重叠执行；profile 的 `needs_bound_card=True`。只有 `/verify` 内部的
  benchmark 与该次 verify 必须使用同一张卡。
- “卡空闲”指该卡对应的服务 worker 当前可接收任务。v1 不调用 `npu-smi`
  检测服务外进程，也不与服务外调度器共享锁；部署必须将配置给本服务的卡
  视为专用资源。若未来需要和外部任务共享 NPU，必须另行增加跨进程卡租约和
  占用探测，不能仅凭 `npu-smi` 的瞬时结果消除竞争。

## 7. Runner 与执行环境

每个选中 case 使用独立 runner 参数执行，runner 负责：

1. 将参考实现和 Triton 实现按文件路径导入。
2. 在 `--describe-cases` 模式下解析输入提供函数，输出
   `{total_cases, cases: [{case_idx, input_desc}]}` 后退出，不运行 kernel。
3. 在 `--case-idx=N` 模式下重新以同一随机种子解析输入，仅取得指定 case，
   再构造 `ModelNew(*get_init_inputs())`。
4. 递归处理 list、tuple、dict 中的 Tensor，并移动到本地 `npu:0`。
5. 在输入枚举前设置确定性随机种子，使 discovery、warmup 和正式采集看到
   相同的 case 顺序、shape 和 dtype。
6. 在任何 warmup 或 kernel launch 之前调用共享
   `enforce_conv_fp32()`；属性存在但回读不是 `False` 时立即失败。
7. 执行一次实现的 forward；deep warmup 由编排器在正式采集前单独启动。

worker 通过 `ASCEND_RT_VISIBLE_DEVICES=<physical-id>` 绑定物理卡，runner 始终
使用可见设备中的本地 `npu:0`。不得在 runner 中重新暴露其他物理卡。

Triton kernel 名从 `@triton.jit` / `@jit` 修饰的函数中提取，保持源码顺序并
去重。无法提取自定义 Triton kernel 名时，case 标记为 `failed`，避免误把
PyTorch/CANN 前置算子作为目标。

## 8. `op` 模式

每个选中 case 执行一次：

```text
msprof op
  --output=<case-output-dir>
  [--launch-count=<1..100>]
  --kernel-name=<name1|name2|...>
  <profile-python> <runner.py> <case-args>
```

约束：

- `msprof op` 的选项使用 `--option=value` 形式。
- 未显式请求 `profile_launch_count` 时省略 `--launch-count`，使用工具默认值。
- runner 不做手工 warmup；`msprof op` 自身负责目标算子的 warmup/replay。
- 遍历输出树中的所有 `OpBasicInfo.csv`，解析其同目录指标文件。
- 结构化结果保留所有成功解析的 kernel instance；文本摘要可以只展示最慢的
  前 5 个，不能从结构化结果中丢弃其余数据。
- case 的 `target_kernel` 是成功解析的自定义 Triton kernel instance 中耗时
  最长者，用于生成主摘要和建议。
- CSV 缺失或单个 kernel 目录损坏时记录 warning；至少一个 kernel 可解析时
  case 可为 `partial`，全部不可解析时为 `failed`。

## 9. `deep` 模式

### 9.1 采集顺序

每个选中 case 独立执行：

1. 正式采集前直接运行 runner 3 次，作为 DVFS warmup。
2. 按以下顺序执行 7 次 task-based msprof：
   - `PipeUtilization`
   - `ArithmeticUtilization`
   - `Memory`
   - `MemoryL0`
   - `MemoryUB`
   - `L2Cache`
   - `ResourceConflictRatio`
3. 再执行 1 次 `--aic-mode=sample-based`，用于读取
   `device_0/sqlite/aicore.db` 中的逐核 `task_cyc`。

每个 task-based 命令必须包含：

```text
msprof
  --output=<metric-output-dir>
  --ai-core=on
  --aic-metrics=<metric>
  --task-time=on
  --ascendcl=on
  <profile-python> <runner.py> <case-args>
```

sample-based 命令还包含：

```text
--aic-metrics=PipeUtilization
--aic-mode=sample-based
--aic-freq=100
```

### 9.2 目标 kernel 与指标合并

- 完成可用的 task-based 采集后，优先从 `PipeUtilization` 的
  `op_summary` 选择目标；该组不可用时，按固定的 7 组顺序使用第一个含
  `Task Duration(us)` 的可用 `op_summary`。
- 目标候选只包含提取出的自定义 Triton kernel 名：接受完整相等或符合
  `msprof op --kernel-name` 语义的名称前缀匹配。
- 选择候选中 task duration 最大的完整 `Op Name` 作为该 case 的
  `target_kernel`，并记录 `target_selection_metric`。
- 其余 6 组 CSV 必须按同一个完整 `Op Name` 精确匹配。
- 禁止每轮各自选择最慢行；这会把不同算子的指标拼成不存在的 kernel。
- 同一 metric 中同名 kernel 出现多次时，在完全匹配的行内选择
  `Task Duration(us)` 最大的一行，并把候选行数、选择字段和值写入 evidence。
  禁止在未定义字段语义的情况下对 ratio 做求和或平均。
- 缺失的 metric 保持缺失并列入 `missing_metrics`，禁止转成 `0`。

### 9.3 主 Bound

取得实际 CSV 字段后，按以下优先级返回第一条成立的规则：

| 优先级 | Bound | 规则 |
|---|---|---|
| 1 | `MTE2 BOUND` | MTE2 `> 80%`，或 MTE2 为比较集合最大值且 `> 70%` |
| 2 | `CUBE BOUND` | CUBE `> 80%`，或 CUBE 为比较集合最大值且 `> 70%` |
| 3 | `VEC BOUND` | PUSHQ `> 80%` |
| 4 | `FIXP BOUND` | FIXP `> 80%` |
| 5 | `MTE3 BOUND` | MTE3 `> 80%` |
| 6 | `SCALAR BOUND` | SCALAR `> 80%`，或 SCALARLDST `> 80%` |
| — | `NO BOUND` | 所有必需字段均可用，但以上规则均不成立 |

只有当判定所需的字段集合完整时才能返回 `NO BOUND`。字段不足以完成判定时
`bound=null`，case 至少为 `partial`，并返回 `missing_metrics`。`bound` 必须
携带 ratios、命中规则和来源字段作为 evidence。

解析层维护“实际 CSV 表头 → 上述规范名称”的显式 alias 表。找不到
PUSHQ、SCALARLDST 或其他必需字段时按缺失处理，不使用 `aiv_vec_ratio` 等
不同语义字段冒充。

### 9.4 per-core

- 使用标准库 `sqlite3` 读取 sample-based `aicore.db`。
- 以数据库中实际出现的 core ID 为准，不硬编码 24 个 AIC 或 48 个 AIV。
- 若 SQLite schema 能将 task 精确关联到 `target_kernel`，只聚合该 kernel，
  返回 `scope="target_kernel"`。
- 若只能取得整次 sample run 的逐核 cycle，则返回
  `scope="profile_run"` 并附 warning，禁止将其表述为 target-only 数据。
- 根据同一作用域的 `aicore_time` 与最大 cycle 反推频率，再换算逐核时长。
- 返回有效核数、min/avg/max、`(max-min)/max` 和 `<10%` 是否达标。
- sample-based 失败或 cycle 与 time 无法采用同一作用域时，
  `per_core=null` 并记录 warning，不伪造逐核数据。

### 9.5 优化建议

建议由主 Bound 和 `optimization_quickref.md` 的固定映射生成。输出只陈述有
指标证据支持的方向，不根据缺失指标推断瓶颈。每条建议包含：

- `bound`
- `evidence`
- `recommendation`
- `source`

## 10. 结果格式

任务的 `result` 直接采用以下顶层格式：

```json
{
  "profile_mode": "deep",
  "requested_case_idx": null,
  "total_cases": 5,
  "selected_cases": 5,
  "completed_cases": 3,
  "partial_cases": 1,
  "failed_cases": 1,
  "remaining_case_indices": [],
  "timed_out": false,
  "summary_text": "...",
  "warnings": [],
  "wallclock_sec": 123.45,
  "per_case_results": [
    {
      "case_idx": 0,
      "status": "complete",
      "input_desc": "...",
      "target_kernel": "softmax_kernel",
      "target_selection_metric": "PipeUtilization",
      "kernels": [],
      "raw_metrics": {},
      "bound": {
        "name": "MTE2 BOUND",
        "ratios": {"mte2": 0.84},
        "rule": "mte2 > 0.80",
        "evidence": {}
      },
      "per_core": {},
      "recommendations": [],
      "missing_metrics": [],
      "warnings": [],
      "error": null,
      "msprof_wallclock_sec": 24.3
    }
  ]
}
```

case 状态：

- `complete`：该模式的必需采集和解析全部成功。
- `partial`：已得到可用诊断，但存在缺失 metric、损坏的次要 kernel 记录或
  sample-based 缺失。
- `failed`：该 case 没有得到任何可靠的目标 kernel 诊断。

计数满足：

```text
completed_cases + partial_cases + failed_cases
  + len(remaining_case_indices) == selected_cases
```

正常结束时 `remaining_case_indices=[]`；超时时它包含当前未完成及尚未开始的
case。已完成 case 的原子快照不会因后续 case 超时而丢失。

## 11. 错误与任务状态

task status 反映服务编排是否正常完成；kernel/profile 结论在 `result` 中读取。

| 场景 | task status | result |
|---|---|---|
| 请求字段不合法 | HTTP 422，不创建任务 | 无 |
| `msprof` 不存在或不可执行 | `FAILED` | 无或仅 preflight 诊断 |
| 模块加载失败、无输入、case 越界 | `FAILED` | 可附结构化输入错误 |
| 一个或多个 case 完成/部分完成/运行失败，编排正常结束 | `SUCCEEDED` | 完整逐 case 结果 |
| 所有 case 的 kernel 均运行失败，但编排正常结束 | `SUCCEEDED` | `failed_cases == selected_cases` |
| engine 崩溃、结果 JSON 无法解析 | `FAILED` | 最后有效快照（如有） |
| 超过总任务预算 | `TIMEOUT` | 最后一个原子快照（如有），`timed_out=true` |

deep 模式 warmup 失败时，该 case 标记为 `failed`，不开始正式采集。某个
task-based metric 失败时记录 stderr 尾部并继续剩余 metric；能得到可靠目标
kernel 但指标不完整时标记 `partial`，无法得到任何可靠目标 kernel 时才标记
`failed`。sample-based 是非关键采集，失败只使原本 `complete` 的 case 变为
`partial`。

## 12. 超时、进程与清理

- `ProfileRequest.timeout` 是整个 job 的 deadline。
- 单次 runner/msprof 默认最多 300 秒，实际超时为
  `min(MSPROF_TIMEOUT_SEC, remaining_job_time)`。
- `MSPROF_TIMEOUT_SEC` 必须是正整数；非法配置在 preflight 阶段使任务
  `FAILED`。
- `ProfileEvaluator` 的外层墙钟为 `spec.timeout + 60` 秒。
- 外层 `run_script` 和 engine 内部每个 msprof 都创建独立进程组；超时时终止
  整个进程组，避免遗留 profiler 或 kernel runner。
- 每完成一个 case 后，以“写临时文件 + `os.replace`”原子更新
  `profile_result.json`。
- 总 deadline 到达时，engine 写入 `timed_out=true` 的最终快照并返回超时
  退出码；若 engine 来不及写，evaluator 使用上一份完整快照。
- runner、metric 输出、sample 输出和 group 目录均在 `finally` 中清理。
- worker 现有的 job 临时目录清理保持不变。

## 13. 环境配置

新增并记录以下变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `MSPROF_BIN` | `msprof` | profiler 可执行文件 |
| `MSPROF_PROFILE_PYTHON` | 当前 Python | msprof 下执行 runner 的 Python |
| `MSPROF_TIMEOUT_SEC` | `300` | 单次 msprof 上限；仍受 job 剩余时间限制 |

worker 启动后无需重启上层 API 即可处理新任务，但 worker 进程必须在包含
`msprof`、CANN 环境和正确 Python 依赖的环境中启动。ProfileEvaluator 在运行
前使用 `shutil.which` 做 preflight。

## 14. 测试策略

### 14.1 纯单元测试（无需 NPU）

- `ProfileRequest` 默认值、Literal 校验、`case_idx`、conditional launch-count。
- Job JSON round-trip 和 registry 仅含 `verify`、`profile`。
- `get_input_groups` 的 list、tuple、generator、空迭代器、异常和 fallback。
- runner discovery 与正式执行使用同一种子并保持 case 描述一致。
- `case_idx=None` 全量、合法单 case、越界。
- Triton kernel 名提取、去重和无 kernel fail-closed。
- `msprof op` 的 `--output=`、`--kernel-name=`、`--launch-count=` 参数。
- OpBasicInfo、PipeUtilization、Memory、L2 CSV 的正常、缺失和畸形输入。
- deep 7 组命令的固定顺序和 sample-based 参数。
- 同一完整 Op Name 的跨 metric 合并；拒绝跨 kernel 拼接。
- Bound 六级优先级、边界值、`PUSHQ`、`SCALARLDST`、缺失字段和
  `NO BOUND`。
- SQLite per-core、空数据库、未知 schema 和非连续 core ID。
- result 计数不变量、summary、warning 和 error 序列化。

### 14.2 evaluator/API/worker 测试（无需 NPU）

- fake profile engine 覆盖 complete、partial、failed、timeout、坏 JSON。
- API 入队字段完整透传；非法组合返回 422。
- `ProfileEvaluator` 绑定设备、传总 timeout、读取直接 result。
- 超时后保留部分 result；无快照时仍正确返回 `TIMEOUT`。
- profile 注册成功，benchmark 继续不注册。
- profile 与 verify 进入同一队列；单卡 worker 在前一任务完成前不启动后一任务。
- 单个 case 失败后继续后续 case。
- 原子快照和所有临时目录清理。

### 14.3 fake-msprof 集成测试（无需 NPU）

在临时 `PATH` 中提供可执行 fake msprof，生成固定 CSV/SQLite，验证：

- op 模式真实 subprocess 命令和输出树遍历。
- deep 模式 3 次 warmup、7 次 task-based、1 次 sample-based。
- 非零退出、超时和子进程树终止。
- 多 case 默认全量与 `case_idx` 单选。
- runner 在首次 kernel launch 前执行 HF32 关闭与回读。

### 14.4 真实 NPU smoke

使用已知正确、无占位符的 Triton add/relu/identity fixture：

1. 单 case 分别执行 op 和 deep。
2. 至少两个输入组的任务执行默认全量。
3. 同一多 case 任务执行 `case_idx=1`，确认只返回该 case。
4. 校验目标 kernel 名、Bound evidence、case 计数和无残留 msprof 进程。
5. 校验 HF32 回读通过。

当前开发 shell 中 `msprof` 不在 `PATH`，因此实现可以先完成前三层；配置好
CANN/msprof 环境并通过本节 smoke 之前，不得声称真实 NPU 验收完成。

## 15. 文档与兼容性

- README 只列出公开的 `/verify` 和 `/profile`。
- README 说明 `/verify.measure_speed` 是验证通过后的内部 benchmark。
- README 给出全量 profile、单 case profile、op、deep 和轮询示例。
- `.env.example` 记录 msprof 环境变量。
- `engine/__init__.py`、`config.py` 的陈旧 verify/benchmark/validate 描述同步
  更新。
- 现有 Job JSON 没有 profile 字段时通过 dataclass 默认值保持兼容。
- verify、benchmark 的已有 JSON 和调用方式保持不变。

## 16. 完成标准

必须同时满足：

1. 现有测试和所有新增无 NPU 测试通过。
2. Python 语法、项目 lint/format 检查通过。
3. `/verify` 行为和内部 benchmark 回归不变。
4. `/profile` 默认全量，`case_idx` 可选择单 case。
5. op/deep 的命令、指标和 Bound 规则与权威来源一致。
6. 不发生跨 kernel 指标拼接，不把缺失指标当成零。
7. 超时能终止进程树并返回最后的有效快照。
8. 所有移植代码带正确许可证和来源说明。
9. profile 与 verify 在同卡 worker 上严格串行排队。
10. 在具备 CANN/msprof 的真实 NPU 环境完成 smoke。
