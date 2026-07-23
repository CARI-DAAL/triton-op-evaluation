# triton-evaluation 设计文档

- 日期：2026-07-23
- 状态：设计已确认，待 spec 审阅 → 实现计划
- 位置：`e84381970/projects/triton-evaluation/`

## 1. 目标与定位

把 `triton-op-verifier` skill 的评测能力（正确性 verify + 性能 benchmark）包装成一个**独立部署的 evaluator service**，通过 **FastAPI + Redis** 对外提供异步评测。以后可插入 **profile**（simulator 指令级诊断）等新 evaluator。

**定位**：独立小服务，只按路径 subprocess 调用 skill 的脚本（`validate_triton_impl.py` / `verify.py` / `benchmark.py`），**不依赖 kernelgym**，不 import skill。

**主要消费方**：批量跑分，取代 `run_benchmark_triton.sh` 那种并行批量。单机多卡、中等并发。

### 非目标（v1 明确不做）

- 不做多机分布式（Redis 架构上留了扩展空间，但 v1 worker 与卡同机）。
- 不做鉴权 / 多租户配额。
- 不重新发明结果格式：`result` 直接透传 skill 原生 JSON。
- 不实现 profile evaluator（只预留接入点）。
- 不做可靠队列（`BRPOPLPUSH`）；v1 用 reaper 兜底，列为后续硬化项。

## 2. 已确认的关键决策

| 决策 | 选择 |
|---|---|
| 架构方向 | 新建独立 evaluator service，wrap skill 脚本，不依赖 kernelgym |
| 交互模式 | 异步提交 + 轮询；Redis = 任务队列 + 结果存储 |
| evaluator 组织 | 每种 evaluator 单独端点（`/verify`、`/benchmark`，以后 `/profile`） |
| 命名 | `verify`（对不对）/ `benchmark`（多快，vs torch）/ `profile`（为什么慢，留给 simulator） |
| 绑卡 | 一卡一常驻 worker，启动时 `ASCEND_RT_VISIBLE_DEVICES=<card>` 静态钉死 |
| 调用方式 | subprocess 调 skill 脚本（沿用其超时 kill / 崩溃隔离），不 import |
| 输入 | 代码走字符串，worker 每个 job 落到临时 `verify_dir` |

**核心不变量**：一卡一 worker、一次一 job ⟹ 任意时刻一张卡最多一个 job。benchmark 独占性由结构保证，**无需锁**。

## 3. 架构

```
                       ┌─────────────┐
  client ──POST /verify┤             │ LPUSH job
         ──POST /benchmark   FastAPI  ├────────────┐
         ──GET  /tasks/{id}  (api)    │            ▼
                       └──────┬──────┘        ┌─────────┐
                              │ 读状态/结果    │  Redis  │
                              └───────────────┤ queue + │
                                              │ results │
   worker[npu:0] ◀── BRPOP ────────────────── └────┬────┘
   worker[npu:1] ◀── BRPOP ───────────────────────┤
   worker[npu:k] ◀── BRPOP ───────────────────────┘
        │ 每个 worker：绑定 1 张卡，串行取 job
        ▼
   materialize verify_dir → subprocess 调 skill 脚本
   (ASCEND_RT_VISIBLE_DEVICES=k) → 解析 JSON → 写回 Redis
```

### 模块边界

| 模块 | 职责 | 依赖 |
|---|---|---|
| `api/` | FastAPI 路由 + 请求/响应模型。只做：校验→入队→返回 task_id；查状态 | broker |
| `broker/` | Redis 封装：`enqueue` / `dequeue` / `store_result` / `get_status`。Redis 细节藏在接口后 | redis |
| `worker/` | 常驻进程：绑卡→BRPOP→派发给 evaluator→写回结果；reaper | broker, evaluators |
| `evaluators/` | 可插拔 runner：`Evaluator` 基类 + `VerifyEvaluator` / `BenchmarkEvaluator`（以后 `ProfileEvaluator`）。落盘输入→subprocess→解析 JSON | skill 脚本（按路径） |
| `models/` | 共享 job/result 数据结构 | — |
| `config.py` | redis url、skill 脚本路径、卡列表、超时 | — |

## 4. API 契约

```
POST /verify
  { op_name, reference_code, kernel_code,
    non_compute?: bool, triton_impl_name?: "triton_ascend_impl", timeout?: 900 }
  → 200 { task_id }

POST /benchmark
  { op_name, reference_code, kernel_code,
    verify_task_id?: str,          # 闸门用，见 §6
    warmup?: 5, repeats?: 50, triton_impl_name?, verify_not_required?: bool }
  → 200 { task_id }
  # 既无 verify_task_id 又无 verify_not_required → 422

GET /tasks/{task_id}
  → { task_id, type, status, result?, error?, device_id?,
      created_at, started_at?, finished_at? }
     status ∈ QUEUED | RUNNING | SUCCEEDED | FAILED | TIMEOUT | REJECTED

GET /healthz  → { redis: ok, workers: [{card, alive, current_task}] }
```

`result` 透传 skill JSON：verify 给 `verify_result.json` 全文（passed/total/failures/metrics）；benchmark 给 `perf_result.json` 全文（speedup_vs_torch/per_shape_results 等）。

## 5. Redis 键布局

| 键 | 类型 | 内容 |
|---|---|---|
| `eval:queue` | list | job payload `{task_id, type, spec}`，worker `BRPOP` |
| `eval:task:{id}` | hash | `status / type / result(JSON) / error / device_id / created_at / started_at / finished_at`，TTL 24h |

一个共享队列，job 带 `type` 标签；worker 不挑 type，取到什么跑什么。

## 6. verify → benchmark 闸门（跨端点）

端点独立 ⟹ 闸门跨 task 边界。按优先级：

1. **`verify_task_id` 播种（推荐）**：benchmark 请求带上前序 verify 的 task_id。worker 从 `eval:task:{verify_task_id}` 取出存好的 `verify_result.json`，写进本 job 的 `verify_dir` → benchmark.py 自带 L1 闸门天然通过。若那个 verify 非全过 → benchmark.py `exit 2` → task `REJECTED`。
2. **`verify_not_required=true` 透传**：客户端明确不要闸门时直接传给 benchmark.py。
3. 两者都不给 → API 层 422 拒绝，不给"裸跑 benchmark"留后门。

闸门语义与 skill 完全一致，只是把"读同目录文件"变成"从 Redis 取前序 task 结果再落盘"。

## 7. 可插拔 Evaluator 接口

```python
class Evaluator(Protocol):
    name: str                    # "verify" | "benchmark" | 以后 "profile"
    needs_bound_card: bool       # verify/benchmark=True; simulator=False（离线跑 host）
    def run(self, spec, verify_dir, device_id) -> EvaluatorResult: ...
    # 内部：落盘输入 → subprocess 调对应 skill 脚本 → 解析 JSON → 归一化 status
```

worker 持一张 `{type: Evaluator}` 注册表，BRPOP 到 job 后按 `type` 派发。

**加 profile 三步**：① 写 `ProfileEvaluator`（包 `triton-simulator-optimizer` 的 `msprof op simulator` + `parse_prof.py`）；② 注册；③ 加 `POST /profile` 端点。simulator 类 `needs_bound_card=False`（host 仿真、不上卡），将来可不占绑卡 worker，不跟 benchmark 抢卡。

## 8. 错误处理与状态映射

**核心区分**：task 状态只反映"评测有没有正常完成"，算子好坏在 result 里读。

| evaluator | 退出码 | task status | 含义 |
|---|---|---|---|
| verify | 0 | `SUCCEEDED` | 跑完，`result.passed_cases == total`（全对） |
| verify | 1 | `SUCCEEDED` | 跑完，`passed < total`（算子错了，但 job 成功） |
| benchmark | 0 | `SUCCEEDED` | perf 采完，有 speedup |
| benchmark | 2 | `REJECTED` | L1 闸门：verify 没全过 / 没播种 |
| 任意 | 崩溃 / 无 JSON / 落盘失败 | `FAILED` | 评测本身没跑起来 |
| 任意 | 超服务级墙钟 | `TIMEOUT` | worker kill 子进程树 |

> verify exit 1 不是错误，是"算子不对"这个正常结论。判定读 JSON 的 `passed_cases==total_cases`，不看退出码猜。OOM / 单 shape 失败由 skill 脚本内部 try/except 记进 result，task 仍 `SUCCEEDED`。

**超时**：两层——skill 脚本自带子进程超时（到点 kill 进程树）；服务再包一层墙钟（略大于脚本超时）兜底 kill → `TIMEOUT`。

**Redis 不可用**：API 返回 503；worker 带退避重连。

## 9. worker 生命周期

- **启动**：从 config 读卡列表，每张卡 fork 一个常驻 worker，`ASCEND_RT_VISIBLE_DEVICES=<card>` 钉死，连 Redis 进循环。可选开机自检 `torch.npu.is_available()`；坏卡的 worker 记日志退出，不拖累其它卡。
- **循环**：`BRPOP eval:queue` → 置 `RUNNING`+started_at+device_id → `evaluator.run()` → 写 result+终态 → 下一轮。
- **崩溃兜底**：worker 挂了那条 job 已被取走 → task 卡 RUNNING。reaper 把 `RUNNING` 且超过（timeout+余量）的 task 标 `FAILED(worker_lost)`，client 重提；worker 由外部 supervisor（systemd/docker restart）拉起。
- **优雅退出**：SIGTERM 时跑完当前 job 再退出。

## 10. 测试策略（CPU 上覆盖大部分）

- **`evaluators/`（重点，无需 NPU）**：注入假脚本执行器，验证 ① 输入落盘（`{op}_torch.py`/`{op}_{impl}.py` 文件名、内容）；② 命令拼装（参数 + env 里有 `ASCEND_RT_VISIBLE_DEVICES`）；③ 喂预置 verify_result.json/perf_result.json + 各退出码 → 断言 status/result 映射。用 SKILL.md 的 JSON 样例做 fixture。
- **`broker/`**：fakeredis/真 Redis 测 enqueue/dequeue/store/get 往返、TTL、状态流转。
- **`api/`**：FastAPI TestClient——POST 入队返 task_id；GET 读状态；坏输入 422；benchmark 缺闸门参数 422。
- **`worker/`**：假 evaluator + 假 broker 测派发、退出码映射、超时 kill。
- **集成（需真卡，CI 无卡跳过）**：平凡 kernel（如逐元素 add）端到端 /verify→SUCCEEDED 全过→/benchmark 带 verify_task_id→有 speedup。

## 11. 目录结构

```
triton-evaluation/
├── api/         # FastAPI app + routes + pydantic models
├── broker/      # Redis 封装（队列+结果）
├── worker/      # 常驻 worker + 派发 + reaper
├── evaluators/  # Evaluator 基类 + Verify/Benchmark（以后 Profile）
├── models/      # job/result 数据结构
├── config.py
├── run_api.py / run_worker.py
├── docs/specs/  # 本设计文档
└── tests/
```

## 12. 后续硬化项（非 v1）

- 可靠队列 `BRPOPLPUSH`（防 worker 崩溃丢 job）。
- profile evaluator（simulator 指令级诊断）。
- 多机 worker。
- 鉴权 / 配额。
