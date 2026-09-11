# Kernel Eval Service(昇腾)优化设计

- **日期**:2026-09-03
- **状态**:Draft(待用户审阅)
- **范围**:`triton-op-evaluation` 服务演进;对外部《Kernel Agent Evaluation Sandbox》技术规格的对比分析结论一并收录
- **硬件目标**:仅华为昇腾 NPU(CANN + torch_npu + triton-ascend,910B3 attested)。外部文档中的 CUDA 概念(nvcc / sm_80 / NVIDIA Container Toolkit / `nvidia-smi --gpu-reset`)一律映射为昇腾等价物,不做 CUDA 实现

---

## 1. 第一性原理

本服务存在的唯一理由:把 `(candidate kernel, reference, checks)` 变成**可信的类型化判定与测量**,并满足四条不变量:

1. **判定不丢**(durability):任何已确认受理(首个 2xx)的评测,调用方永远能通过 GET 取回终态
2. **舰队不中毒**(isolation & health):恶意或有缺陷的 kernel 不能拖垮训练循环之外的共享设施(其他卡、其他调用方)
3. **卡秒不浪费**:最稀缺资源(NPU 卡秒)只花在"值得"的候选上,且在卡上的每一秒都在做只有卡能做的事
4. **判定可追溯**:同一输入永远得到同一分类;测量方法显式标注,绝不无声混比

三个痛点(已确认)对应该原理的三处违反:编译失败占卡(违反 3)、批量接入困难(配额模型按交互流量设计)、卡故障缺自动闭环(违反 2)。

**演进而非重写**:现有 durable 核心(冻结契约 + 摘要 pin、 leased/fenced CAS 状态机、多层可辨识 watchdog、~600 测试)是资产,全部保留。本 spec 的增量精确对准三个痛点。

## 2. 现状 vs 外部设计文档:对比结论

| 外部文档主张 | 结论 |
|---|---|
| CPU/NPU 两阶段过滤 | **采纳**(昇腾版,见 §8),并以"产物缓存复用"补全文档缺失的另一半收益 |
| `POST /v1/evaluate/batch` + batch_id | **拒绝**。GRPO 对组内样本数变化天然容忍,per-evaluation 幂等键 + GET 已覆盖;批量是新的有状态实体,纯增熵。替代:rollout 流量等级(§7) |
| gRPC + REST 双协议 | **拒绝**(YAGNI;内网 REST keep-alive 开销毫秒级,对比每次评测几十秒) |
| Celery / Ray 编排 | **拒绝**。现有 Redis leased/fenced 队列是定制的、durability 语义正确;Celery 会倒退 |
| 每请求临时容器 | **拒绝**(现阶段)。维持进程组隔离 + 临时目录;观察到污染实例再议 |
| 服务端 reward 表(§6 错误分类 → R 值) | **拒绝**。服务只出类型化 outcome;打分留在训练侧 / Eval Hub(现有边界正确)。错误分类本身已比文档更完备(`correct/wrong/compile_error/runtime_error/execution_timeout` + phase 归因) |
| 调用方提供 `baseline_latency_ms` | **拒绝**。坚持现场实测 reference(同硬件同环境,公平) |
| watchdog 超时 SIGKILL | **已有且更强**(SIGALRM 带内归因 + 整轮 exit-3 + 外层强杀三层,PR #12) |
| 驱动挂起检测 / 设备重置 / 排水 | **采纳并补全**(§10:canary → 排水 → 恢复阶梯 → 毒丸隔离) |
| `sha256(kernel_code)` 编译缓存 | **采纳并升级**(§8:判定缓存含环境指纹 + 负缓存;产物缓存直接用 Triton 原生共享 cache) |
| `compute-sanitizer` 预检 | **搁置**(昇腾侧等价物待现场调研;profile check 的 msprof 已覆盖诊断场景) |

## 3. 非目标

- gRPC 双协议;Celery/Ray;批量端点;服务端 reward;调用方自带 baseline
- 多租户计费、Web 控制台、跨硬件抽象层
- 改动任何 2.0 已冻结语义(状态枚举、错误码、数值口径)

## 4. 总体架构

新增组件(全部落在现有进程模型与单宿主机部署内,零新增基础设施):

1. **无卡编译 worker**(`run_compile_worker.py`):与 NPU worker 同宿主机、同 conda 环境的额外进程,**不持卡、无 `ASCEND_RT_VISIBLE_DEVICES`**,每机 N 个(可配)
2. **编译产物缓存**:宿主机共享 `TRITON_CACHE_DIR` 目录(bind-mount 给所有 worker)
3. **判定缓存(verdict cache)**:Redis
4. **卡健康状态表**:Redis,worker 维护

数据流:

```
POST /evaluations ──▶ admission(durable + 幂等,不变)
                        │
                        ▼
            ready:compile lane ←── 无卡 CPU worker 认领
            │  命中 verdict 缓存 → 直接 compile_ok / compile_error
            │  AOT 编译(triton-ascend,显式 target)→ 产物落共享缓存
            │    compile_error → 合成终态(与引擎带内归因逐字段同形)← 不占卡
            │    compile_ok → 重入 exec lane(running→queued 围栏转移)
            ▼
            ready:exec lane ←── NPU worker(每卡一个)认领
            │  TRITON_CACHE_DIR 指共享缓存 → 卡上尽力免重编译
            │  correctness →(过)→ performance(event 计时)→ profile(msprof)
            ▼
            terminal(durable,不变)
```

`compile_stage` 部署配置枚举:`"cpu_lane" | "on_card" | "off"`(spike 决定默认;`on_card` 为退化模式:不省卡时但缓存照常生效;`off` 为回滚开关)。未勾选 `compile` 检查的任务在 `cpu_lane` 模式下仍经过 compile 阶段(它是管线优化;对 2.0 请求合成结果保证契约不变,见 §6)。

## 5. 状态机演进

- **对外 5 状态枚举不动**:`queued|running|completed|failed|timed_out`
- 对内新增 `stage ∈ {compile, exec}` 字段,决定 lane 归属与恢复语义
- 新增一条围栏转移:`running → queued`(stage 推进),仅限 compile worker 完成 compile_ok 后把任务重新入 exec lane;CAS 复核 attempt id + fencing token + revision + deadline,与现有 `_fenced_write` 同机制。注意与 reaper 的 lease 过期重入队(已存在)区分:前者是**成功推进**,后者是**失败回收**,两者走同一 CAS 围栏、语义不同
- lease 过期恢复**按 stage 回对应 lane**(compile 阶段死 → 回 compile lane;exec 阶段死 → 回 exec lane,不重跑 compile);`MAX_ATTEMPTS=3` 总尝试语义不变
- ready 拆双 zset(`ready:compile` / `ready:exec`),共用现有全局 `queue-sequence` 计数器
- readiness 门禁按 lane 独立:CPU worker 心跳与 NPU worker 心跳分开计数;某类全死时对应请求停在 `queued`(503 NOT_READY),deadline watchdog 照常兜底
- `claim_next` 参数化 lane,其余逻辑(移出 zset、置 running、装 lease/token)不变

## 6. 契约演进(v2.0 → v2.1,两级发布)

### 发布切分原则

Release 1 的全部内容**契约不可见**(纯内部优化);只有下表右列需要 v2.1。

| 优化 | 契约影响 |
|---|---|
| compile 预检阶段 + 产物/判定缓存(内部) | 无。对 2.0 请求,编译失败合成与引擎带内归因逐字段一致的 `result.correctness.outcome=compile_error` 终态 |
| rollout 配额(经 capability 声明 quota class) | 无 |
| 卡健康闭环 / 毒丸隔离(2.0 退化为现有形状) | 无 |
| `checks` 枚举 + `"compile"`(`checks:["compile"]` 单独合法,永不占卡) | v2.1 |
| `purpose` 枚举 + `"rollout"` | v2.1 |
| `performance.method: "event"\|"profiler"`(默认 `event`)+ 结果 `measurement_method` | v2.1 |

### v2.1 增量(additive)

**Request**:
- `checks` + `"compile"`;包含 `correctness` 时 compile 阶段隐式执行(不请求数值就不上报);`performance` 仍要求 `correctness`(不变)
- `purpose` + `"rollout"`(无额外上下文;`grade_context` 仍仅 grade 必填)
- `performance.method`,默认 `"event"`

**Result**:
- 可选 `result.compile`:`{outcome: "compiled"|"compile_error", compiler_stage?: "frontend"|"backend", error_excerpt?: <有界已脱敏摘录>, duration_ms}`;域码复用 `SUBJECT_COMPILE_ERROR`
- `result.performance` 每 case 与 overall 增加 `measurement_method: "npu_event_v1" | "profiler_kernel_sum_v1"`;同一评测内 reference/impl 恒同方法,speedup 自洽;**跨方法绝对值不可比**
- compile 失败时后续检查沿用现有 skip 码(`PRIOR_CHECK_FAILED`);不新增 ErrorEnvelope HTTP 码
- 新增检查码 `SUSPECTED_POISON_CANDIDATE`(不可重试,见 §10);2.0 请求触达毒丸条件时退化为现有 attempts 耗尽的 `WORKER_LOST` 形状,契约零改动

**机制**:
- 新目录 `contracts/evaluation-api/2.1/`,自带 pinned digests;启动 fail-closed 校验不变
- 新增 golden fixtures:compile-only 成功/失败、rollout 请求、`measurement_method` 标注、毒丸合成终态
- 双版本窗口:2.0 与 2.1 并行接受;2.0 沿用 legacy routes 同款 time-box 机制(发布 sunset instant、窗口有上限)
- `checks:["compile"]` 不返回编译产物本身(产物只留缓存服务后续执行)

## 7. 调度与配额

### 配额(quota class 机制)

quota class 由 **capability 授权声明**(registry 服务端配置,契约不可见),2.1 的 `purpose=rollout` 默认映射 rollout class。桶:

| 桶 | rate/min | max_inflight | 默认 deadline |
|---|---|---|---|
| trial(交互) | 60 | 4 | 900s(不变) |
| grade(可信 hub) | 不变 | 不变 | 不变 |
| rollout(新) | 600 | 256(≈一步 rollout) | 300s |

> 实现注记(Plan B,2026-09-04):300s 默认 deadline 未实现——rollout 以 trial purpose 准入,沿用调用方自报的 deadline(上限 900s);且 rollout caller 仍共享 session 级 in-flight 上限(默认 2),每步 rollout 应使用独立 X-Session-ID。

rollout 的 in-flight 给到整步规模:**整个 rollout 步一次性 durable 入队**,RL 客户端不做滑窗 pacing;公平性在调度端解决。

### 调度:严格优先级双派发队列

exec lane 拆两个派发队列,NPU worker 认领时先交互后 bulk。**lane 归属跟随解析后的 quota class**(rollout class → bulk lane),而非请求字段本身——Release 1 由 capability 声明驱动,Release 2 的 `purpose=rollout` 默认解析为 rollout class:

```
ready:exec:interactive  ←  trial + grade(FIFO)
ready:exec:bulk         ←  rollout class(FIFO)
```

不饿死的不变量:交互流量被自身配额结构性封顶(每 trial caller ≤4 in-flight),8 卡必然腾得出手跑 bulk;反向地,bulk 洪峰不再队头阻塞交互请求。不做加权公平队列 / 预留卡。

### 单飞编译(single-flight)

compile lane 按 `kernel_digest` 单飞:首个到达者真编译,并发同摘要任务挂起复用判定与产物。跨请求由 verdict cache 兜住(§8)。

### 卡亲和规避

infra 可疑死亡时在 evaluation 记 `last_failed_card`;worker 认领时同卡号则跳过取下一条。单卡部署配置关闭。

## 8. 编译预检与产物缓存

### 诚实的收益定位

> compile 阶段 = **过滤(确定性收益)** + **产物预热(best-effort 收益)**,两者分开度量。

两个技术细微点:
- **specialization**:Triton JIT 按实参属性特化(指针 16B 对齐、标量整型),CPU 阶段无真实张量。对策:按规范特化编译;NPU 张量通常 64B 对齐,命中率高但不保证 100%,未命中的 case 卡上照常重编译、引擎带内归因兜底
- **autotune**:CPU 阶段不做 sweep(只编译默认 config)。过滤 best-effort:只会漏报,不会误报

### 两层缓存

1. **产物缓存 = Triton 原生 `TRITON_CACHE_DIR` 挂宿主机共享目录**:triton 缓存键已含 source + signature + 特化 + 选项 + 版本 + target,compile worker 编译即落盘、NPU worker 天然命中。**不发明自定义 artifact 键**。加 LRU 容量上限(默认 20GB,可配)+ 定期清理
2. **verdict cache(Redis)**:`sha256(kernel_code + triton_impl_name + triton-ascend 版本 + CANN 版本 + target) → {compiled | compile_error + 摘录}`。**负判定也缓存**(编译失败确定性);版本指纹变化自然失效;TTL 7 天

### 确定性保证

同一 triton-ascend/CANN 版本 + 同一 target,编译结果确定。compile worker 与 NPU worker 同宿主机同 conda 环境;readiness 心跳的 compatibility 指纹纳入 triton-ascend/CANN 版本。CPU 判 compile_error 时合成的终态与引擎今天在带内产生的形状**逐字段一致**(`outcome=compile_error`、`SUBJECT_COMPILE_ERROR`、phase=compile)——CPU 判定永远是"引擎也会这么判"的提前兑现,不是第二套真相。

### Spike 验收标准(实施第一步,硬闸门)

1. 无卡容器(无 `/dev/davinci*`,strace 证实未打开设备文件)编译 ≥3 个代表性 op 模块(含多 kernel、含 autotune):import + frontend + backend 全通过,产物落入共享 TRITON_CACHE_DIR
2. 上述模块随后在 NPU worker 执行:量化卡上免重编译命中率(目标 >50%,stretch >80%)
3. ≥5 个历史真实编译失败样本:CPU 侧与卡上同分类同判定(frontend/backend 归因一致)
4. 量化语料编译墙钟时间分布(决定 compile worker 数)
5. 任一条不满足 → `compile_stage="on_card"` 退化:verdict cache + 共享产物缓存照上,仅放弃"省卡时"收益

### compile worker 机制

- 预算 `min(剩余 deadline, 上限默认 300s)`;watchdog 复用 PR #12 的 exit-3 可辨识死亡模式,归因 `DEADLINE_EXCEEDED`,不误判为可重试崩溃

## 9. 性能测量轻量化

### 引擎改动(`engine/benchmark.py`)

- 新增 event 计时路径:warmup 预热(含既有 `clear_l2_cache`)后,每迭代 `start.record(); test_fn(); end.record()` + `synchronize()`,收集 `elapsed_time()` 样本。事件是流序设备时间戳,`test_fn` 内部既有 sync 不污染测量区间
- 统计口径:每 case 延迟 = 样本**中位数**;`CV = std/mean > 0.3(可配)` 或有效样本不足 → `NO_STABLE_MEASUREMENT`(取代"CSV 解析不出即不可测")
- 现有 profiler 路径(`run_profiler_with_config` + CSV 解析)保留,服务 `method="profiler"` 显式请求;peak_memory 采集(`max_memory_allocated`)两路径都保留
- 引擎输出 JSON 增加 `measurement_method + {median, cv, n}`,shaping 层投影进 2.1 契约

### 方法与版本对应

- **2.0 默认保持 profiler**(绝对数值口径冻结)
- **2.1(含全部 rollout 流量)默认 event**;`measurement_method` 字段保证跨方法数据不被无声混比
- 验收做 **event vs profiler 秩一致研究**(同批 kernel 双方法):目标是加速比判定秩一致(same speedup>1 verdict),不是数值相等(语义上 event ≥ kernel-sum)

## 10. 卡健康闭环

**触发条件**:infra 可疑死亡(整轮 watchdog 死亡、lease 被收割的 WORKER_LOST、外层强杀且无法归因)。干净的业务失败(compile_error / wrong / timeout)不触发。

1. **Canary 探针**:worker 在可疑死亡后、认领下一条前,跑固定已知良好小 kernel(秒级)。通过 → 继续认领;失败 → 卡标记 `unhealthy`
2. **排水**:不健康卡的 worker 停止认领;`/internal/metrics` 暴露每卡健康 gauge,告警挂在现有指标体系上(不建新告警设施)
3. **恢复阶梯**(每步后 canary 复检):① 进程内清理(cleanup + canary 重试)→ ② 设备级 reset(命令可配、现场验证;容器本就特权)→ ③ worker 进程重启。自动上限默认 ②(可配),**宿主机重启永远人工 + 页呼**
4. **毒丸隔离**:evaluation 跨 attempt 累计 infra 可疑死亡次数,**第 2 次即终止并隔离**(检查码 `SUSPECTED_POISON_CANDIDATE`,不可重试;2.0 请求退化为 attempts 耗尽的 `WORKER_LOST`)。配合卡亲和规避,毒丸最多吃掉 2 张卡的恢复时间,不可能连环排水全舰队

## 11. 测试与验收

沿用现有纪律(fakeredis + 注入时钟 + fake engine;~600 测试零回归是底线)。

**单元**:
- 双 lane 状态机 `running→queued` 围栏转移:CPU/NPU 并发认领竞态、stage 推进恰好单赢家(扩展 `test_resource_state_machine` / `test_fencing`)
- verdict cache 命中 / 负缓存 / 版本失效;单飞(两等待者一编译者)
- rollout 配额桶;严格优先级派发的"交互有界 → bulk 不饿死"不变量
- 卡亲和规避;canary 状态机;毒丸阈值(干净失败不计数)
- event 统计与稳定性门(fake event sampler)
- 全部新形状 golden fixtures + 摘要 pin 更新(跨仓契约测试)

**集成**:fake compile worker + fake NPU worker 端到端 durable 流程;**在每个 stage 边界注入崩溃**,验证恢复回正确 lane 且幂等重放成立。

**NPU-only(release-gate 标记)**:spike 脚本本体;无卡编译实证;缓存命中率;真卡 canary;reset 钩子;event/profiler 秩一致研究。

**验收数字(Release 1)**:
- compile 失败任务的 NPU 占用降为 0(结构性保证,`cpu_lane` 模式)
- 产物缓存命中率 ≥ spike 目标(>50%)
- 2.1 benchmark 检查墙钟时间下降(实测报告)
- 现有测试套件零回归;readiness / fencing 不变量全部保持

## 12. 发布顺序与回滚

- **Release 1(契约不动)**:内部 compile 阶段 + 双 lane + 产物/判定缓存 + rollout quota class(capability 配置)+ 卡健康闭环 + 毒丸隔离(2.0 退化形状)。2.0 消费者零感知
- **Release 2(v2.1)**:`compile` check、`purpose=rollout`、`performance.method` + `measurement_method`、`SUSPECTED_POISON_CANDIDATE`;发布 2.0 sunset instant(窗口有上限)
- **回滚开关**:`compile_stage="off"`(单 lane,行为回到今天);`compile_stage="on_card"`(保留缓存收益);rollout quota class 可独立回退为 trial 配额;卡健康各环节均有独立禁用配置;v2.1 可停发(2.0 行为完整保留)
