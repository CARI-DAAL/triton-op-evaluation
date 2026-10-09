# Eval service NPU 全量验证方案

审查日期：2026-10-08，2026-10-09 更新输入协议部分。输入协议的验收对象为 main `e4f68e5`（含 PR #54 部署修复与 #55 官方 definition/workloads 协议）加上其后的上游准入对齐 follow-up。原功能基线为 main `1f48857`（含 PR #50 的单实现评测）；[PR #51](https://github.com/CARI-DAAL/triton-op-evaluation/pull/51) 的显式依赖组装、环境快照和 metrics 并发修复也纳入待验收范围；这些项目以该 PR 合并后的部署版本为验收对象。先固定实际部署的合并提交和 evaluator build，再开始实机测试。本文是测试方案；下列“需补充”项目尚未实现为实机测试工具，也未在 NPU 上执行。

## 1. 当前测试实际上覆盖到哪里

| 已有入口 | 实际执行内容 | 能证明什么 / 不能证明什么 |
| --- | --- | --- |
| `python -m pytest -m 'not npu' -q`；CI Python 3.10–3.12 | HTTP、Pydantic、输入生成、幂等/状态机、lease/fencing、缓存、worker、结果整形；大量使用 fakeredis、stub runner 或 fake NPU | 覆盖软件协议与编排；不能替代真实 NPU 算子、时延、profiling、故障恢复或压力验证 |
| [scripts/npu_release_gate.py](../../scripts/npu_release_gate.py) → [tests/test_npu_release_calibration.py](../../tests/test_npu_release_calibration.py) | 真实服务 `/livez`、`/readyz`，两组 float32 add：4096、8192；只执行 correctness，检查执行环境非空；再提交一个上游必然拒绝的 workload，须返回不可重试的 `INPUT_GENERATION_FAILED`，随后正确任务照常通过；同一个带 4e-3 偏差的 add 候选，不声明 tolerance 判 wrong、声明 `max_atol=0.01` 判通过 | 是 readiness + add correctness + 请求错误归因 + 声明 tolerance smoke；不是完整 release calibration，不能证明性能、缓存、profile、恢复或稳定性 |
| [tests/test_event_timing_npu.py](../../tests/test_event_timing_npu.py) | 真实 NPU Event 测量一个 64×64 ReLU；需显式启用 | 验证基础计时机制，不代表完整 HTTP 性能链路或复杂题目覆盖 |
| [tests/test_profile_npu_integration.py](../../tests/test_profile_npu_integration.py) | 真实 msprof op/deep，直接运行 engine 的 add 测试；需显式启用 | 已有实机 profile 集成测试，但不是 HTTP、admission 和 worker 的端到端验收 |
| [scripts/npu_batchmatmul_gate.py](../../scripts/npu_batchmatmul_gate.py) | 一题 51 cases，通过 HTTP 运行 correctness、event/profiler 计时、op/deep profile | 新增官方输入协议的实机验收入口；本地 CPU 测试证明编排，硬件结果仍待目标主机执行 |
| [scripts/test_eval_service.py](../../scripts/test_eval_service.py) | 同样两组 add，可默认请求 performance | 退出码只检查 correctness；performance 失败仍可能退出 0，不能作为性能验收门槛 |
| [tests/test_card_health_npu.py](../../tests/test_card_health_npu.py) | 指定真实 NPU 的健康 canary | 证明健康卡上的 canary 可执行；不证明故障隔离、恢复阶梯或复位后再接单 |
| [scripts/probe_profiler_latency.py](../../scripts/probe_profiler_latency.py) | 本机 NPU profiler 诊断，256×256 matmul，8 次有效观测及中位数，保留采集产物 | 诊断本机 profiler 链路；未经过 HTTP admission、worker lease 和结果恢复 |
| [scripts/replay_profiler_export.py](../../scripts/replay_profiler_export.py) | 对已有 profiler 产物重新导出 | 不会运行新的 NPU 计算 |
| [tests/README.md](../../tests/README.md) 的缓存与单实现步骤 | 冷→热→旁路→热、改变 shape/seed；单实现始终新测 | 当前是人工操作说明，尚非自动实机门禁 |

没有发现现成的持续并发、阶梯压测、长时间 soak 或 worker crash soak 工具。单实现、baseline cache、异常分类和恢复的大多数覆盖目前仍是 CPU 模拟。

来源：仓库 [tests/README.md](../../tests/README.md)、[.github/workflows/evaluation-api-v2.yml](../../.github/workflows/evaluation-api-v2.yml)、上述脚本及 `tests/test_single_*.py`、`test_baseline_*.py`、[test_worker.py](../../tests/test_worker.py)、[test_recovery.py](../../tests/test_recovery.py)、[test_card_health.py](../../tests/test_card_health.py)。

## 2. 现有代码现在就能运行的实机步骤

从对应部署版本的仓库根目录执行。URL 必须指向实际可达的测试实例；capability 文件为测试调用方凭证。

```bash
python -m pytest -m 'not npu' -q

python scripts/test_eval_service.py \
  --url "$EVALUATION_URL" \
  --token "$(cat "$EVALUATION_CAPABILITY_FILE")" \
  --no-speed

python scripts/npu_release_gate.py \
  --service-url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" \
  --output /tmp/npu-smoke-report.json

# 在 NPU 主机执行，逐张测试卡替换 card id；不要与该卡的 worker 负载同时运行。
ASCEND_RT_VISIBLE_DEVICES=0 \
  python -m pytest tests/test_card_health_npu.py -m npu -q

# 配好 profiler 和 sandbox 环境；输出目录使用本次运行的新目录。
python scripts/probe_profiler_latency.py \
  --device 0 --sandbox --output /tmp/profiler-probe
```

完整方案不能只执行 `pytest -m npu`：现有 release_gate 有自己的凭证、URL、超时和“不允许 skip”的入口；压力与恢复场景还缺驱动工具。

## 2.1 本轮只用一题：官方 BatchMatmul

测试材料固定为 [BatchMatmul fixture](../../tests/fixtures/npu_kernelbench/batchmatmul/README.md)，
保留全部 **51 cases**。官方 GitCode 提交为
`0607a29d231cde384c9dfdb8a39f54f91837c327`；`definition.json` 和 `workload.jsonl`
字节未改，其组合 SHA256（中间一个 NUL 字节）为
`57e902365bd9b2fa437cc500cf0d862989b5881f439b2157de7341b127126c73`，
与 open-evo 归档声明的 `official_digest` 一致。

| 官方 UUID | A × B | dtype | 覆盖点 |
| --- | --- | --- | --- |
| BatchMatmul-003 | `[2,7,15]` × `[2,15,7]` | float16 | 小尺寸、非对齐维度 |
| BatchMatmul-006 | `[1,2,3]` × `[1,3,2]` | float32 | 最小输入规模 |
| BatchMatmul-010 | `[9,607,1213]` × `[9,1213,607]` | bfloat16 | BF16 和非对齐 K |
| BatchMatmul-044 | `[128,1024,2048]` × `[128,2048,1024]` | float32 | A+B 为 2 GiB，加输出约 2.5 GiB；不是峰值显存估计 |
| BatchMatmul-051 | `[40,2590,5181]` × `[40,5181,2590]` | float16 | 大尺寸、非对齐尾块 |

以上是边界示例，完整入口仍执行全部 51 cases，含 FP32 18、FP16 18、BF16 15 组。
服务现在以 `definition/workloads` 为唯一输入协议，按固定版本官方生成策略在 NPU
生成数据，默认 seed 200。归档 adapter 的 seed 0/标准正态路径与此不同，历史 tensor
hash 和旧时延不能直接作为当前输入或速度结论。输入生成对齐不更换服务自己的精度
判定算法，也不把历史候选视为已通过当前构建。

准入以固定版本上游代码为规范：API 只拒绝安全/资源上限和上游对任何 seed 都必然
失败的参数。其余请求原样交给生成器，行为与上游一致，包括取决于 seed 的生成失败。
声明了 per-workload tolerance 的 workload 按固定版本官方判定（声明字段 + dtype 标准、
0.99 比例、32 ULP 误差上限、`-inf` 选项），未声明的仍用服务自己的精度判定。固定提交
的 105 个官方题均可原样提交；其中 DequantSwigluQuant、TopKTopP 和 3 个
MiniMaxSparseAttention 依赖声明的 tolerance，需在实机上单独验收其正确性判定。

先离线验证 fixture，随后对部署的新接口运行完整链路：

```bash
python scripts/npu_batchmatmul_gate.py \
  --check-only --output /tmp/batchmatmul-fixture-check.json

python scripts/npu_batchmatmul_gate.py \
  --url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" \
  --deadline-seconds 3600 --job-timeout 19000 \
  --output /tmp/batchmatmul-npu-report.json
```

服务须配置 `TRITON_EVAL_REFERENCE_DEVICE_POLICY=npu-strict`，并为大 case 配置足够的
设备内存、verify/benchmark/profile 时间预算。引擎按 case 在 NPU 生成输入：benchmark
（含历史官方 adapter 的 reference cache 查询）与 profile 在生成下一组前释放上一组，
verify 最多同时持有相邻两组，不会一次物化全部 51 组。客户端 deadline 不会扩大服务端已有预算。计时阶段明确关闭 baseline cache，以得到新鲜 Torch/候选测量。报告保留
资源 ID、输入/reference digest、部署 provenance 以及逐阶段结果；任一阶段缺失、
失败或 profile partial 都不会标记全量通过。`--check-only` 只检查文件一致性，
`hardware_verified` 明确为 false。

已有底层 Event/profile 测试也可单独执行：

```bash
TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 ASCEND_RT_VISIBLE_DEVICES=0 \
  python -m pytest tests/test_event_timing_npu.py tests/test_profile_npu_integration.py -m npu -q
```

这些底层测试与 HTTP BatchMatmul 验收分别记录，不合并推断覆盖范围。

## 3. 测试环境与证据

使用专用测试实例、Redis 命名空间和 NPU 卡；故障注入只针对该实例。卡数记为 C，每张卡一个 NPU worker，另有 CPU worker。全部 worker 使用相同的部署配置及 baseline namespace。

每次保存：Git SHA、evaluator build、OpenAPI hash、NPU 型号/卡列表、驱动/CANN/Torch/torch_npu/Triton 版本、sandbox/overlay/source pins、配置摘要、测试语料版本、种子、请求/结果 ID、客户端提交/完成时刻、服务端 created/started/finished 时刻、检查结果、cache hit、资源采样和日志路径。凭证及私人源码不进入报告。

设备隔离检查、时延稳定性检查、吞吐压力测试分开进行。时延基准在空闲卡上测；压力下的端到端延迟用于容量分析，不当作 kernel 的固有时延。

通过、失败、未执行/不支持必须分开记录。必需项目出现 skip 或 missing evidence，整体不能标为通过。HTTP 200、resource completed、CLI 退出 0 都不能单独证明全部 requested checks 成功。

## 4. 全功能实机矩阵（需补充自动化）

| 组 | 场景 | 关键断言 |
| --- | --- | --- |
| 接入与契约 | `/evaluations`、`/verify`、`/benchmark`、`/profile`、GET、健康、OpenAPI、metrics；缺失/错误凭证、权限、非法 JSON/字段、体积上限、超限 | 状态码和 typed error 正确；被拒请求不执行代码；observer-only metrics；错误流量不阻塞健康检查 |
| 生命周期 | wait=0 / bounded wait；客户端断连；相同 key 重放；相同 key 不同内容冲突；跨 caller 读取 | 一个已接受逻辑请求对应一个可恢复 ID；重放不重新执行；权限隔离；客户端 timeout 不取消已接受任务 |
| 运行模式 | comparison、Torch-only、Triton-only；compile、correctness、performance、profile 及所有合法组合 | comparison 才给正确性和 speedup；单实现结果明确 unverified、无伪造 counterpart；单实现 `/verify` 拒绝；compile 检查实际提供的源码 |
| 输入生成 | 多 case、seed、shape、dtype、range、标量参数、选择子集、空张量等边界；官方 105 题原样提交（含 `perf_baseline`、未声明输入、浮点 `range`、per-workload tolerance）；上游取决于 seed 的生成失败（如未指定分布、`high` 小于默认 `low`、seed 0 抽到 uniform family） | 正确映射原 case index；相同 seed 可复现；不支持的设备能力给明确失败/未测，不能伪造成功；不要求硬件不支持的 dtype 通过；生成失败须返回不可重试的 `INPUT_GENERATION_FAILED` 并指明 workload，卡不被标为 suspicious、不重排；torch `RuntimeError`/OOM 仍按引擎故障处理，不能被归为请求错误 |
| 历史官方 reference cache | 启用 reference cache 的 seed 0 归档任务：命中、miss 回退、哨兵重测 | 逐 case 查询、miss 回退与哨兵测量时设备上只持有一组输入；case 编号与缓存 manifest 一致 |
| 数值正确性 | 已知正确 pointwise/reduction/matmul 语料；错误输出、shape/dtype 不符、NaN/Inf、输入修改；non_compute 分支；声明 tolerance 的官方题（DequantSwigluQuant、TopKTopP 的 `-inf`、MiniMaxSparseAttention 的 0.02） | 精度判定与配置一致；错误 candidate 是有效失败 evidence，不能变成基础设施成功或漏跑；正确性失败后的依赖检查按规则 skipped；声明 tolerance 的 case 按固定版本官方判定，未声明的 case 判定不变 |
| compile / JIT / runtime | Python 语法、import 错误、Triton JIT 错误、执行异常 | 语法/import verdict 与设备编译失败分清；negative verdict 缓存不能错误接管依赖 shape 的 JIT 失败 |
| 性能 | event、profiler；不同 warmup/repeats；case subset；部分/全部不稳定；单实现与comparison | 正时延、正确 method/coverage；comparison speedup 可从分母重算；没有稳定样本时显式未测/失败；不能只检查 correctness 退出码 |
| Torch baseline cache | 冷→热→旁路→热；仅换 candidate；改 Torch/cases/seed/method/warmup/repeats；部分覆盖；并发冷 miss；重启worker | 只有 reference 标量入缓存；hit 节省参考计时但仍做正确性及候选计时；旁路不读写/覆盖；不同任务/协议不串用；单实现两种 flag 均不访问 comparison cache |
| precheck verdict cache | 正/负缓存、TTL、toolchain fingerprint、并发正负发布、关闭缓存、Redis advisory 超时 | key 作用域正确、负 verdict 优先、晚到 positive 不延长 negative TTL；缓存故障退回执行；不能访问其他 cache 的私有状态 |
| Profiling | op/deep；三种 implementation；部分case；缺失msprof、无效或部分产物、超时 | implementation 归属正确；partial/missing evidence 如实保留；Torch native profile 的初始化开销及捕获上限不能被宣称为 run-only 或全覆盖 |
| 执行配置 | 自定义 engine_dir；不同 reference policy；显式 Settings 与冲突 ambient env；启动后 env 变化；错误pin或overlay | 所有检查使用同一启动配置；precheck 不偷偷换回 bundled engine；配置不一致和无效 pin 在执行前失败，不能回退到无隔离模式 |
| 隔离与清理 | sandbox sentinel 文件访问、路径/结果 symlink、任务间隔离、子进程/孙进程 timeout 清理 | 候选读不到受保护的测试 sentinel；只接收本 attempt 的普通文件；超时/失租后无遗留进程占卡；未启用 sandbox 的部署不能通过隔离门槛 |
| 观测 | observer 抓取 metrics、慢/故障 metrics 连接、错误响应计数、CPU/NPU worker 指标 | `/livez` 不被同步观测阻塞；指标故障不改变 admission、续租和终态提交；同一行为无错误重复记账 |

语料按功能分类并冻结源码及期望结果。先用小工作量语料验证所有分支，再用中/大工作量评估性能与压力，避免把算子自身不稳定误判成调度问题。

## 5. 实机恢复与故障注入（单独阶段）

每个场景先让一个已知正确任务完成，再注入一次故障，最后再次提交 canary 和正确任务验证恢复。不能只证明任务失败，而不证明实例恢复。

- 运行中结束 engine 子进程、结束 worker、停止 worker 再恢复：检查资源恢复、lease 重领、旧 attempt 写入被 fencing 拒绝，以及无遗留 NPU 占用。
- 排队超时和运行超时：验证准确 terminal state、已产生 evidence 的保留、未执行检查的 skip 原因。
- API 重启 / 客户端断连：同一实例上的已接受请求保持可恢复；不自动向另一实例重复提交。
- Redis 暂时断开和进程重启：使用测试 Redis，验证已确认接受的数据可恢复、失租 worker 不继续发布；区分 durable Redis 故障与仅 advisory 连接故障。
- 可控 OOM / 设备错误：验证卡健康状态、停止接新任务、canary 与恢复阶梯；恢复命令按实际测试主机配置，不能假定所有 Ascend 都有相同 reset 命令。
- 压力中单 worker 故障：验证其他卡的进展，以及故障卡恢复后的接单；跨 host 不要求自动迁移，当前设计是实例自有状态。

恢复时间以配置推导并记录：lease 过期、maintenance 扫描、排队和 canary/recovery 时间分别计量；不预设一个与部署配置无关的固定秒数。默认不启用的 engine failure requeue 与启用后的行为分开验证。

## 6. 压力与稳定性方案

先测单卡串行的服务耗时与吞吐，再确定目标负载。以下时长是建议起点，可随实机预算调整。

| 阶段 | 负载 | 建议持续时间 | 判定重点 |
| --- | --- | --- | --- |
| 空载与预热 | 各卡串行固定任务集 | 10–15 min | 初始化、首次编译与冷cache稳定；建立内存/时延基线 |
| 并发阶梯 | in-flight = 1、C、2C、4C | 每档 10–15 min | 吞吐、排队增长、P50/P95/P99、卡利用率；找饱和点 |
| 定速负载 | 根据测得容量依次发 50%、80%、100%、120% 的请求率 | 每档约 15 min | 使用独立到达率驱动，避免闭环客户端掩盖排队；超载时可观测且有界 |
| 突发 | 短时 4C–8C 请求，再停止提交并排空 | 10轮 | 配额与拒绝码符合预期；已接受资源不丢失；队列可以排空 |
| 混合业务 | 默认建议60% comparison benchmark、15% verify、10%单实现benchmark、10%compile、5%profile | 1 h | CPU/NPU队列、长profile与短任务混跑、GET轮询与metrics；按真实业务调整比例 |
| steady soak | 固定语料集，约实测饱和吞吐的60%–70% | 先8 h，再24 h | 持续正确性、无意外超时、无僵尸进程、worker内存/FD/线程不持续增长 |
| recovery soak | 稳态负载下，分阶段注入上一节故障 | 1–2 h | 恢复后的成功率与容量回归；每次故障之间留恢复窗口 |

不要直接把默认配额当设备容量：当前默认 trial 60 requests/min、4 inflight、每session 2 inflight。先单独验收这些限额；再给专用压测 caller 配置足够的受控额度并使用多个 session，以真正压到 worker。预期的配额拒绝单独统计，不能混入正常请求错误率，也不能被算成成功执行。

闭环并发测试和定速到达测试都需要：前者找最大并行吞吐，后者暴露积压与尾延迟。正常流量测量与故障注入数据分组，不混算一个成功率。

## 7. 稳定性验收门槛

以下是拟定门槛，实机开始前冻结；不是已取得的结果。

1. 必需功能场景全通过，无跳过、错归属、伪造正确性或伪造速度比。
2. 专用稳定环境下，已接受的已知正确任务零意外数据错误、零重复终态提交、零丢失资源；所有接受的任务最终有合法终态。
3. 配额内稳态阶段零非预期5xx/worker crash/任务超时；预期 quota rejection 和注入故障单列。
4. 停止负载后队列在按剩余任务耗时和 deadline 推导的排空窗口内归零；正常健康 worker 继续接单。
5. 子进程/FD/线程/NPU占用在预热后无持续泄漏趋势。对固定语料观察 parent worker RSS 与设备常驻内存；一次性JIT/cache增长不直接判泄漏。
6. Redis容量按“保留的评测资源 + 固定baseline/verdict keys”解释。baseline无TTL、评测结果按保留期增长是设计行为；不能把预期持久化增长当泄漏，也不能通过清空持久库掩盖增长。
7. Kernel计时稳定性另设空闲卡实验：固定 task/protocol 至少重复30次，报告CV、中位数、P95及前后窗口变化。可先用CV≤5%作为足够长、确定性任务的候选阈值；微秒级/噪声任务按测量分辨率单独定门槛，不能把引擎现有CV过滤阈值当作发布SLO。
8. API延迟和恢复时间先基于单卡/满卡基线定SLO，再比较稳态和恢复后窗口；不存在适合所有硬件/任务的统一绝对毫秒门槛。

## 8. 后续实现顺序与产物

建议保留现有 smoke 入口，补一套与业务源码分开的测试工具：

1. 在已有 BatchMatmul HTTP 入口上补齐异常与恢复矩阵，复用已有纯逻辑断言；对必需用例禁止 skip。将性能成功条件纳入正式门禁，不能沿用 smoke 的 correctness-only 退出判定。
2. 增加负载驱动：固定并发与定速两种模式、提交台账、幂等/恢复、轮询限速、排空、资源采样和结果汇总。可使用标准 Python + 仓库已有 httpx；暂不需要部署新的压测平台。
3. 增加显式选择的故障注入阶段，驱动只操作测试实例的具体 worker/Redis，保存故障前后证据。
4. 统一输出 report JSON、JUnit、请求台账 JSONL、资源时间序列与失败样本；report分别给出functional、performance、recovery、load、soak的状态，缺项不生成“全量通过”。

建议第一次实机按“现有smoke → 全功能矩阵 → 单故障恢复 → 容量阶梯 → 8h soak”推进。每一阶段有明确退出条件，失败先保留最小复现，不继续累积不可解释的压力数据。
