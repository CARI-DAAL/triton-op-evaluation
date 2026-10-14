# NPU 验证与反馈提速测试指南 草案

本指南验收 [EVF 01 至 08](../plans/evaluation-efficiency-tickets.json) 的设备行为和实际收益。
代码基线为 `312bbc7`；新版以实际部署的 commit、镜像 digest 和 evaluator build 为准。
CPU 验收记录为 **2217 passed、22 skipped、7 deselected**；测试机提供的首轮 NPU 实测
见[第 9 节](#9-npu-实测记录2026-10-14分析机-atlas-800t-a3)。完整设备验收仍有待办。

测试按“部署核对 → 小算子功能验收 → 联合执行对照 → 代表性工作量 → 复用与恢复”推进。
先取得完整、口径一致的证据，再判断提速。一次请求返回 `completed` 不代表候选正确，
也不代表每个 profile case 都完成。

## 1 测量目标与范围

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| 首份可行动反馈时间 | 提交到所需 correctness 或 performance 证据可读的时间 | 衡量 agent 等待时间 |
| 整轮墙钟时间 | 提交到请求的全部检查终止的时间，包含排队、初始化和采集 | 衡量服务执行成本 |
| 服务流程加速比 | separate 墙钟中位数 / combined 墙钟中位数 | 衡量联合执行收益，大于 1 表示 combined 更快 |
| 算子 speedup | 同方法测得的 reference latency / candidate latency | 衡量 kernel 性能，单独报告 |
| 测量漂移 | combined / separate 的逐 case 候选、reference 和 trace 耗时比 | 检查提速是否伴随测量口径变化 |

本轮合并的是 **timing 与 `profile.mode=trace`**。Event 计时先在 profiler 关闭时完成，
随后采集 trace；profiler timing 对重叠 case 复用同一份采集。`trace` 提供 kernel 耗时，
不提供 `msprof op/deep` 的硬件利用率指标。

`op/deep` 支持多 case 共用外层 runner、元数据 discovery 和按需选择指标；原生采集进程
仍独立运行。Correctness 仍有自己的 engine 进程；显式 `correctness_from` 可以复用
符合条件的原始完整验证结果。Reference-only baseline 生产继续单独运行。

## 2 固定部署与实验条件

在专用测试实例上部署匹配的 API、worker 和客户端，按
[升级流程](recovery.md#upgrading)更新 evaluator build，并重新生成受源码变更影响的
profiler、sandbox、runtime overlay 和 cache source pins。完整客户端版本为 `0.4.0`。
新旧版本分别保留配置与证据；不要用旧 build 标识运行新源码。

第一轮使用 **单物理卡、单 NPU worker、串行请求**，固定如下条件：

- 卡型号与卡号、CANN/driver、Torch/torch_npu/Triton 版本、设备频率策略和背景负载。
- Reference device policy、输入源码、workloads 顺序、seed、tolerance 和 `non_compute`。
- 测量方法、warmup、repeats、baseline/JIT cache 状态、请求 deadline 和 worker 阶段预算。

版本信息应来自实际 worker 环境；客户端 Python 的版本不能代替 worker 版本。
保存 `npu-smi info`、部署配置中的非敏感部分、commit/镜像/build 标识以及报告中的
`provenance.execution_environment`。凭据只通过 capability 文件传入。

以下命令在仓库根目录执行，客户端机器只需能访问测试服务：

```bash
: "${EVALUATION_URL:?请先配置测试服务地址}"
: "${EVALUATION_CAPABILITY_FILE:?请先配置 capability 文件路径}"
export TRITON_EVAL_CLI="$PWD/plugins/triton-eval/skills/triton-eval/scripts/triton_eval.py"
export NPU_TEST_OUT="/tmp/triton-eval-npu-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$NPU_TEST_OUT"

python3 scripts/npu_release_gate.py \
  --service-url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" \
  --output "$NPU_TEST_OUT/smoke.json"
```

退出条件：readiness 和 add correctness smoke 通过。此 gate 的通过范围是 smoke，
不能代替下面的联合执行、定向 profile、复用或恢复验收。

## 3 生成可复现的小算子语料

从仓库内的固定 add smoke 生成客户端文件；该步骤只写源码字符串，不执行候选：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
from scripts.test_eval_service import build_request

out = Path(os.environ["NPU_TEST_OUT"])
body = build_request()
code = body["triton_code"]
assert "x + y" in code
(out / "definition.json").write_text(json.dumps(body["definition"], indent=2))
(out / "workloads.jsonl").write_text(
    "".join(json.dumps(case) + "\n" for case in body["workloads"])
)
(out / "candidate.py").write_text(code)
(out / "wrong.py").write_text(code.replace("x + y", "x + y + 1000"))
(out / "import-error.py").write_text("import evf_missing_module\n" + code)
(out / "request.json").write_text(json.dumps(body, indent=2))
PY
```

语料包含 4096 和 8192 两个 float32 case。先执行完整验证并保存复用句柄：

```bash
python3 "$TRITON_EVAL_CLI" verify \
  --definition "$NPU_TEST_OUT/definition.json" \
  --workloads "$NPU_TEST_OUT/workloads.jsonl" \
  --triton "$NPU_TEST_OUT/candidate.py" \
  --save-handle "$NPU_TEST_OUT/verify-handle.json" \
  > "$NPU_TEST_OUT/verify.json"
```

预期：CLI 退出 0，`result.correctness.data.outcome=correct`，`passed_cases=total_cases=2`。

## 4 验收正确性控制和错误反馈

重复上面的 verify 命令，按表替换候选或追加参数，并分别保存 stdout JSON。
这些是预期失败场景；批量脚本应显式记录退出码，避免 `set -e` 提前结束整个测试。

| 场景 | 候选与附加参数 | 预期 |
| --- | --- | --- |
| 完整正确 | `candidate.py` | 退出 0，完整 `correct` |
| 只验证一个正确 case | `candidate.py --case-indices 0` | 退出 2，`incomplete`，coverage 仅含 case 0 |
| 首错停止 | `wrong.py --stop-on first_failure` | 退出 1，`wrong`；selected 为 `[0,1]`，evaluated 为 `[0]` |
| 只验证后一个错误 case | `wrong.py --case-indices 1 --stop-on first_failure` | 退出 1，failure 的原始 `case_idx=1` |
| 收集全部错误 | `wrong.py --stop-on complete` | 退出 1，两个 case 均有实际差异证据 |
| 导入失败 | `import-error.py` | 退出 1，`compile_error`；不是卡故障或数值误差 |

检查 `coverage.selected_case_indices`、`evaluated_case_indices`、`stop_reason` 和失败位置。
未显式设置 correctness 控制参数时，默认全量结果可以没有 `coverage` 字段。
未执行 case 不应被计为失败；部分通过不应授权完整 comparative performance。
若要证明“未生成后续输入”，需结合输入生成计数或测试实例日志；单靠更短墙钟不能证明。

## 5 验收 timing 与 trace 联合执行

请求完整 correctness、event timing 和 case 0 的 trace，并在 timing 可读时先返回：

```bash
python3 "$TRITON_EVAL_CLI" evaluate \
  --definition "$NPU_TEST_OUT/definition.json" \
  --workloads "$NPU_TEST_OUT/workloads.jsonl" \
  --triton "$NPU_TEST_OUT/candidate.py" \
  --checks correctness performance profile \
  --measurement-method event --warmup 5 --repeats 20 --no-baseline-cache \
  --profile-mode trace --profile-cases 0 \
  --wait-for performance --save-handle "$NPU_TEST_OUT/joint-handle.json" \
  > "$NPU_TEST_OUT/early-performance.json"

JOINT_ID=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["NPU_TEST_OUT"])/"joint-handle.json").read_text())["evaluation_id"])')
python3 "$TRITON_EVAL_CLI" progress "$JOINT_ID" > "$NPU_TEST_OUT/progress.json"
python3 "$TRITON_EVAL_CLI" get "$JOINT_ID" > "$NPU_TEST_OUT/joint-final.json"
```

检查以下证据：

1. 运行中快照通过 `completed_checks.performance` 返回 timing；已终止请求也可能直接返回
   带 `result` 的完整 resource，两种情况都合法。
2. Event performance 的方法为 `npu_event_v1`；trace 的方法为 `profiler_kernel_sum_v1`。
   两种时间分开解释。
3. Correctness/performance 覆盖两个 case，trace 只覆盖 case 0，选择范围没有被扩大。
4. 以同一 evaluation ID 可以取回最终结果；`attempt_id` 和 `progress_revision` 能区分
   旧 attempt 失效与新 attempt 的证据。

小任务可能在客户端第一次观察前就全部完成。需要观察提前反馈时，可增加 trace 工作量；
不能为了展示先后顺序而把未产生的结果标为完成。

## 6 测量 separate 与 combined 的实际收益

先测 add smoke，再测至少一个计算量较大的 workload 和一个带宽主导的 workload：

```bash
python3 scripts/npu_feedback_comparison.py \
  --url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" \
  --method event --pairs 5 --warmup-pairs 1 \
  --output "$NPU_TEST_OUT/feedback-event.json"

python3 scripts/npu_feedback_comparison.py \
  --url "$EVALUATION_URL" \
  --capability-file "$EVALUATION_CAPABILITY_FILE" \
  --method profiler --pairs 5 --warmup-pairs 1 \
  --output "$NPU_TEST_OUT/feedback-profiler.json"
```

脚本在同一个新版实例上交替运行：

- **separate**：一次 correctness＋timing 请求，然后一次独立 trace 请求。
- **combined**：一次 correctness＋timing＋trace 请求。

两边均验证一次，使用相同的 warmup 5/repeats 20、case 范围和新鲜 Torch baseline。
指定的 warmup 配对保留在报告中，但不进入摘要；只对完整通过的配对计算比值。
任意一边失败会令本轮 `passed=false`，不能丢掉失败一半后继续宣称提速。

使用 `--request request.json` 测自选任务。`--cases 0 3 7` 只适用于实际包含这些索引的
请求，而且只缩小 timing/trace 范围：correctness 仍覆盖该请求的全部 workloads。
官方 BatchMatmul 全量验收及其 `npu-strict` 配置见
[NPU 验证计划](npu-validation.md#21-本轮只用一题官方-batchmatmul)。

当前脚本是 **新版内部两个执行策略的对照**。基线 `312bbc7` 没有 trace mode；对比
旧版与新版时，应另用两版都支持的相同 verify、benchmark、op/deep 请求。
两个版本分别记录 build，保持 reference policy 和测量方法一致。

脚本为每个资源使用相同的 deadline；separate 有两个资源，combined 有一个资源。
大任务需事先固定合适预算，并报告超时。客户端 timeout 不会扩大 worker 的阶段预算。

### 读取报告

重点查看：

| 字段 | 含义 |
| --- | --- |
| `passed` | 请求成功、方法和覆盖范围满足脚本检查；若未设漂移阈值，不代表测量等价性已通过 |
| `execution_identity` | 整轮固定的 evaluator build 与环境身份 |
| `summary.*.median_wall_seconds` | 各策略墙钟中位数 |
| `summary.*.median_time_to_performance_seconds` | performance 可读时间的中位数 |
| `summary.separate_over_combined_wall_ratio` | 服务流程加速比 |
| `rounds[*].measurement_comparison` | 逐 case 候选/reference/trace 耗时比和 kernel 名称变化 |
| `rounds[*].*.requests` | 每次资源 ID、幂等键、阶段结果和恢复信息 |

先在空闲卡上标定噪声，再为正式验收选择 `--max-relative-drift FRACTION`。
未传此参数时脚本只报告漂移；传入后，候选/reference/trace 的相对漂移超限会失败。
阈值应在正式比较前冻结。

默认轮询间隔 0.2 秒，首份反馈时间含轮询与网络观察误差。若 attempt 被替换，脚本将
首份 timing 的时间保守回退到终态观察时间，避免把旧 attempt 的早期结果记到新 attempt。
脚本不清 JIT cache：当前测量主要观察预热后的执行；冷 cache 使用单独目录另做实验。
5 个配对适合首次检查，可靠的尾延迟结论需要更多有效样本。

## 7 验收定向 profile 和显式正确性复用

只对 case 0 采集两个 deep 指标组，关闭 per-core sampling：

```bash
python3 "$TRITON_EVAL_CLI" profile \
  --definition "$NPU_TEST_OUT/definition.json" \
  --workloads "$NPU_TEST_OUT/workloads.jsonl" \
  --triton "$NPU_TEST_OUT/candidate.py" \
  --profile-mode deep --case-indices 0 \
  --metric-groups Memory L2Cache --no-per-core \
  --save-handle "$NPU_TEST_OUT/deep-handle.json" \
  > "$NPU_TEST_OUT/deep-selected.json"
```

预期 `capture_requested` 与请求一致，`raw_metrics` 保留实际采到的值，不把未请求指标
计为缺失。通过 worker 命令/进程日志核对两次指定指标采集、没有 per-core sampling。
现有三个预热进程仍保留；“两次指标采集”不等于“总共启动两次进程”。
再用 `--case-indices 0,1` 验证多 case 只有一个外层 profile engine，原生采集按 case 执行。
`max_kernels_per_case` 限制返回数量，截断应标记 partial，不能算采集成本下降。

复用第 3 节取得的原始完整验证：

```bash
VERIFY_ID=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["NPU_TEST_OUT"])/"verify-handle.json").read_text())["evaluation_id"])')
python3 "$TRITON_EVAL_CLI" benchmark \
  --definition "$NPU_TEST_OUT/definition.json" \
  --workloads "$NPU_TEST_OUT/workloads.jsonl" \
  --triton "$NPU_TEST_OUT/candidate.py" \
  --correctness-from "$VERIFY_ID" --no-baseline-cache \
  --save-handle "$NPU_TEST_OUT/reuse-handle.json" \
  > "$NPU_TEST_OUT/reused-correctness.json"
```

预期 `correctness.data.reused_from` 指向该原始验证，candidate timing 仍新鲜执行。
该请求可能等待原物理卡。源码、seed、tolerance、环境或有效期不匹配时应明确拒绝，
而不是静默重新验证。分别用 `wrong.py`、不同 `--seed` 和部分验证的 ID 检查拒绝路径。
缺失关键版本信息、不同 owner、链式复用和 grade 复用也应拒绝。

## 8 恢复 多卡和结果交付

单卡基本功能通过后，在独立阶段验证：

- 已发布 timing 后 trace 失败或超时：timing 仍可从 progress/最终资源恢复。
- Worker replacement：旧 attempt 的阶段证据失效，revision 递增，新 attempt 重新建立证据。
- 客户端断连：用保存的 handle 在原实例恢复，不提交重复工作。
- 多卡调度：在卡 0 验证并排队超过 10 个绑定卡 0 的复用请求；卡 1 仍能领取后面的普通任务。
  使用测试 caller 的明确配额，记录 FIFO 和等待时间。

故障注入与性能样本分开统计，按[恢复测试方案](npu-validation.md#5-实机恢复与故障注入单独阶段)
在专用实例执行。恢复后再次运行已知正确任务，确认实例能够继续接单。

每次实验至少交付以下材料：

| 项目 | 待填写内容 |
| --- | --- |
| 版本与环境 | 基线/新版 commit、build、镜像、NPU 卡号、完整软件栈 |
| 输入与协议 | task、输入摘要、seed、正确性策略、方法、warmup/repeats、cache 状态 |
| 功能 | 全量正确性、coverage、profile scope、复用拒绝与恢复是否通过 |
| 速度 | 有效配对数、墙钟/首份反馈中位数、流程加速比、逐 case 测量漂移 |
| 稳定性 | 失败/超时/partial 数及原因，是否有 attempt replacement |
| 产物 | JSON 报告、handles、必要的 worker 日志、测试配置 |

验收结论分为“功能证据通过”“测量等价性通过”“达到预定提速目标”三项。
提速目标在测量前约定；下节按测试范围记录已取得的首轮数据。

## 9 NPU 实测记录（2026-10-14，分析机 Atlas 800T A3）

本节来自测试机执行者提供的汇总记录；原始 JSON 与 handles 仍在测试机，尚未归档到
本仓库。测量版本为 `analysis/eval-feedback` @ `f54b9a0`，evaluator build 为
`git:f54b9a0…`。环境为 openEuler 22.03 SP4 aarch64、kernelgym Python 3.11、
CANN 9.0.0.0430、driver/npu-smi 25.5.0。

服务采用 host 部署，16 个 NPU worker 加 1 个 CPU worker。执行者报告设备拓扑为
8 NPU × 2 die，即 16 个运行时设备，对照期间无其他负载。本轮未提供逐 attempt 的
实际设备号，是否满足第 2 节固定同一物理设备的条件仍待核对。

测试机 CPU 回归为 **2226 passed / 13 skipped / 7 deselected**；
`npu_release_gate.py` smoke **passed**。此前 `312bbc7` 的完整五阶段 gate 通过，
属于基线版本记录，不能代替 `f54b9a0` 尚未完成的设备验收项目。

### 联合执行对照

语料为 add smoke 两个 case；`--pairs 5 --warmup-pairs 1`，warmup 5 / repeats 20。
两组报告均为 `passed=true`，本轮未冻结 `--max-relative-drift`。

| 方法 | separate 墙钟中位数 | combined 墙钟中位数 | 报告中的流程加速比 | 首份 performance 中位数 |
| --- | --- | --- | --- | --- |
| event | 45.9 s | 32.9 s | **1.39×** | 24.4 s → 20.0 s |
| profiler | 65.1 s | 43.4 s | **1.50×** | 43.7 s → 38.7 s |

按展示的中位数计算，墙钟分别减少约 **28.3% / 33.3%**，首份 performance 提前
约 **18.0% / 11.4%**。加速比沿用原报告，展示的耗时已经过四舍五入。
这些是服务流程收益，不是候选 kernel 相对 Torch 的加速比。

### 测量漂移与统计口径

执行者汇总的 combined/separate 逐 case 耗时比：

- profiler：**0.933–1.071**，观察范围约为 **−6.7% 至 +7.1%**。
- event：**0.773–1.21**，观察范围约为 **−22.7% 至 +21.0%**。

未预先冻结漂移阈值，因此当前只能报告观察范围，不能据此宣称测量等价性正式通过。
Event 波动可能涉及微秒级计时噪声；需要结合原始样本、预热分组和设备归属进一步核对，
本记录不将波动原因直接归为固有噪声。

原汇总标记 `n=54`，其计数单位、包含哪些指标/方法、是否纳入 warmup 尚待确认。
脚本对每个配对的每个 case 产生 candidate、reference、trace 三种比值；按所述
5 个正式配对 × 2 case × 3 指标，应为每方法 30 个正式比值，含 1 个 warmup 配对则
为 36 个。待核对前，不把 `n=54` 当作 54 次独立重复实验。

原始报告与 handles 路径（位于测试机）：

```text
/tmp/triton-eval-npu-20261014/feedback-event.json
/tmp/triton-eval-npu-20261014/feedback-profiler.json
```

### 当前结论与剩余验收

**首轮 add smoke 的功能证据通过；在所报告部署条件下，combined 流程加速为
1.39–1.50×，首份 performance 提前约 11–18%。** 结论限定于本轮两 case 小算子语料。
PR #61 保持草稿，完整验收还需：

1. 归档原始 JSON、实际运行命令和 `n=54` 的生成脚本；区分 warmup 与正式配对，
   分别统计 candidate/reference/trace 比值。
2. 导出逐 evaluation/attempt 的运行时设备号；核对是否同 die 对照，必要时补固定单卡
   或按设备分层的对照。
3. 补计算密集和带宽主导 workload，并在正式比较前标定噪声、冻结漂移阈值。
4. 完成第 7 节定向 profile 与显式复用的拒绝路径。
5. 完成第 8 节恢复及多卡调度项目。

### 本轮总结采纳与局部修正

本轮以执行者提供的总结作为改动判断依据，不再要求补传原始报告；上述未覆盖范围保留为
后续验证项，不据此扩大当前小算子结果的结论。

已收到的记录片段包含 `triton_version=null`，会令显式正确性证据复用因环境身份不完整
而被拒绝。版本采集现优先查询安装包元数据；缺少元数据时，读取实际 Triton 模块声明的
`__version__`。导入失败、版本缺失或值无效时仍保留未知状态，不授权复用。
该修正经源码安装场景的本地回归验证；历史记录不追补版本，已有 NPU 数字仍对应
`f54b9a0`。计时方法、采样参数、容差和联合执行流程保持不变。
