# REPORT — nb5-uni-grade-failure-fix-20261008

## Summary
1. **根因（双层，均已证实）**：(A) uni 被评分的 artifact 内核（kernel sha 5a3bb8be，SPLIT-fp16 Triton bmm）在 case 1 候选执行时**确定性触发 AI Core 硬件异常**（driver EZ9999 "fftsplus aicore error, core id is 3"，chipId:2/dieId:1）→ 507014 aicore timeout → C++ abort → engine exit=1 无结果；生产异步模式下错误浮现在下一处 device sync（case 2 的 reference 输入 digest，被误标为 aclnnInplaceNormal——先前报告的"reference 物化 aicore 楔死"归因被证伪，CANN 自身已警告异步栈不准）。实证：kernel 5a3bb8be 在 service 全部 6/6 次执行皆死（3+1 旧 grade + 1 次 in-trial + 今日 attempt-5 的 3 次=7 次执行尝试），同窗同卡同 reference 下其他框架内核反复 51/51 通过；ASCEND_LAUNCH_BLOCKING=1 直接复现（空闲卡5，worker@5 暂停期间）精确死在 verify.py `_run_candidate_forward`（case 1，`测试第1/51`+`输入类型判定` 之后、`[输出概览]` 之前），拿到 fftsplus aicore dump。(B) 2026-10-07 12:28:31–13:45:16Z 另有 devdrv "Conflict open udevid" 风暴（kernel journal 6834 行，npu-smi 也被拒）杀死 ambient 枚举（507033，uni attempt-1 与 12 次 in-trial 请求属此类）；风暴持有者低于 service，仍未知。
2. **修复 commit**：service 仓库（生产 checkout=/home/e84381970/open-evo-work/runtime-artifacts/nb3-nb7-cache-integrate-20261006/checkout，即 e74b1f5 所在 worktree）独立分支 **nb5-uni-grade-failure-fix-20261008 = c49f53a**：可重试 pre-subject 引擎失败（engine_failure 形状：retryable+execution+无 data）不再单次终态，改为按 reaper 同款 replacement 语义 requeue（共享 MAX_ATTEMPTS=3 预算）+ 写 last_failed_card 使既有 card_affinity_avoidance 生效（即 F1）；开关 TRITON_EVAL_ENGINE_FAILURE_REQUEUE（默认 true）。判分数学/reference/case/阈值零改动；预算耗尽/过期/被 fence 时保持旧终态语义。测试：新增 4 + 更新 1，1252 passed / 4 failed（4 个在基线 e74b1f5 同样失败，环境性）。
3. **部署状态**：已部署生产——17 个 worker 单元逐一重启（仅在其卡无 lease 时，进行中 gate trial 零扰动）；API pid 2874980 未重启、readyz 全程 200；evaluator_build pin 089a0261 未改（诚实记录：worker 运行 c49f53a 代码但心跳报 089a0261；新 build 应为 sha256:71fb6c42…，pin 更新+API 重启留给 lead 下个静默窗）。回滚：checkout 回 e74b1f5 分支并重启 worker，或设 TRITON_EVAL_ENGINE_FAILURE_REQUEUE=false。
4. **验证 EID**：ev_5b5a1160ddb8419c9ea1cfd6afb44354（同一 grade 操作 attempt-5，request_digest 04b5e546 逐字节相同，salt=4，HTTP 202，1/3 请求预算）：修复机械性**生效**——3 次 service 尝试、卡 8→10→4（journal 两条 "requeued retryable engine failure" + attempt_count=3、last_failed_card=10 实证），但 3 次全部死于同一确定性 AI Core 异常 ⇒ **未取得正常成绩**（acceptance "成功" 未达成：该 artifact 内核本身杀死 aicore，任何重试预算都无法为其评分；不改 artifact/引擎隔离即无解，二者均超出本任务授权）。
5. **无回归证据**：部署窗口（≥22:15Z）30 个完成 resource 中 13 个 51/51 correctness 成功（重启后的 worker 卡 0/1/2/6/8/9/11/12/13 等），唯一失败即上述确定性 artifact；gate trial 全程连续运行未受滚动重启影响。
6. **遗留给 lead**：①uni 需换 artifact（内核修复后重交，需人类裁定）或授权"逐 case 候选子进程隔离"引擎改造（使 aicore 崩溃内核得到正常 FAILED 成绩而非引擎死亡，工作量另立任务）；②conflict 风暴持有者归属 NPU/驱动 owner；③F2/F3 仍未实现（对本次根因非必需）。

## Verbatim constraints and identifiers (copied from the brief)
- "定位并修复uni-agent候选在真实eval service评分时于参考/设备阶段失败（NPU错误507014/507033）的根因，并经真实service请求证明修复有效。"
- "uni旧条件trial：request 04b5e546 / artifact 5a3bb8be；3次原评分 ev_84715ee1、ev_f13ab07e、ev_5df7f1b5 与1次追加评分 ev_1efb701d2fee4049b13aa6d4ce0e7106 均在候选执行前的reference/设备阶段失败（507014/507033）；trial内12次请求失败均终态。"
- "已撤回的归因：devdrv队列、Paseo/inotify队列（inotify溢出是tailscaled/gonotify问题，已修，与NPU无关）。不要重走这两条。"
- "未实现的service缺口（来自先前报告）：F1 last_failed_card、F3 health-key消费、F2 canary提案。"
- "关键对照：同一生产service（e74b1f5/BUILD089a0261，API pid2874980，HTTP 12907）上，pi/claude/dsh的评分请求能完成（claude ev_a9882cfb 51/51，dsh ev_21562aa2 已评分）。先用记录找出uni请求与这些成功请求在'请求内容/任务/卡/worker/引擎子进程/reference路径/时间'上的差异，定位首个失败边界，再决定修复。"
- "2026-10-07 04:01至2026-10-08 04:01有一个孤儿staging worker空闲占用card14（已退役），核对uni失败是否落在card14。"
- "证实原因后实施最小修复（service/engine/worker配置或代码均可）；不改判分数学、reference、case、阈值。改动在service仓库独立分支提交；部署到生产前后对比/readyz、16 NPU+CPU单元、API健康；保留回滚路径。"
- "经真实service对原uni artifact提交评分请求验证（本任务总计最多3次评分请求，含复现）；每次记录EID、卡、结果。成功=请求进入候选执行并得到正常成绩（正确与否都算评分成功）。"
- "若在预算内无法证实原因：停止，报告已排除项、首个失败边界和下一步需要的具体资源，不扩大范围。"
- "Summary<=12行：根因、修复commit与部署状态、验证EID与结果、对其他框架评分无回归的证据。"
- Budget: "一owner，thinking=max，150分钟；真实评分请求<=3；NPU只用于这些请求及其必要的复现。服务重启前确认没有在跑的评测lease。根盘<9GiB时不做大写入。429停止报reset。"
- Out of scope: "运行uni新条件gate slot或任何Agent trial（修复验证后lead另派）、改判分数学/reference/阈值、驱动/固件更改、主机重启、改gate/ATIF判据、/home/e84381970/open-evo仓库检出（nb5-gate-hq-accepted-rule-20261008独占）、与其他worker通信。"
- Human authorization: GOAL.md Human rulings 2026-10-08 "那你派一个agent 去修复uni 的问题"。

## AC1 — 观察 / 未知 / 假设 分离 + 差异表 + 首个失败边界

### 差异表（uni 失败 vs 同 service 成功 grade；全部来自留存记录）
| 维度 | uni（7 次执行全死） | pi/claude/dsh（成功） |
|---|---|---|
| 请求形状 | official_l3_BatchMatmul / reference_digest 3115965c / checks=[correctness,performance] / perf=profiler 20+5 / deadline 1800 | **逐字段相同**（store canonical_request 对比） |
| kernel_code | **artifact 5a3bb8be**（fp32→fp16 SPLIT 三项 tl.dot + do_not_specialize×19） | 各自框架内核（如 8d9093db、8a98305b 等，均有 51/51 成功记录） |
| 卡 | Oct7: 10/6/4/9；Oct8: 8/10/4 | 9(13:57Z)、0/6/8(17:0x)、1(20:56)、10(21:13)——与失败卡重叠 |
| worker/引擎子进程 | 同一 systemd fleet（2026-10-07T02:25:52Z 起 pids 32848xx 未断），同 bwrap sandbox manifest 2e2daf23 + runtime overlay 2f5808d1，同 ENGINE_DIR | 相同 |
| reference 路径 | 同一 model-bound cache（index v2prime） | 相同 |
| 时间 | 13:41–15:43Z(Oct7) + 22:22–22:51Z(Oct8) | 13:57Z（与 uni attempt-3 同窗同时长！）、17:07–17:21、20:56、21:13 |
| 结果 | 7/7 engine exit=1（507014 或 ambient 507033） | 51/51 正常成绩 |

关键对照修正：pi ev_c77ab78b（13:57–14:04Z card9 成功）与 uni ev_5df7f1b5（13:55–14:04Z card4 死亡）**同时刻、同 reference、同缓存**——唯一差异是内核字节 ⇒ 排除"时间段/卡/缓存状态"类解释。card14 孔儿 worker：uni 失败卡为 10/6/4/9/8/10/4，**均不在 card14**（card14 仅今晚 20:41 一次 trial 失败，且该 trial 属其他 agent）。

### 首个失败边界（文件:行）
- **模式 B（主导，7 次全属此类除 attempt-1）真实边界**：`triton_evaluation/engine/verify.py` `_run_candidate_forward`（约 1111 行调用，1046–1079 定义）case 1 候选 launch → AI Core 异常。生产异步模式下的表面栈（verify.py:1466 load_case → reference_cache.py:708/730/700/406/386/365/355 `_tensor_bytes .cpu()`）是**下一处 stream sync 的误归因**（CANN stderr 自注 "the operator is called asynchronously, the stacktrace may be inaccurate"；引擎 stdout 缓冲在 C++ abort 时丢失 ≤8KB，致表面看像 case1 已完成）。决定性证据：ASCEND_LAUNCH_BLOCKING=1 直接复现死于 `测试第 1/51`+`[输入类型判定]` 之后、`[输出概览]` 之前 = 候选 forward 内；driver dump：`EZ9999 … fftsplus aicore error, core id is 3 … mte error info: 0x4603000013, ifu 0x65c18406a6000, ccu 0xc52c0b9f23000013 … [aicore timeout] davinci_kernel_task.cc:1318`。
- **模式 A（Oct7 attempt-1 + 12 次 in-trial）**：`engine/_common_utils.py:237-289` ambient 枚举等待全 240s 耗尽——kernel journal `[ascend][uda][uda_occupy_dev_by_ns 941] Conflict open udevid`（udevid=0..15 全卡；npu-smi 亦被拒）风暴窗 **2026-10-07T12:28:31–13:45:16Z**（6834 行；另 Oct6 18:45–18:48Z、Oct7 06:49–07:02Z）。持有者未知（低于 service）。
- 次要修正：先前报告"17 心跳同时过期=单次 fleet 停止(17:43:59Z)"、"执行 harness=svc-validation-p0@bc4906c 无 bwrap"均误读——systemd fleet pids 32848xx 自 02:25:52Z 连续在役；run_worker.py 由 WorkingDirectory(checkout) 解析、sys.path[0]=checkout ⇒ 实际 harness=checkout e74b1f5（含 bwrap sandbox，Oct8 03:58 CST 进程树实证）。

### 观察/未知/假设
- 观察：见差异表与上两节；store 全量 749 resource 按内核 hash 聚类：5a3bb8be=6/6 失败 0 成功（唯一从未成功过的多请求内核）；20:40/20:41Z 两 trial 同死"case 43 载入"= 同样的异步误归因（实为 case 42 候选崩溃），其 agent 换内核后通过——与"随机设备 flake"不相容。
- 未知：conflict 风暴持有者（需 NPU/驱动 owner+厂商，沿用 nb5-driver-queue 报告的厂商诉求）；fftsplus 异常的编译器侧根因（triton-ascend 3.2.0 codegen 或内核自身访问模式）——修复它=改 artifact，超授权。
- 假设（已证伪不再持有）："参考物化 aicore 楔死=偶发设备态"、"uni 只是调度运气差"。

## AC2 — 最小修复与部署
- 分支 **nb5-uni-grade-failure-fix-20261008**，commit **c49f53ababadd7cd59de2f425b1b02ec4444bfc9**（基于 e74b1f5）：worker.py 增 `_retryable_engine_failure` 判定 + requeue 分支；broker.py 增 `requeue_retryable_engine_failure(lease, failed_card)`（镜像 reaper replacement：bump fence、replacement_pending=1、ready 重入队、共享 MAX_ATTEMPTS=3、写 last_failed_card）；config.py 增 `engine_failure_requeue=True`。F1 就此落地；F2/F3 未做（对本根因非必需：canary 无法预测内核触发的 aicore 崩溃）。
- 判分面零改动；非 retryable（deadline/watchdog/poison）、产出任何 verdict、预算耗尽、被 fence——全部保持原终态语义。
- 测试：tests/test_engine_failure_requeue.py（4 个新测试：决策判定/replacement+卡回避/共享预算/陈旧 lease 拒绝）+ test_compile_check.py 更新为新契约（3 次耗尽后终态）；全量 1252 passed / 4 failed（4 个在基线 e74b1f5 相同失败：integer_metrics×2、reference_policy×2，环境性）。
- 部署：确认运行中 lease 后，17 个 worker 单元逐一"仅空闲卡时"重启（22:15–22:22Z；card2/8/14 等其空闲再重启，进行中 gate trial 零失败重启）；API 2874980 未动。前后对比：/readyz {"ready":true} 前后一致；17/17 units active（含 worker@5 诊断后恢复）；API pid/start 不变；心跳 17 条新鲜。回滚路径：`git checkout nb3-nb7-cache-integrate`（e74b1f5）+ worker 重启，或 env `TRITON_EVAL_ENGINE_FAILURE_REQUEUE=false`（无需回代码）。
- 诚实缺口：TRITON_EVAL_EVALUATOR_BUILD 仍 pin 089a0261（未重启 API 故未 bump）；正确值应为 sha256:71fb6c42cca4ab96493f24539ed037919e05e6d710c984ae5f98aec843e52c78（BUILD 公式对 c49f53a 计算，存 evidence/new-evaluator-build.txt）——留 lead 静默窗执行（bump 需 API+worker 同时重启，会打断在跑 gate）。

## AC3 — 真实 service 评分请求验证（1/3 预算已用）
- 提交：attempt-5（同一 grade 操作，kernel/artifact 5a3bb8be、request_digest 04b5e546 逐字节校验相同、idempotency salt=4、grade token 走真实 HTTP 12907）。
- **EID ev_5b5a1160ddb8419c9ea1cfd6afb44354**：2022-10-07T22:22:45Z 创建 → 22:51:45Z 终态 completed；**卡轨迹 8 → 10 → 4**（journal 06:32:22/06:42:01 CST 两条 `requeued retryable engine failure of attempt at_273985e1… on card 8` / `at_eed6b0e0… on card 10`；store attempt_count=3、last_failed_card=10、attempt_card=4）——**修复机械性验证通过**（旧代码下第 1 次失败即终态，如今 3 次尝试+失败卡回避）。
- 结果：3 次尝试全部死于同一确定性 AI Core 崩溃（engine_crashes.log 第 1246/1313/1384 块，均为 case-1 候选边界 507014）⇒ correctness=EVALUATOR_EXECUTION_FAILED(engine exit=1)、performance=PRIOR_CHECK_FAILED——**未获正常成绩**。原因不在 service：artifact 内核本身杀死 aicore（AC1 模式 B），重试预算无法救。判定：本 AC 的"成功=正常成绩"**未达成**，且在本任务授权内（不改 artifact、不做大引擎改造）不可达成；剩余 2 次请求预算未使用（再交同一 artifact 必然同果，故不再消耗）。
- 附加诊断复现（非评分请求，NPU 用于"必要的复现"，空闲卡5、期间 worker@5 暂停后恢复）：ASCEND_LAUNCH_BLOCKING=1 同步复现，死亡点精确定位候选 forward，取得 fftsplus aicore dump（evidence/direct-repro-ascend-launch-blocking/）。

## 对其他框架无回归的证据
- 滚动部署期间与之后（≥22:15Z）store 完成的 30 个 resource：13 个 correctness 51/51 成功（卡 0/1/2/6/8/9/11/12/13/15，多为重启后 worker 执行），16 个在途，唯一失败=上述确定性 artifact 内核；gate trial 提交流从未因重启中断（3 条在飞 lease 全程未被触碰，重启仅作用于无 lease 的卡）。
- 单元/集成套件 1252 passed（4 个基线共有的环境性失败），含 recovery/fencing/api_v2/compile_check/worker 全绿。

## 已排除项（本任务新增）
- "reference/输入物化路径故障"（异步误归因；reference 路径对所有框架同一且健康）；
- "偶发设备 flake/坏卡"（7/7 同内核确定性复现+同窗他内核 51/51；卡在成功/失败间反复）；
- "评分请求形状/任务/缓存/引擎/沙箱差异"（逐字段对比相同）；
- "card14 孤儿 worker 争用"（uni 失败均不在 card14）；
- "devdrv/inotify 队列"（brief 已撤回，未重走）。

## 下一步需要的具体资源（供 lead）
1. 人类裁定：uni 换 artifact（内核修复后重交 grade——旧 4+1+1 终态不动，新 EID）或授权引擎"逐 case 候选子进程隔离"改造（使 aicore 崩溃内核获得正常 FAILED 成绩；含 bwrap 内 fork 语义与 51×崩溃恢复成本评估，建议另立任务）。
2. NPU/驱动 owner：12:28:31–13:45:16Z "Conflict open udevid" 风暴持有者识别（npu-smi 同被拒；需厂商 devdrv 队列/占用视图，诉求同 nb5-driver-queue-readonly-capture 报告）。
3. 静默窗：bump TRITON_EVAL_EVALUATOR_BUILD→sha256:71fb6c42… 并同启 API+17 worker（~10s 中断）。

## Evidence index（本任务产物）
- evidence/uni_grade_attempt5.py + WORK/raw/grade/fanout/uni-agent-extra-grade-attempt-5.json（EID/digest/attempt-5 契约记录）
- evidence/direct-repro-ascend-launch-blocking/{stdout,stderr}.log + evidence/aicore-error-dump.txt（fftsplus dump 摘录）
- evidence/pre-submit-npu-smi.txt（提交前 16 卡健康单次读取）+ evidence/new-evaluator-build.txt
- 生产留痕：engine_crashes.log @22:32:22/22:42:01/22:51:45（行 1246/1313/1384）；journal worker "requeued…" ×2；redis resource ev_5b5a1160（attempt_count=3, last_failed_card=10）
- git：service 仓库分支 nb5-uni-grade-failure-fix-20261008（c49f53a）；基线 nb3-nb7-cache-integrate（e74b1f5）未动
