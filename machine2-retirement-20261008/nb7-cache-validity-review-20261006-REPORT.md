# nb7-cache-validity-review-20261006 — REPORT

## Summary
- **结论：YES**。整合 c03fd80 + f5e43f1 后，仅靠既有机制合法重钉（index v2' + timing policy v7）即可推进下一次真实 service 验收；不改预算、不削弱 fallback、不动 sealed 数据、不关指纹检查。
- 按人类裁决"硬件+软件一样就可复用"逐项核对：合并候选上**未发现任何实际测量不等价**；当前阻碍全部是**机械 pin 失败**（对测量中性修复的字节指纹），且两处收敛位置都是现成机制：① `build_model_bound_index()`（reference_cache.py:866，algorithm_files 重钉当前引擎，binding key 本就不含引擎文件 hash）；② policy v7 按 v5→v6 先例（reference_timing_reuse.py target_service_source_sha256，仅需更新 benchmark.py 一个 pin）。
- 实测合并（本 scratch tmp/mrg，三方 merge）：唯一冲突 `tests/test_reference_cache.py:415` 测试 double 签名（合并解 `lambda _t, **_k: True`），`engine/verify.py` 自动合并；生产代码零冲突。
- 最小顺序：合并分支→现有测试→重导 v2'/v7/BUILD/overlay→staging 容器工具→native /verify→完整 grade/perf/sentinel→准入。预算 7200/600/1500/900、哨兵>3%、8 样本全不动。
- 未知边界如实：canary B 超时只证明"policy失效→fresh 8×51 撑爆 600s 腿"机制成立（每阶段耗时未留存）；v2' 索引的 timing-hit 路径在 NPU 未跑过（policy 路径有 card2 51/51 先例）；NPU-init 根因未证明；容器边界 verify 未在合并候选上做过——验收必须含。

## Verbatim constraints and identifiers (copied)
- 原问题："把已完成NB3修复c03fd80与NB7缓存f5e43f1整合后，是否可以仅通过合法重新验证/重钉timing policy，在不改既定预算、不削弱cache可选回退协议的前提下推进下一次真实service验收？若NO，指出必须先修的最小具体缺口及证据。本任务是候选整合前的有效性审查，不是重开驱动诊断。"
- "不能靠恢复命中掩盖退化的miss路径"；"不可放宽7200/工具/性能腿预算、数学/容差/任务/reference或3%哨兵/8次中位规则"；"保持benchmark.py字节或只替换sha不是合法证明；不关闭指纹检查"；"不要凭摘要里的'8样本'就擅自改采样数"；"不能建议删npu-strict以保命中、改预算或伪造指纹"。
- 人类最新裁决："其实就是 硬件 + 软件是一样就可以复用"；"同NPU型号而非卡号、运行依赖/reference计算/计时协议；同workload/input是条目身份。工作目录、卡号、无测量影响的整文件修复应只作出处，不应仅因hash变化一律禁用缓存。保留原当前卡>3%哨兵，不改数学/预算，不盲目忽略真正依赖"；"区分当前代码的机械pin失败与实际测量不等价，并指出最小的判据/绑定收敛位置；不要设计新通用框架/做全量1046重测，不把当前policy机械规则误当人类不可更改验收"；"不碰迁移owner2c16551d的配置"。
- 证据锚点：NB3 报告 `/root/.paseo/worktrees/0u9809pj/nb3-prod-deploy-input-fix-20261006/.scratch/nb3-official-npu-repair-20261006/REPORT.md`；NB7 报告/UNAPPLIED-PACKAGE `/root/.paseo/worktrees/0u9809pj/nb7-model-bound-cache-20261006/.scratch/nb7-model-bound-cache-20261006/`；af31c25 报告 `/root/.paseo/worktrees/0u9809pj/ops-service-pr-af31c25-20261006/.scratch/ops-service-pr-af31c25-20261006/REPORT.md`；归档索引 `/home/e84381970/open-evo-work/evidence-archive/paseo-scratch-20261006/INDEX.md`。
- EID：NB3 canary A `ev_d6a2874641cb4338822d860c31bf04a7`（card5，51/51 fresh 正确、输入hash与sealed 51/51吻合、181.5s）；canary B `ev_f0a12f1e33504d85ae2d02ea0859ec7f`（correctness 51/51、performance=timeout、736s）；NB7 staging `ev_ce6024f97126454098f42317a323923b`（card14，51/51，1 cache+50 fresh）；af31c25 probe `ev_27cf61f3a6774ddf8909c5755671bbed`（card2，51/51 correctness、51/51 cross-card timing_reuse hits、sentinel +0.05%）。分支：NB3 tip c03fd80（cc0b22c→1d45e74→d175a65→c03fd80），NB7 tip f5e43f1b777a747636c02939ffb6e65ea7d90223（903d27f→f5e43f1），均自 7b008d25；af31c25 是 7b008d25 的祖先（中间仅 10ef9c1/5cce7ef/7b008d2 三个输入物化重做工提交）。

## Evidence

### 1. 机械 pin 失败 vs 实际测量依赖（人类裁决的判据落位）

| 绑定面 | 合并候选状态 | 定性 | 证据位置 |
|---|---|---|---|
| NPU 型号（硬件） | Ascend910_9362 门不变，全队同型 | **真实依赖，保留** | v2 index `deployment.npu_model`；policy v6 `npu_model` 字段 |
| 运行时/工具链指纹（软件运行依赖） | CANN 9.1.1、patched triton driver.py 3942188b…/npu_utils.cpp 6bd908f6…、profiler runtime——两分支均未触碰 | **真实依赖，等价** | policy v6 `target_patched_runtime_files`；`CacheDeployment.check_runtime`（reference_cache.py:117，含 runtime_files 校验） |
| reference 计算身份 | reference_source_sha256 / official_digest / adapter_version（任务 torch 文件），两分支未触碰 | **真实依赖，等价** | `ModelBoundBinding`（reference_cache.py:204-247） |
| 计时协议 | profiler / warmup5 / repeats20 / 8 样本 / 中位 / cv / clear_l2 全不变；每评 `check_measurement` 实测核对（reference_cache.py:842） | **真实依赖，等价** | benchmark.py:57-79 默认值未被任何分支 diff 触及；policy 加载时 contract 与 sealed deployment measurement 强校验（reference_timing_reuse.py:289-297） |
| workload/input（条目身份） | d175a65 恢复 sealed 输入契约：canary A card5 实测逐 case 输入 hash 与 sealed 51/51 吻合（有因果作用但**收敛**——不是不等价，是修复回归） | **真实依赖，已恢复等价** | NB3 报告 canary A；NB7 报告"on that baseline this cache serves 51/51" |
| algorithm_files pin（verify.py/benchmark.py/_common_utils.py） | 合并树字节变化 → v2 index `check_runtime` applicable=False → 全 miss；policy target pin 不匹配 → 不授权 | **机械 pin 失败**（改动为测量中性守护/trace，或输入恢复修复；human 裁决：只作出处） | 见 §2 变更逐项 |
| 工作目录/卡号 | v2 key 已无卡号；index/policy 路径是部署出处 | **非测量依赖** | `CacheIndexV2` 注释（reference_cache.py:265-275） |

**结论行**：合并候选与封存代之间不存在已识别的测量不等价；被禁用的是出处指纹，不是复用前提。

### 2. 合并树变更逐项（对计时/输入的因果作用）

| commit | 文件 | 测量因果 | 定性 |
|---|---|---|---|
| cc0b22c/1d45e74 | verify.py / _common_utils.py | 预执行设备初始化有界等待（≤240s，受整次预算 min 封顶），不进任何 kernel/reference 计时 | 测量中性 |
| d175a65 | verify.py（+tests） | 恢复 NPU RNG snapshot（"npu:0" 设备类型）→ 输入回到 sealed 契约 | 因果但**收敛**（51/51 吻合） |
| c03fd80 | benchmark.py:1170-1190 | npu-strict 下 `_ensure_npu_ambient_shared` 失败→有界等待→仍失败 RuntimeError 大声失败；不触碰 WARMUP/REPEATS/采样/中位 | 测量中性（纯失败门） |
| 903d27f | verify.py(+3)/reference_cache 等 | correctness 每 case `reference_source` trace；v2 解析/回退 | 测量中性 |
| f5e43f1 | evaluators/sandbox.py | overlay 接受 v1 **或** v2 pinned index（同一 digest pin 校验） | 测量中性 |

合并树（tmp/mrg 实测）pin 面 hash：verify.py `f1f5372b6c79ca05…`、benchmark.py `d65b54d46647f41b…`、_common_utils.py `c705f9aa53f588cd…`；sandbox.py `6d473280…`、reference_cache.py `efccb418…`、reference_timing_reuse.py `c5930ec2…`、reference_diagnostics.py `8fe1181d…` 与 v6 的四个 target pin **逐字节相同**——policy v7 相对 v6 仅 benchmark.py 一个字段变化（683cc5fb→d65b54d4）。

### 3. 最小判据/绑定收敛位置（不设计新框架）

1. **正确性+计时缓存索引**：`build_model_bound_index()`（reference_cache.py:866-880+，NB7 已用其产出 v2 f3365d0b 的同一工具）在合并 checkout 上重导出 v2'：`algorithm_files` 重钉合并树字节（出处），`ModelBoundBinding.key` 不含引擎文件 hash（key=npu_model+sealed runtime fingerprint+reference 身份+input hash+case+policy），sealed 1046 manifests 与 source index 4c165dfc / deployment 84154dfb 逐字节不动。
2. **timing policy**：按 v5→v6 完全相同的先例（NB7 UNAPPLIED-PACKAGE.md 记录该先例的成因与断言"pins asserted equal to branch bytes"）重导出 v7：仅 `target_service_source_sha256["engine/benchmark.py"]` 更新；`npu_model`/`source_devices`/`cross_card_scope`/`timing_contract`（profiler/w5/r20/8）/source index+deployment digest 不动；当前卡 fresh 8 样本哨兵 + >3% 单调 marker（`needs_remeasurement` reference_cache.py:812-840）不动。
3. 合法性链（非"只换 sha"）：policy 加载即强校验 contract==source deployment measurement（reference_timing_reuse.py:289-297）；`check_measurement` 每次评估实测核对协议；哨兵每评在当前卡实测；完整验收再走真实 grade/perf/sentinel。指纹检查全程开启，仅授权目标换为新字节，且新字节 diff 证明测量中性（§2）。
4. 若不做 v2'（严格"仅 policy"读法）：correctness 走 fresh 51/51（canary A 181.5s 在 1500s verify 帽内）+ timing 走 policy v7 hits，同样可验收——两种读法均 YES；推荐 v2'+v7 完整配置（NB7 包设计）。

### 4. 三类 fallback 与原路径对照（合并树现行代码，benchmark.py:958-1160）

| 情形 | 路径 | 成本 |
|---|---|---|
| cache+policy 均未配置 | `from_environment`→None→原 e4a36f7 路径（benchmark.py:1082-1090 注释明引 e4a36f7） | 参考每 case 1×(w5/r20 profiler)（`_measure_framework` benchmark.py:563-615） |
| 未配置 cache，policy v7 匹配 | `_try_validated_timing_reuse`→hit：首 case 哨兵 8×，≤3% 服务 sealed 中位；miss→原路径 1×51 | af31c25 probe card2 实测 51/51 hits、sentinel +0.05%；准入 bmm gate "perf ~445s original path" 在 600s 腿内 |
| cache(v2') 配置且 check_runtime 失败（pin 不符） | applicable=False→**每 lookup miss→fresh，不 raise**（reference_cache.py:520-560 v2 dispatch） | 正确性 fresh；timing 落回 policy 路径 |
| cache(v2') 配置、per-case 身份 miss（型号/输入/条目缺） | `ReferenceCacheMiss`→先试 policy→仍 miss→`reference_measure_eight=True`→8×/case fresh remeasure | **8×51 为缓存逻辑新增成本**（非 e4a36f7 原成本），BMM51 撑爆 600s 腿→timeout=fail-loud（canary B 机制）；正确性结果仍正确 |
| pinned 篡改/索引身份矛盾（cache 或 policy） | `ReferenceCacheError`/`TimingReuseError`→大声 infra 失败，绝不静默 | fail-closed |
| 哨兵 drift>3% | 单调 marker `card-<id>.invalid.json` + 全量 fresh remeasure | 现行规则，不动 |
| npu-strict 设备初始化失败（c03fd80 新增） | 有界等待≤240s（仅预执行探测）→仍失败 RuntimeError，绝不退 CPU | 大声失败 |

vs af31c25（optional-cache-fallback，生产部署过、probe 如上）：未配置→原路径的语义即由此引入，合并树保留全部语义且 v2 化了正确性查找。vs e4a36f7 原计时协议：参考每 case 单次 w5/r20 profiler；8 样本/中位仅存在于封存代、remeasure 与哨兵路径——采样数无人改动，v7 亦不改动。

### 5. 预算数字与 EID 佐证
- 腿预算来源：`TRITON_EVAL_BENCHMARK_TIMEOUT_SECONDS=600`、`TRITON_EVAL_VERIFY_BUDGET_CAP_SECONDS=1500`（baseline api.env 与 NB7 staging-env.sh 同值）、facade /verify 900s、总 7200（用户规则）。canary B：correctness 51/51 fresh 成功后 performance 在 736s envelope timeout——与"v1 索引在 card5 无 BMM51 条目→per-card miss→policy v5 指纹对 c03fd80 benchmark 字节 decline→8×51 fresh>600s"机制一致；每阶段耗时未留存，精确 split **unknown**（不编造根因；驱动等待≤240s 未触发有 NB3 记录）。
- 复用命中后的腿成本先例：af31c25 probe card2（policy 路径 51/51 hits + 哨兵）与准入 bmm gate ~445s——均在 600s 内。

### 6. 下一次整合+验收最小可执行顺序（不搭新 runner、不全量重测）
1. 新分支自 7b008d25 合入 c03fd80+f5e43f1（唯一冲突 tests/test_reference_cache.py:415，解 `lambda _t, **_k: True`）；跑现有套件（NB3 31/9/18 + NB7 16 及受影响 suite；--basetemp 在自有 scratch 并清理）。
2. 合并 checkout 上重导出：index v2'（build_model_bound.py 配方）、policy v7（v5→v6 先例；evidence 注记修复出处与不变契约）、BUILD/manifest/overlay（build_deploy_package.py + NB3 deploy 机制）。
3. staging **容器工具→native service /verify 真实请求**（补 NB3 只 host POST 的缺口）：fresh 正确性 51/51、逐 case 输入 hash==sealed、per-case reference_source。
4. 完整官方 evaluation：grade+perf+sentinel，留存 per-case reference_source/reference_path、哨兵 drift、8 样本中位、cache 来源（cached_manifest/historical_validated/fresh）；perf 腿须 <600s。
5. 准入 gates+fleet，EID 全记录；任何哨兵>3% 或 NPU-init 复现按大声失败处理并停（不弱化、不加新门槛）。
6. 全程不改 7200/600/1500/900、数学/容差/reference、3%/8 样本；不碰 owner 32dffe59 与迁移 owner 2c16551d 的任何文件/配置；不改生产/staging/main。

### 7. 边界与开放项
- v2' 索引 timing-hit（cached_manifest 状态）路径尚未在 NPU 上跑过（NB7 staging 是 verify-only）；首跑即上述第 4 步。policy 路径 hit 有 card2 全量先例。
- canary B 归因中"每阶段精确耗时"unknown；机制链（decline→8×51→600s 腿超时）由代码+600s 配置+一次观测支撑，不升级为根因证明。
- NPU-init 瞬态根因未证明（NB3 同结论）；c03fd80 仅缓解+大声失败。
- 本审查为只读：源 worktree/生产/staging/sealed 数据零修改；合并 dry-run 在自有 scratch `tmp/mrg`（16MB 总量，留作证据）。
