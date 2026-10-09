# 官方 BatchMatmul NPU 验证 fixture

`definition.json` 和 `workload.jsonl` 原样取自 Ascend/ascendc-kernelgen-data 的固定提交；上游拥有任务、Torch reference 和 51 组 workload 定义。保留上游 Apache-2.0 `LICENSE`，两份官方文件的组合 digest 与归档 `official_digest` 完全一致，证明文件字节一致；来源链接及字节 hash 见 `manifest.json`。`candidate.py` 原样提取自 open-evo 退休归档 dsh 请求的 `kernel_code`，保留 `ModelNew` 接口，仅作为待验证候选。

case index 为零基：index 0 对应 `BatchMatmul-001`，index 50 对应 `BatchMatmul-051`，顺序以 workload 行顺序为准。每组输入顺序为 A、B；dtype 分布为 float32 18 组、float16 18 组、bfloat16 15 组。

官方固定版本生成器按 workload 重置指定 seed，在目标设备生成共享 normal/uniform 分布族的 A/B；normal 的均值与标准差另行随机采样，bfloat16 normal 先生成 float32 再转换。历史归档 adapter 的 `torch.randn` 未指定 device，使用环境中的 Torch 默认设备与全局 RNG；历史审计 seed 为 0，按顺序消耗前序 case 的随机数。NB0 绑定为 `npu-strict`，记录 selected_reference_device 为 `npu:0`，不能把未指定 device 解读成必然使用 CPU。两者不等价。官方 CLI 默认 seed 为 200，历史参考审计 seed 为 0。fixture 保留官方 specs，不携带原始 tensor 数据；验证结果须注明采用哪种生成策略、seed 与设备，不能复用历史输入 hash 或速度结论。

按 shape × dtype 字节数估算，51 组中 A+B 输入合计最小 48 bytes，最大 2147483648 bytes；这是每组输入静态体积，**不是峰值显存**，未含输出、累加、候选临时量和分配器开销。测试应按 case 分别物化并释放输入。

此目录是验证材料，不代表当前服务已经在 NPU 通过。不得通过导入历史请求或执行下载脚本来恢复凭据或环境。
