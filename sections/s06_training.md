## 6. 训练系统：MoE 的稳定性、精度与流水线

### 6.1 稳定性：从 aux loss 到 aux-loss-free

MoE 训练的第一大问题是**路由坍缩**：模型把 token 集中到少数专家，其余专家收不到梯度、逐渐退化。三代方案：

| 方案 | 机制 | 代表 | 副作用 |
|---|---|---|---|
| aux loss | 惩罚负载不均，梯度回传 | Switch/GShard | 干扰语言建模梯度；batch 依赖 |
| aux + z-loss | 再加 logits 尺度惩罚 | ST-MoE | 超参敏感；z-loss 系数需调 |
| aux-loss-free | 偏置项按负载更新，不进梯度 | DeepSeek-V3 | 偏置漂移需监控 |

DeepSeek-V3 的 aux-loss-free 是 2024-2025 年最重要的训练技巧之一。它的更新规则极其简单：

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\text{load}_i - \bar{\text{load}})$$

但工程上有三个细节：

1. **偏置只加在 top-k 选择上，不加在门控权重上**。也就是说，$b_i$ 改变"谁被选中"，但不改变"选中后权重多大"。这样语言建模的梯度完全不受均衡目标干扰。
2. **$\gamma$ 要按层调**。浅层路由变化快，深层路由稳定；统一 $\gamma$ 会让浅层振荡、深层不收敛。
3. **序列级均衡损失作为补充**。V3 后来加了 sequence-wise balance loss（只在序列维度统计），防止长序列内部的局部坍缩。V4 沿用并加强了这一点。

2026 年的后续研究（如 ReLibra、Router Drift Control）指出：**RL 阶段的路由漂移是新的不稳定源**。RL 会改变数据分布，导致训练时的专家负载分布与推理时不一致；解决方法是在 RL 时"重放"路由（routing replay），或者用轻量的负载预测器做在线校正。

### 6.2 FP8 训练：DeepSeek-V3 的配方

V3 是第一个在超大规模（671B）上验证 FP8 训练可行性的模型。它的配方可以概括为"**大部分算 FP8，关键处保高精度**"：

| 组件 | 精度 | 原因 |
|---|---|---|
| GEMM（Fprop/Dgrad/Wgrad） | FP8（E4M3） | 算力翻倍；Wgrad 的激活也用 FP8 存储，省显存 |
| Embedding / Output Head | BF16 | 词表大，量化误差会累积 |
| MoE Gating | FP32 | 路由对精度极敏感，FP8 会让 top-k 排序错乱 |
| Normalization | BF16/FP32 | 数值范围大 |
| Attention | BF16 | softmax 对精度敏感 |
| 残差/累加 | FP32 | 防止长链累加误差 |

**Fine-grained quantization** 是 V3 的核心技巧：

- 激活按 **1×128 tile** 分组，每组一个缩放因子；
- 权重按 **128×128 block** 分组，每组一个缩放因子；
- 这样比"整个张量一个 scale"精度高得多，又比"per-element scale"开销小。

V3 报告称 FP8 训练与 BF16 训练的 loss 曲线"几乎重合"，且训练速度提升约 1.5-1.8×（取决于 GEMM 占比）。

### 6.3 DualPipe：把 all-to-all 藏进计算

MoE 训练的通信量极大：跨节点 EP 的 all-to-all 让"计算:通信"降到约 **1:1**。V3 的 DualPipe 做了三件事：

1. **双向流水线**：从流水线两端同时注入 micro-batch，让前向和反向在时间上重叠；
2. **组件重排**：把一个 chunk 拆成 attention、dispatch、MLP、combine 四段，手动调整各段占用的 SM 比例，让 all-to-all 与计算并行；
3. **PP 通信与 EP 通信融合**：把流水线的 send/recv 和 all-to-all 的 dispatch/combine 放在同一个通信窗口，减少等待。

结果是 **all-to-all 和 PP 通信都被完全隐藏**，流水线气泡也减少。这套设计后来被 DeepEP、UniEP 等系统继承。

### 6.4 Muon 与 MuonClip：优化器层面的稳定性

Kimi K2 用 **MuonClip** 解决了一个具体问题：Muon 优化器的 token 效率高，但 attention 的 QK logits 容易爆炸（因为 Muon 对矩阵参数的更新幅度大）。MuonClip 的做法是在每次更新后对 Q/K 矩阵做 **QK-clip**：如果 $\|QK^\top\|$ 超过阈值，就按比例缩小。K2 用这套配方训练 15.5T token，**零 loss spike**。

DeepSeek-V4 和 Qwen3.8-Next 也转向了 Muon。Qwen3.8 的报告说 Muon 让"最优学习率和 batch size 上移、batch-size warmup 不再必要、稳定性显著提升"。**Muon 正在取代 AdamW 成为大 MoE 训练的默认优化器**，这是 2025-2026 年一个容易被忽略但很重要的趋势。

### 6.5 Upcycling：把稠密模型"长成"MoE

Upcycling 的思路是：拿一个训练好的稠密模型，把它的 FFN 复制成多个专家，再继续训练。好处是省掉从零预训练的成本；坏处是**专家初始化完全相同**，需要额外的扰动/路由训练来分化。

2025 年的 scaling law 研究（arXiv:2502.03009）给出一个反直觉结论：upcycling 的收益不是单调的——**稠密模型和 upcycled MoE 之间存在一个"交互项"**：稠密模型越大，upcycling 的边际收益越小；数据量越大，upcycling 越划算。工程上的经验法则是：**upcycling 适合"中等稠密模型 + 大量新数据"的场景**，不适合"超大稠密模型 + 少量新数据"。

2026 年出现了多种 upcycling 变体：Expert Upcycling（按专家重要性选择性复制）、DivMoE（跨域专家组合）、STAPLE（静态稀疏 upcycling + embedding 模块）、Router Upcycling（复制 router 而非专家）。核心问题都是**如何让复制出来的专家尽快分化**。

### 6.6 RL 阶段：MoE 的新战场

2025 年之后，MoE 的训练重心从"预训练"转向"后训练 + RL"。三个代表性系统：

- **DeepSeek-R1**：GRPO（Group Relative Policy Optimization），用规则奖励做可验证任务的 RL。R1 证明了"纯 RL 可以激发推理能力"，但 R1 的 MoE 主干在 RL 中会出现路由漂移。
- **GLM-5**：异步 RL 基础设施，把"生成"和"训练"解耦到不同进程/设备，大幅提升 RL 吞吐；同时提出异步 agent RL 算法，支持长程交互。
- **MiniMax-M2**：Forge 系统，agent-native RL，windowed-FIFO 调度 + prefix-tree 合并 + 推理优化，支持白盒/黑盒 agent。

**MoE + RL 的三个特殊问题**：

1. **路由不一致**：训练时的路由（top-k 索引）和推理时的路由可能不同，导致 RL 的 importance ratio 失真。解法是 routing replay（把训练时的 top-k 索引存下来，推理时强制复用）。
2. **专家负载漂移**：RL 数据分布与预训练不同，某些专家会突然过载。解法是在线负载监控 + 动态冗余。
3. **长序列**：agentic RL 的轨迹动辄 100K+ token，KV cache 和 MoE 权重争抢 HBM。解法是 PD 分离 + KV 压缩（V4.1 的 FP4 KV 就是为此设计）。

### 6.7 训练成本：一个可复用的估算框架

DeepSeek-V3 的公开数字：**2.788M H800 GPU hours**，按当时 H800 租金约 $2/GPU-hour 估算，总成本约 **$5.58M**。这个数字常被引用为"MoE 训练便宜"的证据，但需要注意口径：

- 它只算**预训练**，不含 SFT/RL/数据工程/失败重跑；
- 它假设**极高的 MFU**（V3 报告称约 40%），普通团队很难达到；
- 它依赖**极致的工程优化**（DualPipe、FP8、细粒度量化、通信重叠），这些优化本身就是数月的人力成本。

一个更现实的估算：**同等规模 MoE 的训练总成本 = GPU 租金 × (预训练小时 × 1.5~2.0) + 数据成本 + 人力成本**。1.5~2.0 的系数覆盖 SFT/RL/重跑/调试。
