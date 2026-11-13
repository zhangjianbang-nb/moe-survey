## 4. 架构演进：八个里程碑与"瓶颈迁移"

MoE 的历史不是"模型一代比一代大"的线性叙事，而是一部**瓶颈迁移史**：每一代架构解决一个核心矛盾，同时把瓶颈推到下一个位置。2026 年 8 月的一篇技术综述（arXiv:2608.08650）把这个过程整理成八个节点，每个节点标注"新设计变量 / 解决什么问题 / 把瓶颈推到哪里"。下表在此基础上补充了代表模型与关键数据。

| # | 里程碑 | 新设计变量 | 解决的核心问题 | 转移出的新瓶颈 | 代表 |
|---|---|---|---|---|---|
| 1 | 统计分工 | gating + 局部专家 | 单一模型难以划分输入空间 | 所有专家都参与计算，容量绑定 FLOPs | Jacobs 1991, Jordan & Jacobs 1994 |
| 2 | 稀疏条件计算 | top-k + capacity | 总参数与激活 FLOPs 解耦 | 离散路由、坍缩、token drop | Shazeer 2017, GShard 2020, Switch 2021 |
| 3 | 集群级训练 | EP + all-to-all + 稳定损失 | Transformer MoE 能上集群 | 通信、straggler、数值/负载稳定性 | GShard 600B / Switch 1.6T |
| 4 | 开源产品化 | decoder-only 闭环 | 经典 MoE 进入开放生态 | 大专家知识冗余、组合粒度粗 | Mixtral 8×7B (2023) |
| 5 | 知识切分 | fine-grained + shared + device-limit | 公共知识重复、专家组合扇出不足 | 小 GEMM、共享路径固定成本 | DeepSeekMoE (2024), V2 (2024) |
| 6 | 超稀疏扩容 | 更大 N、tiny experts、检索路由 | 固定激活预算下继续扩容量 | 加权 I/O、检索、低 token/专家 | DeepSeek-V3 (2024), Kimi K2 (2025) |
| 7 | 动态计算 | adaptive-k / zero-compute experts | 不同难度 token 用同一预算 | 控制开销、尾延迟、训练稳定性 | LongCat-Flash (2025), Stepped MoE (2026) |
| 8 | 语义-物理解耦 | 异构/跨层拓扑、结构化通信 | 语义路由被设备拓扑与通信绑死 | 调度复杂度、缓存一致性、大规模验证 | DeepSeek-V4 (2026), Kimi K3 (2026) |

> 来源：arXiv:2608.08650 的八节点表（Table 3）经本文核对与补充；代表模型与参数为各官方技术报告数据。

### 4.1 节点 1-2：从"统计分工"到"稀疏激活"

1991 年 Jacobs 等人的 "Adaptive Mixtures of Local Experts" 提出用门控网络把输入分配给不同子网络。这时所有专家都参与计算，MoE 只是**统计意义上的分工**（不同子网络拟合输入空间的不同区域），不是计算意义上的稀疏。

2017 年 Shazeer 等人的 Sparsely-Gated MoE 引入 top-k 稀疏门控与容量因子，第一次把"总参数"和"每 token 计算量"解耦。GShard（2020）把它推到 600B 参数、2048 个 TPU v3、4 天训练 100 语言翻译；Switch Transformer（2021）把 top-k 简化为 top-1，用 2048 专家做到 1.6T 参数，并首次证明 MoE 可以用 bfloat16 训练。

**这一阶段的核心工程遗产**是三个至今仍在用的机制：容量因子（capacity factor）、负载均衡辅助损失（aux loss）、专家并行（EP）。它们同时也是后续所有麻烦的源头。

### 4.2 节点 3-4：集群训练与开源产品化

GShard 的关键贡献不只是模型，而是**编译器层面的自动分片**：把 MoE 层表达成"可分片的计算图"，让 XLA 自动处理跨设备的 all-to-all。Switch 则证明 MoE 可以在 bfloat16 下稳定训练，并给出"选择性精度"（router 用 fp32、专家用 bf16）的配方。

2023 年 12 月 Mixtral 8×7B 是一个转折点：它用 8 个专家、top-2、47B 总参数/13B 激活，在开源社区证明了 decoder-only MoE 的产品化可行性。它的架构极其简单（就是 Mistral 7B 的 FFN 换成 8 个专家），但工程上第一次把"MoE 推理"变成普通开发者能跑的东西。

### 4.3 节点 5：DeepSeekMoE 的"知识切分"

DeepSeekMoE（2024-01）提出两个至今被广泛模仿的设计：

1. **细粒度专家（fine-grained expert segmentation）**：把每个专家切小，专家数从 8/16 增加到 64/160/256。这样同样激活参数下，token 可以由更多"小专家"组合，组合空间指数级增长；
2. **共享专家隔离（shared expert isolation）**：保留 1-2 个"总是激活"的共享专家，承载通用知识，让路由专家专注于细分模式。

DeepSeek-V2（2024-05）把这两点与 **MLA（Multi-head Latent Attention）** 结合：236B 总参数/21B 激活、128K 上下文、8.1T token。相比 DeepSeek 67B，训练成本降 42.5%、KV cache 减 93.3%、最大生成吞吐提升 5.76×。**MLA + MoE 的组合是 DeepSeek 系列成本优势的根基**：MLA 把 KV cache 压成 latent 向量，MoE 把 FFN 计算稀疏化，两者分别解决"长上下文显存"和"大模型算力"两个不同问题。

V2 还引入了 **device-limited routing**：每个 token 的候选专家限制在少数设备上，控制跨设备 fan-out。这是"语义路由受物理拓扑约束"的第一个显式设计，为节点 8 埋下伏笔。

### 4.4 节点 6：超稀疏扩容（V3 / K2）

DeepSeek-V3（2024-12）是这一阶段的集大成者：

- 671B 总参数、37B 激活（稀疏度约 18×）；
- 61 层、hidden 7168、MLA（128 heads × 128 dim，KV 压缩 512，query 压缩 1536）；
- 1 个共享专家 + 256 个路由专家，专家中间维 2048，每 token 激活 8 个路由专家；
- 每 token 最多发到 4 个节点（node-limited routing）；
- aux-loss-free 负载均衡 + MTP（多 token 预测，深度 1）；
- FP8 混合精度训练（fine-grained quantization）；
- 14.8T token，2.788M H800 GPU hours，全程无不可恢复 loss spike。

Kimi K2（2025-07）把稀疏度推到另一个量级：1.04T 总参数、32.6B 激活、384 个路由专家、top-8。它的关键创新是 **MuonClip**：在 Muon 优化器上加了 QK-clip，解决 attention logit 爆炸导致的训练不稳定，15.5T token 训练零 loss spike。

**这一阶段的共同特征**：专家数从几十增加到几百，专家宽度从 4096 降到 2048/1536，激活专家数稳定在 6-8。换句话说，**稀疏度提升靠"增加专家总数"，而不是"增加每 token 激活数"**。这带来两个直接后果：单专家 GEMM 变小（效率下降）、专家负载更不均（通信和调度更难）。

### 4.5 节点 7：动态计算（LongCat-Flash / Stepped MoE）

LongCat-Flash（2025-09，美团）提出 **Zero-computation Experts**：一部分专家是"零计算"的（直接输出零或恒等），路由可以根据上下文动态决定激活多少专家。560B 总参数，每 token 激活 18.6B-31.3B（平均 27B），而不是固定值。

它的另一个设计是 **Shortcut-connected MoE**：把 MoE 层的输入直接连到输出（类似残差），扩大计算-通信重叠窗口。

动态计算的核心问题是**难度估计**：模型怎么知道这个 token 需要几个专家？LongCat 用路由分数的分布（top-k 与 top-(k+1) 的差距）作为代理。2026 年的 Stepped MoE、ConceptMoE 等工作进一步把"计算预算"变成可配置的超参。

### 4.6 节点 8：语义-物理解耦（V4 / K3）

2026 年的两个旗舰模型把 MoE 推到了新阶段：

**DeepSeek-V4**（2026-04）：

- V4-Pro：1.6T 总参数、49B 激活、61 层、hidden 7168、1 shared + 384 routed、top-6、专家中间维 3072；
- V4-Flash：284B 总参数、13B 激活、43 层、hidden 4096、1 shared + 256 routed、top-6、专家中间维 2048；
- 1M 上下文；CSA（Compressed Sparse Attention）+ HCA（Heavily Compressed Attention）混合注意力；
- mHC（Manifold-Constrained Hyper-Connections）改造残差；
- Muon 优化器；
- 门控从 Sigmoid 改为 **Sqrt(Softplus)**；
- 前 3 层 MoE 用 **Hash routing**（固定哈希，无路由参数）；
- 32T token。

**Kimi K3**（2026-07）：

- 2.8T 总参数、104B 激活、896 个路由专家、每 token 激活 16 个；
- KDA（Kimi Delta Attention）+ Attention Residuals；
- Stable LatentMoE；
- 1M 上下文、原生视觉；
- 相比 K2 的 scaling efficiency 提升约 2.5×。

**这一阶段的关键词是"解耦"**：

- **语义路由与物理执行解耦**：Hash routing、可学习的拓扑、跨层专家共享，让"哪个专家处理这个 token"不再被"专家在哪张卡上"绑死；
- **注意力与 FFN 解耦**：CSA/HCA 把长上下文注意力压到极低复杂度，FFN 的 MoE 独立扩展；
- **训练与推理解耦**：V4.1-Flash 的 CED 架构让 prefill 只激活 8B、decode 激活 16B，同一模型在不同阶段用不同计算量。

### 4.7 演进的三条主线

把八个节点压缩，可以看到三条同时推进的主线：

1. **容量线**：总参数从 600B → 1.6T → 2.8T，稀疏度从 4× → 18× → 27×；
2. **效率线**：每 token 激活参数从 97B（GLaM）→ 37B（V3）→ 13B（V4-Flash）→ 6B（Qwen3.8-Next），单位算力的能力持续提升；
3. **系统线**：从"单机 EP"到"跨节点 EP"到"分离式部署"到"语义-物理解耦"，每一步都是被通信和负载不均逼出来的。

**一个反直觉的观察**：2026 年的旗舰模型（V4-Pro 49B、K3 104B）激活参数比 2025 年的 V3（37B）更大，但单位激活参数的效率更高（V4 在 1M 上下文下单 token FLOPs 只有 V3.2 的 27%）。这说明"稀疏度"不再是唯一指标，**"有效计算密度"**（每 FLOP 产生的有效上下文/推理能力）成为新的竞争维度。
