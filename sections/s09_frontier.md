## 9. 前沿方向：2026 年的五个战场

### 9.1 长上下文：注意力与 MoE 解耦

2025 年之后，长上下文不再是"把 RoPE 拉长"，而是"重新设计注意力"。四条路线：

**路线 A：稀疏选择（DSA / QSA）。** DeepSeek Sparse Attention（V3.2、GLM-5）用一个轻量 indexer 为每个 query 选 top-k 的 KV 块，只对选中的块做注意力。Qwen Sparse Attention（Qwen3.8-Next）类似，但用"微块粒度"打分。核心是 **indexer 要足够便宜**——它的计算量必须远小于主注意力。

**路线 B：压缩 + 稀疏（CSA/HCA）。** DeepSeek-V4 的 CSA 先把序列压缩 $m$ 倍（$m=4$），再在压缩后的 KV 上做稀疏选择（top-512/1024）；HCA 用更激进的压缩（$m'=128$）。两者交替使用：CSA 保细节，HCA 保长程。V4 报告称 1M 上下文下 FLOPs 只有 V3.2 的 27%。

**路线 C：线性注意力（KDA / GDN / Lightning）。** Kimi Delta Attention（K3、Kimi Linear）用"对角+低秩"转移矩阵做线性注意力，Kimi Linear 报告"公平对比下超过全注意力"。Gated DeltaNet（Qwen3-Next）用门控 Delta 规则。MiniMax-01 的 Lightning Attention 把线性注意力做到 4M 上下文。

**路线 D：混合（Hybrid）。** Qwen3-Next 用"每 4 层一个全注意力 + 其余 GDN"；Kimi Linear 用"KDA + MLA"层间混合；Jamba/Nemotron-H 用"Mamba + Transformer + MoE"混合。混合的目的是**用少量全注意力层保表达力，用大量线性层省成本**。

**MoE 在其中的角色**：注意力负责"上下文效率"，MoE 负责"参数容量"，两者独立扩展。V4 的 CSA/HCA 与 MoE 是正交的；K3 的 KDA 与 896 专家也是正交的。**"注意力-MoE 解耦"是 2026 年架构设计的默认范式**。

### 9.2 多模态 MoE

多模态 MoE 的核心问题是"**模态间的专家分配**"：

- **DeepSeek-VL2**：动态 tiling 视觉编码 + DeepSeekMoE + MLA，处理任意长宽比的高分辨率图像；
- **Qwen3-Omni**：全模态（文本/图像/音频/视频），MoE 主干；
- **Kimi K3**：原生视觉，896 专家；
- **DeepSeek-V4.1-Flash**：多模态，CED 架构，prefill 8B / decode 16B；
- **VisMMOE**（2026）：视觉 token 剪枝改善 offloading，利用 visual-expert affinity。

**关键发现**：视觉 token 数量远大于文本 token（一张高分辨率图可能几千 token），会**放大专家负载不均**。所以多模态 MoE 需要"模态感知的负载均衡"——V4.1 明确引入了 modality-specific load balancing（图像和文本 token 分开均衡）。

### 9.3 MoE + RL：从"能训"到"训得稳"

2025-2026 年，RL 成为 MoE 后训练的核心。三个系统性问题：

1. **路由一致性**：训练时 top-k 索引 vs 推理时 top-k 索引不一致 → importance ratio 失真。解法：routing replay（存下训练时的索引）。
2. **负载漂移**：RL 数据分布变化 → 专家负载突变。解法：在线监控 + 动态冗余 + ReLibra 的"路由重放引导均衡"。
3. **长轨迹**：agentic RL 的轨迹 100K+ token，KV cache 与 MoE 权重争 HBM。解法：PD 分离 + KV 压缩 + 异步 RL（GLM-5 的 generation/training 解耦）。

**Router Drift Control**（2026）给出一个诊断指标：训练时与推理时的 top-k 索引重合率。低于 80% 就需要干预。

### 9.4 动态计算：让 token 决定预算

固定 top-k 的问题是"简单 token 和困难 token 用同样的计算量"。动态计算让预算随 token 变化：

- **LongCat-Flash**（2025）：zero-computation experts，激活 18.6-31.3B（平均 27B）；
- **Stepped MoE**（2026）：segment-level routing，可配置推理复杂度；
- **ConceptMoE**（2026）：multi-ratio，一个模型多个计算预算；
- **Training-Free Halving**（2026）：推理时把激活专家数减半，无需重训。

**工程难点**：动态分支对 GPU 不友好（不同 token 走不同路径 → warp divergence、负载不均）。所以生产系统通常用"分段"（segment）而不是"逐 token"动态。

### 9.5 权重外置：把"死重"请出显存

MoE 的 99% 权重在单 token 推理中是"死重"。2026 年出现"权重外置"思路：

- **Qwen3.8-Flash-Next**：51B 参数的 n-gram embedding 表放在 host memory，按需 prefetch；
- **DeepSeek-V4.1-Flash**：Engram（稀疏访问的条件记忆），把部分记忆放 SSD/host；
- **KTransformers**：专家放 CPU，attention 放 GPU。

**核心权衡**：外置省显存，但引入 PCIe/SSD 带宽瓶颈。适合"显存受限 + 请求稀疏"的场景，不适合"高并发 + 低延迟"。

### 9.6 理论：专家为什么会专业化

2022 年的理论工作（arXiv:2208.02813）证明：**MoE 的成功依赖两个条件**——(1) 数据有内在的簇结构；(2) 专家是非线性的。Router 学到的是"簇中心特征"，把复杂问题分解成多个线性子问题，每个专家解决一个。

这个理论解释了为什么：
- 专家数不能无限增加（簇的数量有限）；
- 专家必须非线性（线性专家无法拟合簇内结构）；
- 路由坍缩是"簇合并"的表现（数据簇不够分明时，多个专家退化成同一个）。

2025 年的统计学习分析（arXiv:2505.10860）进一步证明：**共享专家提高样本效率**（因为它承担了跨簇的通用成分），**normalized sigmoid 门控比 softmax 更适合细粒度专家**（因为 softmax 的全局竞争会压制小专家）。

### 9.7 一个开放问题：专家到底学到了什么？

2024-2026 年有多篇"专家可解释性"工作，结论并不一致：

- OLMoE 的分析显示路由有"高专业化"（不同专家处理不同领域）；
- 但也有工作发现"专家冗余"（多个专家输出高度相似）；
- DeepSeek-V3 的报告说"专家确实分化，但分化程度低于预期"。

**目前的共识**：专家专业化是"统计意义上的"，不是"语义意义上的"。一个专家可能同时处理"Python 代码"和"法律条文"，因为它们在表示空间里接近。**不要指望 MoE 的专家像"部门"一样分工**。
