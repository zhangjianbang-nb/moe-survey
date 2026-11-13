## 2. MoE 的数学基础：稀疏激活到底省了什么

### 2.1 从稠密 scaling 到条件计算

稠密 Transformer 的成本结构可以用两个数字概括：

- **训练算力**：$C \approx 6ND$（N 为参数量，D 为 token 数）；
- **推理算力**：每 token 前向约 $2N$ FLOPs，且**每个参数都必须被读到**。

这意味着稠密模型扩大参数时，训练和推理成本同步线性上升。MoE 的核心主张是**把"参数容量"和"每 token 计算量"解耦**：把 Transformer 的 FFN 层复制成 $E$ 个专家，每个 token 只路由到其中 $k$ 个（$k \ll E$）。于是：

| 量 | 稠密模型 | MoE |
|---|---|---|
| 参数总量 | $N$ | $\approx E \cdot N_{\text{expert}} + N_{\text{shared}}$ |
| 每 token 激活参数 | $N$ | $\approx k \cdot N_{\text{expert}} + N_{\text{shared}}$ |
| 每 token FLOPs | $2N$ | $\approx 2(k \cdot N_{\text{expert}} + N_{\text{shared}})$ |
| 显存占用 | $\propto N$ | $\propto$ 总参数（全部专家都要驻留） |

**关键结论：MoE 省的是"计算"，不是"显存"。** 这是理解 MoE 推理所有工程难题的起点——权重必须全量驻留 HBM，但每个 token 只用到其中一小部分。以 DeepSeek-V3 为例：671B 总参数、37B 激活参数，稀疏度约 18×；以 BF16 存储需约 1.3 TB 显存，而单 token 只读约 74 GB 权重。

### 2.2 一层 MoE 的前向：公式与实现

设输入 $x \in \mathbb{R}^d$，门控矩阵 $W_g \in \mathbb{R}^{d \times E}$。

**(1) 门控打分**

$$s = x W_g \in \mathbb{R}^{E}$$

**(2) Top-k 选择**

$$\mathcal{T} = \operatorname{TopK}(s, k)$$

**(3) 门控权重（softmax 版，Switch/GShard 路线）**

$$g_i = \begin{cases} \dfrac{\exp(s_i)}{\sum_{j \in \mathcal{T}} \exp(s_j)} & i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases}$$

**(4) 加权求和**

$$y = \sum_{i \in \mathcal{T}} g_i \cdot \operatorname{FFN}_i(x)$$

DeepSeek 系列从 V2 起改用 **sigmoid 门控 + top-k 内归一化**：

$$g_i = \frac{\sigma(s_i)}{\sum_{j \in \mathcal{T}} \sigma(s_j)}, \quad i \in \mathcal{T}$$

差异不只是换了个激活函数。softmax 让所有专家在同一个分母下**互相竞争**（一个专家分数高会压低其他所有专家），sigmoid 让每个专家**独立判断**"我是否适合这个 token"，再由 top-k 归一化保证权重和为 1。DeepSeek 的报告明确说这样"缓解了专家之间的竞争"，也让门控对专家数的扩展更稳定。2025 年的一篇分析（arXiv:2505.10860）进一步给出了统计解释：sigmoid 门控的梯度不会被其他专家的 logits 稀释，共享专家则承担了"通用知识"的容量，使路由专家更专注于细分模式。

### 2.3 容量因子与 token dropping

训练时每个专家的容量被固定为：

$$\text{capacity} = \text{CF} \times \frac{T_{\text{batch}} \cdot k}{E}$$

- **CF > 1**：每个专家留余量，超出部分 padding，浪费算力；
- **CF < 1**：超出容量的 token 被**丢弃**（不经过该专家，残差直通），信息损失；
- **dropless**：不设上限，按实际负载动态分配（MegaBlocks 的块稀疏路线）。

Switch Transformer 训练用 CF=1.0~1.25，推理时 dropless；GShard 用 CF=1.25。**token dropping 是 MoE 训练不稳定和评测波动的常见来源**：同一句话在不同 batch 组成下可能被路由到不同专家、甚至被丢弃，导致输出不可复现。2025 年的 "Demons in the Detail"（arXiv:2501.11873）专门分析了负载均衡损失的实现细节，指出不同框架对 $f_i$ 的计数口径（是否含 padding、是否按 top-k 重复计数）不同，会让"同样超参"训出完全不同的专家分布。

### 2.4 负载均衡损失：为什么需要，以及它的代价

路由是**自利**的：每个 token 独立选专家，模型会倾向于把 token 集中到少数"好用"的专家，形成**负载坍缩（load collapse）**。后果是过载专家丢 token、闲置专家浪费参数。

Switch Transformer 的辅助损失：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$

其中：

$$f_i = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\left[i \in \operatorname{TopK}(s_t, k)\right], \quad P_i = \frac{1}{T}\sum_{t=1}^{T} g_i(x_t)$$

- $f_i$ 是**离散**的"分到专家 i 的 token 比例"（不可微，当作常数）；
- $P_i$ 是**连续**的"路由概率均值"（可微，梯度从这里回传）；
- 乘 $E$ 是为了让均匀分布时 $\mathcal{L}_{\text{aux}} = \alpha$，与专家数无关。

这个设计有一个根本矛盾：**负载均衡是"batch 级"目标，语言建模是"token 级"目标**。$\alpha$ 调大，专家分布均匀但语言建模变差；$\alpha$ 调小，专家坍缩。而且 $f_i$ 依赖当前 batch 的组成，batch 大小一变，最优 $\alpha$ 就变。

### 2.5 z-loss：稳定 router 的第二个旋钮

Router logits 过大会让 softmax 进入饱和区，梯度接近 0 或爆炸。ST-MoE 引入 **router z-loss**：

$$\mathcal{L}_z = \frac{1}{B}\sum_{b=1}^{B} \left(\log \sum_{j=1}^{E} \exp(s_j^{(b)})\right)^2$$

它惩罚 logits 的绝对尺度，不改变路由的相对排序。ST-MoE 用系数 0.001，报告称这是"训练 269B 参数 MoE 不崩"的关键之一。DeepSeek-V3 则用更直接的手段：门控 logits 用 FP32 计算、sigmoid 有界、再加 aux-loss-free 的偏置控制。

### 2.6 专家并行的通信量：一个必须记住的公式

当专家被切分到不同 GPU（Expert Parallelism, EP），每个 token 要"发"到它选中的专家所在设备，算完再"收"回来。单层、单 batch 的通信量：

$$V_{\text{comm}} = 2 \cdot T \cdot k \cdot d \cdot b$$

- $T$：batch 内 token 数；$k$：每 token 激活专家数；$d$：hidden 维度；$b$：每元素字节数；系数 2 = dispatch + combine。

代入 DeepSeek-V3 的量级（$T=4096$，$k=8$，$d=7168$，BF16 $b=2$）：

$$V = 2 \times 4096 \times 8 \times 7168 \times 2 \approx 939\ \text{MB / layer}$$

61 层就是约 **57 GB** 的跨设备流量——这还没算 all-to-all 的路由元数据、padding 和对端排队。这就是为什么 MoE 推理的瓶颈往往不在 GEMM，而在**通信**。

### 2.7 稀疏度该取多少：scaling law 的答案

Clark et al. (2022) 的 "Unified Scaling Laws for Routed Language Models" 给出了路由模型的 scaling 形式，核心结论是：**在固定算力下，最优的专家数/稀疏度随算力增长而增长**，但存在收益递减。工程上的经验区间：

- 2021-2022：稀疏度 4×~16×（GShard 600B 约 4×、GLaM 1.2T/97B 约 12×、Switch-C 1.6T/约 100B 激活约 16×）；
- 2023-2024：3.6×~18×（Mixtral 46.7B/12.9B 约 3.6×、DeepSeek-V3 671B/37B 约 18×）；
- 2025-2026：20×~33×（LongCat-Flash 560B/27B 约 20.7×、Kimi K3 2.8T/104B 约 27×、DeepSeek-V4-Pro 1.6T/49B 约 33×）。

稀疏度越高，同样算力下参数容量越大，但**通信占比、负载不均、专家利用率**三个问题同时恶化。这就是 2026 年研究的重心从"怎么更稀疏"转向"怎么把稀疏 MoE 跑得更快"的原因。
