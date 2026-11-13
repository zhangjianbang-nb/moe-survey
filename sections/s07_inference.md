## 7. 推理系统：MoE 真正的主战场

### 7.1 三个根本矛盾

MoE 推理的困难可以归结为三个矛盾，所有系统工作都是在解这三个矛盾：

**矛盾一：容量与带宽。** 671B 参数（BF16 约 1.3 TB）必须全部驻留 HBM，但每个 token 只读 37B（约 74 GB）。这意味着 **99% 的权重在单 token 推理中是"死重"**——它们占着显存，却不产生计算。HBM 容量决定"能不能装下"，HBM 带宽决定"读得多快"。

**矛盾二：稀疏与并行。** 专家切分到多卡（EP）后，每个 token 要去不同卡上找专家，产生 all-to-all 通信。通信量与 token 数、激活专家数、hidden 维度成正比，且**随 EP 规模线性增长**。EP 越大，单卡负载越轻，但通信越贵。

**矛盾三：均衡与效率。** 真实请求的专家偏好高度不均（代码请求偏代码专家、数学请求偏推理专家）。EP 假设"每个专家负载差不多"，但现实中热门专家可能过载 3-5 倍，导致 straggler。均衡需要冗余专家/动态重排，但副本占显存、重排有开销。

### 7.2 并行策略：为什么 MoE 偏爱 EP

| 并行方式 | 切分对象 | 通信模式 | 适用 |
|---|---|---|---|
| TP（张量并行） | 单个矩阵的列/行 | all-reduce（每层 2 次） | dense 层、attention；跨节点效率低 |
| EP（专家并行） | 不同专家到不同卡 | all-to-all（dispatch/combine） | MoE FFN；跨节点可扩展 |
| PP（流水线并行） | 不同层到不同卡 | p2p send/recv | 超大模型；有气泡 |
| DP（数据并行） | 不同 batch 到不同卡 | 无（推理时） | attention 复制；提升吞吐 |
| SP（序列并行） | 序列维度切分 | all-gather/reduce-scatter | 长序列 attention |

MoE 的核心选择是 **EP 而不是 TP**，原因：

1. **TP 的通信量与 hidden 维度成正比，且每层都要 all-reduce**；EP 只在 MoE 层做 all-to-all，且通信量与"被路由的 token 数"成正比——稀疏时更省。
2. **TP 要求每张卡都有完整的层输入**，无法利用"专家稀疏"；EP 让每张卡只放部分专家，天然匹配稀疏性。
3. **TP 跨节点效率低**（all-reduce 对延迟敏感）；EP 的 all-to-all 可以流水线化、可以与计算重叠。

但 EP 不是免费的：**EP 度越高，单专家 batch 越小，GEMM 效率越低**。V3 的 prefill 用 EP32（每专家 batch 足够大），decode 用 EP320（每卡一个专家，靠大 batch 摊薄）。这是一个典型的"按阶段选并行度"的设计。

### 7.3 all-to-all：MoE 的通信命脉

一次 MoE 层的前向包含两次 all-to-all：

1. **dispatch**：每个 token 的 hidden 向量发送到它选中的 $k$ 个专家所在设备；
2. **combine**：专家算完后，结果按门控权重加权求和，送回原设备。

通信量（单层、单 batch）：

$$V = 2 \cdot T \cdot k \cdot d \cdot b$$

代入 V3（$T=4096$，$k=8$，$d=7168$，BF16 $b=2$）：单层约 **939 MB**，61 层约 **57 GB**。如果每层都同步等待，这个开销会吃掉大部分算力。所以所有高性能实现都做**通信-计算重叠**：

- **DeepEP**（DeepSeek 开源）：NVLink + RDMA 双通道；LL（low-latency）模式面向 decode 的小 batch，HT（high-throughput）模式面向 prefill/训练的大 batch；用 hook 机制把 dispatch/combine 拆成细粒度步骤，与 GEMM 交错执行。
- **NCCL EP**（NVIDIA，2026）：基于 NCCL Device API 的官方 EP 库，提供 `ncclEpDispatch` / `ncclEpCombine`；LL 模式针对 1-128 token 的 decode，HT 模式针对训练/prefill。它的意义是**把 EP 通信标准化**，让上层框架不必自己造轮子。
- **UBEP**（2026）：面向 NVL72/576 和 CloudMatrix384 这类"超节点"，解决 BSP 串行化、同步开销、距离无关调度导致的负载不均。
- **MegaScale-Infer 的 M2N 库**：消除 GPU-CPU 拷贝、组初始化开销和 GPU 同步，专门服务"attention-FFN 分离"后的 token 传输。

**工程要点**：all-to-all 的性能不只取决于带宽，还取决于**路由元数据的开销**。每个 token 要携带"目标专家 ID、原设备 ID、门控权重"，这些元数据的打包/解包可能比 hidden 向量本身还贵。DeepEP 的做法是把元数据压缩成 32-bit 的 token 索引，并在 GPU 上做 prefix-sum 排序。

### 7.4 负载均衡：推理侧的三层机制

训练侧的 aux-loss-free 解决"平均负载"，但推理侧的负载是**动态、偏斜、不可预测**的。生产系统通常有三层：

**第一层：冗余专家（redundant experts）。** 把高负载专家复制多份，放到不同 GPU。V3 的 prefill 部署有 32 个冗余专家（每 GPU 多放 1 个），decode 部署有 64 个 GPU 专门放冗余专家和共享专家。冗余专家的集合**每 10 分钟按在线统计更新**。

**第二层：动态重排（rebalancing）。** 在不增加跨节点通信的前提下，把专家在节点内重新排列，让每张卡的负载尽量均匀。V3 的 prefill 每 10 分钟重排一次；decode 因为"每卡一个专家"不需要重排。

**第三层：请求级调度。** 把"专家偏好相似"的请求放在同一 batch，减少跨节点 fan-out；或者用预测器预判下一层的专家激活，提前预取（PROBE）。2026 年的 LLEP 更进一步：**允许把过载设备的 token 和专家参数一起"搬"到空闲设备**，报告 5× 加速、4× 峰值显存下降。

**一个反直觉的事实**：2026 年的 LLEP 论文指出，**负载不均是"自然的、甚至有益的"**——领域专家集中处理领域 token，正是 MoE 想要的。强行均衡会破坏专业化。所以生产系统的目标不是"完全均匀"，而是"在 SLO 内接受不均衡"。

### 7.5 Kernel：grouped GEMM 与 fused MoE

MoE 的计算模式与 dense 完全不同：每个专家收到的 token 数不同（动态），需要 **grouped GEMM**（把多个不同大小的 GEMM 打包成一个 kernel 启动）。

三种实现路线：

1. **Token-major grouped GEMM**：按 token 分组，每个专家一个 tile。问题是当每专家 token 很少时，tile padding 严重、启动的 grid 很短命、带宽利用率低。
2. **Weight-major persistent kernel**（MonoMoE，2026）：把整个 decode step 的 token tile 放在 tensor-core 的 N 维度，CTA 按专家权重 tile 划分。一个 persistent grid 融合 routing、top-k、量化、两个 expert projection、激活、reduction。好处是**消除 expert-local token materialization、减少 padding**。
3. **Mega-kernel**（UniEP，2026）：把通信和计算融合进一个 kernel，用统一抽象适配不同 EP 优化策略（压缩、重叠、重排）。

**融合（fusion）是另一个关键**：gate + top-k + softmax + scatter + GEMM + weighted-sum 如果分开做，会有多次 kernel launch 和中间张量读写。vLLM 的 `fused_moe`、SGLang 的 `fused_experts`、TensorRT-LLM 的 MoE plugin 都在做这件事。MonoMoE 把这个思路推到极致：**整个 decode step 一个 kernel**。

### 7.6 量化：MoE 的"显存救命稻草"

MoE 对量化比 dense 模型更友好，原因有二：

1. **专家层比普通 FFN 更鲁棒**：MoQE（2023）发现，专家权重降到 2-bit，性能损失远小于同等 dense FFN；
2. **专家是"独立"的**：可以对不同专家用不同精度（热专家高精度、冷专家低精度），而不影响其他专家。

代表性工作：

| 工作 | 方法 | 效果 |
|---|---|---|
| MoQE (2023) | 只量化专家权重，2-bit | 显存大降，性能损失小 |
| QMoE (2023) | 子 1-bit 压缩 + 定制解码 kernel | 1.6T SwitchTransformer → <160 GB（0.8 bit/param，20×） |
| DynaExq (2025) | 运行时混合精度分配（按专家热度） | 单 GPU 硬显存约束下的在线优化 |
| MonoMoE (2026) | block-wise 量化 + weight-major megakernel | decode 吞吐提升 |
| V4.1-Flash (2026) | FP4 KV cache | 890 B/token（V4-Flash 的 1/4） |

**工程要点**：MoE 量化的难点不在"怎么量化"，而在"**量化后的 kernel 怎么写**"。低比特权重的解码需要专门的解包/反量化指令，且要和 grouped GEMM 融合。QMoE 的贡献一半是算法，一半是 bespoke GPU decoding kernel。

### 7.7 Offloading：当显存装不下时

单卡/少卡部署大 MoE 的标准做法是 **offloading**：把冷专家放在 CPU 内存/SSD，热专家留在 GPU，按需预取。

- **FineMoE**（2024）：从专家选择模式和 prompt 语义中提取"预取提示"，做细粒度 prefetch/cache/offload；
- **MoE-Lightning**（2024）：内存受限 GPU 上的高吞吐 MoE 推理，用分层 cache 和 pipeline；
- **KTransformers**（2025）：CPU-GPU 混合，把 MoE 专家放 CPU、attention 放 GPU，用 llama.cpp 风格的高效 CPU kernel；
- **VisMMOE**（2026）：视觉 token 剪枝不仅省计算，还**重塑专家需求**（visual-expert affinity），让专家访问更集中、更可预测，从而提升 offloading 命中率；
- **Specter**（2025）：量化 + 多步投机 offloading，把专家加载藏在投机解码的 draft 阶段。

**offloading 的核心指标是"命中率"**：如果热专家集合稳定，GPU 命中率高，延迟可控；如果请求分布漂移，命中率下降，延迟爆炸。所以生产系统通常把 offloading 作为"降级方案"，而不是首选。

### 7.8 分离式部署（disaggregation）：2025-2026 的主线

这是过去两年 MoE 推理系统最重要的架构变化。核心观察：**attention 和 FFN 的资源需求完全不同**。

- attention：prefill 时 compute-bound（大矩阵乘），decode 时 KV-cache-bound（读 KV）；
- MoE FFN：decode 时 memory-bound（读专家权重），prefill 时 compute-bound（大 GEMM）。

把它们放在同一组 GPU 上，只能取"最大公约数"的资源配置，浪费严重。于是出现 **attention-FFN 分离**：

| 系统 | 分离方式 | 关键机制 |
|---|---|---|
| MegaScale-Infer (2024) | attention 和 FFN 分到不同 GPU 组 | ping-pong pipeline；M2N 通信库 |
| Step-3 AFD (2025) | Attention-FFN Disaggregation | 专用子系统；MFA 注意力 |
| JANUS (2025) | attention/MoE 分到不同 worker pool | 两阶段通信；微秒级 activation scheduler；SLO-aware scaling；4.7× per-GPU 吞吐 |
| FinDEP (2025) | DEP + 细粒度任务调度 | 支持 shared experts；任务重叠 |
| AInfer-PD (2026) | prefill-decode 多路复用 | 通信安全的 in-place 复用 |

**分离的代价是"token 要在两组 GPU 之间来回传"**。所以分离系统必须解决：

1. **低延迟传输**：M2N、两阶段通信、零拷贝；
2. **流水线**：把 batch 切成 micro-batch，在 attention 组和 FFN 组之间"乒乓"；
3. **负载匹配**：attention 组和 FFN 组的 GPU 数量要按实际负载比例配置，否则一组等另一组。

### 7.9 PD 分离与 wide EP：V3 的生产部署

DeepSeek-V3 的公开部署方案是理解"大规模 MoE 推理"的最好样本：

**Prefill 阶段**：

- 最小部署单元：**4 节点 / 32 GPU**；
- attention：TP4 + SP，DP8；
- MoE：**EP32**（每 GPU 8 个专家）；
- **32 个冗余专家**（每 GPU 多放 1 个）；
- 同时处理两个 micro-batch，让一个的 attention/MoE 与另一个的 dispatch/combine 重叠；
- 每 10 分钟按在线统计更新冗余专家集合。

**Decode 阶段**：

- 最小部署单元：**40 节点 / 320 GPU**；
- attention：TP4 + SP，DP80；
- MoE：**EP320**（每 GPU 1 个专家）；
- **64 个 GPU 专门放冗余专家和共享专家**；
- all-to-all 用 IB 点对点 + **IBGDA**（GPU 直接发起 RDMA）；
- 每专家 batch 通常 < 256 token，**瓶颈是内存访问而非计算**；
- 只给 dispatch+MoE+combine 分配少量 SM，把大部分 SM 留给 attention。

**为什么 decode 要 EP320？** 因为 decode 时每 token 只激活 8 个专家，如果 EP 度低，单卡要装很多专家、读很多权重；EP320 让每卡只装 1 个专家，权重读取量最小。代价是 all-to-all 的 fan-out 变大，所以用 IBGDA 和点对点传输压延迟。

**wide EP 的边界**：EP 度不是越高越好。当 EP 大到"每卡一个专家"后，再增加卡只会增加通信跳数，不会减少单卡权重读取。V3 的 EP320 基本是"每卡一专家"的极限；再往上需要"专家复制 + 请求路由"（EPLB）或"分离式部署"。

### 7.10 长上下文：KV cache 与 MoE 权重的显存战争

长上下文推理中，KV cache 和 MoE 权重争抢 HBM：

- 128K 上下文、BF16 KV、MLA 压缩后约 0.5-1 GB/请求；
- 671B MoE 权重约 1.3 TB；
- 一张 H100 80GB，装完权重后所剩无几。

解法是"**双向压缩**"：

1. **压 KV**：MLA（latent 压缩）、DSA（稀疏选择）、CSA/HCA（压缩 + 稀疏）、FP4 KV（V4.1，890 B/token）；
2. **压权重**：FP8/FP4 量化、专家 offloading、专家复制减少跨卡。

V4 的报告给出一个关键数字：**1M 上下文下，V4-Pro 的单 token 推理 FLOPs 只有 V3.2 的 27%，KV cache 只有 10%**。这不是靠"更大的模型"，而是靠 CSA/HCA 把注意力的复杂度从 $O(L^2)$ 压到接近 $O(L)$，再靠 mHC 和 Muon 保持训练稳定性。

### 7.11 端侧与边缘：另一条战线

MoE 在端侧有天然优势（激活参数小、算力需求低），但劣势是**权重总量大、显存不够**。2026 年的几条路线：

- **CPU-GPU 混合**（清华，2026）：SLP 流式加载 prefill，双 RTX 5090 上 1800 token/s、45K prompt 30 秒内完成；AVX-512 FP8 GEMV 让 CPU 原生跑 FP8；本地部署达到云级 SLO。
- **Apple Silicon**：多节点 EP 在 Mac Studio 集群上跑私有 LLM；Apple NPU 上的 MoE 推理。
- **3D-MoE**（2026）：3D 堆叠芯片上的 MoE decode 硬件-软件协同设计。
- **CoMoE**（2026）： commodity GPU 上的民主化 MoE 推理。

端侧的核心矛盾是"**权重总量 vs 显存带宽**"：即使只激活 3B 参数，也要从 100B+ 权重中挑出这 3B。所以端侧 MoE 的瓶颈几乎总是"读权重"，而不是"算"。

### 7.12 定量分析：MoE decode 的 roofline

用 roofline 模型可以精确解释"为什么 MoE decode 是 memory-bound"。

单 token、单层的算术强度（arithmetic intensity）：

$$AI = \frac{2 \cdot W_{\text{act}}}{W_{\text{act}} \cdot b} = \frac{2}{b}\ \text{FLOP/byte}$$

- BF16（$b=2$）：$AI = 1$ FLOP/byte；
- FP8（$b=1$）：$AI = 2$；
- FP4（$b=0.5$）：$AI = 4$。

H100 SXM 的 BF16 峰值约 989 TFLOPS，HBM 带宽 3.35 TB/s，**拐点（ridge point）约 295 FLOP/byte**。MoE decode 的 $AI=1$，**比拐点低近 300 倍**——完全 memory-bound。

**batch 的作用**：如果一次处理 $B$ 个 token，权重只读一次，$AI = 2B/b$。要让 BF16 达到拐点，需要 $B \approx 295$。这解释了为什么 V3 decode 要把"每专家 batch"做到 256 左右：**低于这个数，GPU 算力大量闲置；高于这个数，才进入 compute-bound 区间**。

**这也解释了为什么"小 batch 的 MoE 推理"是灾难**：batch=1 时，GPU 的算力利用率可能只有 1-2%，绝大部分时间在等 HBM。所有"低延迟 MoE 推理"的系统工作（DeepEP LL 模式、MonoMoE、投机解码）本质上都在解决"如何在小 batch 下仍把带宽用满"。
