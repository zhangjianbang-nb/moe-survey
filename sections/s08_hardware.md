## 8. 硬件协同：MoE 对芯片的要求

### 8.1 一张表看懂硬件瓶颈

| 指标 | H100 SXM | H200 | B200 | GB200 NVL72（单 GPU 等效） |
|---|---|---|---|---|
| HBM 容量 | 80 GB | 141 GB | 192 GB | 192 GB |
| HBM 带宽 | 3.35 TB/s | 4.8 TB/s | 8 TB/s | 8 TB/s |
| BF16 峰值 | 989 TFLOPS | 989 TFLOPS | ~2250 TFLOPS | ~2250 TFLOPS |
| FP8 峰值 | 1979 TFLOPS | 1979 TFLOPS | ~4500 TFLOPS | ~4500 TFLOPS |
| NVLink 带宽 | 900 GB/s | 900 GB/s | 1.8 TB/s | 1.8 TB/s（域内） |
| 节点间 | IB 400 Gb/s | IB 400 Gb/s | IB 800 Gb/s | NVLink 域内 72 GPU |

**MoE 的核心矛盾在这张表里一目了然**：

- 671B BF16 权重需要 1.3 TB → 需要 17 张 H100 或 7 张 B200 才能装下（还不算 KV）；
- 但 decode 时每 token 只读 37B → 单卡带宽利用率极低；
- 所以必须"多卡装权重 + EP 并行 + 大 batch"三件事一起做。

### 8.2 算术强度的三档

| 精度 | 每参数字节 | decode AI（batch=1） | H100 拐点 | 差距 |
|---|---|---|---|---|
| BF16 | 2 | 1 FLOP/byte | 295 | 295× |
| FP8 | 1 | 2 FLOP/byte | 590（FP8 峰值/带宽） | 295× |
| FP4 | 0.5 | 4 FLOP/byte | 1180 | 295× |

**结论**：无论用什么精度，单 batch decode 都是 memory-bound，差距稳定在约 300 倍。精度提升（FP8/FP4）只是把"分子"和"分母"同比例放大，**不改变 memory-bound 的本质**。真正改变本质的是 batch size：$AI = 2B/b$，batch 上去才进入 compute-bound。

### 8.3 NVLink 与 IB：两种通信的物理基础

- **NVLink**：节点内 GPU 互联，H100 900 GB/s、B200 1.8 TB/s。all-to-all 的节点内部分走 NVLink。
- **IB / RoCE**：节点间互联，400/800 Gb/s per port。跨节点 all-to-all 走 IB。
- **IBGDA**：GPU 直接发起 RDMA，绕过 CPU，降低延迟。V3 的 decode 用它。
- **NVLink SHARP / 交换机内归约**：把 all-reduce 的部分计算下放到交换机，减少 GPU 工作量。

**拓扑决定性能**：EP 的 all-to-all 是"多对多"模式，对拓扑敏感。胖树（fat-tree）拓扑下，跨节点跳数少；环状/3D torus 拓扑下，跳数多。V3 的 node-limited routing（每 token 最多 4 个节点）就是为了控制跳数。

### 8.4 计算-通信重叠的硬件基础

MoE 的高性能实现依赖三个硬件特性：

1. **异步拷贝引擎（Copy Engine）**：NVLink 的 copy engine 可以在不占用 SM 的情况下搬数据。FEPLB（2026）利用这一点做"近零成本"的专家重排——重排的数据搬运完全在后台进行，不抢计算资源。
2. **TMA（Tensor Memory Accelerator）**：Hopper 起，TMA 可以在全局内存和共享内存之间做异步块拷贝，减少地址计算开销。MoE 的 grouped GEMM 用它加载专家权重。
3. **分布式共享内存（DSMEM）**：Hopper 的 thread block cluster 允许同一 cluster 内的 SM 直接访问彼此的共享内存。DeepEP 用它做节点内 all-to-all 的低延迟路径。

### 8.5 DeepSeek 的硬件建议（ISCA'25）

DeepSeek 在 ISCA'25 的论文（arXiv:2505.09343）里直接给芯片设计提了建议，核心有三条：

1. **HBM 容量比带宽更重要**：MoE 的瓶颈是"装不下"，不是"读不快"（虽然两者都重要）。他们建议未来芯片的 HBM 容量增长速度要快于带宽。
2. **互联要支持"多对多"模式**：MoE 的 all-to-all 不是规整的 all-reduce，需要更灵活的互联拓扑和更低的延迟。
3. **计算单元要支持"小 GEMM 高效"**：MoE 的单专家 GEMM 很小（batch 可能只有几十），需要硬件对小矩阵乘更友好。

这三条在 2026 年的硬件里部分实现了：B200 的 192GB HBM、NVLink 域内 72 GPU、SM100 的 FP4 支持，都是在往这个方向走。

### 8.6 一个被低估的指标：显存带宽利用率

MoE decode 的性能可以用一个简单公式估算：

$$\text{吞吐} \approx \frac{\text{HBM 带宽} \times \text{带宽利用率}}{\text{每 token 读取字节数}}$$

以 H100（3.35 TB/s）、BF16、每 token 读 74 GB（37B 激活参数）为例：

$$\text{吞吐} \approx \frac{3.35 \times 10^{12} \times 0.7}{74 \times 10^9} \approx 32\ \text{token/s}$$

这与实测的 30-40 token/s 吻合。**要提高吞吐，只有三条路**：提高带宽利用率（更好的 kernel）、减少每 token 读取字节（量化/更稀疏）、增加 batch（摊薄权重读取）。
