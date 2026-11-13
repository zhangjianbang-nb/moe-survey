| 缩写/术语 | 全称 | 中文解释 |
|---|---|---|
| **MoE** | Mixture-of-Experts | 混合专家：把 FFN 拆成多个"专家"，每个 token 只激活其中一部分 |
| **Expert** | Expert Network | 专家：通常是一个独立的 FFN 子网络 |
| **Router / Gate** | Gating Network | 路由/门控：决定每个 token 送给哪些专家的线性层 |
| **Top-k** | Top-k Routing | 每个 token 选分数最高的 k 个专家 |
| **Capacity Factor (CF)** | Capacity Factor | 容量因子：每个专家最多处理的 token 数上限的倍率 |
| **Token Dropping** | Token Dropping | 专家容量满后，多余 token 被丢弃（不经过该专家） |
| **Aux Loss** | Auxiliary Loss | 辅助损失：惩罚专家负载不均的额外损失项 |
| **Z-loss** | Router Z-loss | 惩罚 router logits 尺度过大的损失项，稳定训练 |
| **Aux-loss-free** | Auxiliary-Loss-Free Balancing | 用偏置项（不进梯度）做负载均衡，DeepSeek-V3 提出 |
| **Shared Expert** | Shared Expert | 总是激活的专家，承载通用知识 |
| **Fine-grained Expert** | Fine-grained Expert Segmentation | 把专家切小、数量增多，增加组合空间 |
| **EP** | Expert Parallelism | 专家并行：把不同专家放到不同 GPU |
| **All-to-all** | All-to-all Communication | 多对多通信：每个设备向所有设备发送不同数据 |
| **Dispatch / Combine** | Dispatch / Combine | MoE 的两次 all-to-all：发送 token 到专家 / 收回结果 |
| **TP / PP / DP / SP** | Tensor / Pipeline / Data / Sequence Parallelism | 张量/流水线/数据/序列并行 |
| **DeepEP** | Deep Expert Parallelism | DeepSeek 开源的 EP 通信库（LL/HT 双模式） |
| **EPLB** | Expert Parallelism Load Balancer | DeepSeek 开源的推理侧专家负载均衡器 |
| **Grouped GEMM** | Grouped GEMM | 把多个不同大小的 GEMM 打包成一个 kernel |
| **Fused MoE** | Fused MoE Kernel | 把 gate/top-k/GEMM/加权求和融合成一个 kernel |
| **PD 分离** | Prefill-Decode Disaggregation | 把预填充和解码放到不同 GPU 组 |
| **AFD** | Attention-FFN Disaggregation | 把注意力和 FFN 放到不同 GPU 组（Step-3） |
| **KV Cache** | Key-Value Cache | 自回归推理缓存的历史 K/V |
| **MLA** | Multi-head Latent Attention | 多头潜在注意力：把 KV 压成 latent 向量（DeepSeek） |
| **DSA** | DeepSeek Sparse Attention | 用轻量 indexer 选 top-k KV 块的稀疏注意力 |
| **CSA / HCA** | Compressed Sparse Attention / Heavily Compressed Attention | V4 的两种压缩注意力，交替使用 |
| **KDA** | Kimi Delta Attention | Kimi 的线性注意力（对角+低秩转移矩阵） |
| **GDN** | Gated DeltaNet | 门控 Delta 规则线性注意力（Qwen3-Next） |
| **Upcycling** | Sparse Upcycling | 把稠密模型复制成专家，继续训练成 MoE |
| **Muon / MuonClip** | Muon Optimizer / Muon with QK-clip | 矩阵级优化器；K2 加 QK-clip 防 logit 爆炸 |
| **Routing Replay** | Routing Replay | 存下训练时的 top-k 索引，推理时复用，保证一致性 |
| **Roofline** | Roofline Model | 用算术强度与硬件拐点判断 compute/memory bound 的模型 |
| **Arithmetic Intensity** | Arithmetic Intensity | 算术强度：每字节内存访问对应的 FLOPs |
| **MFU** | Model FLOPs Utilization | 模型算力利用率：实际 FLOPs / 峰值 FLOPs |
| **IBGDA** | InfiniBand GPU Direct Async | GPU 直接发起 RDMA，绕过 CPU |
| **TTFT / TPOT** | Time To First Token / Time Per Output Token | 首 token 延迟 / 每输出 token 时间 |
| **SLO** | Service Level Objective | 服务等级目标（延迟/吞吐承诺） |
