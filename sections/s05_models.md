## 5. 模型代际全景：从 600B 到 2.8T

下表汇总 2020-2026 年主要 MoE 模型的公开参数。**所有数字以官方技术报告为准**；标注"官方博客"的条目来自厂商发布页而非 arXiv 论文；"未披露"表示报告未给出该项。

### 5.1 奠基期（2020-2024）：把 MoE 跑通

| 模型 | 时间 | 总参数 | 激活 | 路由专家 | 激活专家 | 层数 | 训练 token | 上下文 | 注意力 | 关键设计 |
|---|---|---|---|---|---|---|---|---|---|---|
| GShard | 2020-06 | 600B | 未披露 | 2048 | top-2 | — | — | — | — | 首个大规模 Transformer MoE；2048 TPU v3 / 4 天 |
| Switch-C | 2021-01 | 1.6T | 未披露 | 2048 | top-1 | 32 | — | — | — | top-1 简化；bf16 训练；选择性精度 |
| GLaM | 2021-12 | 1.2T | 97B | 64 | top-2 | 32 | 1.6T | — | — | 首个"激活参数"叙事 |
| ST-MoE-32B | 2022-02 | 269B | ~32B 等效 | 32 | top-2 | — | — | — | — | router z-loss；训练稳定性设计指南 |
| Mixtral 8×7B | 2023-12 | 46.7B | 12.9B | 8 | top-2 | 32 | — | 32K | GQA | 开源产品化拐点 |
| DeepSeekMoE-16B | 2024-01 | 16.4B | 2.8B | 64+2 | 6+2 | 28 | 2T | 4K | MHA | 细粒度 + 共享专家 |
| DeepSeek-V2 | 2024-05 | 236B | 21B | 160+2 | 6+2 | 60 | 8.1T | 128K | MLA | device-limited routing；KV cache −93.3% |
| DBRX | 2024-03 | 132B | 36B | 16 | top-4 | 40 | 12T | 32K | GQA | 16 专家粗粒度 |
| Grok-1 | 2024-03 | 314B | 86B | 8 | top-2 | 64 | — | 8K | — | 粗粒度、高激活比 |
| Snowflake Arctic | 2024-04 | 480B | 17B | 128 | — | — | 3.5T | — | — | 极致稀疏（28×） |
| OLMoE | 2024-09 | 6.9B | 1.3B | 64 | top-8 | 16 | 5T | 4K | MHA | 全开源（权重/数据/日志） |
| Jamba | 2024-03 | 52B | 12B | 16 | top-2 | — | — | 256K | Mamba 混合 | Transformer-Mamba 混合 |
| Hunyuan-Large | 2024-11 | 389B | 52B | 16 | top-1 | — | — | — | — | 腾讯；top-1 粗粒度 |
| Qwen1.5-MoE | 2024-02 | 14.3B | 2.7B | 60+4 | 4+4 | 24 | — | 32K | — | 细粒度 + 共享 |
| Qwen2-57B-A14B | 2024-06 | 57.3B | 14B | 64 | top-8 | 28 | — | 128K | GQA | 中等规模 |

### 5.2 主力期（2025-2026）：稀疏度竞赛与系统协同

| 模型 | 时间 | 总参数 | 激活 | 路由专家 | 激活专家 | 层数 | 训练 token | 上下文 | 注意力 | 关键设计 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 2024-12 | 671B | 37B | 256+1 | 8+1 | 61 | 14.8T | 128K | MLA | aux-loss-free、MTP、FP8、DualPipe |
| Qwen3-235B-A22B | 2025-05 | 235B | 22B | 128 | top-8 | 94 | 36T | 128K | GQA | thinking/non-thinking 统一 |
| Kimi K2 | 2025-07 | 1.04T | 32.6B | 384 | top-8 | 61 | 15.5T | 128K | MLA | MuonClip（QK-clip）；零 loss spike |
| GLM-4.5 | 2025-08 | 355B | 32B | 160 | top-8 | 92 | 23T | 128K | — | ARC 能力 |
| GLM-5 | 2026-02 | 744B | 40B | 256 | top-8 | 80 | 28.5T | 200K | DSA | DSA 稀疏注意力；异步 RL |
| MiniMax-01 | 2025-01 | 456B | 45.9B | 32 | top-2 | 64 | — | 4M | Lightning | 线性注意力 + MoE |
| MiniMax-M2 | 2026-05 | 229.9B | 9.8B | 未披露 | 未披露 | — | — | — | — | agentic 端到端；Forge RL |
| Step-3 | 2025-07 | 321B | 38B | 未披露 | 未披露 | 61 | — | — | MFA | AFD（Attention-FFN 分离） |
| LongCat-Flash | 2025-09 | 560B | 27B 均值 | — | 动态 | — | — | — | MLA | Zero-computation Experts；18.6-31.3B 动态 |
| Hunyuan-TurboS | 2025-05 | 389B | 52B | 16 | top-1 | — | — | — | Mamba 混合 | Mamba-Transformer 协同 |
| Llama 4 Scout | 2025-04 | 109B | 17B | 16 | top-1 | 48 | 官方博客 | 10M | iRoPE | early fusion；top-1 |
| Llama 4 Maverick | 2025-04 | 400B | 17B | 128 | top-1 | 48 | 官方博客 | 1M | iRoPE | early fusion |
| Qwen3-Next-80B-A3B | 2025-09 | 80B | 3B | 未披露 | 未披露 | — | — | 256K | GDN 混合 | Gated DeltaNet + 全局注意力 |
| Kimi Linear | 2025-10 | 48B | 3B | 未披露 | 未披露 | — | — | — | KDA+MLA | 线性注意力首次全面超越全注意力 |
| DeepSeek-V3.2 | 2025-12 | 671B（沿用 V3 主干） | 37B | 256+1 | 8+1 | 61 | — | 128K | DSA | DSA 稀疏注意力；RL 规模化 |
| Kimi K3 | 2026-07 | 2.8T | 104B | 896 | top-16 | — | — | 1M | KDA | Stable LatentMoE；原生视觉 |
| DeepSeek-V4-Pro | 2026-04 | 1.6T | 49B | 384+1 | 6+1 | 61 | 32T | 1M | CSA/HCA | mHC、Muon、Hash routing（前 3 层） |
| DeepSeek-V4-Flash | 2026-04 | 284B | 13B | 256+1 | 6+1 | 43 | 32T | 1M | CSA/HCA | 同上；1M 上下文 FLOPs 仅 V3.2 的 27% |
| DeepSeek-V4.1-Flash | 2026-09 | 552B | 16B decode / 8B prefill | — | — | — | 45T | 1M | CSA2 | CED、FP4 KV（890 B/token）、SWA Bounded Replay |
| Qwen3.8-Flash-Next | 2026-08 | 125B（+51B n-gram 外置） | 6B | — | — | 48 | — | — | GDN+QSA | 1/9 训练 FLOPs；Gated Residual |
| Nemotron 3.5 Super/Lightning | 2026-10 | — | — | — | — | — | — | — | Mamba 混合 | 开放高效 MoE |

### 5.3 五个可读出的规律

**(1) 激活参数不涨，总参数暴涨。** 2024 年 V3 是 37B 激活；2026 年 V4-Pro 是 49B、K3 是 104B——激活参数缓慢上升，但总参数从 671B 涨到 1.6T/2.8T。**稀疏度（总/激活）从 18× 涨到 33×（V4-Pro）、27×（K3）。**

**(2) 专家数持续增加，单专家变窄。** V3 的 256 专家、专家中间维 2048；V4-Pro 的 384 专家、中间维 3072；K3 的 896 专家。专家越窄，组合空间越大，但单专家 GEMM 越小，GPU 利用率越难做。

**(3) 激活专家数稳定在 6-8（K3 是 16）。** 这是工程折中：激活太多，通信量和计算量都上去；激活太少，路由容量不足。K3 敢用 16，是因为它的 896 专家足够细，单专家足够小。

**(4) 注意力与 MoE 解耦。** 2025 年后的旗舰模型几乎都用了"混合注意力"：MLA（DeepSeek）、DSA（V3.2/GLM-5）、CSA/HCA（V4）、KDA（K3/Kimi Linear）、GDN（Qwen3-Next）、Lightning（MiniMax）。MoE 负责 FFN 容量，注意力负责上下文效率，两者独立演进。

**(5) 训练 token 量级跳变。** 2024 年 14.8T（V3）、2025 年 15.5T（K2）/23T（GLM-4.5）/36T（Qwen3）、2026 年 28.5T（GLM-5）/32T（V4）/45T（V4.1-Flash）。**数据量已经超过" Chinchilla 最优"数倍**，说明 MoE 的容量需要海量数据才能填满——这也是为什么 MoE 训练越来越像"数据工程"而不是"模型工程"。

### 5.4 三个值得单独说的模型

**DeepSeek-V3.2（2025-12）**：它是"V3 主干 + DSA 稀疏注意力"的组合。DSA（DeepSeek Sparse Attention）用轻量 indexer 为每个 query 选 top-k 的 KV 块，把长上下文注意力的复杂度从 $O(L^2)$ 降到 $O(L \cdot k)$。V3.2 的意义在于：**它证明了 MoE 主干可以和稀疏注意力解耦升级**，不需要重新训练整个模型。

**Kimi K3（2026-07）**：2.8T 参数、896 专家、top-16，是当前公开的最大 MoE。它的 "Stable LatentMoE" 把专家路由放在低维 latent 空间做，减少路由本身的计算；KDA 则把线性注意力的表达力推到"公平对比下超过全注意力"。K3 的 2.5× scaling efficiency 提升，主要来自"专家更细 + 注意力更省 + 数据更好"三者叠加。

**DeepSeek-V4.1-Flash（2026-09）**：它不是"更大的模型"，而是"更省的模型"。552B 主干，prefill 只激活 8B、decode 激活 16B；CED 架构让 decoder 的 global KV 从 encoder 的 final hidden 投影而来；CSA2 做跨层 KV 复用；FP4 KV cache 把每 token 的全局 KV 压到 **890 字节**（约为 V4-Flash 的 1/4）。**这是"推理成本优先"的设计哲学**：模型能力靠 post-training 和 RL 提升，推理成本靠架构压缩。
