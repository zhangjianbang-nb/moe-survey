#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 arxiv_raw.json 打分排序，输出 candidates.json（含分组标签）"""
import json, re

RAW = "/home/whu/workspace/ai_workspace/moe-survey-20261103/arxiv_raw.json"
OUT = "/home/whu/workspace/ai_workspace/moe-survey-20261103/candidates.json"

# 里程碑论文：标题片段 -> (加分, 主题标签)
CURATED = {
    "Switch Transformers": (40, "架构/路由"),
    "GShard": (36, "架构/路由"),
    "GLaM": (34, "架构/路由"),
    "DeepSeekMoE": (40, "架构/路由"),
    "DeepSeek-V2": (38, "架构/推理"),
    "DeepSeek-V3": (42, "架构/训练/系统"),
    "DeepSeek-R1": (34, "训练/RL"),
    "Mixtral of Experts": (36, "架构"),
    "Qwen3": (32, "架构"),
    "Kimi K2": (36, "架构/训练"),
    "Llama 4": (30, "架构"),
    "GLM-4.5": (30, "架构"),
    "MiniMax-01": (28, "架构/长上下文"),
    "OLMoE": (30, "开源/架构"),
    "Jamba": (26, "混合架构"),
    "Hunyuan-Large": (28, "架构"),
    "Hunyuan-TurboS": (26, "架构"),
    "Step-3": (28, "架构/系统"),
    "ERNIE 4.5": (26, "架构"),
    "LongCat-Flash": (30, "架构/推理"),
    "Qwen3-Next": (30, "混合架构"),
    "Kimi Linear": (28, "混合架构"),
    "Nemotron-H": (24, "混合架构"),
    "Phi-3.5-MoE": (22, "架构"),
    "DBRX": (22, "架构"),
    "Arctic": (20, "架构"),
    "ST-MoE": (34, "训练/稳定性"),
    "MegaBlocks": (34, "系统/训练"),
    "Tutel": (30, "系统"),
    "DeepSpeed-MoE": (32, "系统"),
    "FastMoE": (26, "系统"),
    "BASE Layers": (30, "路由"),
    "Hash Layers": (24, "路由"),
    "Expert Choice": (32, "路由"),
    "Soft MoE": (28, "路由"),
    "Unified Scaling Laws for Routed": (30, "scaling"),
    "A Review of Sparse Expert Models": (28, "综述"),
    "Upcycling": (26, "训练"),
    "Loss-Free Balancing": (30, "路由"),
    "Mixture of Tokens": (24, "路由"),
    "DeepEP": (36, "推理系统/通信"),
    "EPLB": (32, "推理系统/均衡"),
    "DualPipe": (30, "训练系统"),
    "MoE-Infinity": (26, "推理/offload"),
    "FasterMoE": (26, "系统"),
    "Lina": (24, "系统"),
    "MoE-Lightning": (24, "推理/offload"),
    "KTransformers": (28, "推理/offload"),
    "QMoE": (26, "量化"),
    "MoQ": (24, "量化"),
    "LIMoE": (24, "多模态"),
    "MoE-LLaVA": (22, "多模态"),
    "DeepSeek-VL2": (26, "多模态"),
    "MMoE": (20, "推荐"),
    "PLE": (22, "推荐"),
    "Towards Understanding Mixture of Experts": (26, "理论"),
    "Mixture of Experts in Large Language Models": (24, "综述"),
    "A Survey on Mixture of Experts": (26, "综述"),
    "Sparse MoE as the New Dropout": (22, "训练"),
    "HyperRouter": (18, "路由"),
    "Sinkformers": (18, "路由"),
    "MoE Jetpack": (22, "训练"),
    "MoE-Pruner": (20, "压缩"),
    "ExpertFlow": (22, "推理/调度"),
    "MoE-Lightning": (22, "推理/offload"),
    "Kimi K2 Thinking": (26, "推理/RL"),
    "Qwen3-Omni": (22, "多模态"),
    "Seed-OSS": (20, "架构"),
    "Ling 2.0": (24, "架构"),
}

KW_SCORE = [
    ("survey", 12), ("review", 10), ("scaling law", 14), ("load balanc", 12),
    ("all-to-all", 14), ("expert parallel", 14), ("inference", 10),
    ("quantiz", 10), ("offload", 10), ("throughput", 8), ("latency", 8),
    ("training", 6), ("fine-grained", 12), ("shared expert", 10),
    ("router", 10), ("gating", 8), ("capacity factor", 8), ("token drop", 10),
    ("efficient", 6), ("fp8", 10), ("kernel", 8), ("serving", 10),
    ("long context", 8), ("multimodal", 6), ("hybrid", 6), ("mamba", 6),
    ("reinforcement", 6), ("rlhf", 4), ("grpo", 8), ("distill", 6),
    ("pruning", 6), ("compression", 6), ("communication", 10), ("bandwidth", 8),
]


def score(r):
    s = 0.0
    title = r["title"].lower()
    abst = r["abstract"].lower()
    tags = []
    for frag, (pts, tag) in CURATED.items():
        if frag.lower() in title:
            s += pts
            tags.append(tag)
    for kw, pts in KW_SCORE:
        if kw in title:
            s += pts * 1.5
        elif kw in abst:
            s += pts * 0.5
    # 命中查询数
    s += min(len(r["hits"]), 8) * 2.5
    # 新近度（2024 之后加分）
    yr = int(r["published"][:4])
    s += max(0, yr - 2022) * 3
    if yr >= 2025:
        s += 6
    # 明确排除：非 LLM 的 MoE（推荐/传统 ML）轻度降权
    if any(k in title for k in ["recommend", "click-through", "ctr", "multi-task learning"]):
        s -= 10
    if any(k in title for k in ["graph", "federated", "spiking", "wireless"]):
        s -= 12
    r["score"] = round(s, 1)
    r["tags"] = sorted(set(tags))
    return r


def main():
    data = json.load(open(RAW))
    data = [score(r) for r in data]
    data.sort(key=lambda x: -x["score"])
    json.dump(data, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"total={len(data)}")
    for i, r in enumerate(data[:60]):
        print(f"{i+1:3d} {r['score']:6.1f} {r['published'][:7]} {r['id']:14s} {r['title'][:88]}")


if __name__ == "__main__":
    main()
