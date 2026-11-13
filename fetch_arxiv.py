#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 调研：arXiv 多关键词批量检索 -> arxiv_raw.json"""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

OUT = "/home/whu/workspace/ai_workspace/moe-survey-20261103/arxiv_raw.json"
BASE = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# (label, query, max_results, sort_mode)
QUERIES = [
    # ---- 基础/里程碑 ----
    ("base:moe-core", 'all:"mixture of experts"', 100, "relevance"),
    ("base:sparse-expert", 'all:"sparse expert"', 60, "relevance"),
    ("base:conditional-compute", 'all:"conditional computation"', 50, "relevance"),
    ("m:switch", 'ti:"Switch Transformer"', 20, "relevance"),
    ("m:gshard", 'ti:"GShard"', 20, "relevance"),
    ("m:glam", 'ti:"GLaM"', 20, "relevance"),
    ("m:deepseekmoe", 'ti:"DeepSeekMoE"', 20, "relevance"),
    ("m:deepseekv2", 'ti:"DeepSeek-V2"', 20, "relevance"),
    ("m:deepseekv3", 'ti:"DeepSeek-V3"', 20, "relevance"),
    ("m:r1", 'ti:"DeepSeek-R1"', 20, "relevance"),
    ("m:mixtral", 'ti:"Mixtral"', 20, "relevance"),
    ("m:qwen3", 'ti:"Qwen3"', 30, "relevance"),
    ("m:kimi-k2", 'ti:"Kimi K2"', 20, "relevance"),
    ("m:llama4", 'ti:"Llama 4"', 20, "relevance"),
    ("m:glm45", 'ti:"GLM-4.5"', 20, "relevance"),
    ("m:minimax", 'ti:"MiniMax-01" OR ti:"MiniMax-M1" OR ti:"MiniMax-M2"', 20, "relevance"),
    ("m:olmoe", 'ti:"OLMoE"', 20, "relevance"),
    ("m:jamba", 'ti:"Jamba"', 20, "relevance"),
    ("m:hunyuan", 'ti:"Hunyuan-Large" OR ti:"Hunyuan-TurboS" OR ti:"HunyuanVideo"', 20, "relevance"),
    ("m:step3", 'ti:"Step-3"', 20, "relevance"),
    ("m:ernie45", 'ti:"ERNIE 4.5"', 20, "relevance"),
    ("m:longcat", 'ti:"LongCat-Flash"', 20, "relevance"),
    ("m:qwen3next", 'ti:"Qwen3-Next"', 20, "relevance"),
    ("m:kimi-linear", 'ti:"Kimi Linear"', 20, "relevance"),
    ("m:ling", 'ti:"Ling" AND all:"mixture of experts"', 20, "relevance"),
    ("m:nemotron", 'ti:"Nemotron-H" OR ti:"Nemotron-4"', 20, "relevance"),
    ("m:seedoss", 'ti:"Seed-OSS"', 20, "relevance"),
    ("m:phi-moe", 'ti:"Phi-3.5-MoE" OR ti:"Phi-3" AND all:MoE', 20, "relevance"),
    ("m:grok", 'ti:"Grok"', 20, "relevance"),
    ("m:dbrx", 'ti:"DBRX"', 20, "relevance"),
    ("m:arctic", 'ti:"Arctic" AND all:"mixture of experts"', 20, "relevance"),
    # ---- 路由/负载均衡 ----
    ("r:expert-choice", 'all:"expert choice routing"', 40, "relevance"),
    ("r:load-balance", 'all:"load balancing" AND all:"mixture of experts"', 60, "relevance"),
    ("r:aux-loss-free", 'all:"loss-free balancing" OR all:"auxiliary-loss-free"', 40, "relevance"),
    ("r:router-zloss", 'all:"router z-loss" OR all:"router collapse"', 30, "relevance"),
    ("r:soft-moe", 'all:"soft mixture of experts" OR ti:"Soft MoE"', 30, "relevance"),
    ("r:optimal-transport", 'all:"optimal transport" AND all:"mixture of experts"', 30, "relevance"),
    ("r:expert-specialization", 'all:"expert specialization" AND all:"mixture of experts"', 40, "relevance"),
    ("r:dynamic-experts", 'all:"dynamic expert" AND all:"mixture of experts"', 40, "relevance"),
    # ---- 训练/系统 ----
    ("s:expert-parallel", 'all:"expert parallelism"', 60, "relevance"),
    ("s:all-to-all", 'all:"all-to-all" AND all:"mixture of experts"', 50, "relevance"),
    ("s:deepep", 'all:"DeepEP" OR ti:"DeepEP"', 20, "relevance"),
    ("s:eplb", 'all:"EPLB" OR ti:"EPLB"', 20, "relevance"),
    ("s:dualpipe", 'all:"DualPipe"', 20, "relevance"),
    ("s:upcycling", 'all:"upcycling" AND all:"mixture of experts"', 30, "relevance"),
    ("s:fp8-moe", 'all:"FP8" AND all:"mixture of experts"', 40, "relevance"),
    ("s:moe-training", 'all:"mixture of experts" AND all:"training" AND all:"language model"', 80, "relevance"),
    ("s:moe-rl", 'all:"mixture of experts" AND (all:"reinforcement learning" OR all:"RLHF" OR all:"GRPO")', 50, "relevance"),
    # ---- 推理 ----
    ("i:moe-inference", 'all:"MoE inference" OR all:"mixture of experts" AND all:"inference"', 80, "relevance"),
    ("i:offloading", 'all:"expert offloading" OR all:"MoE offloading"', 40, "relevance"),
    ("i:quantization", 'all:"mixture of experts" AND (all:"quantization" OR all:"quantized")', 60, "relevance"),
    ("i:grouped-gemm", 'all:"grouped GEMM" OR all:"fused MoE"', 40, "relevance"),
    ("i:serving", 'all:"mixture of experts" AND (all:"serving" OR all:"scheduling")', 40, "relevance"),
    ("i:speculative", 'all:"mixture of experts" AND all:"speculative decoding"', 30, "relevance"),
    ("i:batch", 'all:"mixture of experts" AND all:"batch size"', 30, "relevance"),
    # ---- 架构前沿 ----
    ("a:fine-grained", 'all:"fine-grained experts" OR all:"fine-grained mixture of experts"', 40, "relevance"),
    ("a:shared-expert", 'all:"shared expert" AND all:"mixture of experts"', 40, "relevance"),
    ("a:long-context", 'all:"mixture of experts" AND all:"long context"', 50, "relevance"),
    ("a:multimodal", 'all:"mixture of experts" AND (all:"multimodal" OR all:"vision-language")', 60, "relevance"),
    ("a:scaling-law", 'all:"mixture of experts" AND all:"scaling law"', 40, "relevance"),
    ("a:hybrid", 'all:"mixture of experts" AND (all:"hybrid" OR all:"linear attention" OR all:"Mamba")', 60, "relevance"),
    ("a:sparse-attention", 'all:"sparse attention" AND all:"mixture of experts"', 40, "relevance"),
    ("a:moe-survey", 'ti:"mixture of experts" AND (ti:"survey" OR ti:"review")', 30, "relevance"),
    # ---- 最新（按时间） ----
    ("new:moe-2026", 'all:"mixture of experts"', 100, "newest"),
    ("new:moe-2025", 'all:"MoE" AND all:"large language model"', 100, "newest"),
    ("new:moe-efficiency", 'all:"mixture of experts" AND (all:"efficient" OR all:"throughput")', 80, "newest"),
]


def fetch(query, max_results, sort_mode):
    sort_by = "submittedDate" if sort_mode == "newest" else "relevance"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "moe-survey/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            root = ET.fromstring(raw)
            out = []
            for e in root.findall("a:entry", NS):
                aid = e.find("a:id", NS).text.strip()
                m = re.search(r"abs/([^v]+)(v\d+)?$", aid)
                base_id = m.group(1) if m else aid
                title = " ".join(e.find("a:title", NS).text.split())
                summ = " ".join(e.find("a:summary", NS).text.split())
                pub = e.find("a:published", NS).text
                authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
                cats = [c.get("term") for c in e.findall("a:category", NS)]
                out.append({"id": base_id, "title": title, "abstract": summ,
                            "published": pub, "authors": authors[:8], "cats": cats})
            return out
        except Exception as ex:
            print(f"  retry {attempt+1}: {ex}", file=sys.stderr)
            time.sleep(5 + attempt * 5)
    return []


def main():
    db = {}
    for label, q, n, mode in QUERIES:
        print(f"[{label}] {q} (n={n}, {mode})", flush=True)
        rows = fetch(q, n, mode)
        print(f"  -> {len(rows)}", flush=True)
        for r in rows:
            if r["id"] not in db:
                r["hits"] = [label]
                db[r["id"]] = r
            else:
                if label not in db[r["id"]]["hits"]:
                    db[r["id"]]["hits"].append(label)
        time.sleep(3.2)
    with open(OUT, "w") as f:
        json.dump(list(db.values()), f, ensure_ascii=False, indent=1)
    print(f"total unique = {len(db)} -> {OUT}")


if __name__ == "__main__":
    main()
