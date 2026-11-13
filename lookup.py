#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按标题精确搜索 arXiv，输出 id + title"""
import sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
TITLES = [
    "BASE Layers: Simplifying Training of Large, Sparse Models",
    "Hash Layers For Large Sparse Models",
    "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts",
    "Tutel: Adaptive Mixture-of-Experts at Scale",
    "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models",
    "DBRX: A New Open Source LLM",
    "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts",
    "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model",
    "Kimi K3",
    "GLM-5",
    "DeepSeek-V3.2",
    "Mixtral 8x22B",
    "FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models",
    "MoE-Infinity: Offloading-Efficient MoE Model Serving",
    "KTransformers: A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations",
    "Janus: Disaggregating Attention and Experts",
    "DeepEP",
    "EPLB",
    "MoE-Lightning",
]

def search(q):
    p = {"search_query": f'ti:"{q}"', "start": 0, "max_results": 5,
         "sortBy": "relevance", "sortOrder": "descending"}
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(p)
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            root = ET.fromstring(r.read())
        out = []
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.split("/abs/")[-1]
            t = " ".join(e.find("a:title", NS).text.split())
            out.append((aid, t))
        return out
    except Exception as ex:
        return [("ERR", str(ex))]

for t in TITLES:
    print(f"\n## {t}")
    for aid, title in search(t):
        print(f"  {aid:24s} {title[:100]}")
    time.sleep(3.2)
