#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF 可编程断言：页数/空白/溢出/关键串/墨水覆盖"""
import os
import re
import sys

import pymupdf

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
PDF = os.path.join(D, "MOE_SURVEY.pdf")
doc = pymupdf.open(PDF)
N = doc.page_count
W, H = doc[0].rect.width, doc[0].rect.height
fails, warns = [], []


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


# 1) 页数
if not (45 <= N <= 80):
    fail(f"页数 {N} 不在 45-80")

# 2) 每页墨水覆盖率 + 文本边界
ink, bbox_bad, blank, bottom_gap = [], [], [], []
for i, pg in enumerate(doc):
    pix = pg.get_pixmap(dpi=40, colorspace=pymupdf.csGRAY)
    data = pix.samples
    dark = sum(1 for b in data if b < 245)
    ratio = dark / len(data)
    ink.append(ratio)
    if ratio < 0.004:
        blank.append(i + 1)
    blocks = [b for b in pg.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if blocks:
        x0 = min(b[0] for b in blocks); x1 = max(b[2] for b in blocks)
        y1 = max(b[3] for b in blocks)
        if x1 > W - 8 or x0 < 8:
            bbox_bad.append((i + 1, round(x0, 1), round(x1, 1)))
        gap = (H - 40) - y1   # 页脚上方 40pt 视为页脚区
        bottom_gap.append((i + 1, round(gap, 1)))
    else:
        bottom_gap.append((i + 1, 0))

if blank:
    fail(f"疑似空白页: {blank}")
if bbox_bad:
    fail(f"文本越界页: {bbox_bad[:8]}")

# 大面积底部留白（> 260pt 且非章节末页）——仅提示
big_gap = [(p, g) for p, g in bottom_gap if g > 260]
if big_gap:
    warn(f"底部留白 >260pt 的页: {big_gap[:12]}")

# 3) 关键串
full = "\n".join(pg.get_text() for pg in doc)
KEYS = [
    "MoE 大模型深度调研", "执行摘要", "术语表",
    "为什么 MoE 成为", "MoE 的数学基础", "路由与负载均衡", "架构演进",
    "模型代际全景", "训练系统", "推理系统", "硬件协同", "前沿方向",
    "产业落地", "趋势判断",
    "56 篇核心论文深度卡", "80 篇相关工作速览",
    "DeepSeek-V3", "DeepSeek-V4", "Kimi K3", "GLM-5", "Qwen3",
    "all-to-all", "EPLB", "DeepEP", "aux-loss-free", "MuonClip",
    "grouped GEMM", "PD 分离", "roofline", "FP8", "FP4",
    "2.788M H800", "939 MB", "2Tkdb",
]
for k in KEYS:
    if k not in full:
        fail(f"缺少关键串: {k}")

# 4) 深度卡与速览数量
n_arxiv = len(re.findall(r"arXiv:\d{4}\.\d{4,5}", full))
if n_arxiv < 56:
    fail(f"深度卡 arXiv 编号仅 {n_arxiv} 个（应 56）")
quick_ids = len(re.findall(r"\b(?:25|26)\d{2}\.\d{4,5}\b", full))
if quick_ids < 80:
    fail(f"速览 arXiv 编号仅 {quick_ids} 个（应 ≥80）")

# 5) 页脚
for i, pg in enumerate(doc):
    t = pg.get_text()
    if f"第 {i+1} 页" not in t:
        fail(f"第 {i+1} 页缺页脚页码")
        break

# 6) 封面/目录
if "1337" not in doc[0].get_text() or "56" not in doc[0].get_text():
    fail("封面统计数字缺失")
if "目 录" not in doc[1].get_text():
    fail("第 2 页不是目录页")

print(f"pages={N}  ink_min={min(ink):.4f}  ink_mean={sum(ink)/N:.4f}")
print(f"blank={blank}  bbox_bad={bbox_bad}  arxiv_cards={n_arxiv}  quick_ids={quick_ids}")
print(f"WARN {len(warns)}:")
for w in warns:
    print("  -", w)
print(f"FAIL {len(fails)}:")
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
