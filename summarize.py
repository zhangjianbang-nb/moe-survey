#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为每篇论文生成压缩摘要：标题/摘要/关键句/表格行，便于批量精读"""
import glob, json, os, re, sys

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
PDF = os.path.join(D, "pdfs")
OUT = os.path.join(D, "notes", "condensed")
os.makedirs(OUT, exist_ok=True)

KEY_PATS = [
    r"[^.]*\b(total|activated)\b[^.]*param[^.]*\.",
    r"[^.]*\bexperts?\b[^.]*\.",
    r"[^.]*\b(top-?\d+|top-?k)\b[^.]*\.",
    r"[^.]*\b(tokens?|token)\b[^.]*\b(trillion|T)\b[^.]*\.",
    r"[^.]*\b(context|sequence length)\b[^.]*\.",
    r"[^.]*\b(GPU|H800|H100|B200|A100)\b[^.]*\b(hours|MFU)\b[^.]*\.",
    r"[^.]*\b(throughput|latency|tokens/s|TFLOPS)\b[^.]*\.",
    r"[^.]*\b(load balanc|auxiliary loss|aux-loss-free|capacity factor|token drop)[^.]*\.",
    r"[^.]*\b(all-to-all|expert parallel|EP|TP|PP)\b[^.]*\.",
]


def sentences(text):
    # 简单分句：句号/换行；保留长度 40~400 的句子
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip().replace("\n", " ") for p in parts if 40 <= len(p.strip()) <= 400]


def main():
    files = sorted(glob.glob(os.path.join(PDF, "*.txt")))
    for txt in files:
        aid = os.path.basename(txt)[:-4]
        t = open(txt).read()
        head = t[:4000]
        body = t[4000:60000]
        # 摘要：取 Abstract 后到 Introduction 前
        m = re.search(r"abstract", head, re.I)
        abstract = head[m.start():m.start() + 1400] if m else head[:1200]
        sents = sentences(body)
        picked = []
        for p in KEY_PATS:
            for s in sents:
                if re.search(p, s, re.I) and s not in picked:
                    picked.append(s)
                    if len(picked) >= 14:
                        break
            if len(picked) >= 14:
                break
        # 表格：找含 "Table" 的行及其后 3 行
        tbl = []
        lines = t.split("\n")
        for i, ln in enumerate(lines):
            if re.match(r"^(Table|TABLE)\s*\d", ln.strip()) and len(tbl) < 3:
                tbl.append(" | ".join(x.strip() for x in lines[i:i + 5] if x.strip())[:600])
        with open(os.path.join(OUT, aid + ".md"), "w") as f:
            f.write(f"# {aid}\n\n## 摘要\n{abstract}\n\n## 关键句\n")
            for s in picked[:12]:
                f.write(f"- {s}\n")
            if tbl:
                f.write("\n## 表格片段\n")
                for s in tbl:
                    f.write(f"- {s}\n")
    print(f"condensed {len(files)} papers -> {OUT}")


if __name__ == "__main__":
    main()
