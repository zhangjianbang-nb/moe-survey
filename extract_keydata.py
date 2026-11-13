#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从每篇 PDF 文本中抽取关键定量数据，输出 keydata.json 供人工核对"""
import glob, json, os, re

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103/pdfs"
OUT = "/home/whu/workspace/ai_workspace/moe-survey-20261103/keydata.json"

PATS = {
    "total_params": [
        r"(\d[\d,\.]*)\s*(?:B|billion)\s*(?:total\s*)?param",
        r"total\s*(?:of\s*)?(\d[\d,\.]*)\s*(?:B|billion)\s*param",
        r"(\d[\d,\.]*)\s*(?:B|billion)[- ]parameter",
    ],
    "act_params": [
        r"(\d[\d,\.]*)\s*(?:B|billion)\s*(?:param\w*\s*)?activat",
        r"activat\w*\s*(?:with\s*)?(\d[\d,\.]*)\s*(?:B|billion)",
        r"(\d[\d,\.]*)\s*(?:B|billion)\s*activated",
    ],
    "experts": [
        r"(\d[\d,]*)\s*(?:routed\s*)?experts",
        r"(\d[\d,]*)\s*experts?\s*(?:in total|per layer)",
    ],
    "act_experts": [
        r"(?:activates?|selects?|chooses?|top-?)\s*(\d[\d,]*)\s*(?:out of\s*\d+\s*)?(?:routed\s*)?experts",
        r"(\d[\d,]*)\s*experts?\s*(?:are\s*)?activat",
    ],
    "tokens": [
        r"(\d[\d,\.]*)\s*(?:T|trillion)\s*(?:high-quality\s*)?tokens",
        r"(\d[\d,\.]*)\s*(?:T|trillion)\s*tokens",
    ],
    "context": [
        r"(\d[\d,\.]*)\s*(?:K|k)\s*(?:context|tokens?)\s*(?:length|window)?",
        r"context\s*(?:length|window)\s*(?:of\s*)?(\d[\d,\.]*)\s*(?:K|k)",
        r"(\d[\d,\.]*)\s*(?:million|M)\s*(?:tokens?\s*)?context",
    ],
    "gpu_hours": [
        r"([\d,\.]+)\s*(?:H800|H100|A100|B200|GPU)[- ]?(?:GPU[- ]?)?hours",
        r"([\d,\.]+)\s*GPU\s*hours",
    ],
    "mfu": [
        r"(\d[\d\.]*)\s*%\s*(?:MFU|model\s*FLOPs\s*utilization)",
    ],
}


def find(text, pats, maxn=3):
    out = []
    for p in pats:
        for m in re.finditer(p, text, re.I):
            s = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            out.append(s.strip())
            if len(out) >= maxn:
                return out
    return out


def main():
    res = {}
    for txt in sorted(glob.glob(os.path.join(D, "*.txt"))):
        aid = os.path.basename(txt)[:-4]
        t = open(txt).read()
        # 只看前 6 页 + 全文（参数通常在摘要/表1）
        head = t[:30000]
        rec = {"id": aid, "chars": len(t)}
        for k, pats in PATS.items():
            rec[k] = find(head, pats)
        res[aid] = rec
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"papers={len(res)} -> {OUT}")
    for aid, r in list(res.items())[:6]:
        print(aid, {k: v[:1] for k, v in r.items() if k != "chars" and v})


if __name__ == "__main__":
    main()
