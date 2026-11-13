#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按候选列表下载 arXiv PDF（断点续传 + 限速），并提取全文 txt"""
import json, os, subprocess, sys, time, urllib.request

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
PDF = os.path.join(D, "pdfs")
CAND = os.path.join(D, "candidates.json")
PICK = os.path.join(D, "deep_ids.txt")   # 每行一个 arxiv id

def dl(aid):
    out = os.path.join(PDF, aid.replace("/", "_") + ".pdf")
    if os.path.exists(out) and os.path.getsize(out) > 30000:
        return out, "skip"
    url = f"https://arxiv.org/pdf/{aid}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                raise RuntimeError("not a pdf")
            open(out, "wb").write(data)
            return out, f"ok {len(data)}"
        except Exception as e:
            print(f"  retry {attempt+1} {aid}: {e}", flush=True)
            time.sleep(4 + attempt * 4)
    return None, "fail"

def extract(p):
    txt = p[:-4] + ".txt"
    if os.path.exists(txt) and os.path.getsize(txt) > 2000:
        return txt, "skip"
    try:
        import pymupdf
        doc = pymupdf.open(p)
        t = "\n".join(pg.get_text() for pg in doc)
        open(txt, "w").write(t)
        return txt, f"ok {len(t)}"
    except Exception as e:
        return None, f"fail {e}"

def main():
    ids = [l.strip() for l in open(PICK) if l.strip() and not l.startswith("#")]
    print(f"to download: {len(ids)}", flush=True)
    ok = fail = 0
    for i, aid in enumerate(ids):
        p, msg = dl(aid)
        if p:
            ok += 1
            _, m2 = extract(p)
            print(f"[{i+1}/{len(ids)}] {aid} {msg} | {m2}", flush=True)
        else:
            fail += 1
            print(f"[{i+1}/{len(ids)}] {aid} FAIL", flush=True)
        time.sleep(1.2)
    print(f"done ok={ok} fail={fail}", flush=True)

if __name__ == "__main__":
    main()
