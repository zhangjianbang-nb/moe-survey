#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二遍：从 PDF 反查各章起始页，写入 toc_pages.json"""
import json
import os
import re

import pymupdf

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
PDF = os.path.join(D, "MOE_SURVEY.pdf")
HTML = os.path.join(D, "MOE_SURVEY.html")

html = open(HTML, encoding="utf-8").read()
names = []
for m in re.finditer(r'<h2[^>]*>(.*?)</h2>', html, re.S):
    txt = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if txt and txt not in ("目 录",):
        names.append(txt)

doc = pymupdf.open(PDF)
pages = {}
for name in names:
    key = name[:24]
    for i in range(2, doc.page_count):   # 跳过封面与目录
        t = doc[i].get_text()
        if key in t:
            pages[name] = i + 1
            break

old = {}
pj = os.path.join(D, "toc_pages.json")
if os.path.exists(pj):
    old = json.load(open(pj, encoding="utf-8"))

json.dump(pages, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("entries:", len(pages))
for k, v in pages.items():
    flag = "" if old.get(k) == v else "  <-- changed"
    print(f"  {v:>3}  {k[:56]}{flag}")
