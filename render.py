#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTML → PDF（Chromium via playwright，带页码页脚）"""
import os
from playwright.sync_api import sync_playwright

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
EXE = "/home/whu/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome"
HTML = os.path.join(D, "MOE_SURVEY.html")
PDF = os.path.join(D, "MOE_SURVEY.pdf")

FOOTER = ('<div style="width:100%;font-size:8px;color:#94a3b8;'
          'font-family:Noto Sans CJK SC,sans-serif;padding:0 10mm;'
          'display:flex;justify-content:space-between;">'
          '<span>MoE 大模型深度调研 · 2026-11-03 · 56 篇深度 + 80 篇速览</span>'
          '<span>第 <span class="pageNumber"></span> 页 / 共 <span class="totalPages"></span> 页</span>'
          '</div>')
HEADER = '<div></div>'

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=["--no-sandbox", "--disable-gpu"])
    pg = b.new_page(viewport={"width": 1100, "height": 1500})
    pg.goto("file://" + HTML, wait_until="load")
    pg.wait_for_timeout(1500)
    pg.pdf(path=PDF, format="A4", print_background=True,
           display_header_footer=True, header_template=HEADER, footer_template=FOOTER,
           margin={"top": "12mm", "bottom": "15mm", "left": "10mm", "right": "10mm"})
    b.close()
print("pdf:", PDF, os.path.getsize(PDF), "bytes")
