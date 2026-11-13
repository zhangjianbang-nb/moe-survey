#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 大模型深度调研：装配 HTML"""
import importlib
import json
import os
import re
import sys

D = "/home/whu/workspace/ai_workspace/moe-survey-20261103"
sys.path.insert(0, D)

import markdown

CSS = r"""
@page{size:A4;margin:12mm 10mm 15mm 10mm}
*{box-sizing:border-box}
body{font-family:"Noto Sans CJK SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
  margin:0;background:#eef1f5;color:#1b2027;line-height:1.72;font-size:13.6px}
.wrap{max-width:1040px;margin:0 auto;padding:22px 16px 70px}
.cover{background:linear-gradient(152deg,#080d1f 0%,#101a3a 38%,#1b2f6b 72%,#2b4ea8 100%);
  color:#fff;border-radius:16px;padding:54px 50px 40px;margin:0 0 26px;position:relative;overflow:hidden;
  page-break-after:always;break-after:page;min-height:calc(100vh - 120px);display:flex;flex-direction:column}
.cover:before{content:"";position:absolute;right:-90px;top:-70px;width:330px;height:330px;border-radius:50%;
  background:radial-gradient(circle,rgba(96,165,250,.30),rgba(96,165,250,0) 68%)}
.cover:after{content:"";position:absolute;left:-60px;bottom:-110px;width:300px;height:300px;border-radius:50%;
  background:radial-gradient(circle,rgba(251,191,36,.20),rgba(251,191,36,0) 70%)}
.cover .cover-top{display:flex;justify-content:space-between;align-items:baseline;gap:18px;margin-bottom:20px}
.cover .kicker{letter-spacing:.22em;font-size:11px;opacity:.74;text-transform:uppercase}
.cover h1{font-size:31px;line-height:1.34;margin:0 0 12px;font-weight:800;letter-spacing:.01em}
.cover .sub{font-size:14.4px;opacity:.93;margin:0 0 16px;line-height:1.72}
.cover .rule{width:66px;height:4px;background:#fbbf24;border-radius:2px;margin:0 0 16px}
.cover .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:14px}
.cover .stat{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.20);
  border-radius:11px;padding:9px 14px}
.cover .stat b{display:block;font-size:21px;font-weight:800}
.cover .stat span{font-size:11px;opacity:.85}
.cover .cover-quote{font-size:16.6px;line-height:1.7;opacity:.92;margin:0 0 10px;padding-left:14px;
  border-left:3px solid #fbbf24;max-width:700px}
.cover .famstrip{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px;font-size:11px}
.cover .famstrip span{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.18);
  border-radius:999px;padding:4px 11px;opacity:.9}
.cover .meta-block{margin-top:auto;font-size:11.8px;opacity:.92;line-height:1.8}
.cover .meta-block b{opacity:1}
.cover .cover-foot{position:absolute;left:44px;right:44px;bottom:22px;display:flex;justify-content:space-between;
  gap:14px;font-size:10.4px;letter-spacing:.06em;opacity:.55;padding-top:9px;border-top:1px solid rgba(255,255,255,.14)}
.toc{background:#fff;border-radius:11px;padding:14px 20px;margin-bottom:20px;
  box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:12.2px;line-height:1.8;
  page-break-after:always;break-after:page}
.toc .toc-h{font-size:16px;font-weight:800;color:#101a3a;letter-spacing:.08em;margin-bottom:8px}
.toc ol{margin:0 0 0 4px;padding:0;list-style:none;column-count:2;column-gap:30px}
.toc li{margin:2px 0}
.toc a{color:#1b2f6b;text-decoration:none}
.toc .pg{float:right;color:#94a3b8;font-size:10.8px;font-weight:600}
.toc .n{display:inline-block;min-width:34px;color:#94a3b8;font-weight:700}
.toc .sub{font-size:10.8px;color:#64748b;line-height:1.65;margin-bottom:4px}
h2.sec{margin:30px 0 10px;padding-left:11px;border-left:5px solid #1b2f6b;font-size:20px;color:#101a3a}
h3.grp{margin:22px 0 8px;padding:8px 13px;background:#dfe7fb;border-radius:7px;color:#101a3a;font-size:15.5px}
.survey{background:#fff;border-radius:11px;padding:22px 26px;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:13.7px}
.survey h2{color:#101a3a;border-bottom:1px solid #dfe7fb;padding-bottom:4px;margin-top:24px;font-size:18.5px}
.survey h2:first-child{margin-top:2px}
.survey h3{color:#1b3f8f;margin-top:17px;font-size:15.2px}
.survey h4{color:#334155;margin:14px 0 4px;font-size:13.8px}
.survey p{margin:7px 0;orphans:2;widows:2}
.survey ul,.survey ol{margin:6px 0 10px 21px;padding:0}
.survey li{margin:3px 0}
.survey table{border-collapse:collapse;width:100%;font-size:12.4px;margin:9px 0}
.survey table th{background:#e6ecfa;padding:6px 7px;text-align:left;border-bottom:1px solid #cbd5e1}
.survey table td{border-bottom:1px solid #eef1f5;padding:6px 7px;vertical-align:top}
.survey blockquote{margin:9px 0;padding:8px 13px;background:#f8fafc;border-left:4px solid #94a3b8;
  color:#334155;border-radius:0 6px 6px 0}
.survey code{font-family:"DejaVu Sans Mono","Noto Sans Mono CJK SC",monospace;font-size:11.4px;
  background:#f1f5f9;border:1px solid #e2e8f0;border-radius:4px;padding:0 4px}
.survey pre{background:#f1f5f9;border:1px solid #cbd5e1;border-radius:7px;padding:9px 11px;
  font-size:10.6px;line-height:1.6;overflow-x:auto}
.survey pre code{background:none;border:0;padding:0}
.eq{font-family:"Cambria Math","Latin Modern Math",Georgia,"Noto Serif CJK SC",serif;
  background:#f8fafc;border-left:3px solid #64748b;padding:8px 13px;margin:9px 0;font-size:13.6px;
  line-height:2.0;overflow-x:auto;border-radius:0 6px 6px 0}
.eq .tag{float:right;color:#94a3b8;font-size:10.8px;font-family:sans-serif}
.imath{font-family:"Cambria Math","Latin Modern Math",Georgia,"Noto Serif CJK SC",serif;
  font-style:italic;white-space:nowrap;padding:0 1px}
.card{background:#fff;border-radius:11px;padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:2px}
.rank{font-weight:700;color:#94a3b8;font-size:12.5px}
.badge{color:#fff;border-radius:9px;padding:1px 8px;font-size:11px;white-space:nowrap}
.venue{color:#64748b;font-size:12px}
.stars{color:#b45309;font-size:12px;font-weight:700}
.card h3{margin:4px 0 2px;font-size:16.0px;line-height:1.45;color:#101a3a}
.meta{font-size:11.4px;color:#7c8b9c;margin-bottom:4px}
.meta a{color:#1d4ed8;text-decoration:none}
.one{background:#fff8e6;border-left:4px solid #f59e0b;padding:6px 10px;border-radius:4px;
  font-size:13px;margin:8px 0}
table.detail{width:100%;border-collapse:collapse;font-size:12.3px;margin-top:4px}
table.detail th{width:74px;text-align:left;color:#fff;background:#3457c0;padding:4px 7px;
  vertical-align:top;white-space:nowrap;border-radius:2px;font-weight:600}
table.detail td{padding:4px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}
table.detail th.idea{background:#0f766e}
table.detail th.train{background:#7c3aed}
table.detail th.infer{background:#b45309}
table.quick{width:100%;border-collapse:collapse;font-size:11.4px;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}
table.quick th{background:#1b2f6b;color:#fff;padding:6px 6px;text-align:left;font-size:11.4px}
table.quick td{padding:4px 6px;border-bottom:1px solid #eef1f5;vertical-align:top;line-height:1.5}
table.quick td.id{white-space:nowrap;color:#64748b;font-size:10.6px}
table.quick tr:nth-child(even){background:#f8fafc}
table.quick tr.grp td{background:#dfe7fb;color:#101a3a;font-weight:700;font-size:12px;padding:5px 8px}
.qh{font-size:12.6px;color:#475569;margin:0 0 8px}
.endblock{text-align:center;margin-top:36px;padding:26px 18px;background:#fff;border-radius:11px;
  box-shadow:0 1px 3px rgba(0,0,0,.08);page-break-inside:avoid;break-inside:avoid}
.endline{font-size:15px;font-weight:700;color:#101a3a;letter-spacing:.2em;margin-bottom:7px}
.endnote{font-size:11.4px;color:#7c8b9c;line-height:1.9}
footer{text-align:center;color:#94a3b8;font-size:11.4px;margin-top:32px}
@media print{
 .cover{min-height:0;width:auto;height:266mm;border-radius:0;margin:0;padding:30px 38px 0;overflow:hidden;
   page-break-inside:avoid;break-inside:avoid;page-break-after:always;break-after:page}
 .cover .meta-block{margin-top:22px}
 .wrap{padding-top:0;padding-bottom:0}
 body{background:#fff}
 .card,.survey,table.quick{box-shadow:none;border:1px solid #e2e7ee}
 .card{padding:10px 14px;margin:8px 0}
 .card-head,.card h3,.card .one{page-break-after:avoid;break-after:avoid}
 .card h3{font-size:14.6px;margin:2px 0 1px}
 .one{font-size:12.1px;padding:4px 8px;margin:5px 0}
 table.detail{font-size:11.2px}
 table.detail th{padding:3px 6px;width:68px}
 table.detail td{padding:3px 7px}
 .eq,pre,.fig{page-break-inside:avoid;break-inside:avoid}
 tr{page-break-inside:avoid;break-inside:avoid}
 table.quick thead{display:table-header-group}
 h2.sec{page-break-before:always;break-before:page;page-break-after:avoid;break-after:avoid}
 h3.grp{page-break-after:avoid;break-after:avoid}
 .survey h2{page-break-before:always;break-before:page;page-break-after:avoid;break-after:avoid}
 .survey h2:first-child{page-break-before:avoid;break-before:auto}
 .survey h3,.survey h4{page-break-after:avoid;break-after:avoid}
}
"""

# ---------------------------------------------------------------- 公式渲染
FORMULA = {
 "s = x W_g \\in \\mathbb{R}^{E}":
   's = x·W<sub>g</sub> ∈ ℝ<sup>E</sup>'
   '<span class="tag">(2-1) 路由打分</span>',

 "\\mathcal{T} = \\operatorname{TopK}(s, k)":
   '𝒯 = TopK(s, k)'
   '<span class="tag">(2-2) 选中集合</span>',

 "g_i = \\begin{cases} \\dfrac{\\exp(s_i)}{\\sum_{j \\in \\mathcal{T}} \\exp(s_j)} & i \\in \\mathcal{T} \\\\ 0 & \\text{otherwise} \\end{cases}":
   'g<sub>i</sub> = exp(s<sub>i</sub>) ⁄ Σ<sub>j∈𝒯</sub> exp(s<sub>j</sub>)　（i ∈ 𝒯）；　'
   'g<sub>i</sub> = 0　（i ∉ 𝒯）'
   '<span class="tag">(2-3) softmax 门控</span>',

 "y = \\sum_{i \\in \\mathcal{T}} g_i \\cdot \\operatorname{FFN}_i(x)":
   'y = Σ<sub>i∈𝒯</sub> g<sub>i</sub> · FFN<sub>i</sub>(x)'
   '<span class="tag">(2-4) 专家加权求和</span>',

 "g_i = \\frac{\\sigma(s_i)}{\\sum_{j \\in \\mathcal{T}} \\sigma(s_j)}, \\quad i \\in \\mathcal{T}":
   'g<sub>i</sub> = σ(s<sub>i</sub>) ⁄ Σ<sub>j∈𝒯</sub> σ(s<sub>j</sub>)，　i ∈ 𝒯'
   '<span class="tag">(2-5) sigmoid 门控</span>',

 "\\text{capacity} = \\text{CF} \\times \\frac{T_{\\text{batch}} \\cdot k}{E}":
   'capacity = CF × (T<sub>batch</sub> · k) ⁄ E'
   '<span class="tag">(2-6) 专家容量</span>',

 "\\mathcal{L}_{\\text{aux}} = \\alpha \\cdot E \\sum_{i=1}^{E} f_i \\cdot P_i":
   'ℒ<sub>aux</sub> = α · E · Σ<sub>i=1</sub><sup>E</sup> f<sub>i</sub> · P<sub>i</sub>'
   '<span class="tag">(2-7) 负载均衡损失</span>',

 "f_i = \\frac{1}{T}\\sum_{t=1}^{T} \\mathbf{1}\\left[i \\in \\operatorname{TopK}(s_t, k)\\right], \\quad P_i = \\frac{1}{T}\\sum_{t=1}^{T} g_i(x_t)":
   'f<sub>i</sub> = (1/T) Σ<sub>t=1</sub><sup>T</sup> 𝟙[i ∈ TopK(s<sub>t</sub>, k)]，　'
   'P<sub>i</sub> = (1/T) Σ<sub>t=1</sub><sup>T</sup> g<sub>i</sub>(x<sub>t</sub>)'
   '<span class="tag">(2-8) 负载占比与概率质量</span>',

 "\\mathcal{L}_z = \\frac{1}{B}\\sum_{b=1}^{B} \\left(\\log \\sum_{j=1}^{E} \\exp(s_j^{(b)})\\right)^2":
   'ℒ<sub>z</sub> = (1/B) Σ<sub>b=1</sub><sup>B</sup> ( log Σ<sub>j=1</sub><sup>E</sup> '
   'exp(s<sub>j</sub><sup>(b)</sup>) )²'
   '<span class="tag">(2-9) z-loss</span>',

 "V_{\\text{comm}} = 2 \\cdot T \\cdot k \\cdot d \\cdot b":
   'V<sub>comm</sub> = 2 · T · k · d · b'
   '<span class="tag">(2-10) all-to-all 通信量</span>',

 "V = 2 \\times 4096 \\times 8 \\times 7168 \\times 2 \\approx 939\\ \\text{MB / layer}":
   'V = 2 × 4096 × 8 × 7168 × 2 ≈ 939 MB / layer'
   '<span class="tag">(2-11) V3 单层实算</span>',

 "s'_i = s_i + b_i, \\quad \\mathcal{T} = \\operatorname{TopK}(s', k)":
   "s′<sub>i</sub> = s<sub>i</sub> + b<sub>i</sub>，　𝒯 = TopK(s′, k)"
   '<span class="tag">(3-1) 加偏置后选 top-k</span>',

 "b_i \\leftarrow b_i + \\gamma \\cdot \\operatorname{sign}(\\text{load}_i - \\bar{\\text{load}})":
   "b<sub>i</sub> ← b<sub>i</sub> + γ · sign(load<sub>i</sub> − load̄)"
   '<span class="tag">(3-2) 偏置更新（不进梯度）</span>',

 "V = 2 \\cdot T \\cdot k \\cdot d \\cdot b":
   'V = 2 · T · k · d · b'
   '<span class="tag">(7-1) 单层 all-to-all 字节数</span>',

 "AI = \\frac{2 \\cdot W_{\\text{act}}}{W_{\\text{act}} \\cdot b} = \\frac{2}{b}\\ \\text{FLOP/byte}":
   'AI = (2 · W<sub>act</sub>) ⁄ (W<sub>act</sub> · b) = 2 ⁄ b　FLOP/byte'
   '<span class="tag">(7-2) MoE decode 算术强度</span>',

 "\\text{吞吐} \\approx \\frac{\\text{HBM 带宽} \\times \\text{带宽利用率}}{\\text{每 token 读取字节数}}":
   '吞吐 ≈ (HBM 带宽 × 带宽利用率) ⁄ 每 token 读取字节数'
   '<span class="tag">(8-1) 带宽受限下的吞吐上界</span>',

 "\\text{吞吐} \\approx \\frac{3.35 \\times 10^{12} \\times 0.7}{74 \\times 10^9} \\approx 32\\ \\text{token/s}":
   '吞吐 ≈ (3.35 × 10<sup>12</sup> × 0.7) ⁄ (74 × 10<sup>9</sup>) ≈ 32 token/s'
   '<span class="tag">(8-2) 以 V3 decode 为例</span>',
}

SYM = [
 ("\\mathcal{L}", "ℒ"), ("\\mathcal{T}", "𝒯"), ("\\mathcal{O}", "𝒪"),
 ("\\mathbb{R}", "ℝ"), ("\\mathbf{1}", "𝟙"),
 ("\\ll", "≪"), ("\\gg", "≫"), ("\\propto", "∝"), ("\\rightarrow", "→"),
 ("\\leftarrow", "←"), ("\\to", "→"), ("\\Rightarrow", "⇒"), ("\\top", "ᵀ"),
 ("\\cdot", "·"), ("\\times", "×"), ("\\approx", "≈"), ("\\in", "∈"),
 ("\\notin", "∉"), ("\\le", "≤"), ("\\ge", "≥"), ("\\neq", "≠"),
 ("\\sum", "Σ"), ("\\prod", "∏"), ("\\exp", "exp"), ("\\log", "log"),
 ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
 ("\\epsilon", "ε"), ("\\theta", "θ"), ("\\lambda", "λ"), ("\\mu", "μ"),
 ("\\sigma", "σ"), ("\\tau", "τ"), ("\\phi", "φ"), ("\\omega", "ω"),
 ("\\pi", "π"), ("\\rho", "ρ"), ("\\eta", "η"), ("\\kappa", "κ"),
 ("\\ldots", "…"), ("\\cdots", "⋯"), ("\\quad", "　"), ("\\qquad", "　　"),
 ("\\,", " "), ("\\ ", " "), ("\\left", ""), ("\\right", ""),
]


def tex2html(s):
    """把单行 LaTeX 转成可打印的 HTML（上下标 + Unicode 符号）。"""
    s = s.strip()
    # 上下标先做，避免花括号干扰后面的 \frac
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", s)
    # 结构命令
    for _ in range(3):
        s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) ⁄ (\2)", s)
    s = re.sub(r"\\(?:text|operatorname|mathrm|mathcal|mathbb|mathbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", "\\1\u0304", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", "\\1\u0302", s)
    # 符号
    for a, b in SYM:
        s = s.replace(a, b)
    # 残余清理
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mathify(src):
    """把 $...$ 与 $$...$$ 转成可直接打印的 HTML（不依赖 MathJax）。"""
    def disp(m):
        raw = m.group(1).strip()
        if raw in FORMULA:
            return '<div class="eq">' + FORMULA[raw] + "</div>"
        return '<div class="eq">' + tex2html(raw) + "</div>"

    def inl(m):
        return '<span class="imath">' + tex2html(m.group(1)) + "</span>"

    src = re.sub(r"\$\$(.+?)\$\$", disp, src, flags=re.S)
    src = re.sub(r"(?<!\$)\$([^$\n]{1,90})\$(?!\$)", inl, src)
    return src


# ---------------------------------------------------------------- 章节
MODULES = ["s01_exec", "s02_foundation", "s03_routing", "s04_evolution",
           "s05_models", "s06_training", "s07_inference", "s08_hardware",
           "s09_frontier", "s10_industry"]

CARD_GROUPS = [
    (0, 3, "综述与总览", "#475569"),
    (3, 12, "奠基与路由机制", "#0f766e"),
    (12, 18, "训练系统与并行", "#7c3aed"),
    (18, 34, "模型架构代际", "#1d4ed8"),
    (34, 46, "推理系统与部署", "#b45309"),
    (46, 50, "量化与压缩", "#be123c"),
    (50, 53, "多模态与注意力", "#0891b2"),
    (53, 56, "结构创新与均衡细节", "#4338ca"),
]


def card_group(i):
    for a, b, name, color in CARD_GROUPS:
        if a <= i < b:
            return name, color
    return "其他", "#475569"


def clean_title(t):
    t = t.replace("$^{2}$", "²").replace("$^{3}$", "³").replace("$_{2}$", "₂")
    return t.replace("$", "").replace("\\", "")


def load_quick():
    """解析 quick_data.py，保留主题分组注释。"""
    txt = open(os.path.join(D, "quick_data.py"), encoding="utf-8").read()
    raw = {p["id"].split("v")[0]: p for p in json.load(open(os.path.join(D, "arxiv_raw.json")))}
    rows, group = [], "其他"
    for ln in txt.split("\n"):
        s = ln.strip()
        if s.startswith("#") and not s.startswith("#!"):
            group = s.lstrip("# ").strip()
            continue
        m = re.match(r'\("(\d{4}\.\d{4,5})",\s*"(.*?)",\s*(\d{4}),\s*"(.*?)"\)', s)
        if m:
            pid, title, year, one = m.groups()
            canon = clean_title(raw.get(pid, {}).get("title", title).replace("\n", " "))
            rows.append((group, pid, canon, year, one))
    return rows


def render_toc(tokens):
    pages = {}
    pj = os.path.join(D, "toc_pages.json")
    if os.path.exists(pj):
        pages = json.load(open(pj, encoding="utf-8"))
    out = ['<div class="toc"><div class="toc-h">目 录</div><ol>']
    for t in tokens:
        if t["level"] != 2:
            continue
        pg = pages.get(t["name"], "")
        pgs = f'<span class="pg">{pg}</span>' if pg else ""
        out.append(f'<li><a href="#{t["id"]}">{t["name"]}</a>{pgs}</li>')
        subs = [c for c in t.get("children", []) if c["level"] == 3]
        if subs:
            out.append('<li class="sub">' + " · ".join(
                f'<a href="#{c["id"]}">{c["name"]}</a>' for c in subs) + "</li>")
    for name, anchor in [("附录 A · 术语表", "glossary"),
                         ("附录 B · 56 篇核心论文深度卡", "cards"),
                         ("附录 C · 80 篇相关工作速览", "quick")]:
        pg = pages.get(name, "")
        pgs = f'<span class="pg">{pg}</span>' if pg else ""
        out.append(f'<li><a href="#{anchor}">{name}</a>{pgs}</li>')
    out.append("</ol></div>")
    return "\n".join(out)


def render_cards():
    import cards_data
    importlib.reload(cards_data)
    out = ['<h2 class="sec" id="cards">附录 B · 56 篇核心论文深度卡</h2>',
           '<p class="qh">按主题分组，每张卡给出「一句话结论 / 核心问题 / 方法 / 数据与实验 / 局限」。'
           '卡片顺序即阅读顺序：先综述建立框架，再看奠基工作、训练系统、模型代际、推理系统、量化、'
           '多模态与理论。</p>']
    last = None
    for i, c in enumerate(cards_data.CARDS):
        pid, title, org, year, stars, one, core, method, data, limit = c
        gname, gcolor = card_group(i)
        if gname != last:
            out.append(f'<h3 class="grp">{gname}（{sum(1 for j in range(len(cards_data.CARDS)) if card_group(j)[0]==gname)} 篇）</h3>')
            last = gname
        out.append(
            '<div class="card">'
            f'<div class="card-head"><span class="rank">#{i+1:02d}</span>'
            f'<span class="badge" style="background:{gcolor}">{gname}</span>'
            f'<span class="venue">{year} · {org}</span>'
            f'<span class="stars">★ {stars}.0</span></div>'
            f'<h3>{clean_title(title)}</h3>'
            f'<div class="meta">arXiv:{pid} · <a href="https://arxiv.org/abs/{pid}">arxiv.org/abs/{pid}</a></div>'
            f'<div class="one">{one}</div>'
            '<table class="detail">'
            f'<tr><th>核心问题</th><td>{core}</td></tr>'
            f'<tr><th class="idea">方法</th><td>{method}</td></tr>'
            f'<tr><th class="train">数据/实验</th><td>{data}</td></tr>'
            f'<tr><th class="infer">局限</th><td>{limit}</td></tr>'
            "</table></div>")
    return "\n".join(out)


def render_quick():
    rows = load_quick()
    out = ['<h2 class="sec" id="quick">附录 C · 80 篇相关工作速览</h2>',
           f'<p class="qh">共 {len(rows)} 篇，按主题分组；每篇一句中文结论，便于按方向检索。'
           '标题取自 arXiv 元数据。</p>',
           '<table class="quick"><thead><tr><th style="width:22px">#</th>'
           '<th style="width:78px">arXiv</th><th style="width:56px">年份</th>'
           '<th>论文标题</th><th style="width:250px">一句话</th></tr></thead><tbody>']
    last, n = None, 0
    for group, pid, title, year, one in rows:
        if group != last:
            out.append(f'<tr class="grp"><td colspan="5">{group}</td></tr>')
            last = group
        n += 1
        out.append(f'<tr><td>{n}</td><td class="id">{pid}</td><td>{year}</td>'
                   f'<td>{title}</td><td>{one}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def cover():
    return """
<div class="cover">
  <div class="cover-top">
    <div class="kicker">Deep Survey · Mixture-of-Experts</div>
    <div class="cover-date">2026-11-03</div>
  </div>
  <h1>MoE 大模型深度调研<br>原理、架构演进、训练系统与推理部署</h1>
  <div class="rule"></div>
  <div class="sub">从 GShard 到 DeepSeek-V4.1 / Kimi K3 / GLM-5：<br>
  稀疏路由的数学、八次瓶颈迁移、EP 通信与 all-to-all、量化与分离式部署</div>
  <div class="stats">
    <div class="stat"><b>1337</b><span>arXiv 检索候选</span></div>
    <div class="stat"><b>56</b><span>篇深度精读</span></div>
    <div class="stat"><b>80</b><span>篇速览</span></div>
    <div class="stat"><b>11</b><span>章正文</span></div>
  </div>
  <div class="cover-quote">MoE 的本质，是把「参数容量」与「每 token 计算量」解耦。<br>
  它省算力，但不省显存——这是所有推理工程难题的起点。</div>
  <div class="famstrip">
    <span>GShard</span><span>Switch</span><span>GLaM</span><span>ST-MoE</span><span>Mixtral</span>
    <span>DeepSeekMoE</span><span>DeepSeek-V2 / V3 / V3.2</span><span>DeepSeek-V4 / V4.1</span>
    <span>Qwen3 / Qwen3.8-Next</span><span>Kimi K2 / K3</span><span>GLM-4.5 / GLM-5</span>
    <span>MiniMax-M2</span><span>Step-3</span><span>LongCat-Flash</span><span>Llama 4</span>
    <span>OLMoE</span><span>Nemotron-H</span>
  </div>
  <div class="meta-block">
    <b>覆盖范围</b>　1991 年自适应 Mixture of Experts → 2026 年 10 月最新技术报告<br>
    <b>数据来源</b>　arXiv 元数据与全文、官方技术报告、开源实现（DeepEP / EPLB / Tutel / MegaBlocks）<br>
    <b>面向读者</b>　推理引擎、CUDA 算子与系统工程师；假设读者熟悉 Transformer 与分布式训练基础
  </div>
  <div class="cover-foot">
    <span>MoE SURVEY · 2026-11-03</span>
    <span>56 deep reads · 80 quick scans · 1337 candidates</span>
  </div>
</div>
"""


def endblock():
    return """
<div class="endblock">
  <div class="endline">报 告 完</div>
  <div class="endnote">
    本报告基于截至 2026-11-03 的公开论文与技术报告整理；模型参数、训练成本等数字均标注来源口径，<br>
    未披露的数据不做推测。arXiv 编号可直接在 arxiv.org/abs/&lt;id&gt; 访问原文。<br>
    报告由 ZCode 生成，正文与卡片数据全部落盘于 ai_workspace/moe-survey-20261103。
  </div>
</div>
"""


def build():
    md = markdown.Markdown(extensions=["tables", "toc", "sane_lists", "md_in_html"])
    body_parts = []
    for m in MODULES:
        txt = open(os.path.join(D, "sections", m + ".md"), encoding="utf-8").read()
        body_parts.append(mathify(txt))
    body_md = "\n\n".join(body_parts)
    body_html = md.convert(body_md)
    toc_html = render_toc(md.toc_tokens)

    md2 = markdown.Markdown(extensions=["tables", "sane_lists"])
    gloss = md2.convert(mathify(open(os.path.join(D, "sections/s00_glossary.md"),
                                     encoding="utf-8").read()))

    parts = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>MoE 大模型深度调研 · 2026-11-03</title>'
             f"<style>{CSS}</style></head><body><div class=\"wrap\">"]
    parts.append(cover())
    parts.append(toc_html)
    parts.append('<div class="survey">')
    parts.append(body_html)
    parts.append('</div>')
    parts.append('<h2 class="sec" id="glossary">附录 A · 术语表</h2>')
    parts.append('<div class="survey">' + gloss + "</div>")
    parts.append(render_cards())
    parts.append(render_quick())
    parts.append(endblock())
    parts.append("</div></body></html>")

    out = os.path.join(D, "MOE_SURVEY.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("written:", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    build()
