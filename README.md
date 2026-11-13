# MoE 大模型深度调研

**原理、架构演进、训练系统与推理部署** —— 一份面向推理引擎与系统工程师的中文技术综述，外加一份写给所有人看的「故事书」网页。

- 检索 **1337 篇** arXiv 候选论文 → 深度精读 **56 篇** + 速览 **80 篇**
- **59 页** A4 PDF，11 章正文 + 术语表 + 56 张深度卡片
- 配套故事书网页：用「一家医院」的比喻讲完 MoE 十年，含 7 篇专业导读、17 条术语小抄

## 在线阅读

| 内容 | 文件 |
|---|---|
| 综述 PDF（59 页） | [`MOE_SURVEY.pdf`](MOE_SURVEY.pdf) |
| 综述网页版（与 PDF 同源） | [`MOE_SURVEY.html`](MOE_SURVEY.html) |
| 故事书网页（精读版，零依赖） | [`story/index.html`](story/index.html) |

> 两个 HTML 都是**单文件、零外部依赖**（无外链 CSS/JS/图片），双击即可离线阅读，也可直接部署到任意静态托管。

## 核心结论

MoE 把**参数容量**与**每 token 计算量**解耦：省算力，但不省显存。

报告把 MoE 的十年归纳为一部**瓶颈迁移史**——每一代架构解决一个核心矛盾，同时把瓶颈推到下一个位置：

```
容量 → 路由 → 通信 → 显存 → 带宽 → 精度 → 调度 → 存储
```

- DeepSeek-V3 训练 2.788M H800 小时；V4.1 推理约 890 B/token
- EP/all-to-all 通信量约 939 MB/layer，是 MoE 推理的第一瓶颈
- 推理优化主线：grouped GEMM、EPLB、PD 分离、FP8/FP4

## 仓库结构

```
.
├── MOE_SURVEY.pdf / .html      # 综述成品（59 页）
├── sections/                   # 11 章 Markdown 源（含术语表）
├── notes/condensed/            # 57 篇精读论文的结构化笔记（233 KB）
├── story/                      # 故事书网页 v2（index.html）+ v1 备份
│   ├── index.html              #   精读版：序 + 6 章 + 尾声 + 数字速查 + 术语小抄
│   ├── index-v1.html           #   初版 40 KB 六章故事
│   └── scripts/                #   断言脚本 verify.js / 截图脚本 shoot.js
├── *.py                        # 检索 → 打分 → 下载 → 抽取 → 装配 → 分页 → 校验流水线
├── keydata.json                # 从论文中抽取的关键数据（参数/成本/通信量等）
├── toc_pages.json              # 目录页码映射
├── deep_ids.txt                # 56 篇精读论文的 arXiv ID 清单
└── verify.py                   # 成品校验（页数/空白/越界/卡片编号）
```

## 复现流水线

```bash
python3 fetch_arxiv.py      # 拉取 arXiv 候选（1337 篇）
python3 score.py            # 关键词打分排序
python3 download.py         # 下载入选论文 PDF
python3 extract_keydata.py  # 抽取关键数据 → keydata.json
python3 summarize.py        # 精读笔记 → notes/condensed/
python3 make_report.py      # 装配 HTML（CSS / 公式 / 卡片 / 速览 / 目录）
python3 render.py           # 渲染
python3 paginate.py         # 分页 + 目录页码
python3 verify.py           # 断言校验（59 页、0 空白、0 越界）
```

> 流水线依赖 `cards_data.py`（56 张深度卡）与 `quick_data.py`（80 条速览）两套人工整理的数据。
> 原始论文 PDF 与渲染预览图未纳入仓库（体积与版权原因）。

## 故事书网页

`story/index.html` 是同一批 100 篇论文的通俗版本：把 MoE 写成一家医院——分诊台是 router，256 个科室是总参数，每位病人只挂 8 个号（激活参数），跑腿费是 all-to-all 通信，医生档案压缩是量化，门诊/住院分楼是分离式部署。

网页自带明暗主题、阅读进度条、目录高亮与移动端胶囊导航，`node story/scripts/verify.js` 可跑 26 项结构断言。

## 引用

```bibtex
@techreport{moe-survey-2026,
  title  = {MoE 大模型深度调研：原理、架构演进、训练系统与推理部署},
  author = {Zhang, Jianbang},
  year   = {2026},
  note   = {59-page Chinese technical survey, 1337 arXiv candidates, 56 deep-read papers},
  url    = {https://github.com/zhangjianbang-nb/moe-survey}
}
```

## 许可

- 代码（`*.py`、`story/scripts/`）：MIT，见 [LICENSE](LICENSE)
- 文档与网页正文（PDF / HTML / `sections/` / `notes/`）：CC BY 4.0

论文版权归各原作者所有；本仓库仅收录笔记与综述文本，不收录论文原文 PDF。
