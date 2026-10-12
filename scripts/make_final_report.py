"""Stage: make_final_report. Tier: cpu. Builds the final deliverable folder from artifacts_committed/final/:
  reports/final_<date>/data/      tidy CSV + JSON for analysis / visualisation (+ DATA_DICTIONARY.md)
  reports/final_<date>/figures/   PNG figures (300 dpi) used in the PDF
  reports/final_<date>/AuscultRAG_results_report.pdf
Run scripts/aggregate_results.py first. Usage: python scripts/make_final_report.py [--out reports/final_2026-10-10]
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auscultrag.config import ARTIFACT_ROOT  # noqa: E402

FINAL = ROOT / "artifacts_committed" / "final"
OUT = ROOT / "reports" / f"final_{date.today().isoformat()}"
FONT = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"
TASKS = ["circor_outcome", "icbhi_adventitious", "icbhi_chronic", "kauh_adventitious", "sprsound_adventitious",
         "tbscreen_cough", "coswara_cough", "coswara_respiratory"]
SHORT = {"circor_outcome": "CirCor", "icbhi_adventitious": "ICBHI-adv", "icbhi_chronic": "ICBHI-chr",
         "kauh_adventitious": "KAUH", "sprsound_adventitious": "SPRSound", "tbscreen_cough": "TBScreen",
         "coswara_cough": "Coswara-cough", "coswara_respiratory": "Coswara-resp"}
# reference palette (dataviz skill, references/palette.md): first three categorical slots validate all-pairs
# in light mode; ordinal blue ramp steps 250/450/650; neutral gray for the no-evidence baseline
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9c9a93"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------------------------- data
def ci(x):
    try:
        lo, hi = ast.literal_eval(x) if isinstance(x, str) else x
        return float(lo), float(hi)
    except Exception:
        return np.nan, np.nan


def tidy() -> pd.DataFrame:
    frames = []
    for exp in ("main", "controls", "xcorpus", "textret", "robustness"):
        d = pd.read_csv(FINAL / f"{exp}.csv")
        d["experiment"] = exp
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d[["ci95_low", "ci95_high"]] = d.report_score_ci95.apply(lambda x: pd.Series(ci(x)))
    d["perturbation"] = d.condition.str.split("@").str[1].where(d.experiment == "robustness")
    d["input"] = np.where(d.run.str.contains("_text"), "text-only",
                          np.where(d.model.str.contains("extract"), "audio (two-step)",
                                   np.where(d.model == "kNN", "retrieval vote", "audio")))
    d = d.drop_duplicates(["experiment", "run", "model", "condition", "task"])
    cols = ["experiment", "model", "input", "condition", "perturbation", "task", "report_score", "ci95_low", "ci95_high",
            "n_queries", "n_informative_fields", "json_ok_rate", "complete_rate", "coverage", "run"]
    return d[cols]


def extras() -> dict[str, pd.DataFrame]:
    sp = pd.read_csv(FINAL / "specialist_probe.csv")
    spx = pd.read_csv(FINAL / "specialist_probe_xcorpus.csv")
    enc = pd.read_csv(FINAL / "encoders.csv")
    car = pd.read_csv(FINAL / "careaqa_supported_fields.csv")
    rk = pd.read_csv(FINAL / "robustness_knn.csv")
    rk[["ci95_low", "ci95_high"]] = rk.report_score_ci95.apply(lambda x: pd.Series(ci(x)))
    rk = rk.drop(columns=["report_score_ci95"])
    ev = pd.read_csv(ROOT / "artifacts_committed" / "analysis" / "accuracy_by_agreement.csv")
    ov = pd.read_csv(ROOT / "artifacts_committed" / "analysis" / "override_analysis.csv")
    return {"specialist_probe": sp, "specialist_probe_xcorpus": spx, "encoder_ablation_knn": enc,
            "careaqa_supported_fields": car, "robustness_knn_all_perturbations": rk,
            "evidence_use_accuracy_by_agreement": ev, "evidence_use_overrides": ov}


def field_scores() -> pd.DataFrame:
    rows = []
    for run, model in [("report_test_v3", m) for m in ["gpt-audio-2025-08-28", "gpt-audio-1.5", "gemini-3-flash-preview",
                                                       "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]] + \
                      [("report_test_v3_text", m) for m in ["gpt-5.4-2026-03-05", "gemini-3.5-flash", "medgemma-4b", "medgemma-27b"]]:
        p = ARTIFACT_ROOT / "llm" / run / f"report_field_scores_{model}.csv"
        if p.exists():
            f = pd.read_csv(p)
            f["model"] = np.where(f.condition.str.startswith("knn"), "kNN", model)
            f["run"] = run
            rows.append(f)
    return pd.concat(rows).drop_duplicates(["run", "model", "condition", "task", "field"])


# ----------------------------------------------------------------------------------------------- figures
def style():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "font.size": 9,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def bars(ax, cats, series, width=0.8):
    """Grouped bars with a 2px-ish surface gap; series = [(label, values, color)]."""
    n = len(series)
    w = width / n
    x = np.arange(len(cats))
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.88, label=lab, color=col, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, cats, rotation=0)
    return x


def fig_main(d, path):
    m = d[(d.experiment == "main") & d.condition.isin(["norag", "rag", "text_none", "text_rag"])]
    order = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3-flash-preview", "gpt-audio-1.5", "gpt-audio-2025-08-28",
             "gemini-3.5-flash-lite", "af3+extract", "qwen2-audio-7b+extract", "medgemma-27b", "medgemma-4b"]
    lab = {"gemini-3-flash-preview": "gemini-3-flash", "gpt-audio-2025-08-28": "gpt-audio", "af3+extract": "AF3 (two-step)",
           "qwen2-audio-7b+extract": "Qwen2-Audio (two-step)", "medgemma-27b": "MedGemma-27B (text)",
           "medgemma-4b": "MedGemma-4B (text)"}
    means = m.groupby(["model", "condition"]).report_score.mean().unstack()
    pick = lambda o, a, b: means.loc[o, a] if pd.notna(means.loc[o].get(a)) else means.loc[o, b]
    pre = [pick(o, "norag", "text_none") for o in order]
    post = [pick(o, "rag", "text_rag") for o in order]
    knn = d[(d.experiment == "main") & (d.condition == "knn_fieldwise")].drop_duplicates("task").report_score.mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))[::-1]
    for yi, a, b in zip(y, pre, post):
        ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(pre, y, s=36, color=GRAY, zorder=2, label="without retrieval", edgecolor="white", linewidth=1)
    ax.scatter(post, y, s=36, color=BLUE, zorder=3, label="with AuscultRAG evidence", edgecolor="white", linewidth=1)
    ax.axvline(knn, color=INK2, lw=0.8, ls="--", label="kNN vote (same evidence)")
    ax.set_yticks(y, [lab.get(o, o) for o in order])
    ax.set_xlabel("report score (mean over 8 tasks)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_controls(d, path):
    """Small multiples: 2 models x 8 tasks = 16 panels; each panel has its own y-scale (Coswara is ~0.1)."""
    models = [("gemini-3.5-flash", "gemini-3.5-flash"), ("gpt-audio-1.5", "gpt-audio-1.5")]
    conds = [("no evidence", GRAY), ("random cases", RAMP[0]), ("far neighbours", RAMP[1]), ("true top-k", RAMP[2])]
    fig, axes = plt.subplots(2, 8, figsize=(7.4, 3.3), sharey=False)
    for i, (m, lab) in enumerate(models):
        c = d[(d.experiment == "controls") & (d.model == m)].set_index(["condition", "task"]).report_score
        mn = d[(d.experiment == "main") & (d.model == m)].set_index(["condition", "task"]).report_score
        for j, t in enumerate(TASKS):
            ax = axes[i, j]
            v = [mn["norag", t], c["rag_random", t], c["rag_far", t], mn["rag", t]]
            ax.bar(range(4), v, 0.82, color=[col for _, col in conds], edgecolor="white", linewidth=0.5)
            ax.set_xticks([])
            ax.set_ylim(0, max(v) * 1.18)
            ax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            mono = v[1] < v[2] < v[3]
            if not mono:
                ax.text(1.5, max(v) * 1.08, "far > top-k" if v[2] >= v[3] else "", ha="center", fontsize=5.5, color=INK2)
            if i == 0:
                ax.set_title(SHORT[t], fontsize=6.5, color=INK, pad=3)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5)
    handles = [matplotlib.patches.Patch(color=col, label=l) for l, col in conds]
    fig.legend(handles=handles, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_textret(d, path):
    tk = TASKS
    t = d[(d.experiment == "textret")].set_index(["condition", "task"]).report_score
    m = d[(d.experiment == "main") & (d.model == "gpt-audio-1.5")].set_index(["condition", "task"]).report_score
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bars(ax, [SHORT[x].replace("-", "\n") for x in tk], [("no evidence", [m["norag", x] for x in tk], GRAY),
                                      ("text description -> retrieval", [t["rag@t2a", x] for x in tk], ORANGE),
                                      ("audio -> audio retrieval (AuscultRAG)", [m["rag", x] for x in tk], BLUE)])
    ax.set_ylabel("report score")
    ax.legend(ncol=3, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.16))
    fig.savefig(path)
    plt.close(fig)


def fig_robust(d, path):
    r = d[d.experiment == "robustness"].copy()
    r["cond"] = r.condition.str.split("@").str[0]
    p = r.groupby(["perturbation", "cond"]).report_score.mean().unstack()
    order = ["clean", "mp38", "trunc0.25", "white0", "ambient0", "babble10", "babble0"]
    lab = {"clean": "clean", "mp38": "MP3\nlow bit-rate", "trunc0.25": "25%\nduration", "white0": "white noise\n0 dB",
           "ambient0": "clinic noise\n0 dB", "babble10": "babble\n10 dB", "babble0": "babble\n0 dB"}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bars(ax, [lab[o] for o in order], [("no evidence", p.loc[order, "norag"], GRAY),
                                       ("AuscultRAG", p.loc[order, "rag"], BLUE),
                                       ("kNN vote", p.loc[order, "knn_rag"], AQUA)])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel("report score (mean, 8 tasks)")
    ax.legend(ncol=3, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.16))
    fig.savefig(path)
    plt.close(fig)


def fig_xcorpus(d, path, model="gemini-3.5-flash"):
    x = d[(d.experiment == "xcorpus") & d.run.str.contains("xcorpus")]
    tk = ["icbhi_adventitious", "kauh_adventitious", "sprsound_adventitious"]
    v = x[x.model.isin([model, "kNN"])].drop_duplicates(["condition", "task"]).set_index(["condition", "task"]).report_score
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bars(ax, [SHORT[t] for t in tk], [("no evidence", [v["norag@xc", t] for t in tk], GRAY),
                                      ("kNN vote, cross-dataset", [v["knn_rag@xc", t] for t in tk], AQUA),
                                      ("AuscultRAG, cross-dataset", [v["rag@xc", t] for t in tk], BLUE),
                                      ("AuscultRAG, in-dataset", [v["rag@in", t] for t in tk], RAMP[2])])
    ax.set_ylabel("report score")
    ax.legend(ncol=2, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.savefig(path)
    plt.close(fig)


def fig_specialists(d, ex, path):
    sp = ex["specialist_probe"].groupby("model").report_score.mean()
    main = d[d.experiment == "main"]
    rows = [("HeAR + trained probe", sp["hear+linear-probe"], AQUA), ("CLAP + trained probe", sp["clap+linear-probe"], AQUA),
            ("OPERA-CT + trained probe", sp["operaCT+linear-probe"], AQUA),
            ("AuscultRAG (gemini-3.5-flash)", main[(main.model == "gemini-3.5-flash") & (main.condition == "rag")].report_score.mean(), BLUE),
            ("kNN vote", main[main.condition == "knn_fieldwise"].drop_duplicates("task").report_score.mean(), GRAY),
            ("gemini-3.5-flash, no evidence", main[(main.model == "gemini-3.5-flash") & (main.condition == "norag")].report_score.mean(), GRAY)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], 0.6, color=[r[2] for r in rows], edgecolor="white")
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.005, yi, f"{r[1]:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("report score (mean over 8 tasks)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.56)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- pdf
def build_pdf(d, ex, figs, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont("CJK", FONT))
    H1 = ParagraphStyle("h1", fontName="CJK", fontSize=16, leading=22, spaceAfter=8, textColor=colors.HexColor(INK))
    H2 = ParagraphStyle("h2", fontName="CJK", fontSize=12.5, leading=18, spaceBefore=10, spaceAfter=5)
    B = ParagraphStyle("b", fontName="CJK", fontSize=9.5, leading=15, alignment=TA_LEFT, wordWrap="CJK")
    S = ParagraphStyle("s", parent=B, fontSize=8, leading=12, textColor=colors.HexColor(INK2))

    def table(rows, widths=None):
        t = Table([[Paragraph(str(c), S if i else ParagraphStyle("th", parent=S, textColor=colors.HexColor(INK)))
                    for c in r] for i, r in enumerate(rows)], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK2)),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor(GRID)),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def img(p, w=16.5):
        from PIL import Image as PI
        iw, ih = PI.open(p).size
        return Image(str(p), width=w * cm, height=w * cm * ih / iw)

    main = d[d.experiment == "main"]
    mm = main.groupby(["model", "condition"]).report_score.mean().unstack()
    knn_mean = main[main.condition == "knn_fieldwise"].drop_duplicates("task").report_score.mean()
    f = lambda x: f"{x:.3f}"
    st = []
    st += [Paragraph("AuscultRAG 审稿补充实验结果报告", H1),
           Paragraph(f"生成日期 {date.today().isoformat()}。数据：artifacts_committed/final/，代码：scripts/。"
                     "所有数字均为测试集（patient-disjoint，folds 1-4 合并）结果，所有设置只在 dev（fold 0）上确定。", S),
           Spacer(1, 6), Paragraph("一、核心结论", H2)]
    for t in [
        "<b>1. 音频到音频检索稳定提升诊断报告质量。</b>6 个商业音频大模型加上检索证据后，报告分平均提升约 0.12–0.14；"
        "开源音频模型（Qwen2-Audio、AF3，两步法）和医学大模型 MedGemma-4B/27B 同样明显提升。",
        "<b>2. 提升来自声学相似度，而不是标签泄漏。</b>证据格式完全不变，只把病例换成随机病例或较远的邻居，分数呈单调下降"
        "（随机 &lt; 远邻 &lt; 真实邻居），两个模型的结论一致。",
        "<b>3. 音频检索优于文本检索。</b>先把声音写成文字再去检索，多数任务明显变差，有时还不如不用检索。",
        "<b>4. 对噪声鲁棒。</b>人声、诊室环境声、白噪声、MP3 压缩、截短录音等 7 种扰动下，加检索都仍然优于不加。",
        "<b>5. 跨数据集依然有效。</b>只用其他数据集的病例做检索仍然有帮助；更强的 Gemini 在证据来自其他数据集时超过了 kNN 标签迁移。",
        "<b>6. 优于医学专用大模型。</b>心肺听诊专用模型 CaReAQA 即使训练时见过我们的部分测试病人，分数仍远低于 AuscultRAG；"
        "Google 医学大模型 MedGemma（4B/27B）本身无法听音频，单独使用时很弱，但接入 AuscultRAG 检索到的病例后分数接近翻倍，"
        "说明 AuscultRAG 能把声学信息“转交”给不能听音频的医学大模型。",
        "<b>7. 需要正视的点。</b>每个字段单独训练的监督分类器是性能上限（约 0.48 vs 0.41）；纯文本模型加检索证据与音频模型加检索相当。"
        "因此论文贡献应定位为“音频检索作为外部声学记忆”，强调免训练、可增量扩充、可解释的完整报告和跨数据集稳定性。"]:
        st.append(Paragraph(t, B))
    st += [Paragraph("指标说明", H2), Paragraph(
        "报告分 = 各“信息字段”宏 F1 的平均值（信息字段：真值至少有 2 个取值的字段，规则在 dev 上固定）。模型按提示词写出"
        "Findings / Impression 和一行 JSON 字段；字段来自专家标注（例如 CirCor 的杂音时相、分级，ICBHI 的诊断、crackles/wheezes）。"
        "每个任务约 200 个查询（ICBHI 102、KAUH 87、TBScreen 98），一个病人只取一条录音。置信区间为病人级 bootstrap，见数据文件。", B),
        PageBreak()]

    st += [Paragraph("二、主结果：所有模型加上检索后都提升", H2), img(figs["main"]),
           Paragraph("灰点为不加检索，蓝点为加 AuscultRAG 证据，虚线为 kNN 按字段投票。MedGemma 为纯文本模型（不听音频），"
                     "作为医学大模型对照；AF3 和 Qwen2-Audio 使用两步法（自由写报告，再由中立文本模型抽取字段）。", S)]
    rows = [["模型", "输入", "不加检索", "加检索", "提升"]]
    for mdl, inp, a, b in [("gemini-3.5-flash", "音频", "norag", "rag"), ("gemini-3.8-flash", "音频", "norag", "rag"),
                           ("gemini-3-flash", "音频", "norag", "rag"), ("gpt-audio-1.5", "音频", "norag", "rag"),
                           ("gpt-audio (2025-08-28)", "音频", "norag", "rag"), ("gemini-3.5-flash-lite", "音频", "norag", "rag"),
                           ("Qwen2-Audio-7B（两步法）", "音频", "norag", "rag"), ("AF3（两步法）", "音频", "norag", "rag"),
                           ("MedGemma-27B", "纯文本", "text_none", "text_rag"), ("MedGemma-4B", "纯文本", "text_none", "text_rag"),
                           ("gpt-5.4", "纯文本", "text_none", "text_rag")]:
        key = {"gemini-3-flash": "gemini-3-flash-preview", "gpt-audio (2025-08-28)": "gpt-audio-2025-08-28",
               "Qwen2-Audio-7B（两步法）": "qwen2-audio-7b+extract", "AF3（两步法）": "af3+extract",
               "MedGemma-27B": "medgemma-27b", "MedGemma-4B": "medgemma-4b", "gpt-5.4": "gpt-5.4-2026-03-05"}.get(mdl, mdl)
        rows.append([mdl, inp, f(mm.loc[key, a]), f(mm.loc[key, b]), f"+{mm.loc[key, b] - mm.loc[key, a]:.3f}"])
    rows.append(["kNN 按字段投票（相同证据）", "—", "—", f(knn_mean), ""])
    st += [Spacer(1, 6), table(rows, [5.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]),
           Paragraph("解读：不同厂商、不同规模、开源与闭源、医学与通用模型都得到一致的提升，说明检索到的相似病例提供了模型自身"
                     "听不出来的信息。仅看不加检索的结果，所有模型都很弱（约 0.11–0.28），这也是审稿人认为“直接用音频大模型不可靠”的原因。", B),
           PageBreak()]

    st += [Paragraph("三、提升来自声学相似度（回应“标签泄漏”）", H2), img(figs["controls"]),
           Paragraph("两个模型 × 8 个任务 = 16 组对照。每组四个条件的提示词格式、病例数量和显示的相似度数值完全相同，只换病例本身。"
                     "每个小图的纵轴独立（Coswara 两个任务分数约 0.1，与其他任务共用纵轴会被压扁）。", S),
           Paragraph("解读：如果模型只是“看到了答案选项和标签词”，随机病例应该和真实邻居一样有效。实际上 <b>16 组中随机病例全部低于真实邻居，"
                     "也全部低于远邻</b>；13 组呈完全单调的剂量反应（随机 &lt; 远邻 &lt; 真实）。3 个例外都是远邻略高于真实邻居"
                     "（Coswara-cough 的两个模型、TBScreen 的 gpt-audio-1.5），这几个任务单个邻居的声学信息本来就弱（见第四节 TBScreen 分析），"
                     "但随机病例仍然最低。这直接回应了审稿人 R1 和 AI 审稿关于 label-token priming 的质疑。", B),
           Paragraph("四、音频检索优于文本检索", H2), img(figs["textret"]),
           Paragraph("gpt-audio-1.5。文本检索：模型先在不看证据的情况下写出对声音的描述，再用 CLAP 文本编码器以该描述检索同一个音频库。", S),
           Paragraph("解读：把声音“翻译成文字”会丢失关键的声学细节，多数任务比音频直接检索低 0.10–0.18，KAUH、SPRSound 上甚至不如不检索。"
                     "Coswara 两个任务整体分数都很低，但音频检索仍不低于文本检索（Coswara-cough 0.083 vs 0.049，Coswara-resp 0.058 vs 0.042）。TBScreen 是例外：在这个任务上音频检索和文本检索"
                     "找到的邻居都与随机抽样相当（标签一致率 0.61 vs 0.59），证据只起到类别先验的作用，差异在置信区间内。这验证了审稿人 R1 要求对比的核心设计选择。", B),
           PageBreak()]

    st += [Paragraph("五、噪声鲁棒性", H2), img(figs["robust"]),
           Paragraph("gpt-audio-1.5，每个任务 50 个查询；只扰动查询，知识库保持干净（模拟真实诊室录音、干净病例库）。"
                     "kNN 在全部 785 个查询 × 19 种扰动上的结果见 robustness_knn_all_perturbations.csv。", S),
           Paragraph("解读：每一种扰动下加检索都仍然优于不加。人声干扰影响最大（检索带来的提升从约 0.10 缩小到约 0.05），"
                     "MP3 压缩和截短录音几乎没有影响。", B),
           Paragraph("六、跨数据集检索（分布偏移）", H2), img(figs["xcorpus"]),
           Paragraph("gemini-3.5-flash。三个肺音语料库字段已统一；“跨数据集”指只在另外两个数据集中检索（采集机构、设备、人群和标注规范均不同，SPRSound 为儿童）。", S),
           Paragraph("解读：只用其他数据集的病例仍然有帮助。证据来自其他数据集时，Gemini 在 ICBHI 和 KAUH 上超过 kNN 直接照搬标签，"
                     "在 SPRSound 上持平；gpt-audio-1.5 则与 kNN 相当。说明证据不完美时，强模型能结合音频做出更好的判断，"
                     "回应了审稿人 R2 的核心问题。", B),
           PageBreak()]

    sp = ex["specialist_probe"].groupby("model").report_score.mean()
    car = ex["careaqa_supported_fields"].groupby(["model", "condition"]).supported_field_score.mean()
    mg = {m: (mm.loc[m, "text_none"], mm.loc[m, "text_rag"]) for m in ("medgemma-4b", "medgemma-27b")}
    g35 = (mm.loc["gemini-3.5-flash", "norag"], mm.loc["gemini-3.5-flash", "rag"])
    st += [Paragraph("七、与医学专用模型的比较", H2),
           Paragraph("7.1 医学大模型：心肺听诊专用模型 CaReAQA 与 Google 医学大模型 MedGemma", ParagraphStyle("h3", parent=B, fontSize=10.5, spaceBefore=4, spaceAfter=3)),
           table([["方法", "能否听音频 / 能否接收证据", "评分范围", "不加检索", "加 AuscultRAG"],
                  ["CaReAQA（CHIL 2025，心肺听诊问答）", "能 / 不能", "仅其支持的字段（5 个心肺任务）", f(car["careaqa", "norag"]), "—"],
                  ["gpt-audio-1.5", "能 / 能", "与 CaReAQA 同一批字段", f(car["gpt-audio-1.5", "norag"]), f(car["gpt-audio-1.5", "rag"])],
                  ["gemini-3.5-flash", "能 / 能", "与 CaReAQA 同一批字段", f(car["gemini-3.5-flash", "norag"]), f(car["gemini-3.5-flash", "rag"])],
                  ["MedGemma-4B（Google 医学大模型）", "不能 / 能", "8 个任务、全部字段", f(mg["medgemma-4b"][0]), f(mg["medgemma-4b"][1])],
                  ["MedGemma-27B（Google 医学大模型）", "不能 / 能", "8 个任务、全部字段", f(mg["medgemma-27b"][0]), f(mg["medgemma-27b"][1])],
                  ["gemini-3.5-flash", "能 / 能", "8 个任务、全部字段", f(g35[0]), f(g35[1])]],
                 [4.4 * cm, 3.0 * cm, 4.2 * cm, 1.8 * cm, 2.8 * cm]),
           Spacer(1, 4),
           Paragraph("解读：(1) CaReAQA 远低于 AuscultRAG，甚至低于通用音频模型不加检索的结果；而我们的 ICBHI、CirCor 测试病人 100% 出现在它的训练集中，"
                     "SPRSound 为 32%。(2) <b>为什么 CaReAQA 只出现在这里：</b>它是短答案问答模型（音频 + 一个短问题 → 一句话），"
                     "写不出完整的结构化报告，输入中也没有放置检索证据的位置，并且没有训练过咳嗽数据。因此它无法进入主表和后续"
                     "“加/不加证据”的对照、跨数据集、鲁棒性实验，只能在它能回答的字段上单独比较。这一点本身也说明：现有心肺专用模型"
                     "输出格式固定、无法利用外部证据，而 AuscultRAG 可以接入任何大模型。(3) MedGemma 不能听音频，单独使用只有约 0.18–0.19；"
                     "接入 AuscultRAG 按声学相似度检索到的病例后提升到约 0.34，即通过检索间接获得了声学信息。27B 与 4B 几乎相同，"
                     "说明模型规模不能替代检索到的声学证据。能听音频的模型加 AuscultRAG（gemini-3.5-flash，0.41）仍然最好。", B),
           Paragraph("7.2 监督专用分类器（性能上限）", ParagraphStyle("h3b", parent=B, fontSize=10.5, spaceBefore=6, spaceAfter=3)),
           img(figs["specialists"], 12.5),
           Paragraph("冻结的专用编码器 + 每个字段单独训练的线性探针（OPERA 论文的标准协议），训练数据与 AuscultRAG 的知识库完全相同，C 在 dev 上选择；8 个任务、全部字段。", S),
           Paragraph("解读：(1) 每个字段单独训练的分类器是性能上限，超过免训练的 AuscultRAG；但迁移到其他数据集后，专用编码器的分类器下降明显"
                     "（HeAR 在 KAUH 上 0.30，OPERA-CT 在三个数据集上 0.31–0.35），跨数据集 AuscultRAG 保持在约 0.44–0.49。"
                     "(2) 检索编码器消融：通用 CLAP 在 8 个任务中 6 个优于 OPERA-CT 和 HeAR，专用编码器只在 Coswara 上更好。"
                     "(3) StethoLM 只公开权重、没有推理代码，复现失败，论文中说明；RespLLM、RespiraMFM 无公开权重。", B),
           Paragraph("八、模型如何使用证据", H2),
           Paragraph("邻居意见完全一致时字段准确率约 0.84，意见分裂时约 0.39：邻居一致性是可靠的置信度，可用于拒答或转人工。"
                     "模型给出证据里没有出现的值的比例只有 0.3–4%（幻觉少）；越强的模型越少推翻邻居多数意见。"
                     "同一个 gemini-3.5-flash，在邻居意见分裂的样本上，听音频比不听音频更准（0.393 vs 0.339）。", B),
           Paragraph("九、局限与论文定位建议", H2),
           Paragraph("1) 监督分类器分数更高、纯文本模型加检索证据与音频模型相当：贡献应写为“音频到音频检索作为外部声学记忆”，"
                     "LLM 的价值在于生成完整可解释报告、利用不完美证据（跨数据集）、以及让开源和医学模型也能受益。"
                     "2) 7B 开源音频模型无法直接处理长篇结构化证据，需要两步法。3) gemini-3.1-pro 受每日调用限额未评测。"
                     "4) 所有 OpenAI/Gemini 模型版本、提示词与随机种子均已记录在代码与数据中，可复现。", B)]

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="AuscultRAG results report", author="AuscultRAG")
    doc.build(st)


def main() -> None:
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="output folder (default reports/final_<today>)")
    a = ap.parse_args()
    if a.out:
        OUT = a.out if a.out.is_absolute() else ROOT / a.out
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    d = tidy()
    ex = extras()
    fs = field_scores()
    d.to_csv(OUT / "data" / "all_results_long.csv", index=False)
    fs.to_csv(OUT / "data" / "field_scores_long.csv", index=False)
    for k, v in ex.items():
        v.to_csv(OUT / "data" / f"{k}.csv", index=False)
    blob = {"description": "AuscultRAG test-set results; see DATA_DICTIONARY.md",
            "all_results": d.replace({np.nan: None}).to_dict("records"),
            **{k: v.replace({np.nan: None}).to_dict("records") for k, v in ex.items()}}
    (OUT / "data" / "all_results.json").write_text(json.dumps(blob, indent=1, default=str))
    style()
    figs = {k: OUT / "figures" / f"{i}_{k}.png" for i, k in enumerate(
        ["main", "controls", "textret", "robust", "xcorpus", "specialists"], 1)}
    fig_main(d, figs["main"])
    fig_controls(d, figs["controls"])
    fig_textret(d, figs["textret"])
    fig_robust(d, figs["robust"])
    fig_xcorpus(d, figs["xcorpus"])
    fig_specialists(d, ex, figs["specialists"])
    build_pdf(d, ex, figs, OUT / "AuscultRAG_results_report.pdf")
    shutil.copy(ROOT / "docs" / "results_summary.md", OUT / "results_summary_detailed.md")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
