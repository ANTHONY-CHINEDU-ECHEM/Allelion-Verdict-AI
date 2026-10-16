"""README figures from reports/evaluation_summary.json and the model reports (reports/figures/*.png)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allelion.config import FIGURES, MODELS, REPORTS

INK, INK2, INK3, RULE = "#17262b", "#4a5a5f", "#7c8b8f", "#dfe5e3"
TIER = {"Pathogenic": "#a8123f", "Likely pathogenic": "#d4547e", "Uncertain significance": "#b8860b", "Likely benign": "#0f8f7c", "Benign": "#1f5c8f"}
ACCENT = "#1f5c8f"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.titlecolor": INK, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": False})


def _save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", facecolor="white")
    plt.close(fig)


def _hbar(ax, labels, values, colors=None, fmt="{:.3f}", xmax=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors or INK, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + (xmax or max(values)) * 0.01, yi, fmt.format(v), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, (xmax or max(values)) * 1.12)
    ax.tick_params(axis="y", length=0)


def confusion(s, name, title):
    m = np.array(s["rules_points"]["confusion_5tier"])
    cls = s["rules_points"]["classes"]
    short = ["P", "LP", "VUS", "LB", "B"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1.4)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=10, color="white" if norm[i, j] > 0.7 else INK)
    ax.set_xticks(range(5), short)
    ax.set_yticks(range(5), [f"{c}" for c in cls])
    ax.set_xlabel("Engine (point system)")
    ax.set_ylabel("Expert panel reference")
    ax.set_title(title, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, name)


def methods(s, name):
    rows = [("Point system", s["rules_points"]), ("2015 rules", s["rules_2015"]), ("Learned verdict", s["learned_meta"])]
    metrics = [("exact_5tier", "Exact five tier"), ("accuracy_3tier", "Three tier"), ("balanced_accuracy_3tier", "Balanced three tier")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(metrics))
    w = 0.26
    cols = [INK, INK3, ACCENT]
    for k, (lab, m) in enumerate(rows):
        vals = [m[key] for key, _ in metrics]
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=cols[k], label=lab)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.18))
    _save(fig, name)


def panels(s, name):
    bp = s.get("by_panel", {})
    if not bp:
        return
    labels = [f"{k} (n={v['n']})" for k, v in bp.items()]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(labels) + 0.8))
    _hbar(ax, labels, [v["accuracy_3tier"] for v in bp.values()], [INK] * len(labels), "{:.1%}", 1.0)
    ax.set_xlabel("Three tier agreement with the panel")
    _save(fig, name)


def ablations(s, name):
    ab = s["ablations"]
    labels = list(ab)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(labels) + 0.8))
    _hbar(ax, labels, [ab[k]["accuracy_3tier"] for k in labels], [INK if k == "all criteria" else INK3 for k in labels], "{:.1%}", 1.0)
    ax.set_xlabel("Three tier agreement")
    _save(fig, name)


def criteria_heat(s, name):
    freq = s["criteria_frequency_by_reference_class"]
    classes = [c for c in TIER if c in freq]
    codes = sorted({k for c in classes for k in freq[c]}, key=lambda k: (k.lstrip("_")[0] != "P", k))
    m = np.array([[freq[c].get(k, 0) for k in codes] for c in classes])
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(codes)), 3.2))
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1.2, aspect="auto")
    for i in range(len(classes)):
        for j in range(len(codes)):
            if m[i, j] >= 0.05:
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(codes)), codes, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(classes)), classes)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Share of blind variants meeting each criterion, by reference class", loc="left")
    _save(fig, name)


def calibration(kind):
    c = json.loads((MODELS / f"{kind}_calibration.json").read_text())
    g, p, lo, up = map(np.array, (c["grid"], c["posterior"], c["lower"], c["upper"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.fill_between(g, lo, up, color=INK3, alpha=0.25, linewidth=0, label="90% bootstrap band")
    ax.plot(g, p, color=INK, lw=2, label="local posterior")
    prior = c["prior"]
    for s, lr in c["lr"].items():
        if s == "VeryStrong":
            continue
        post = lr * prior / ((lr - 1) * prior + 1)
        ax.axhline(post, color=TIER["Pathogenic"], ls="--", lw=0.9, alpha=0.7)
        ax.text(g.min(), post + 0.01, f"PP3 {s}", color=TIER["Pathogenic"], fontsize=8.5)
        bpost = (1 / lr) * prior / ((1 / lr - 1) * prior + 1)
        ax.axhline(bpost, color=TIER["Benign"], ls=":", lw=0.9, alpha=0.6)
    for s, t in c["pathogenic_thresholds"].items():
        if t is not None:
            ax.axvline(t, color=TIER["Pathogenic"], lw=0.8, alpha=0.5)
    for s, t in c["benign_thresholds"].items():
        if t is not None:
            ax.axvline(t, color=TIER["Benign"], lw=0.8, alpha=0.5)
    ax.set_xlabel(f"{kind.capitalize()} predictor score (out of fold)")
    ax.set_ylabel("Posterior probability of pathogenicity")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, f"calibration_{kind}")


def insilico():
    r = json.loads((MODELS / "insilico_report.json").read_text())
    m = r["missense"]
    labels = ["Allelion missense model"] + list(m["baselines"])
    vals = [m["grouped_cv"]["auroc"]] + [v["auroc"] for v in m["baselines"].values()]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    _hbar(ax, labels, vals, [INK] + [INK3] * (len(labels) - 1), "{:.3f}", 1.0)
    ax.axvline(0.5, color=RULE, lw=1)
    ax.set_xlabel("AUROC, five fold cross validation grouped by gene")
    _save(fig, "insilico_auroc")


def retrieval(res, name):
    rows = list(res.values())
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(labels) + 1), sharey=True)
    for ax, key, title in ((axes[0], "recall@5", "Recall@5"), (axes[1], "ndcg@10", "nDCG@10")):
        _hbar(ax, labels, [r[key] for r in rows], [INK if r["label"].startswith("Full") else INK3 for r in rows], "{:.3f}", 1.0)
        ax.set_title(title, loc="left")
    _save(fig, name)


def verifier():
    r = json.loads((MODELS / "verifier_report.json").read_text())
    pp = r["per_perturbation"]
    labels = list(pp)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(labels) + 0.8))
    _hbar(ax, [l.replace("_", " ") for l in labels], [pp[k]["accuracy"] for k in labels], [INK if k in ("verbatim", "paraphrase") else ACCENT for k in labels], "{:.1%}", 1.0)
    ax.set_xlabel("Accuracy on held out perturbed sentences (supported: verbatim, paraphrase)")
    _save(fig, "verifier_perturbations")


def extraction():
    r = json.loads((MODELS / "extractor_report.json").read_text())
    items = [("CRF, tmVar test", r["crf_exact_tmVar"]["f1"]), ("Rules, tmVar test", r["rules_exact_tmVar"]["f1"]),
             ("CRF, SETH held out", r["crf_exact_SETH"]["f1"]), ("Rules, SETH held out", r["rules_exact_SETH"]["f1"]),
             ("Normalised, MutationFinder test", r["mutationfinder_doc_level_crf_plus_rules"]["f1"])]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _hbar(ax, [a for a, _ in items], [b for _, b in items], [INK if "CRF" in a or "Normalised" in a else INK3 for a, _ in items], "{:.3f}", 1.0)
    ax.set_xlabel("F1 (exact span match; document level for MutationFinder)")
    _save(fig, "extraction_f1")


def kb(k):
    items = [("ClinVar alleles", k["variants"]), ("Seen in gnomAD", k["variants_seen_in_gnomad"]), ("HPO terms", k["hpo_terms"]),
             ("Disease profiles", k["diseases"]), ("Genes", k["genes"]), ("Genes with exon model", k["genes_with_exon_model"]),
             ("Literature documents", k["literature_documents"])]
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v in items], color=INK, height=0.6)
    ax.set_xscale("log")
    for yi, (_, v) in zip(y, items):
        ax.text(v * 1.08, yi, f"{v:,}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [a for a, _ in items])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(1000, 3e6)
    ax.set_xlabel("Count (log scale)")
    _save(fig, "kb_composition")


def reliability(s, name):
    rel = s["posterior_reliability"]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    labels = [r["band"] for r in rel]
    ax.bar(range(len(rel)), [r["reference_pathogenic_share"] for r in rel], color=INK, width=0.6)
    for i, r in enumerate(rel):
        ax.text(i, r["reference_pathogenic_share"] + 0.02, f"{r['reference_pathogenic_share']:.0%}\nn={r['n']}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(rel)), labels, fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Bayesian posterior band")
    ax.set_ylabel("Share pathogenic by panel")
    _save(fig, name)


def dense_training():
    d = json.loads((MODELS / "dual_training.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(np.arange(len(d["loss_curve"])) * 20, d["loss_curve"], color=INK, lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Symmetric InfoNCE loss")
    _save(fig, "dense_training")


def reranker():
    r = json.loads((MODELS / "reranker_report.json").read_text())
    imp = sorted(r["importance_gain"].items(), key=lambda x: -x[1])[:12]
    tot = sum(r["importance_gain"].values())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _hbar(ax, [k for k, _ in imp], [v / tot for _, v in imp], [INK] * len(imp), "{:.1%}")
    ax.set_xlabel("Share of total gain")
    _save(fig, "reranker_importance")


def main():
    s = json.loads((REPORTS / "evaluation_summary.json").read_text())
    blind = s.get("classification_blind")
    main_set = blind or s["classification_dev"]
    confusion(main_set, "blind_confusion", "Blind set: engine against expert panels")
    methods(main_set, "blind_methods")
    panels(main_set, "blind_panels")
    ablations(main_set, "blind_ablations")
    criteria_heat(main_set, "criteria_by_class")
    reliability(main_set, "posterior_reliability")
    confusion(s["classification_dev"], "dev_confusion", "Development set")
    for k in ("missense", "splice"):
        calibration(k)
    insilico()
    rr = s["retrieval"]
    retrieval(rr.get("blind") or rr["dev"], "retrieval_ablation")
    verifier()
    extraction()
    kb(s["knowledge_base"])
    dense_training()
    reranker()
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
