"""Report figures (one per key claim)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import roc_curve

from . import config
from .explain import encoded

NAVY, TEAL, AMBER, GREY, RED = "#0B2545", "#13A89E", "#F2A541", "#8D99AE", "#D1495B"


def _save(fig, name):
    config.FIGS.mkdir(parents=True, exist_ok=True)
    p = config.FIGS / name
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def figure1(models, Xte, yte, final, decile_tbl, metrics):
    """Confusion matrix + ROC curves + reliability by decile for the chosen model (test set)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    m = metrics
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                       color="white" if cm[i, j] > cm.max() / 2 else NAVY)
    ax[0].set_xticks([0, 1], ["Not flagged", "Flagged"]); ax[0].set_yticks([0, 1], ["Repaid", "Defaulted"])
    ax[0].set_title("Confusion matrix (high-risk flag)", loc="left", color=NAVY, fontweight="bold", fontsize=11)
    for (name, mod), c in zip(models.items(), [GREY, AMBER, TEAL]):
        fpr, tpr, _ = roc_curve(yte, mod.raw(Xte))
        ax[1].plot(fpr, tpr, color=c, lw=2, label=name.replace("_", " "))
    ax[1].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate"); ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title("ROC curves (2015 test)", loc="left", color=NAVY, fontweight="bold", fontsize=11)
    ax[2].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax[2].plot(decile_tbl["predicted"], decile_tbl["actual"], "o-", color=TEAL, lw=2)
    ax[2].set_xlabel("Predicted default rate"); ax[2].set_ylabel("Actual default rate")
    ax[2].set_title("Reliability by risk decile", loc="left", color=NAVY, fontweight="bold", fontsize=11)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig1_confusion_roc_reliability.png")


def shap_summary(agg, X):
    E = encoded(X)
    expl = shap.Explanation(values=agg, data=E.to_numpy(), feature_names=list(E.columns))
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(expl, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary: 15 most influential features", x=0.02, ha="left", fontsize=12, color=NAVY, fontweight="bold")
    return _save(fig, "fig2_shap_summary.png")


def shap_waterfalls(agg, base, X, positions, labels):
    paths = []
    for pos, lab in zip(positions, labels):
        row = X.iloc[pos]
        expl = shap.Explanation(values=agg[pos], base_values=base, data=row.astype(object).to_numpy(),
                                feature_names=list(X.columns))
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(expl, max_display=10, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Why this applicant is {lab}", x=0.02, ha="left", fontsize=12, color=NAVY, fontweight="bold")
        paths.append(_save(fig, f"shap_waterfall_{lab.replace(' ', '_')}.png"))
    return paths


def fairness_chart(before, after):
    attrs = list(before)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a_, (key, title, lim) in zip(ax, [("approval_ratio", "Approval ratio (need > 0.80)", 0.8),
                                           ("tpr_gap_pp", "TPR gap, points (need <= 5)", 5),
                                           ("fpr_gap_pp", "FPR gap, points (need <= 5)", 5)]):
        x = np.arange(len(attrs)); w = 0.38
        a_.bar(x - w / 2, [before[k][key] for k in attrs], w, color=GREY, label="Before")
        a_.bar(x + w / 2, [after[k][key] for k in attrs], w, color=TEAL, label="After ThresholdOptimizer")
        a_.axhline(lim, color=RED, ls="--", lw=1)
        a_.set_xticks(x, [k.replace("_", " ") for k in attrs], rotation=20, ha="right", fontsize=8)
        a_.set_title(title, loc="left", color=NAVY, fontweight="bold", fontsize=11)
        a_.spines[["top", "right"]].set_visible(False)
    ax[2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig3_fairness_before_after.png")
