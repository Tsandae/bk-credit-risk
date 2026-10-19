"""Render report images from pipeline outputs (GE results panel, bias charts) for the slide deck."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1] / "reports"
NAVY, TEAL, AMBER, GREY, RED = "#0B2545", "#13A89E", "#F2A541", "#8D99AE", "#D1495B"


def ge_panel():
    d = json.loads((R / "validation_results.json").read_text())
    rows = d["results"]
    half = (len(rows) + 1) // 2
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=170)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0, 1.06, f"Great Expectations  |  suite: {d['suite']}", fontsize=13, fontweight="bold", color=NAVY, transform=ax.transAxes)
    ax.text(1, 1.06, f"{d['passed']}/{d['evaluated']} expectations passed", fontsize=13, fontweight="bold",
            color=TEAL if d["success"] else RED, ha="right", transform=ax.transAxes)
    for i, r in enumerate(rows):
        col, row = divmod(i, half)
        x, y = 0.0 + col * 0.51, 0.97 - row * (0.95 / half)
        name = r["expectation"].replace("expect_", "").replace("_to_", " ")
        label = f"{name}  [{r['column']}]"
        ax.text(x, y, "✓" if r["success"] else "✗", fontsize=12, color=TEAL if r["success"] else RED,
                fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(x + 0.025, y, label, fontsize=8.6, color=NAVY, transform=ax.transAxes, va="top", family="DejaVu Sans")
    fig.savefig(R / "ge_results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bias_charts():
    disp = pd.read_csv(R / "bias_outcome_disparity.csv")
    inc = disp[disp.attribute == "income_band"].set_index("group").loc[["<40k", "40-80k", "80-120k", "120k+"]]
    summ = json.loads((R / "bias_summary.json").read_text())
    yr = pd.DataFrame(summ["year_shift"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=170)
    ax = axes[0]
    bars = ax.bar(inc.index, inc["default_rate"] * 100, color=[AMBER if v == inc["default_rate"].max() else TEAL for v in inc["default_rate"]])
    for b, v in zip(bars, inc["default_rate"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=10, color=NAVY)
    ax.set_title("Default rate by income band", fontsize=12, color=NAVY, fontweight="bold", loc="left")
    ax.set_ylabel("% defaulted", color=GREY)
    ax.set_ylim(0, 22)
    ax = axes[1]
    w = 0.38
    x = range(len(yr))
    ax.bar([i - w / 2 for i in x], yr["share_before"] * 100, w, label="Before label filter", color=GREY)
    ax.bar([i + w / 2 for i in x], yr["share_after"] * 100, w, label="Curated dataset", color=TEAL)
    ax.set_xticks(list(x)); ax.set_xticklabels(yr["year"])
    ax.set_title("Share of loans by issue year", fontsize=12, color=NAVY, fontweight="bold", loc="left")
    ax.set_ylabel("% of rows", color=GREY)
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=NAVY)
    fig.tight_layout()
    fig.savefig(R / "bias_charts.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ge_panel()
    bias_charts()
    print("images written to", R)
