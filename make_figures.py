#!/usr/bin/env python3
"""Figures for the WSP paper. Reads results/results.json only — no hand-entered data.

Two figures:
  1. control-vs-edit: the paper's central claim. Accuracy delta with 95% CI for
     edit-vs-base and control-vs-base, all three benchmarks. The control band
     straddling zero is the whole argument, so it gets the main figure.
  2. tokens: thinking-token reduction, edit vs control, same layout.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results", "results.json")))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

ORDER = [("gpqa_diamond", "GPQA Diamond"), ("lcb_v6", "LiveCodeBench v6"), ("ifbench", "IFBench")]
INK = "#1a1a1a"
EDIT = "#1f4f6b"
CTRL = "#b8b8b8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Libertinus Serif", "DejaVu Serif"],
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel(ax, key, title, unit):
    ys, labels = [], []
    for i, (bench, label) in enumerate(ORDER):
        e = RES["benchmarks"][bench]
        src = e["paired"] if key == "acc" else e["tokens"]
        ed = src["base_to_edit"]
        ct = src["base_to_ctrl"]
        for j, (v, colour, name) in ((0, (ed, EDIT, "edit")), (1, (ct, CTRL, "control"))):
            y = i * 2.6 + (0.45 if j == 0 else -0.45)
            d = v["delta"] if key == "acc" else v["pct"]
            lo, hi = v["ci"]
            ax.plot([lo, hi], [y, y], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
            ax.plot([d], [y], "o", color=colour, ms=6.5, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.9)
            if j == 0:
                labels.append((i * 2.6, label))
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([t for _, t in labels], fontsize=10)
    ax.set_ylim(-1.5, (len(ORDER) - 1) * 2.6 + 1.5)
    ax.set_xlabel(unit, fontsize=9.5)
    ax.set_title(title, fontsize=11, pad=10, loc="left", weight="bold")
    ax.tick_params(axis="x", labelsize=9)


# Figure 1 — the control result.
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=200)
panel(ax, "acc", "Accuracy change vs unmodified base (95% CI)", "percentage points")
h = [plt.Line2D([], [], color=EDIT, lw=2.4, marker="o", ms=6, markeredgecolor="white", label="fitted edit"),
     plt.Line2D([], [], color=CTRL, lw=2.4, marker="o", ms=6, markeredgecolor="white", label="norm-matched random control")]
ax.legend(handles=h, fontsize=8.5, frameon=False, loc="lower right", ncol=1)
fig.tight_layout()
p1 = os.path.join(OUT, "control-vs-edit-2x.png")
fig.savefig(p1, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2 — tokens.
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=200)
panel(ax, "tok", "Thinking-token change vs unmodified base (95% CI)", "percent")
ax.legend(handles=h, fontsize=8.5, frameon=False, loc="lower left")
fig.tight_layout()
p2 = os.path.join(OUT, "tokens-2x.png")
fig.savefig(p2, bbox_inches="tight", facecolor="white")
plt.close(fig)

for p in (p1, p2):
    from PIL import Image
    w, hgt = Image.open(p).size
    print(f"  {os.path.basename(p)}  {w}x{hgt}")
