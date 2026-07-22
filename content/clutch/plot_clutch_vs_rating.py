"""Scatter: clutch (y) vs our PICKLE skill rating (x).

Shows clutch tracks skill — a positive trend (the greats are clutch mostly
because they're great; the residual is the part worth naming, see
model/clutch.md §4). All 182 measured clutch players, no external ratings.

Run: python content/clutch/plot_clutch_vs_rating.py -> content/clutch/clutch_vs_rating.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
clu = list(csv.DictReader((ROOT / "data" / "clutch_players.csv").open()))
P = [dict(name=r["name"], z=float(r["z"]), val=float(r["value"])) for r in clu]

INK = "#16321e"; DOT = "#3f6b45"; LIME = "#5a8f1f"; SAGE = "#6f8560"
CREAM = "#fbfdf3"; GRID = "#dbe7c8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

STARS = ["Anna Leigh Waters", "Ben Johns", "Anna Bright", "Gabriel Tardio"]
fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=200)
fig.patch.set_facecolor(CREAM)

pts = [(p["val"], p["z"], p["name"]) for p in P]
x = np.array([a for a, _, _ in pts]); y = np.array([b for _, b, _ in pts])
ax.set_facecolor(CREAM)
ax.axhline(0, color=SAGE, lw=1, ls=(0, (4, 4)), zorder=1)
ax.scatter(x, y, s=48, c=DOT, alpha=0.6, edgecolor=CREAM, linewidth=0.6, zorder=3)
b1, b0 = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, b0 + b1 * xs, color=LIME, lw=3, zorder=4)
r = np.corrcoef(x, y)[0, 1]
ax.text(0.04, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=20, fontweight="bold",
        color=LIME, va="top")
ax.text(0.04, 0.875, f"n = {len(x)}", transform=ax.transAxes, fontsize=12, color=SAGE, va="top")
for nm in STARS:
    m = [(a, b) for a, b, c in pts if c == nm]
    if m:
        ax.annotate(nm.split()[-1] if nm != "Anna Bright" else "Bright",
                    xy=m[0], xytext=(m[0][0], m[0][1] + 0.6), fontsize=10.5,
                    fontweight="bold", color=INK, ha="center")
ax.set_xlabel("PICKLE score (mine)", fontsize=13, fontweight="bold", labelpad=8)
ax.set_ylabel("CLUTCH  —  wins the big points\nvs. their own baseline", fontsize=12.5, fontweight="bold")
ax.set_title("Clutch vs. my PICKLE score", fontsize=14.5, fontweight="bold", color=INK, pad=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(INK)
ax.tick_params(length=0); ax.grid(color=GRID, lw=0.7, zorder=0)

fig.suptitle("Clutch skill is more than overall skill, and loosely correlates with player skill",
             x=0.012, ha="left", fontsize=15.5, fontweight="bold", color=INK, y=0.985)
fig.text(0.012, 0.02, "PICKLES · by Chad Murphy · chad-murphy-data.github.io/pickleball · "
         "clutch on 162,942 rallies", fontsize=9.5, color=SAGE)
fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.16)
out = ROOT / "content" / "clutch" / "clutch_vs_rating.png"
fig.savefig(out, facecolor=CREAM)
print("saved", out)
