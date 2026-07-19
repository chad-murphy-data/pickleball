"""Render the clutch z-distribution chart (from data/clutch_players.csv).

The lay-audience test: if 'clutch' were pure luck, all 182 pros' scores
would fall under a standard bell curve. They spread wider — the best
players spill into a right tail the bell can't reach.

Run: python content/clutch/plot_zdist.py  ->  content/clutch/zdist.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
rows = list(csv.DictReader((ROOT / "data" / "clutch_players.csv").open()))
z = np.array([float(r["z"]) for r in rows])
names = [r["name"] for r in rows]
n = len(z)

INK = "#16321e"; SAGE = "#6f8560"; LIME = "#4e7a1e"; LIMEBAR = "#a9cc5e"
RED = "#c0492f"; NEUT = "#cdd9bd"; CREAM = "#fbfdf3"; GRID = "#dbe7c8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=200)
fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)

bw = 0.5; bins = np.arange(-4, 8.5, bw)
counts, edges = np.histogram(z, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
colors = [LIMEBAR if c > 1.96 else (RED if c < -1.96 else NEUT) for c in centers]
ax.bar(centers, counts, width=bw * 0.92, color=colors, edgecolor=CREAM, linewidth=1.2, zorder=3)

xs = np.linspace(-4, 8, 400); null = n * bw * stats.norm.pdf(xs)
ax.plot(xs, null, color=INK, lw=2.4, ls=(0, (5, 3)), zorder=4)
ax.fill_between(xs, null, color=INK, alpha=0.05, zorder=1)

ymax = counts.max() * 1.18
ax.axvline(1.96, color=SAGE, lw=1, ls=":", zorder=2)
ax.text(2.05, ymax * 0.82, "beyond\nwhat luck\nexplains →", color=LIME, fontsize=11.5,
        fontweight="bold", ha="left", va="top", linespacing=1.3)

tail = [("Anna Leigh\nWaters", "Anna Leigh Waters", 10.2),
        ("Ben Johns", "Ben Johns", 8.0), ("Anna Bright", "Anna Bright", 6.2),
        ("Gabriel Tardio", "Gabriel Tardio", 4.4), ("Christian Alshon", "Christian Alshon", 3.0)]
for lab, nm, yy in tail:
    zz = z[names.index(nm)]
    ax.annotate(lab, xy=(zz, 0.6), xytext=(zz, yy), fontsize=11.5, fontweight="bold",
                color=INK, ha="center", va="bottom", linespacing=1.0,
                arrowprops=dict(arrowstyle="-", color=SAGE, lw=1.1))

ax.text(0, n * bw * stats.norm.pdf(0) + 0.8, "what pure luck\nwould produce", color=INK,
        fontsize=11.5, ha="center", style="italic", linespacing=1.15)

ax.set_xlim(-4.6, 9.0); ax.set_ylim(0, ymax)
ax.set_ylabel("number of players", fontsize=12.5)
ax.set_xlabel("")
ax.text(-3.1, -ymax * 0.115, "← gives points away", color=RED, fontsize=12, ha="center", fontweight="bold")
ax.text(0.1, -ymax * 0.115, "average", color=SAGE, fontsize=12, ha="center")
ax.text(5.6, -ymax * 0.115, "wins the big points →", color=LIME, fontsize=12, ha="center", fontweight="bold")
fig.text(0.5, 0.075, "each player's rate of winning the biggest points, vs. their own average",
         fontsize=11, color=SAGE, ha="center", style="italic")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(INK)
ax.tick_params(length=0); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)

fig.suptitle("Is “clutch” real? Every pro vs. what luck alone would produce",
             x=0.015, ha="left", fontsize=18.5, fontweight="bold", color=INK, y=0.985)
ax.set_title("If clutch were just luck, all 182 pros would fit under the dashed bell. They "
             "don't — the best spill into the tail.", loc="left", fontsize=12.5, color=SAGE, pad=14)
fig.text(0.015, 0.015, "PICKLES · 162,942 rallies · clutch = winning high-leverage points "
         "above your own rate", fontsize=9.5, color=SAGE)
fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.17)
out = ROOT / "content" / "clutch" / "zdist.png"
fig.savefig(out, facecolor=CREAM)
print("saved", out, "| var(z)=%.2f  n>1.96=%d" % (z.var(), (z > 1.96).sum()))
