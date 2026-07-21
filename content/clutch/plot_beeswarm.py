"""Beeswarm version of the clutch distribution — every pro is one dot.

Lay-audience test #2: instead of a histogram + null curve, each of the 182
pros is a single dot placed at their clutch score.  Most cluster near
average (a zone where luck alone could put them); the best players pull
away to the right, further than luck can explain.

Run: python content/clutch/plot_beeswarm.py  ->  content/clutch/beeswarm.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
rows = list(csv.DictReader((ROOT / "data" / "clutch_players.csv").open()))
z = np.array([float(r["z"]) for r in rows])
names = np.array([r["name"] for r in rows])

INK = "#16321e"; SAGE = "#6f8560"; LIME = "#4e7a1e"; LIMEDOT = "#8fbf3f"
RED = "#c0492f"; NEUT = "#bccca6"; CREAM = "#fbfdf3"; GRID = "#dbe7c8"; BAND = "#eef4e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK})

# --- symmetric bin-stack swarm layout ---
bw = 0.34
order = np.argsort(z)
zx = z[order]; nm = names[order]
ys = np.zeros(len(zx))
bin_id = np.floor(zx / bw).astype(int)
for b in np.unique(bin_id):
    idx = np.where(bin_id == b)[0]
    offs = []
    k = 0
    while len(offs) < len(idx):
        offs.append(k)
        if k > 0:
            offs.append(-k)
        k += 1
    ys[idx] = np.array(offs[:len(idx)]) * 1.0

fig, ax = plt.subplots(figsize=(12, 6.6), dpi=200)
fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)

# "could be luck" band
ax.axvspan(-1.96, 1.96, color=BAND, zorder=0)
ax.text(0, 12.4, "could be random luck", color=SAGE, fontsize=12.5, ha="center",
        style="italic", fontweight="bold")
ax.axvline(0, color=SAGE, lw=1, ls=(0, (4, 4)), zorder=1)

cols = np.where(zx > 1.96, LIMEDOT, np.where(zx < -1.96, RED, NEUT))
ax.scatter(zx, ys, s=95, c=cols, edgecolor=CREAM, linewidth=1.1, zorder=3)

# label the confident tail names (each is alone in its bin)
tag = [("Anna Leigh Waters", "Anna Leigh Waters", 6.4),
       ("Ben Johns", "Ben Johns", 4.6), ("Anna Bright", "Anna Bright", 2.8),
       ("Gabriel Tardio", "Gabriel Tardio", 6.0), ("Christian Alshon", "Christian Alshon", 3.4)]
for lab, who, yy in tag:
    i = np.where(nm == who)[0][0]
    ax.annotate(lab, xy=(zx[i], ys[i] + 0.4), xytext=(zx[i], yy),
                fontsize=11.5, fontweight="bold", color=INK, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=SAGE, lw=1.1))

# count callout on the right, in the open lower-right space
n_beyond = int((z > 1.96).sum())
ax.plot([2.0, 8.3], [-2.6, -2.6], color=LIME, lw=1.4, zorder=2)
ax.plot([2.0, 2.0], [-2.1, -3.1], color=LIME, lw=1.4, zorder=2)
ax.plot([8.3, 8.3], [-2.1, -3.1], color=LIME, lw=1.4, zorder=2)
ax.text(5.15, -4.6, f"{n_beyond} players out here.\nLuck alone would put ~4.",
        fontsize=13, color=LIME, fontweight="bold", ha="center", va="top", linespacing=1.3)

ax.set_xlim(-4.6, 8.7); ax.set_ylim(-12, 14)
ax.set_yticks([])
ax.set_xticks(range(-4, 9, 2))
ax.tick_params(length=0, labelsize=12)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK)

# plain zone labels
ax.text(-3.1, -13.6, "← gives points away", color=RED, fontsize=12, ha="center", fontweight="bold")
ax.text(0, -13.6, "average", color=SAGE, fontsize=12, ha="center")
ax.text(5.6, -13.6, "wins the big points →", color=LIME, fontsize=12, ha="center", fontweight="bold")

fig.suptitle("Is “clutch” real? Every pro is one dot.", x=0.015, ha="left",
             fontsize=19, fontweight="bold", color=INK, y=0.98)
ax.set_title("Most pros sit in the shaded zone, where luck alone could put them. The best players "
             "pull away —\nmore clutch than chance can explain.", loc="left", fontsize=12.5,
             color=SAGE, pad=12, linespacing=1.3)
fig.text(0.015, 0.02, "PICKLES · 162,942 rallies · clutch = winning high-leverage points above "
         "your own rate · each dot = one pro (≥300 rallies)", fontsize=9.5, color=SAGE)
fig.subplots_adjust(left=0.03, right=0.985, top=0.83, bottom=0.14)
out = ROOT / "content" / "clutch" / "beeswarm.png"
fig.savefig(out, facecolor=CREAM)
print("saved", out, "| n beyond 1.96 =", n_beyond)
