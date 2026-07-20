"""Scatter: clutch (y) vs skill rating (x) — DUPR and our PICKLE rating.

Shows clutch is largely explained by skill (positive trend), AND that our
rating explains it far better than DUPR does (r 0.66 vs 0.27).

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
v2 = [(r["full_name"], float(r["value_now_mean"]), r["player_id"])
      for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open())]
dupr = {r["player_id"]: float(r["platform_rating_latest"])
        for r in csv.DictReader((ROOT / "data" / "platform_ratings.csv").open())}


def uuid_for(name, val):
    c = [(abs(v - val), pid) for nm, v, pid in v2 if nm == name]
    return min(c)[1] if c else None


P = []
for r in clu:
    pid = uuid_for(r["name"], float(r["value"]))
    P.append(dict(name=r["name"], z=float(r["z"]), val=float(r["value"]),
                  dupr=dupr.get(pid)))

INK = "#16321e"; DOT = "#3f6b45"; LIME = "#5a8f1f"; SAGE = "#6f8560"
CREAM = "#fbfdf3"; GRID = "#dbe7c8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

STARS = ["Anna Leigh Waters", "Ben Johns", "Anna Bright", "Gabriel Tardio"]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.4), dpi=200, sharey=True)
fig.patch.set_facecolor(CREAM)


def panel(ax, xget, xfilter, xlabel, title):
    pts = [(xget(p), p["z"], p["name"]) for p in P if xget(p) is not None and xfilter(xget(p))]
    x = np.array([a for a, _, _ in pts]); y = np.array([b for _, b, _ in pts])
    ax.set_facecolor(CREAM)
    ax.axhline(0, color=SAGE, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.scatter(x, y, s=60, c=DOT, alpha=0.72, edgecolor=CREAM, linewidth=0.7, zorder=3)
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
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=14.5, fontweight="bold", color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(INK)
    ax.tick_params(length=0); ax.grid(color=GRID, lw=0.7, zorder=0)


panel(axL, lambda p: p["dupr"], lambda v: 4.0 < v < 8.5, "DUPR rating", "Clutch vs. DUPR")
panel(axR, lambda p: p["val"], lambda v: True, "PICKLE rating (mine)", "Clutch vs. my PICKLE rating")
axL.set_ylabel("CLUTCH  —  wins the big points\nvs. their own baseline", fontsize=12.5, fontweight="bold")

fig.suptitle("Clutch skill is more than overall skill, and loosely correlates with player skill",
             x=0.012, ha="left", fontsize=17, fontweight="bold", color=INK, y=0.985)
fig.text(0.012, 0.02, "PICKLES · by Chad Murphy · chad-murphy-data.github.io/pickleball · 182 pros · "
         "each dot = one player · clutch on 162,942 rallies", fontsize=9.5, color=SAGE)
fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.16, wspace=0.06)
out = ROOT / "content" / "clutch" / "clutch_vs_rating.png"
fig.savefig(out, facecolor=CREAM)
print("saved", out)
