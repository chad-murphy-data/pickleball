"""VERIFIER 2 — independent re-derivation of the binned(-2.0pp) -> continuous d
translation, the step-vs-linear projection factor, and the d -> upset map."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b6_frame import build, win_prob  # noqa: E402

G = [g for g in build() if g["wind"] is not None]


def translate(rows, clamp, step, target=-0.020, wlo=14, whi=20):
    calm = [g for g in rows if g["wind"] < 8]
    windy = [g for g in rows if wlo <= g["wind"] < whi]
    def s_of(g):
        p = min(max(g["p"], 0.16), 0.84) if clamp else g["p"]
        return abs(p - 0.5)
    def pred(g, d):
        s = s_of(g)
        w = g["wind"] / 10.0
        mult = (1 + d * (1.0 if w >= 1.4 else 0.0)) if step else (1 + d * w)
        p = 0.5 + s * mult
        p = min(max(p, 0.02), 0.98)
        return win_prob(p, g["T"])
    def drift(d):
        a = np.mean([pred(g, d) - pred(g, 0.0) for g in windy])
        b = np.mean([pred(g, d) - pred(g, 0.0) for g in calm])
        return a - b
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if drift(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, len(calm), len(windy)


for labkey, labname in (("heur", "heuristic"), ("setting", "corrected")):
    rows = [g for g in G if g[labkey] == "outdoor"]
    for clamp in (True, False):
        dl, nc, nw = translate(rows, clamp, step=False)
        ds, _, _ = translate(rows, clamp, step=True)
        print(f"{labname:9s} clamp={clamp!s:5s} calm={nc} windy={nw} "
              f"d_linear={dl:+.4f} d_step={ds:+.4f}")

# --- sanity: phase-1's representative-favourite shortcut -------------------
rows = [g for g in G if g["heur"] == "outdoor"]
calm = [g for g in rows if g["wind"] < 8]
windy = [g for g in rows if 14 <= g["wind"] < 20]
pf = np.mean([max(win_prob(g["p"], g["T"]), 1 - win_prob(g["p"], g["T"]))
              for g in windy])
print("\nmean predicted FAV win prob in 14-20 bin:", round(pf, 4))
# invert to a representative share for a race to 11
grid = np.linspace(0.5, 0.85, 3501)
wp = np.array([win_prob(p, 11) for p in grid])
rep = float(np.interp(pf, wp, grid))
slope = float(np.gradient(wp, grid)[np.searchsorted(grid, rep)])
dw = np.mean([g["wind"] for g in windy]) / 10 - np.mean([g["wind"] for g in calm]) / 10
print(f"representative share={rep:.4f} skill={rep-0.5:.4f} dPwin/dshare={slope:.3f} dw={dw:.3f}")
print("phase-1 style d = ", round(-0.020 / ((rep - 0.5) * dw * slope), 4))

# distribution-averaged sensitivity (the correct linearisation)
def sens(sub, clamp=False):
    v = []
    for g in sub:
        p = min(max(g["p"], 0.16), 0.84) if clamp else g["p"]
        s = abs(p - 0.5)
        h = 1e-4
        dP = (win_prob(0.5 + s + h, g["T"]) - win_prob(0.5 + s - h, g["T"])) / (2 * h)
        v.append(s * dP)
    return float(np.mean(v))
sw = sens(windy)
sc = sens(calm)
ww = np.mean([g["wind"] for g in windy]) / 10
wc = np.mean([g["wind"] for g in calm]) / 10
print(f"E[s*P'] windy={sw:.4f} calm={sc:.4f}; linearised d = "
      f"{-0.020/(sw*ww - sc*wc):+.4f}")

# --- OLS projection of a 14mph step onto the linear interaction -----------
for labkey, labname in (("heur", "heuristic"), ("setting", "corrected")):
    rows = [g for g in G if g[labkey] == "outdoor"]
    for clamp in (True, False):
        s = np.array([(min(max(g["p"], 0.16), 0.84) if clamp else g["p"]) - 0.5
                      for g in rows])
        w = np.array([g["wind"] / 10.0 for g in rows])
        X = np.column_stack([np.ones(len(rows)), s, w, s * w])
        tgt = s * (w >= 1.4)
        proj = np.linalg.lstsq(X, tgt, rcond=None)[0]
        print(f"{labname} clamp={clamp} projection factor = {proj[3]:.4f}")

# --- d -> upset translation for the reference favourite -------------------
print("\n--- d -> upset probability, skill=+0.10, 5->20 mph, race to 11 ---")
p0 = 0.60
base = win_prob(p0, 11)
print("win(0.60,11) =", round(base, 5))
for d in (-0.0020, -0.0376, -0.0400, -0.0933, -0.0996, -0.1448, -0.2062):
    p1 = 0.5 + 0.10 * (1 + d * 1.5)
    print(f"  d={d:+.4f}  share {p0:.4f}->{p1:.5f}  "
          f"win {base:.4f}->{win_prob(p1,11):.4f}  "
          f"upset +{100*(base-win_prob(p1,11)):.2f} pp")
