"""VERIFIER 4 — variance channel, re-derived independently.

  * race-share moments validated by direct Monte-Carlo simulation of games
  * z2 slope on wind, outdoor / indoor / difference, own seeds
  * V3 extra-upset statistic, own implementation
  * sensitivity of the V5 upset translation to how the z2 excess is attributed
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b6_frame import build, win_prob  # noqa: E402
import b6_lib as L  # noqa: E402

rng_global = np.random.default_rng(20260731)


# --------------------------------------------------- validate the moments --
def sim_share(p, T, n=400_000, seed=1):
    rng = np.random.default_rng(seed)
    a = np.zeros(n, int)
    b = np.zeros(n, int)
    live = np.ones(n, bool)
    for _ in range(400):
        if not live.any():
            break
        pt = rng.random(n) < p
        a[live & pt] += 1
        b[live & ~pt] += 1
        done = ((a >= T) | (b >= T)) & (np.abs(a - b) >= 2)
        live = live & ~done
    s = a / (a + b)
    return s.mean(), s.var()


print("--- validate exact race-share moments against simulation ---")
sm = L.ShareMoments(n=481)
for p in (0.40, 0.50, 0.60, 0.70):
    for T in (11, 15):
        m, sd = sm.moments(p, T)
        ms, vs = sim_share(p, T, 200_000, seed=int(p * 100) + T)
        print(f"  p={p} T={T}: exact mu={m:.5f} sd={sd:.5f} | "
              f"sim mu={ms:.5f} sd={math.sqrt(vs):.5f}")

# ---------------------------------------------------------------- frame ----
G = [g for g in build() if g["wind"] is not None
     and g["setting"] in ("outdoor", "indoor")]
for g in G:
    p = min(max(g["p"], 0.16), 0.84)
    mu, sd = sm.moments(p, g["T"])
    g["z2"] = ((g["share"] - mu) / sd) ** 2
    g["pw"] = sm.win(p, g["T"])
    g["pw_exact"] = win_prob(p, g["T"])
    g["sk"] = p - 0.5

for setting in ("outdoor", "indoor"):
    sub = [g for g in G if g["setting"] == setting]
    print(f"{setting}: n={len(sub)} mean z2={np.mean([g['z2'] for g in sub]):.4f}")
print("  |pw - pw_exact| max:",
      max(abs(g["pw"] - g["pw_exact"]) for g in G))


# ------------------------------------------------------- z2 slope + diff ---
def z2_slope(rows):
    w = np.array([g["wind"] / 10 for g in rows])
    y = np.array([g["z2"] for g in rows])
    X = np.column_stack([np.ones(len(rows)), w])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def upset_excess(rows, wlo=14):
    edges = np.linspace(0, 1, 11)
    num = den = 0.0
    for k in range(10):
        a = [g for g in rows if edges[k] <= g["pw"] < edges[k + 1]]
        calm = [g for g in a if g["wind"] < 8]
        windy = [g for g in a if g["wind"] >= wlo]
        if len(calm) < 5 or len(windy) < 5:
            continue
        ec = np.mean([(1.0 if g["won"] else 0.0) - g["pw"] for g in calm])
        ew = np.mean([(1.0 if g["won"] else 0.0) - g["pw"] for g in windy])
        sgn = -1.0 if edges[k] >= 0.5 else 1.0
        wgt = min(len(calm), len(windy))
        num += wgt * sgn * (ew - ec)
        den += wgt
    return num / den if den else np.nan


by = defaultdict(list)
for g in G:
    by[g["event"]].append(g)
keys = list(by)
print(f"\nevents = {len(keys)}")


def stat(rows):
    o = [g for g in rows if g["setting"] == "outdoor"]
    i = [g for g in rows if g["setting"] == "indoor"]
    if len(o) < 500 or len(i) < 500:
        return None
    return [z2_slope(o), z2_slope(i), z2_slope(o) - z2_slope(i),
            upset_excess(o), upset_excess(i),
            upset_excess(o) - upset_excess(i)]


base = stat(G)
rng = np.random.default_rng(555111)
draws = []
for _ in range(1500):
    pick = rng.integers(0, len(keys), len(keys))
    s = []
    for j in pick:
        s.extend(by[keys[j]])
    v = stat(s)
    if v is not None and np.all(np.isfinite(v)):
        draws.append(v)
d = np.array(draws)
lo = np.percentile(d, 2.5, axis=0)
hi = np.percentile(d, 97.5, axis=0)
names = ["z2 slope outdoor", "z2 slope indoor", "z2 slope OUT-IN",
         "upset excess outdoor", "upset excess indoor", "upset excess OUT-IN"]
print(f"({len(draws)} usable resamples, seed 555111)")
for k, nm in enumerate(names):
    scale = 100 if "upset" in nm else 1
    print(f"  {nm:24s} {scale*base[k]:+.4f} [{scale*lo[k]:+.4f}, {scale*hi[k]:+.4f}]"
          + (" pp" if scale == 100 else ""))

# -------------------------------------- V5 attribution sensitivity ---------
print("\n--- V5 upset translation: three attributions of the z2 excess ---")
mu, sd = sm.moments(0.60, 11)
z2b = float(np.mean([g["z2"] for g in G if g["setting"] == "outdoor"]))
print(f"mu={mu:.4f} sd_race={sd:.4f} mean z2 outdoor={z2b:.4f}")


def phi_upset(c):
    return 1 - 0.5 * (1 + math.erf(c / math.sqrt(2)))


c_race = (mu - 0.5) / sd
c_tot = (mu - 0.5) / (sd * math.sqrt(z2b))
print(f"base upset: race-sd framing {100*phi_upset(c_race):.2f}%  "
      f"total-sd framing {100*phi_upset(c_tot):.2f}%  "
      f"exact race DP {100*(1-win_prob(0.60,11)):.2f}%")
for tag, slope in (("point", base[0]), ("upper CI", hi[0]),
                   ("upper CI /0.941", hi[0] / 0.941)):
    dz = slope * 1.5
    f_mix = math.sqrt((z2b + dz) / z2b)          # tester's choice
    f_add = math.sqrt(1 + dz)                    # noise added to race variance
    print(f"  {tag:16s} dz={dz:+.4f}"
          f" | tester (race base, total f): {100*(phi_upset(c_race/f_mix)-phi_upset(c_race)):+.2f} pp"
          f" | total base+total f: {100*(phi_upset(c_tot/f_mix)-phi_upset(c_tot)):+.2f} pp"
          f" | race base+added-noise f: {100*(phi_upset(c_race/f_add)-phi_upset(c_race)):+.2f} pp")
