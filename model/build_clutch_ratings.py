"""Build data/clutch_ratings.csv — the clutch index of record.

Supersedes data/clutch_players.csv (the frozen serve-only index, kept as the
archival record of what was published in content/clutch/).

What this is: the two-regime construction from model/clutch_regimes.py, fitted
unanchored (per-regime league constants, NOT the v2 skill offset — anchoring
to v2 pins each player's overall rally rate and re-imposes the zero-sum
artefact it was built to remove).  Every rally constrains all four players;
each player carries a serve level and a return level in each of two leverage
regimes.  See model/clutch_regimes.md.

Columns
-------
serve_regular / return_regular   levels on ordinary points
serve_big / return_big           levels on big points (top-quartile leverage)
regular / big                    the sums (mL+nL, mH+nH)
lift                             big - regular  == THE CLUTCH RATING
lift_se, z                       precision of the lift
skill_adj                        lift residualised on v2 value (clutch beyond
                                 skill — the defensible per-player cut)
lift_halfA / lift_halfB          the same rating fitted on disjoint halves of
                                 the matches, so anyone can see it replicate

Honest scope, carried in the file's own docs so it cannot be lost:
  * DESCRIPTIVE ONLY.  It does not predict who wins from 9-9, at any entry
    state (7-7 through 10-10) or player subset — model/clutch_endgame_sweep.py
    searched 30 cells and the best one did not survive a shuffled null.
  * It correlates +0.72 with skill.  That is expected, not a defect, but the
    skill_adj column is the right one for "more clutch than their level".
  * Individuals with small |z| are noise, exactly as before.

Run: python model/build_clutch_ratings.py      # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model"))

from sitelib import race                                   # noqa: E402
import clutch_srm as cs                                     # noqa: E402
import clutch_regimes as cr                                 # noqa: E402


def prep(blob, cur, traj):
    rows, _ = cs.build(blob, cur, traj)
    lz = np.array([r[4] for r in rows])
    hi_cut, lo_cut = np.quantile(lz, cr.HI_Q), np.quantile(lz, cr.LO_Q)
    keep = ([r + (True,) for r in rows if r[4] > hi_cut]
            + [r + (False,) for r in rows if r[4] < lo_cut])
    nhi, nlo = defaultdict(int), defaultdict(int)
    for r in keep:
        for u in r[:4]:
            (nhi if r[9] else nlo)[u] += 1
    ok = {u for u in nhi if nhi[u] >= cr.MIN_EACH and nlo[u] >= cr.MIN_EACH}
    keep = [r for r in keep if all(u in ok for u in r[:4])]
    return keep, {u: i for i, u in enumerate(sorted(ok))}, nhi, nlo


def fit(sub, index):
    """Unanchored: per-regime league constants, no v2 offset."""
    P = len(index)
    hi = np.array([r[9] for r in sub])
    y = np.array([r[6] for r in sub], dtype=float)
    rH, rL = y[hi].mean(), y[~hi].mean()
    off = np.where(hi, math.log(rH / (1 - rH)), math.log(rL / (1 - rL)))
    sub2 = [r[:5] + (float(off[i]),) + r[6:] for i, r in enumerate(sub)]
    b, se = cr.fit_regimes(sub2, index, anchor=True)
    return b, se, rH, rL


def main():
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])
    keep, index, nhi, nlo = prep(blob, cur, traj)
    P = len(index)
    uu = sorted(index)
    print(f"  {P} players, {len(keep)} rallies")

    b, se, rH, rL = fit(keep, index)
    mL, nL, mH, nH = b[:P], b[P:2 * P], b[2 * P:3 * P], b[3 * P:]
    lift = (mH + nH) - (mL + nL)
    lse = np.sqrt(se[:P] ** 2 + se[P:2 * P] ** 2
                  + se[2 * P:3 * P] ** 2 + se[3 * P:] ** 2)
    v2 = np.array([cur[u]["v"] for u in uu])
    adj = lift - np.polyval(np.polyfit(v2, lift, 1), v2)

    mids = sorted({r[7] for r in keep})
    half = {m: i % 2 for i, m in enumerate(mids)}
    bA, _, _, _ = fit([r for r in keep if half[r[7]] == 0], index)
    bB, _, _, _ = fit([r for r in keep if half[r[7]] == 1], index)
    lA = (bA[2 * P:3 * P] + bA[3 * P:]) - (bA[:P] + bA[P:2 * P])
    lB = (bB[2 * P:3 * P] + bB[3 * P:]) - (bB[:P] + bB[P:2 * P])
    print(f"  split-half r (lift) {np.corrcoef(lA, lB)[0, 1]:+.3f}; "
          f"skill-adj {np.corrcoef(lA - np.polyval(np.polyfit(v2, lA, 1), v2), lB - np.polyval(np.polyfit(v2, lB, 1), v2))[0, 1]:+.3f}")
    print(f"  league base rates: regular {rL:.4f}  big {rH:.4f}")
    print(f"  corr(v2, lift) {np.corrcoef(v2, lift)[0, 1]:+.3f}")

    ol = list(np.argsort(-lift))
    oa = list(np.argsort(-adj))
    out = []
    for i, u in enumerate(uu):
        out.append({
            "uuid": u, "name": cur[u]["name"], "gender": cur[u]["gender"],
            "n_high": nhi[u], "n_low": nlo[u],
            "serve_regular": round(float(mL[i]), 4),
            "return_regular": round(float(nL[i]), 4),
            "serve_big": round(float(mH[i]), 4),
            "return_big": round(float(nH[i]), 4),
            "regular": round(float(mL[i] + nL[i]), 4),
            "big": round(float(mH[i] + nH[i]), 4),
            "lift": round(float(lift[i]), 4),
            "lift_se": round(float(lse[i]), 4),
            "z": round(float(lift[i] / lse[i]), 2),
            "skill_adj": round(float(adj[i]), 4),
            "lift_rank": ol.index(i) + 1,
            "skill_adj_rank": oa.index(i) + 1,
            "lift_halfA": round(float(lA[i]), 4),
            "lift_halfB": round(float(lB[i]), 4)})
    out.sort(key=lambda d: -d["lift"])
    dest = ROOT / "data" / "clutch_ratings.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {dest} ({len(out)} players)")
    print(f"\n  {'player':24}{'lift':>8}{'z':>7}{'skill_adj':>11}")
    for d in out[:10]:
        print(f"  {d['name']:24}{d['lift']:+8.3f}{d['z']:>7.1f}{d['skill_adj']:+11.3f}")


if __name__ == "__main__":
    main()
