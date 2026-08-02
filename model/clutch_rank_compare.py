"""Rank-order agreement between the frozen serve-only clutch index and the
SRM refit — on the OVERLAPPING players only.

Comparing ranks across the two populations would be meaningless: the frozen
index needs >=300 SERVING rallies (182 players, essentially the top pros),
the SRM needs >=300 rallies ON COURT (1,148, a much wider net).  A player
ranked 80th of 1,148 and 11th of 182 may be in identical shape.  So: keep
only players in both, RE-RANK within that set, and correlate.

Also reports the attenuation ceiling.  Both indices are measured with error,
so the observed correlation cannot reach 1 even if they measure exactly the
same thing; the ceiling is sqrt(rel_frozen * rel_srm).  Without it a modest
correlation reads as disagreement when it may be at the maximum possible.

Run: python model/clutch_rank_compare.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent.parent

# reliability inputs
FROZEN_SPLIT_HALF = 0.15        # clutch.md, all 182 players (half-length)
SRM_NULL_SD = 0.0412            # clutch_srm.py permutation null
SRM_REAL_SD = 0.0928


def load():
    v2 = {}
    for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open()):
        v2[r["player_id"].lower()] = {"name": r["full_name"],
                                      "v": float(r["value_now_mean"]),
                                      "gender": r["gender"]}
    srm = {}
    for r in csv.DictReader((ROOT / "data" / "clutch_srm.csv").open()):
        srm[r["uuid"].lower()] = {k: float(r[k]) for k in
                                  ("attack", "defend", "total")}
        srm[r["uuid"].lower()]["n"] = int(r["n"])

    # frozen csv is keyed by NAME; resolve to uuid via the v2 value it
    # carries, since names are not identity (three Kawamotos, two twins)
    by_name = defaultdict(list)
    for u, d in v2.items():
        by_name[d["name"]].append((d["v"], u))
    pairs, unmatched = [], []
    for r in csv.DictReader((ROOT / "data" / "clutch_players.csv").open()):
        cand = by_name.get(r["name"], [])
        if not cand:
            unmatched.append(r["name"])
            continue
        u = min(cand, key=lambda x: abs(x[0] - float(r["value"])))[1]
        if u not in srm:
            unmatched.append(r["name"] + " (not in SRM)")
            continue
        pairs.append({"uuid": u, "name": r["name"], "gender": r["gender"],
                      "frozen": float(r["clutch"]),
                      "frozen_z": float(r["z"]),
                      "frozen_n": int(r["n_rallies"]),
                      "v": v2[u]["v"], **srm[u]})
    return pairs, unmatched


def boot_spearman(x, y, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for idx in rng.integers(0, len(x), (n, len(x))):
        out.append(spearmanr(x[idx], y[idx]).statistic)
    return np.percentile(out, [2.5, 97.5])


def main():
    pairs, unmatched = load()
    n = len(pairs)
    print(f"Overlap: {n} players in BOTH indices"
          + (f"   ({len(unmatched)} unmatched: {unmatched})" if unmatched else ""))

    fz = np.array([p["frozen"] for p in pairs])
    tot = np.array([p["total"] for p in pairs])
    att = np.array([p["attack"] for p in pairs])
    dfd = np.array([p["defend"] for p in pairs])
    v = np.array([p["v"] for p in pairs])

    print(f"\n{'=' * 70}\nRANK-ORDER AGREEMENT (Spearman, re-ranked within the "
          f"overlap)\n{'=' * 70}")
    for label, y in (("SRM total  (attack+defend)", tot),
                     ("SRM attack (serve side — the like-for-like cut)", att),
                     ("SRM defend (return side — frozen index cannot see it)", dfd)):
        rho = spearmanr(fz, y).statistic
        tau = kendalltau(fz, y).statistic
        lo, hi = boot_spearman(fz, y)
        print(f"  frozen vs {label:52} rho {rho:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}]  tau {tau:+.3f}")

    # ---- skill-adjusted: the comparison that matters, since both indices
    # track skill and that shared component inflates raw agreement
    print(f"\n{'-' * 70}\nSKILL-ADJUSTED (each index residualised on v2 value "
          f"within the overlap)\n{'-' * 70}")

    def resid(y):
        b = np.polyfit(v, y, 1)
        return y - np.polyval(b, v)

    rf, rt, ra = resid(fz), resid(tot), resid(att)
    for label, y in (("SRM total", rt), ("SRM attack", ra)):
        rho = spearmanr(rf, y).statistic
        lo, hi = boot_spearman(rf, y)
        print(f"  frozen-resid vs {label:24} rho {rho:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    print(f"  (raw agreement is inflated by both indices tracking skill: "
          f"corr(v2, frozen) {np.corrcoef(v, fz)[0, 1]:+.2f}, "
          f"corr(v2, SRM total) {np.corrcoef(v, tot)[0, 1]:+.2f})")

    # ---- attenuation ceiling
    rel_f = 2 * FROZEN_SPLIT_HALF / (1 + FROZEN_SPLIT_HALF)   # Spearman-Brown
    rel_s = 1 - (SRM_NULL_SD ** 2) / (SRM_REAL_SD ** 2)
    ceiling = float(np.sqrt(rel_f * rel_s))
    rho_tot = spearmanr(fz, tot).statistic
    print(f"\n{'-' * 70}\nATTENUATION CEILING\n{'-' * 70}")
    print(f"  frozen reliability  ~{rel_f:.2f}  (split-half {FROZEN_SPLIT_HALF} "
          f"over all 182, Spearman-Brown corrected)")
    print(f"  SRM reliability     ~{rel_s:.2f}  (1 - null var / real var, "
          f"permutation refit)")
    print(f"  => max correlation two indices this noisy could show: "
          f"{ceiling:.2f}")
    print(f"  observed {rho_tot:+.3f} vs ceiling {ceiling:.2f} — "
          + ("AT/ABOVE the ceiling: they agree as much as their noise allows"
             if rho_tot >= ceiling * 0.95 else
             "below the ceiling: genuine disagreement beyond measurement error"))

    # ---- where they disagree: reliable-frozen subset only
    print(f"\n{'-' * 70}\nRESTRICTED TO THE FROZEN INDEX'S RELIABLE REGION\n{'-' * 70}")
    for cut in (1.5, 2.5):
        m = np.abs(np.array([p["frozen_z"] for p in pairs])) >= cut
        if m.sum() > 8:
            print(f"  |frozen z| >= {cut}  (n={m.sum():3d})  "
                  f"rho {spearmanr(fz[m], tot[m]).statistic:+.3f}")
    print("  (clutch.md: split-half is 0.15 over all 182 but 0.61 at |z|>1.5 "
          "and 0.81 at |z|>2.5 — so agreement SHOULD rise here if the SRM is "
          "measuring the same thing better)")

    # ---- top-20 set overlap
    fo = [p["name"] for p in sorted(pairs, key=lambda p: -p["frozen"])[:20]]
    so = [p["name"] for p in sorted(pairs, key=lambda p: -p["total"])[:20]]
    both = sorted(set(fo) & set(so))
    print(f"\n{'-' * 70}\nTOP-20 SET OVERLAP: {len(both)}/20 shared\n{'-' * 70}")
    print(f"  in both : {', '.join(both)}")
    print(f"  frozen only: {', '.join(sorted(set(fo) - set(so)))}")
    print(f"  SRM only   : {', '.join(sorted(set(so) - set(fo)))}")

    # ---- biggest movers
    fr_rank = {p["name"]: i + 1 for i, p in
               enumerate(sorted(pairs, key=lambda p: -p["frozen"]))}
    sr_rank = {p["name"]: i + 1 for i, p in
               enumerate(sorted(pairs, key=lambda p: -p["total"]))}
    mv = sorted(pairs, key=lambda p: -abs(fr_rank[p["name"]] - sr_rank[p["name"]]))
    print(f"\n{'-' * 70}\nBIGGEST RANK MOVES (of {n})\n{'-' * 70}")
    print(f"  {'player':24}{'frozen':>8}{'SRM':>6}{'move':>7}{'srv rallies':>13}")
    for p in mv[:10]:
        f_, s_ = fr_rank[p["name"]], sr_rank[p["name"]]
        print(f"  {p['name']:24}{f_:>8}{s_:>6}{f_ - s_:>+7}{p['frozen_n']:>13}")


if __name__ == "__main__":
    main()
