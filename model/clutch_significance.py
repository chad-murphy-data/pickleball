"""Which players are CONFIDENTLY clutch? — significance + distribution.

The clutch z-score for a player is (their high-leverage over-delivery) /
(the SD of that statistic under a within-game leverage shuffle).  Under
the null "this player has no clutch tendency," z ~ N(0,1).  So:

  * z > 1.645  = one-tailed p < 0.05     (the user's threshold)
  * z > 1.96   = two-tailed p < 0.05
  * z > 2.576  = two-tailed p < 0.01

But with ~180 players, some clear those bars by chance.  This script does
it honestly:
  1. observed vs chance-expected counts in each tail,
  2. Benjamini-Hochberg FDR to name a defensible "confidently clutch" list,
  3. the whole z-distribution vs N(0,1) (the population-level test — is the
     spread of z wider than noise allows?),
  4. the same after residualizing clutch on skill (is there clutch BEYOND
     just being good?).

Run: python model/clutch_significance.py   (re-runs the rally reconstruction;
a few minutes).  Writes model/clutch_significance.json + data/clutch_players.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import big_points as bp        # noqa: E402
import spec_shootout as sx     # noqa: E402


def bh_fdr(pvals, q):
    """Benjamini-Hochberg: return boolean 'reject' array at FDR level q."""
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(m, bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        reject[order[: kmax + 1]] = True
    return reject


def main():
    games = sx.load_games()
    players, chem = sx.load_v2()
    per, _ = bp.clutch(games, players, chem)      # {uuid: {name,n,clutch,null_sd,z,...}}

    rows = []
    for u, d in per.items():
        rows.append(dict(uuid=u, name=d["name"], n=d["n"], z=d["z"],
                         clutch=d["clutch"], null_sd=d["null_sd"],
                         value=players[u]["v"], gender=players[u]["gender"]))
    rows.sort(key=lambda r: -r["z"])
    m = len(rows)
    z = np.array([r["z"] for r in rows])
    # one-tailed p (clutch is directional: are you ABOVE zero?)
    p1 = stats.norm.sf(z)

    # ---- 1. tail counts vs chance ----
    tiers = [("z>1.645 (1-tail p<.05)", 1.645), ("z>1.96 (2-tail p<.05)", 1.96),
             ("z>2.326 (1-tail p<.01)", 2.326), ("z>2.576 (2-tail p<.01)", 2.576),
             ("z>3.09 (1-tail p<.001)", 3.09)]
    tail = []
    for label, thr in tiers:
        obs = int((z > thr).sum())
        exp = m * stats.norm.sf(thr)
        tail.append(dict(label=label, thr=thr, observed=obs,
                         expected_by_chance=round(exp, 1),
                         excess=round(obs - exp, 1)))

    # ---- 2. BH-FDR lists ----
    rej05 = bh_fdr(p1, 0.05)
    rej10 = bh_fdr(p1, 0.10)
    bonf = 0.05 / m
    confident = [dict(name=rows[i]["name"], z=round(rows[i]["z"], 1),
                      n=rows[i]["n"], value=round(rows[i]["value"], 2),
                      fdr05=bool(rej05[i]), fdr10=bool(rej10[i]),
                      bonferroni=bool(p1[i] < bonf))
                 for i in range(m) if rej10[i]]

    # ---- 3. distribution vs N(0,1) ----
    ks = stats.kstest(z, "norm")
    dist = dict(n_players=m, mean_z=round(float(z.mean()), 3),
                var_z=round(float(z.var()), 3),
                sd_ratio_vs_null=round(float(z.std()), 3),
                ks_stat=round(float(ks.statistic), 3), ks_p=float(ks.pvalue),
                note="under pure noise: mean 0, var 1. var>1 = real spread of "
                     "clutch across players.")

    # ---- 4. residualize on skill: is there clutch BEYOND being good? ----
    val = np.array([r["value"] for r in rows])
    cl = np.array([r["clutch"] for r in rows])
    nsd = np.array([r["null_sd"] for r in rows])
    b1, b0 = np.polyfit(val, cl, 1)
    resid_z = (cl - (b0 + b1 * val)) / nsd
    rp1 = stats.norm.sf(resid_z)
    rrej10 = bh_fdr(rp1, 0.10)
    resid = dict(corr_clutch_value=round(float(np.corrcoef(val, cl)[0, 1]), 3),
                 var_resid_z=round(float(resid_z.var()), 3),
                 n_pass_fdr10=int(rrej10.sum()),
                 obs_gt_1p96=int((resid_z > 1.96).sum()),
                 exp_gt_1p96=round(m * stats.norm.sf(1.96), 1),
                 survivors=[rows[i]["name"] for i in np.argsort(-resid_z)
                            if rrej10[i]][:20],
                 note="clutch minus the part explained by overall skill. "
                      "var still >1 = SOME clutch beyond skill exists in the "
                      "population; few/no individuals clear FDR.")

    out = dict(n_players=m, threshold_counts=tail,
               n_confident_fdr05=int(rej05.sum()),
               n_confident_fdr10=int(rej10.sum()),
               confidently_clutch=confident,
               z_distribution=dist, beyond_skill=resid)
    (ROOT / "model" / "clutch_significance.json").write_text(json.dumps(out, indent=1))

    with (ROOT / "data" / "clutch_players.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "gender", "n_rallies", "clutch", "null_sd", "z",
                    "p_one_tailed", "value", "fdr05", "fdr10"])
        for i, r in enumerate(rows):
            w.writerow([r["name"], r["gender"], r["n"], round(r["clutch"], 4),
                        round(r["null_sd"], 4), round(r["z"], 2), f"{p1[i]:.2e}",
                        round(r["value"], 3), int(rej05[i]), int(rej10[i])])

    # ---- console report ----
    print(f"\n{m} players (>=300 serving rallies)\n")
    print("TAIL COUNTS (observed vs chance):")
    for t in tail:
        print(f"  {t['label']:26s} observed {t['observed']:3d}   "
              f"chance {t['expected_by_chance']:5.1f}   excess +{t['excess']}")
    print(f"\nCONFIDENTLY CLUTCH (Benjamini-Hochberg FDR): "
          f"{out['n_confident_fdr05']} at q=.05, {out['n_confident_fdr10']} at q=.10")
    print(f"{'player':24s} {'z':>5} {'n':>5} {'val':>5}  FDR.05 FDR.10 Bonf")
    for c in confident:
        print(f"  {c['name']:22s} {c['z']:5.1f} {c['n']:5d} {c['value']:5.2f}"
              f"    {'Y' if c['fdr05'] else '·':^5} {'Y' if c['fdr10'] else '·':^5}"
              f" {'Y' if c['bonferroni'] else '·':^4}")
    print(f"\nZ-DISTRIBUTION vs N(0,1):  var = {dist['var_z']} (noise = 1.0);  "
          f"z's are {dist['sd_ratio_vs_null']}x wider than chance;  "
          f"KS p = {dist['ks_p']:.1e}")
    print(f"\nBEYOND SKILL (clutch minus skill):  corr(clutch,skill) = "
          f"{resid['corr_clutch_value']};  residual var = {resid['var_resid_z']} "
          f"(still >1);  {resid['obs_gt_1p96']} players z>1.96 vs "
          f"{resid['exp_gt_1p96']} by chance;  {resid['n_pass_fdr10']} clear FDR.10")
    print(f"  residual survivors: {', '.join(resid['survivors']) or '(none)'}")


if __name__ == "__main__":
    main()
