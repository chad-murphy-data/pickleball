"""Is clutch FORCED to be zero-sum?  No — that's the baseline's doing.

The question
------------
Both existing constructions measure clutch as a leverage GRADIENT:

    frozen:  mean(levz * resid)   over a player's serving rallies
    SRM:     a coefficient on     levz * (side indicator)

`levz` is centred within each game, so both are covariances, and the residual
is taken against a baseline fitted on ALL of a player's points -- the big ones
included.  That pins their average residual near zero, which means beating
the baseline on high-leverage points MECHANICALLY implies falling short on
low-leverage ones.  "Wins the big points" and "loses the small ones" are then
the same statement, and a uniformly-excellent player scores exactly 0.

That tradeoff is imposed by the estimator.  It is not a fact about pickleball.

The fix
-------
Estimate the baseline OUT OF REGIME.  Give every player two levels, fitted on
DISJOINT sets of rallies:

    logit P(serving side wins) = offset
        + [LOW-leverage rallies ]  (mL_s1 + mL_s2) - (nL_r1 + nL_r2)
        + [HIGH-leverage rallies]  (mH_s1 + mH_s2) - (nH_r1 + nH_r2)

Each rally contributes to exactly one regime, so mL and mH never see the same
data and nothing constrains their difference.  Then

    clutch_u = (mH_u - mL_u) + (nH_u - nL_u)

is "how much better is this player when the point is big than when it isn't",
with no zero-sum artefact.  And the two levels can be correlated across
players -- which is the empirical question the gradient estimators cannot
even ask:

    corr(low-leverage level, high-leverage level) across players
        ~ +1  -> players are just consistently good/bad; clutch is the
                 small residual departure, and NOT intrinsically a tradeoff
        ~  0  -> big-point ability is a genuinely separate dimension
        <  0  -> real tradeoff: the big-point specialists really are worse
                 on the small ones (the only case where zero-sum is REAL)

Run: python model/clutch_regimes.py        # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model"))

from sitelib import race  # noqa: E402
import clutch_srm as cs  # noqa: E402

HI_Q = 0.75      # top quartile of within-game leverage = "big points"
LO_Q = 0.50      # bottom half = the baseline regime
MIN_EACH = 150   # rallies required in EACH regime to be estimated


def fit_regimes(rows, index, lam=0.15, lam_lo=0.25, iters=400,
                anchor=True, league=0.0):
    """Params [mL | nL | mH | nH], P each.  Each rally loads exactly one
    regime, so the low- and high-leverage levels are fitted on disjoint data
    and their difference carries no mechanical constraint."""
    P = len(index)
    srv = np.array([[index[r[0]], index[r[1]]] for r in rows])
    rcv = np.array([[index[r[2]], index[r[3]]] for r in rows])
    hi = np.array([r[9] for r in rows], dtype=bool)
    # anchor=True  -> per-rally offset from v2 (pins each player's OVERALL
    #                 rally rate, which forces high-leverage gains to be paid
    #                 for at low leverage -- the zero-sum constraint smuggled
    #                 back in through the skill anchor)
    # anchor=False -> one league-average constant.  Each player's low- and
    #                 high-leverage levels then float freely relative to the
    #                 field, their sum is pinned by nothing, and "is big-point
    #                 ability opposed to ordinary ability" becomes a real
    #                 empirical question instead of an identity.
    off = (np.array([r[5] for r in rows]) if anchor
           else np.full(len(rows), league))
    y = np.array([r[6] for r in rows], dtype=float)
    pri = np.concatenate([np.full(2 * P, lam_lo), np.full(2 * P, lam)])
    sH, sL = hi.astype(float), (~hi).astype(float)

    def eta_of(b):
        mL, nL = b[:P], b[P:2 * P]
        mH, nH = b[2 * P:3 * P], b[3 * P:]
        return (off
                + sL * (mL[srv[:, 0]] + mL[srv[:, 1]]
                        - nL[rcv[:, 0]] - nL[rcv[:, 1]])
                + sH * (mH[srv[:, 0]] + mH[srv[:, 1]]
                        - nH[rcv[:, 0]] - nH[rcv[:, 1]]))

    def negll(b):
        e = np.clip(eta_of(b), -30, 30)
        p = 1 / (1 + np.exp(-e))
        f = -np.sum(y * np.log(np.clip(p, 1e-12, None))
                    + (1 - y) * np.log(np.clip(1 - p, 1e-12, None)))
        f += 0.5 * np.sum((b / pri) ** 2)
        r = p - y
        g = np.zeros(4 * P)
        rl, rh = r * sL, r * sH
        for c in (0, 1):
            np.add.at(g, srv[:, c], rl)
            np.add.at(g, P + rcv[:, c], -rl)
            np.add.at(g, 2 * P + srv[:, c], rh)
            np.add.at(g, 3 * P + rcv[:, c], -rh)
        g += b / pri ** 2
        return f, g

    res = minimize(negll, np.zeros(4 * P), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxfun": iters * 2})
    b = res.x
    e = np.clip(eta_of(b), -30, 30)
    p = 1 / (1 + np.exp(-e))
    w = p * (1 - p)
    h = np.zeros(4 * P)
    for c in (0, 1):
        np.add.at(h, srv[:, c], w * sL)
        np.add.at(h, P + rcv[:, c], w * sL)
        np.add.at(h, 2 * P + srv[:, c], w * sH)
        np.add.at(h, 3 * P + rcv[:, c], w * sH)
    h += 1 / pri ** 2
    return b, 1 / np.sqrt(h)


def main():
    print("Loading cached rallies ...")
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])
    rows, _ = cs.build(blob, cur, traj)

    lz = np.array([r[4] for r in rows])
    hi_cut, lo_cut = np.quantile(lz, HI_Q), np.quantile(lz, LO_Q)
    print(f"  leverage cuts: HIGH = top {int((1 - HI_Q) * 100)}% (levz > "
          f"{hi_cut:+.2f}),  LOW = bottom {int(LO_Q * 100)}% (levz < {lo_cut:+.2f})")

    keep = []
    for r in rows:
        if r[4] > hi_cut:
            keep.append(r + (True,))
        elif r[4] < lo_cut:
            keep.append(r + (False,))
    print(f"  {len(keep)} rallies in the two regimes "
          f"({sum(r[9] for r in keep)} high / {sum(1 for r in keep if not r[9])} low)")

    nhi, nlo = defaultdict(int), defaultdict(int)
    for r in keep:
        for u in r[:4]:
            (nhi if r[9] else nlo)[u] += 1
    ok = {u for u in nhi if nhi[u] >= MIN_EACH and nlo[u] >= MIN_EACH}
    keep = [r for r in keep if all(u in ok for u in r[:4])]
    index = {u: i for i, u in enumerate(sorted(ok))}
    P = len(index)
    print(f"  {P} players with >= {MIN_EACH} rallies in EACH regime; "
          f"{len(keep)} rallies retained\n")

    b, se = fit_regimes(keep, index)
    mL, nL = b[:P], b[P:2 * P]
    mH, nH = b[2 * P:3 * P], b[3 * P:]
    low = mL + nL                       # overall level on small points
    high = mH + nH                      # overall level on big points
    clutch = high - low

    v2 = np.array([cur[u]["v"] for u in sorted(ok)])
    names = [cur[u]["name"] for u in sorted(ok)]

    print("=" * 74)
    print("THE QUESTION: are big-point and small-point ability opposed?")
    print("=" * 74)
    r_lh = float(np.corrcoef(low, high)[0, 1])
    print(f"  corr(low-leverage level, high-leverage level) = {r_lh:+.3f}")
    print(f"    sd(low) {low.std():.4f}   sd(high) {high.std():.4f}   "
          f"sd(difference) {clutch.std():.4f}")
    print("  => DO NOT read this as 'the tradeoff is real'.  See the warning "
          "below —\n     the unconstrained estimator produces a result that "
          "cannot be true.")
    print(f"\n  For contrast, the gradient estimators FORCE this correlation "
          f"toward -1 by\n  construction, because they measure the residual "
          f"against a baseline fitted on\n  both regimes at once.")

    print(f"\n{'-' * 74}\ncorr(v2 skill, ...)   low {np.corrcoef(v2, low)[0, 1]:+.3f}"
          f"   high {np.corrcoef(v2, high)[0, 1]:+.3f}"
          f"   CLUTCH(high-low) {np.corrcoef(v2, clutch)[0, 1]:+.3f}")
    print("""
  *** THIS RESULT IS NOT CREDIBLE AS CLUTCH ***
  corr(v2, low) is strongly NEGATIVE: the estimator says elite players are
  BELOW their own skill baseline on ordinary points, and make it all back on
  big ones.  Nobody believes Anna Leigh Waters loses routine rallies at a
  below-baseline rate.  Removing the within-game centring removed the
  zero-sum artefact and let a DIFFERENT artefact in through the same door:
  the regime label is a function of the score path, which is itself a
  function of the rally outcomes, so the split is endogenous.  The centred
  (zero-sum) estimators are immune to that precisely BECAUSE they only ever
  compare a player to themselves within the same game.

  So the zero-sum property is not a mistake to be fixed — it is the price
  paid for identification.  An uncentred estimator needs an exogenous
  definition of 'big point' (see the notes in clutch_regimes.md).""")

    # permutation null: shuffle the regime LABEL within game
    print(f"\n{'-' * 74}\nPERMUTATION NULL (regime label shuffled within game, "
          f"full refit)\n{'-' * 74}")
    rng = np.random.default_rng(23)
    bygame = defaultdict(list)
    for i, r in enumerate(keep):
        bygame[(r[7], r[8])].append(i)
    perm = list(keep)
    for idxs in bygame.values():
        lab = [keep[i][9] for i in idxs]
        rng.shuffle(lab)
        for i, L in zip(idxs, lab):
            perm[i] = perm[i][:9] + (L,)
    nb, _ = fit_regimes(perm, index)
    nclutch = (nb[2 * P:3 * P] + nb[3 * P:]) - (nb[:P] + nb[P:2 * P])
    print(f"  real sd(clutch) {clutch.std():.4f}   "
          f"null sd(clutch) {nclutch.std():.4f}   "
          f"ratio {clutch.std() / max(nclutch.std(), 1e-9):.2f}x")

    se_c = np.sqrt(se[2 * P:3 * P] ** 2 + se[3 * P:] ** 2
                   + se[:P] ** 2 + se[P:2 * P] ** 2)
    out = [{"uuid": u, "name": cur[u]["name"], "gender": cur[u]["gender"],
            "n_high": nhi[u], "n_low": nlo[u],
            "low": float(low[i]), "high": float(high[i]),
            "clutch": float(clutch[i]), "clutch_se": float(se_c[i]),
            "z": float(clutch[i] / se_c[i])}
           for i, u in enumerate(sorted(ok))]
    out.sort(key=lambda d: -d["clutch"])
    print(f"\n{'-' * 74}\nTOP 12 — biggest lift from small points to big ones"
          f"\n{'-' * 74}")
    print(f"  {'player':24}{'low':>9}{'high':>9}{'lift':>9}{'z':>7}")
    for d in out[:12]:
        print(f"  {d['name']:24}{d['low']:+9.3f}{d['high']:+9.3f}"
              f"{d['clutch']:+9.3f}{d['z']:+7.1f}")
    print(f"\n  BOTTOM 5 (kept private in any writeup — naming chokers is "
          f"punching down)")
    for d in out[-5:]:
        print(f"  {'(withheld)':24}{d['low']:+9.3f}{d['high']:+9.3f}"
              f"{d['clutch']:+9.3f}{d['z']:+7.1f}")

    dest = ROOT / "data" / "clutch_regimes.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {dest}  ({len(out)} players)")


if __name__ == "__main__":
    main()
