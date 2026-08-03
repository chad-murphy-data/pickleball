"""Does the regime estimator manufacture clutch out of nothing?

The threat
----------
Which rallies count as "big" is computed from the SCORE, and the score is the
result of earlier rallies.  So the set of rallies labelled high-leverage for a
player is not randomly assigned — it is selected by how the game has gone.
Blow someone out and your later rallies are all low-leverage; stay tied and
they are all high-leverage.  The label is endogenous to the outcomes being
measured.

If that selection alone can produce a lift spread — especially one correlated
with skill — then data/clutch_ratings.csv is measuring an artefact, and the
+0.72 skill correlation has an innocent explanation that has nothing to do
with clutch.

The test
--------
Simulate the whole archive under a world with NO CLUTCH AT ALL:

  * same games, same four players, same month-of-game v2 skill, same opening
    serve state
  * each side has ONE constant rally-win probability for the whole game
    (kA, kB from the anchored serve DP) — by construction there is no
    leverage effect, no player has any big-point ability
  * simulate the side-out dynamics rally by rally to a real final score

Then run the ENTIRE regime pipeline on the simulated games — same leverage
computation, same within-game standardisation, same global quantile cut, same
fit.  Any lift spread that appears is pure endogeneity, because the truth is
exactly zero for everyone.

This is a strictly stronger null than the label permutation already run.
Permuting labels within a game breaks the score-to-label link; this keeps that
link fully intact and removes only the clutch.

Run: python model/clutch_endogeneity.py         # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

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

from sitelib import race                                      # noqa: E402
from sitelib.winprob import (_table, eta_anchor,              # noqa: E402
                             serve_probs)
import clutch_srm as cs                                        # noqa: E402
import clutch_regimes as cr                                    # noqa: E402

K_LEAGUE = 0.443
SEED = 1234


def sim_game(rng, kA, kB, side, n, T=11, cap=30):
    """Side-out doubles to T, win by 2.  Returns [(a, b, side, n, won)].

    Serving side wins a rally with its constant k, scores and keeps serving;
    on a loss the serve passes #1 -> #2, and #2 -> side-out (the opening
    service turn starts at #2, the standard first-server exception).
    """
    a = b = 0
    out = []
    while True:
        if (a >= T and a - b >= 2) or (b >= T and b - a >= 2):
            return out
        if a > cap or b > cap:
            return out
        k = kA if side == 0 else kB
        won = 1 if rng.random() < k else 0
        out.append((a, b, side, n, won))
        if won:
            if side == 0:
                a += 1
            else:
                b += 1
        elif n == 1:
            n = 2
        else:
            side, n = 1 - side, 1


def leverage(V, T, a, b, state, side_A_serving):
    def val(x, y, s):
        if x >= T and x - y >= 2:
            return 1.0
        if y >= T and y - x >= 2:
            return 0.0
        return V.get((x, y, s), 0.5)
    if side_A_serving:
        w, l = val(a + 1, b, state), (val(a, b, 1) if state == 0 else val(a, b, 2))
    else:
        w, l = val(a, b + 1, state), (val(a, b, 3) if state == 2 else val(a, b, 0))
    return abs(w - l)


def main():
    print("Loading real games ...")
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])

    # --- real games: players, skill, opening serve state
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]

    specs = []
    for (mid, gn), rs in by_game.items():
        s0, s1 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s0) != 2 or len(s1) != 2:
            continue
        us = s0 + s1
        if not all(u in cur for u in us):
            continue
        rs.sort(key=lambda r: r["rally_number"])
        if any(r["server_score"] is None or r["server_side"] is None for r in rs):
            continue
        month = rs[0]["match_date"][:7]
        v = [traj[u].get(month, cur[u]["v"]) for u in us]
        specs.append((mid, gn, us, v, rs[0]["server_side"],
                      rs[0]["server_number"] or 2))
    print(f"  {len(specs)} games to simulate\n")

    rng = np.random.default_rng(SEED)
    rows = []
    tcache = {}
    for (mid, gn, us, v, side0, n0) in specs:
        eta = race.team_eta(v[0], v[1], v[2], v[3])
        p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
        ea = round(eta_anchor(p0) / 0.05) * 0.05
        kA, kB = serve_probs(ea, K_LEAGUE)
        key = (round(kA, 6), round(kB, 6))
        V = tcache.get(key)
        if V is None:
            V = _table(key[0], key[1], 11, 51)
            tcache[key] = V
        seq = sim_game(rng, kA, kB, side0, n0)
        if len(seq) < 6:
            continue
        levs = []
        for (a, b, side, n, won) in seq:
            st = (0 if n == 1 else 1) if side == 0 else (2 if n == 1 else 3)
            levs.append(leverage(V, 11, a, b, st, side == 0))
        L = np.array(levs)
        if L.std() < 1e-9:
            continue
        lz = (L - L.mean()) / L.std()
        off_A = math.log(kA / (1 - kA))
        off_B = math.log(kB / (1 - kB))
        for i, (a, b, side, n, won) in enumerate(seq):
            ss = us[:2] if side == 0 else us[2:]
            rr = us[2:] if side == 0 else us[:2]
            srv = ss[0] if n == 1 else ss[1]
            srvp = ss[1] if n == 1 else ss[0]
            j = (a + b) % 2
            rcv, rcvp = rr[j], rr[1 - j]
            rows.append((srv, srvp, rcv, rcvp, float(lz[i]),
                         off_A if side == 0 else off_B, int(won), mid, gn))
    print(f"  simulated {len(rows)} rallies\n")

    # --- identical regime pipeline
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
    index = {u: i for i, u in enumerate(sorted(ok))}
    P = len(index)
    uu = sorted(ok)
    print(f"  {P} players clear the rally bar in the simulation; "
          f"{len(keep)} rallies\n")

    hi = np.array([r[9] for r in keep])
    y = np.array([r[6] for r in keep], dtype=float)
    rH, rL = y[hi].mean(), y[~hi].mean()
    off = np.where(hi, math.log(rH / (1 - rH)), math.log(rL / (1 - rL)))
    keep2 = [r[:5] + (float(off[i]),) + r[6:] for i, r in enumerate(keep)]
    b, _ = cr.fit_regimes(keep2, index, anchor=True)
    mL, nL, mH, nH = b[:P], b[P:2 * P], b[2 * P:3 * P], b[3 * P:]
    lift = (mH + nH) - (mL + nL)
    v2 = np.array([cur[u]["v"] for u in uu])

    # split-half on the simulation too
    mids = sorted({r[7] for r in keep2})
    half = {m: i % 2 for i, m in enumerate(mids)}

    def fit_half(f):
        sub = [r for r in keep2 if half[r[7]] == f]
        hh = np.array([r[9] for r in sub])
        yy = np.array([r[6] for r in sub], dtype=float)
        aH, aL = yy[hh].mean(), yy[~hh].mean()
        o = np.where(hh, math.log(aH / (1 - aH)), math.log(aL / (1 - aL)))
        s2 = [r[:5] + (float(o[i]),) + r[6:] for i, r in enumerate(sub)]
        bb, _ = cr.fit_regimes(s2, index, anchor=True)
        return (bb[2 * P:3 * P] + bb[3 * P:]) - (bb[:P] + bb[P:2 * P])

    hA, hB = fit_half(0), fit_half(1)

    print("=" * 72)
    print("SIMULATED WORLD WITH ZERO CLUTCH — what the estimator reports anyway")
    print("=" * 72)
    print(f"  {'quantity':<34}{'SIMULATED':>12}{'REAL':>12}")
    print(f"  {'sd(lift)':<34}{lift.std():>12.4f}{0.3350:>12.4f}")
    print(f"  {'corr(v2 skill, lift)':<34}{np.corrcoef(v2, lift)[0, 1]:>+12.3f}"
          f"{0.721:>+12.3f}")
    print(f"  {'split-half r (lift)':<34}{np.corrcoef(hA, hB)[0, 1]:>+12.3f}"
          f"{0.629:>+12.3f}")
    print(f"  {'big-point league rate':<34}{rH:>12.4f}{0.4001:>12.4f}")
    print(f"  {'regular-point league rate':<34}{rL:>12.4f}{0.4418:>12.4f}")
    ratio = 0.3350 / max(lift.std(), 1e-9)
    print(f"\n  real spread / simulated spread = {ratio:.2f}x")
    verdict = ("ARTEFACT — the estimator invents this much clutch from the "
               "label selection alone"
               if lift.std() > 0.8 * 0.3350 else
               "the endogeneity produces some spread, but well short of the real one"
               if lift.std() > 0.3 * 0.3350 else
               "CLEAN — endogeneity produces almost no spread; the real rating "
               "is not this artefact")
    print(f"  => {verdict}")
    if abs(np.corrcoef(v2, lift)[0, 1]) > 0.3:
        print(f"  !! and it reproduces a skill correlation of "
              f"{np.corrcoef(v2, lift)[0, 1]:+.3f} with NO clutch in the data — "
              f"the +0.72 would then need re-interpreting")


if __name__ == "__main__":
    main()
