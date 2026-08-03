"""Are there players who overperform in clutch moments MORE OFTEN than chance?

A different question from everything else in this thread.  Every prior
construction measured the SIZE of a player's clutch edge.  This one asks about
CONSISTENCY: per game, did your side beat expectation on the clutch rallies —
yes or no — and does anyone hit 1 more often than the coin says?

That is not merely a coarser version of the continuous test.  A player who is
slightly better in clutch nearly every game and one who has two enormous games
can share a mean while having very different hit rates.  Binomial consistency
can see the first; a mean cannot.

Design
------
* Clutch rallies: any rally with a side on 9+ (the plain-language definition).
* Expectation: fitted from the state-FE model with the CLUTCH TERMS REMOVED —
  state fixed effects plus per-player level terms.  So a rally's expected
  outcome is adjusted for the score state and for all four players' overall
  quality, and is NEVER derived from the player's own non-clutch rate.  That
  is what killed the earlier constructions.
* Per player-game: actual clutch rallies won by their side vs expected.
  indicator = 1 if actual > expected.
* Per player: hit rate k/n across their games.

The gate
--------
Identical pipeline on the ZERO-CLUTCH SIMULATION.  Under the null the spread
of hit rates should be binomial noise.  If real and simulated spreads match
again, consistency is as empty as magnitude was.

Run: python model/clutch_binary.py            # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

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

from sitelib import race                                    # noqa: E402
from sitelib.winprob import eta_anchor, serve_probs         # noqa: E402
import clutch_srm as cs                                      # noqa: E402
import clutch_endogeneity as ce                              # noqa: E402

CLUTCH = lambda a, b: max(a, b) >= 9      # noqa: E731
MIN_GAMES = 40
CAP = 16


def baseline_fit(rows, index, states, lam=0.20, iters=400):
    """State FE + player levels, NO clutch term.  Returns per-rally expected
    P(serving side wins)."""
    P, S = len(index), len(states)
    srv = np.array([[index[r[0]], index[r[1]]] for r in rows])
    rcv = np.array([[index[r[2]], index[r[3]]] for r in rows])
    st = np.array([states[r[4]] for r in rows])
    y = np.array([r[6] for r in rows], dtype=float)
    pri = np.concatenate([np.full(S, 25.0), np.full(2 * P, lam)])

    def eta_of(p):
        phi, m, nn = p[:S], p[S:S + P], p[S + P:]
        return phi[st] + m[srv[:, 0]] + m[srv[:, 1]] - nn[rcv[:, 0]] - nn[rcv[:, 1]]

    def negll(p):
        z = np.clip(eta_of(p), -30, 30)
        q = 1 / (1 + np.exp(-z))
        f = -np.sum(y * np.log(np.clip(q, 1e-12, None))
                    + (1 - y) * np.log(np.clip(1 - q, 1e-12, None)))
        f += 0.5 * np.sum((p / pri) ** 2)
        r = q - y
        g = np.zeros(S + 2 * P)
        np.add.at(g, st, r)
        for k in (0, 1):
            np.add.at(g, S + srv[:, k], r)
            np.add.at(g, S + P + rcv[:, k], -r)
        g += p / pri ** 2
        return f, g

    res = minimize(negll, np.zeros(S + 2 * P), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxfun": iters * 2})
    return 1 / (1 + np.exp(-np.clip(eta_of(res.x), -30, 30)))


def rows_from(seqs):
    out = []
    for (mid, gn, us, seq) in seqs:
        for (a, b, side, n, won) in seq:
            sa, sb = (a, b) if side == 0 else (b, a)
            ss = us[:2] if side == 0 else us[2:]
            rr = us[2:] if side == 0 else us[:2]
            j = (a + b) % 2
            out.append((ss[0] if n == 1 else ss[1], ss[1] if n == 1 else ss[0],
                        rr[j], rr[1 - j],
                        (min(sa, CAP - 1), min(sb, CAP - 1), n),
                        1.0 if CLUTCH(sa, sb) else 0.0, int(won),
                        (mid, gn), side, us))
    return out


def analyse(seqs, label, cur, index=None, states=None):
    rows = rows_from(seqs)
    if index is None:
        cnt = defaultdict(int)
        for r in rows:
            for u in r[:4]:
                cnt[u] += 1
        ok = {u for u, c in cnt.items() if c >= 300}
        rows = [r for r in rows if all(u in ok for u in r[:4])]
        index = {u: i for i, u in enumerate(sorted(ok))}
        states = {k: i for i, k in enumerate(sorted({r[4] for r in rows}))}
    else:
        rows = [r for r in rows if all(u in index for u in r[:4])
                and r[4] in states]
    exp = baseline_fit(rows, index, states)

    # aggregate the CLUTCH rallies per (game, side)
    agg = defaultdict(lambda: [0.0, 0.0, None])   # act, exp, players
    for r, e in zip(rows, exp):
        if r[5] < 0.5:
            continue
        key = (r[7], r[8])                 # (game, serving side)
        # credit the SERVING side's outcome to the serving pair's game-side,
        # and the complement to the receiving pair's
        g, side = r[7], r[8]
        us = r[9]
        for s in (0, 1):
            k2 = (g, s)
            a = agg[k2]
            if a[2] is None:
                a[2] = us[:2] if s == 0 else us[2:]
            if s == side:
                a[0] += r[6]; a[1] += e
            else:
                a[0] += 1 - r[6]; a[1] += 1 - e

    hits = defaultdict(lambda: [0, 0])
    for (g, s), (act, ex, pl) in agg.items():
        if ex <= 0 or pl is None:
            continue
        ind = 1 if act > ex else 0
        for u in pl:
            hits[u][0] += ind
            hits[u][1] += 1
    keep = {u: v for u, v in hits.items() if v[1] >= MIN_GAMES}
    rate = np.array([v[0] / v[1] for v in keep.values()])
    n = np.array([v[1] for v in keep.values()])
    p0 = rate.mean()
    # binomial expectation for the spread of rates given each player's n
    exp_var = float(np.mean(p0 * (1 - p0) / n))
    obs_var = float(rate.var())
    print(f"  {label:<10} players={len(keep):>4}  mean hit rate {p0:.4f}")
    print(f"    observed var {obs_var:.6f}   binomial-null var {exp_var:.6f}"
          f"   ratio {obs_var / exp_var:.3f}")
    return keep, rate, obs_var / exp_var, index, states


def main():
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])

    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]
    real, specs = [], []
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
        real.append((mid, gn, us,
                     [(r["server_score"] if r["server_side"] == 0 else r["receiver_score"],
                       r["receiver_score"] if r["server_side"] == 0 else r["server_score"],
                       r["server_side"], r["server_number"] or 2, r["won"]) for r in rs]))
        m = rs[0]["match_date"][:7]
        specs.append((mid, gn, us, [traj[u].get(m, cur[u]["v"]) for u in us],
                      rs[0]["server_side"], rs[0]["server_number"] or 2))

    print("Do players overperform in clutch moments more OFTEN than chance?\n")
    keep, rate, ratioR, index, states = analyse(real, "REAL", cur)

    rng = np.random.default_rng(1234)
    sim = []
    for (mid, gn, us, v, s0_, n0) in specs:
        eta = race.team_eta(v[0], v[1], v[2], v[3])
        p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
        kA, kB = serve_probs(round(eta_anchor(p0) / 0.05) * 0.05, 0.443)
        s = ce.sim_game(rng, kA, kB, s0_, n0)
        if len(s) >= 6:
            sim.append((mid, gn, us, s))
    _, rateS, ratioS, _, _ = analyse(sim, "SIM", cur, index, states)

    print(f"\n{'=' * 66}\n  excess-variance ratio  real {ratioR:.3f}   "
          f"sim {ratioS:.3f}\n{'=' * 66}")
    if ratioR > 1.15 and ratioR > 1.2 * ratioS:
        print("  => SOMETHING IS HERE: real hit rates spread wider than binomial "
              "noise AND\n     wider than the zero-clutch simulation.")
        nm = {u: cur[u]["name"] for u in keep}
        order = sorted(keep.items(), key=lambda kv: -kv[1][0] / kv[1][1])
        print(f"\n  {'player':26}{'games':>7}{'hit rate':>10}")
        for u, (k, n_) in order[:12]:
            print(f"  {nm[u]:26}{n_:>7}{k / n_:>10.3f}")
        print("  ...")
        for u, (k, n_) in order[-5:]:
            print(f"  {nm[u]:26}{n_:>7}{k / n_:>10.3f}")
    else:
        print("  => NULL: consistency is as empty as magnitude. Real hit-rate "
              "spread is\n     within binomial noise and matched by the "
              "no-clutch simulation.")


if __name__ == "__main__":
    main()
