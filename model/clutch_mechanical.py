"""The mechanical null: what leverage-covariance does a league with NO clutch
ability produce?

Why this exists. The clutch statistic in `clutch_leverage.py` is a within-cell
covariance between a rally's leverage and its outcome. Leverage is computed
from the score at the START of the rally — and the score is itself the running
sum of past outcomes. So leverage is NOT exogenous to the player's own rally
history, and a covariance estimator on endogenous regressors is biased in
finite samples even when the true effect is exactly zero.

The mechanism is concrete and unavoidable in side-out scoring: a server keeps
serving while winning, so a service run is a string of wins terminated by
exactly one loss — and that terminating loss sits at the run's highest score,
which is usually its highest leverage. Every run ends badly, by construction.
Receiving is the mirror image: the receiving side loses a string of rallies
and then wins the one that ends the run. So the serve channel carries a
mechanical NEGATIVE tilt and the return channel a mechanical POSITIVE one.

That alone would only shift everyone equally, which is harmless. The danger is
that the size of the tilt depends on how often the player wins their serve —
a better server has longer runs, so the one terminating loss is diluted. That
would make skill masquerade as clutch, and the observed +0.51 correlation
between the raw statistic and v2 rating is exactly what that would look like.

So: simulate a synthetic league under the real side-out rules, with per-player
rally-win rates drawn from the real distribution and NO leverage-dependence
whatsoever, and measure the statistic. Whatever it finds is pure artifact. The
resulting curve b_mech(rate) is then subtracted from the real estimates.

Run:  python model/clutch_mechanical.py
Writes model/clutch_mechanical.json (the correction curves).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

from clutch_leverage import (doubles_leverage, singles_leverage, cells,  # noqa
                            player_stats, slopes)

T_GAME = 11


def sim_doubles(rates, n_games, rng, lev_tab, receiver_rule="alternate"):
    """Simulate side-out doubles games between random foursomes.

    Rules: game to T win-by-2; the serving team's #1 server keeps serving
    while winning; on a loss serve passes #1->#2, and #2 loses to a side-out;
    the game opens on the starting team's #2 server (the standard first-server
    exception, which is what the DP in winprob.py assumes).

    Rally outcomes depend ONLY on the server's own rate. There is no
    leverage term anywhere in this generator.
    """
    npl = len(rates)
    srv, rcv, gid, lev, won = [], [], [], [], []
    cap = T_GAME + 20

    for g in range(n_games):
        four = rng.choice(npl, size=4, replace=False)
        team = (four[:2], four[2:])
        sc = [0, 0]
        s = int(rng.integers(2))          # which side serves first
        sn = 2                            # first-server exception
        who = 0                           # index within the serving pair
        rpos = [int(rng.integers(2)), int(rng.integers(2))]  # receiver cursor
        for _ in range(400):
            a, b = sc[s], sc[1 - s]
            if (a >= T_GAME and a - b >= 2) or (b >= T_GAME and b - a >= 2):
                break
            if a >= cap or b >= cap:
                break
            server = team[s][who]
            r_i = rpos[1 - s]
            receiver = team[1 - s][r_i]
            w = 1 if rng.random() < rates[server] else 0
            srv.append(server); rcv.append(receiver); gid.append(g)
            lev.append(lev_tab[(a, b, sn)]); won.append(w)
            if w:
                sc[s] += 1
                # server switches courts, so the diagonal receiver flips
                if receiver_rule == "alternate":
                    rpos[1 - s] ^= 1
                else:
                    rpos[1 - s] = int(rng.integers(2))
            else:
                if sn == 1:
                    sn, who = 2, 1 - who
                else:
                    s, sn, who = 1 - s, 1, 0
    return (np.array(srv), np.array(rcv), np.array(gid),
            np.array(lev, dtype=float), np.array(won, dtype=float))


def measure(srv, rcv, gid, lev, won, npl):
    """Run the clutch estimator on simulated rallies (both channels)."""
    ng = gid.max() + 1
    out = {}
    for ch, who, y in (("S", srv, won), ("R", rcv, 1.0 - won)):
        cell = who.astype(np.int64) * ng + gid
        out[ch] = player_stats(who, cell, lev, y, npl)
    return out


def run(n_players=400, n_games=60000, seed=99, receiver_rule="alternate"):
    rng = np.random.default_rng(seed)
    tab = doubles_leverage(0.4383, T_GAME)
    lev_sd = 0.0659   # global sd of raw leverage in the real archive

    # spread rates across the real observed range so the curve is estimable
    rates = np.linspace(0.34, 0.56, n_players)
    rng.shuffle(rates)

    srv, rcv, gid, lev, won = sim_doubles(rates, n_games, rng, tab,
                                          receiver_rule)
    lev = lev / lev_sd
    st = measure(srv, rcv, gid, lev, won, n_players)

    res = {"receiver_rule": receiver_rule, "n_games": n_games,
           "n_rallies": int(len(won)), "channels": {}}
    for ch in ("S", "R"):
        b, se, ok = slopes(st[ch], 400)
        x, y, w = rates[ok], b[ok], 1.0 / se[ok] ** 2
        # quadratic in rate, precision-weighted
        A = np.vstack([np.ones_like(x), x - 0.45, (x - 0.45) ** 2]).T
        W = np.diag(w)
        coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
        pred = A @ coef
        res["channels"][ch] = {
            "n_players": int(ok.sum()),
            "mean_b": float(np.average(y, weights=w)),
            "sd_b_observed": float(np.std(y)),
            "sd_b_curve": float(np.std(pred)),
            "mean_se": float(np.mean(se[ok])),
            "coef": [float(c) for c in coef],
            "var_z": float(np.var((b / se)[ok])),
        }
    return res, rates, st


if __name__ == "__main__":
    print("simulating a no-clutch league under real side-out rules...")
    out = {}
    for rule in ("alternate", "random"):
        res, rates, st = run(receiver_rule=rule)
        out[rule] = res
        print(f"\nreceiver rule = {rule}   "
              f"{res['n_rallies']:,} rallies, {res['n_games']:,} games")
        for ch, lab in (("S", "serve "), ("R", "return")):
            c = res["channels"][ch]
            print(f"  {lab}: mean b {c['mean_b']:+.5f}   "
                  f"sd across players {c['sd_b_observed']:.5f}   "
                  f"sd explained by rate {c['sd_b_curve']:.5f}   "
                  f"mean se {c['mean_se']:.5f}   var(z) {c['var_z']:.2f}")
            print(f"          curve: b = {c['coef'][0]:+.5f} "
                  f"{c['coef'][1]:+.5f}*(rate-.45) "
                  f"{c['coef'][2]:+.5f}*(rate-.45)^2")
    with open(ROOT / "model" / "clutch_mechanical.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote model/clutch_mechanical.json")
