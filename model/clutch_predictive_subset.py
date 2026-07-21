"""Does clutch predict games for the RELIABLE players only (|z| >= 1.5)?

The full-population clutch test (clutch_predictive.py) found zero
out-of-sample value.  But clutch is noise for the ~130 middle players and
only repeats for the ~50 with |z| >= 1.5.  So the fair test is: zero out
the noisy middle, keep clutch only for the players who actually have a
signal, and see if THAT predicts games the frozen v2 model misses.

Tests, all fit on Jan-May train, graded on the June+ holdout:
  * clutch feature for all players           (the original null)
  * clutch for |z|>=1.5 only                  (the user's question)
  * clutch for the top only (z>=1.5)          ("does ALW-style clutch win?")
  * clutch for the bottom only (z<=-1.5)      ("do chokers lose extra?")
Plus the direct mechanism: on holdout games that involve such a player,
does their team win MORE/LESS than v2 predicts?

Run: python model/clutch_predictive_subset.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import spec_shootout as sx      # noqa: E402
import big_points as bp         # noqa: E402


def main():
    games = sx.load_games()
    players, chem = sx.load_v2()
    racer = sx.Racer()
    per, _ = bp.clutch(games, players, chem)      # train-only clutch + z per uuid

    clutch = {u: d["clutch"] for u, d in per.items()}
    z = {u: d["z"] for u, d in per.items()}
    hi = {u: clutch[u] if abs(z[u]) >= 1.5 else 0.0 for u in clutch}
    top = {u: clutch[u] if z[u] >= 1.5 else 0.0 for u in clutch}
    bot = {u: clutch[u] if z[u] <= -1.5 else 0.0 for u in clutch}
    n_hi = sum(abs(v) >= 1.5 for v in z.values())
    print(f"players with |z|>=1.5: {n_hi}\n")

    def tdiff(g, tbl):
        c = [tbl.get(u, 0.0) for u in g["us"]]
        return (c[0] + c[1]) - (c[2] + c[3])

    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    hold = [g for g in games if g["date"] >= sx.SPLIT
            and all(u in players and players[u]["games"] >= sx.MIN_TRAIN_GAMES
                    for u in g["us"])]
    Ftr, Fho = sx.Frame(train, players, chem), sx.Frame(hold, players, chem)
    won = Fho.won

    feats = {"all": clutch, "|z|>=1.5": hi, "top z>=1.5": top, "bottom z<=-1.5": bot}
    for nm, tbl in feats.items():
        setattr(Ftr, "f_" + nm, np.array([tdiff(g, tbl) for g in train]))
        setattr(Fho, "f_" + nm, np.array([tdiff(g, tbl) for g in hold]))

    ref = racer.win(Fho.eta_v2(), Fho.T)
    xb = sx.fit_params(Ftr, racer, lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()),
                       [1.0, sx.GAMMA_V2])
    base = racer.win(xb[0] * (Fho.d_sum() + xb[1] * Fho.d_gap()), Fho.T)
    print(f"holdout n={len(hold)}")
    print(f"  {'model':22s} {'Brier':>7} {'dBrier vs base':>16} {'P_better':>9} {'x2 loading':>11}")
    print(f"  {'v2 / baseline':22s} {sx.metrics(base, won)['brier']:.4f}")

    for nm in feats:
        ftr = getattr(Ftr, "f_" + nm)
        fho = getattr(Fho, "f_" + nm)
        Ftr._cur, Fho._cur = ftr, fho
        x = sx.fit_params(Ftr, racer,
                          lambda f, xx: xx[0] * (f.d_sum() + xx[1] * f.d_gap()) + xx[2] * f._cur,
                          [1.0, sx.GAMMA_V2, 0.0])
        pw = racer.win(x[0] * (Fho.d_sum() + x[1] * Fho.d_gap()) + x[2] * fho, Fho.T)
        b = sx.paired_boot(pw, base, won)
        m = sx.metrics(pw, won)
        print(f"  +clutch [{nm:14s}] {m['brier']:.4f}  {b['d_brier']:+.4f} "
              f"[{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] {b['p_better']:6.2f}    {x[2]:+.3f}")

    # direct mechanism on the ACTIVE holdout subset (games with a |z|>=1.5 player)
    resid = won.astype(float) - ref
    fho_hi = Fho.__dict__["f_|z|>=1.5"]
    active = fho_hi != 0
    print(f"\nMechanism on the {active.sum()} holdout games involving a |z|>=1.5 player:")
    print(f"  corr(team clutch, win residual vs v2) = "
          f"{np.corrcoef(fho_hi[active], resid[active])[0, 1]:+.3f}")
    for nm, key in [("top z>=1.5", "f_top z>=1.5"), ("bottom z<=-1.5", "f_bottom z<=-1.5")]:
        f = Fho.__dict__[key]
        a = f != 0
        if a.sum() > 10:
            print(f"  {nm}: corr = {np.corrcoef(f[a], resid[a])[0, 1]:+.3f} "
                  f"(n={a.sum()} games)")


if __name__ == "__main__":
    main()
