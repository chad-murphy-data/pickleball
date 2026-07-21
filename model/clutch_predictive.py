"""Does 'clutch' improve out-of-sample prediction? — the empirical test.

Clutch is measured on Jan-May 2026 rallies (train). We add each team's
clutch differential as a feature, fit its loading on train, and grade it
on the frozen June+ holdout the same way the shootout graded experience /
momentum / durability.  If it beats v2 out-of-sample, it earns a shadow
ledger; if not, clutch is a great story but not a predictive input.

Also runs the targeted mechanism check: do high-clutch teams win MORE
than v2 predicts (residual correlation)?  That's where the user's "wins
the points that matter → wins more games" intuition would show up.

Run: python model/clutch_predictive.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import spec_shootout as sx      # noqa: E402
import big_points as bp         # noqa: E402


def team_diff(g, tbl):
    c = [tbl.get(u, 0.0) for u in g["us"]]
    return (c[0] + c[1]) - (c[2] + c[3])


def main():
    games = sx.load_games()
    players, chem = sx.load_v2()        # frozen train ratings
    racer = sx.Racer()

    per, _ = bp.clutch(games, players, chem)     # train-only (Jan-May 2026)
    clutch = {u: d["clutch"] for u, d in per.items()}
    zc = {u: d["z"] for u, d in per.items()}
    print(f"clutch measured for {len(clutch)} players (train only)\n")

    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    hold = [g for g in games if g["date"] >= sx.SPLIT
            and all(u in players and players[u]["games"] >= sx.MIN_TRAIN_GAMES
                    for u in g["us"])]
    Ftr, Fho = sx.Frame(train, players, chem), sx.Frame(hold, players, chem)
    Ftr.cl = np.array([team_diff(g, clutch) for g in train])
    Fho.cl = np.array([team_diff(g, clutch) for g in hold])
    Ftr.clz = np.array([team_diff(g, zc) for g in train])
    Fho.clz = np.array([team_diff(g, zc) for g in hold])
    sx.Frame.d_cl = lambda self: self.cl
    sx.Frame.d_clz = lambda self: self.clz
    won = Fho.won

    def show(name, pw, ref=None):
        m = sx.metrics(pw, won)
        line = (f"  {name:28s} acc={m['accuracy']:.4f}  Brier={m['brier']:.4f}  "
                f"logloss={m['log_loss']:.4f}")
        if ref is not None:
            b = sx.paired_boot(pw, ref, won)
            line += (f"   dBrier={b['d_brier']:+.4f} "
                     f"CI[{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] P_better={b['p_better']:.2f}")
        print(line)
        return pw

    print(f"holdout n={len(hold)}")
    ref = show("v2 (reference)", racer.win(Fho.eta_v2(), Fho.T))

    xb = sx.fit_params(Ftr, racer, lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()),
                       [1.0, sx.GAMMA_V2])
    base = show("sum + gamma (baseline)",
                racer.win(xb[0] * (Fho.d_sum() + xb[1] * Fho.d_gap()), Fho.T), ref)

    xc = sx.fit_params(Ftr, racer,
                       lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()) + x[2] * f.d_cl(),
                       [1.0, sx.GAMMA_V2, 0.0])
    show("+ clutch (raw)",
         racer.win(xc[0] * (Fho.d_sum() + xc[1] * Fho.d_gap()) + xc[2] * Fho.d_cl(), Fho.T), base)

    xz = sx.fit_params(Ftr, racer,
                       lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()) + x[2] * f.d_clz(),
                       [1.0, sx.GAMMA_V2, 0.0])
    show("+ clutch (z-score)",
         racer.win(xz[0] * (Fho.d_sum() + xz[1] * Fho.d_gap()) + xz[2] * Fho.d_clz(), Fho.T), base)

    print(f"\nfitted clutch loadings (train):  raw x2 = {xc[2]:+.3f}   z x2 = {xz[2]:+.4f}")
    print("  (near zero, or Brier not improving out-of-sample = no predictive value)")

    # mechanism check: do clutch teams win MORE than v2 predicts?
    resid = won.astype(float) - ref
    r_raw = np.corrcoef(Fho.cl, resid)[0, 1]
    r_z = np.corrcoef(Fho.clz, resid)[0, 1]
    print(f"\nMechanism check — corr(team clutch, holdout win residual vs v2):")
    print(f"  raw clutch: r = {r_raw:+.3f}    z clutch: r = {r_z:+.3f}")
    print("  (>0 would mean high-clutch teams win more games than their skill predicts)")


if __name__ == "__main__":
    main()
