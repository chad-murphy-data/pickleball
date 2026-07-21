"""Does the clutch RESIDUAL (clutch beyond skill) predict games? No.

Natural hypothesis: the raw clutch feature failed, but maybe the part of
clutch NOT explained by skill — the residual from clutch ~ rating — is the
real predictor (it's what flags Johns/Bright as "more clutch than their
skill implies").

Test on the frozen June+ holdout: add team residual-diff as a feature.
It shows a tiny gain (dBrier -0.0004, P_better 0.99) — BUT a control with
NO clutch in it (the skill-PREDICTED clutch, b0+b1*rating, on the same
player subset) produces the identical -0.0004. So the "signal" is a
rating recalibration that the residual smuggles in (residual = clutch -
b1*rating), not clutch. Confirmed by: the gain gets WEAKER when restricted
to the reliable |z|>=1.5 players (opposite of a real clutch signal).

Verdict: clutch is real, stable, and descriptive (Johns/Bright genuinely
over-deliver on big points relative to skill, and it repeats) — but it has
no game-prediction value in any form (raw, residual, reliable-only, top,
bottom). It's the DISTRIBUTION of wins across leverage, mean-zero within
games, so it nets the same total points = same games. Skill wins games.

Run: python model/clutch_residual_test.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import spec_shootout as sx      # noqa: E402


def main():
    players, chem = sx.load_v2()
    racer = sx.Racer()
    by_name = {}
    for u, d in players.items():
        by_name.setdefault(d["name"], []).append((d["v"], u))

    def uuid(nm, val):
        c = by_name.get(nm, [])
        return min(c, key=lambda x: abs(x[0] - val))[1] if c else None

    rows = []
    for r in csv.DictReader((ROOT / "data" / "clutch_players.csv").open()):
        u = uuid(r["name"], float(r["value"]))
        if u:
            rows.append((u, float(r["z"]), float(r["value"])))
    z = np.array([r[1] for r in rows]); v = np.array([r[2] for r in rows])
    b1, b0 = np.polyfit(v, z, 1)
    resid = {r[0]: r[1] - (b0 + b1 * r[2]) for r in rows}
    resid_hi = {r[0]: (resid[r[0]] if abs(r[1]) >= 1.5 else 0.0) for r in rows}
    control = {r[0]: (b0 + b1 * r[2]) for r in rows}    # skill-only, NO clutch

    games = sx.load_games()
    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    hold = [g for g in games if g["date"] >= sx.SPLIT
            and all(u in players and players[u]["games"] >= sx.MIN_TRAIN_GAMES for u in g["us"])]
    Ftr, Fho = sx.Frame(train, players, chem), sx.Frame(hold, players, chem)
    won = Fho.won

    def diff(g, t):
        c = [t.get(u, 0.0) for u in g["us"]]
        return (c[0] + c[1]) - (c[2] + c[3])

    ref = racer.win(Fho.eta_v2(), Fho.T)
    xb = sx.fit_params(Ftr, racer, lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()),
                       [1.0, sx.GAMMA_V2])
    base = racer.win(xb[0] * (Fho.d_sum() + xb[1] * Fho.d_gap()), Fho.T)
    print(f"holdout n={len(hold)}   baseline Brier={sx.metrics(base, won)['brier']:.4f}\n")
    for nm, t in [("residual (clutch - skill)", resid),
                  ("residual, |z|>=1.5 only", resid_hi),
                  ("CONTROL: skill only, no clutch", control)]:
        fho = np.array([diff(g, t) for g in hold])
        Ftr._c = np.array([diff(g, t) for g in train]); Fho._c = fho
        x = sx.fit_params(Ftr, racer,
                          lambda f, xx: xx[0] * (f.d_sum() + xx[1] * f.d_gap()) + xx[2] * f._c,
                          [1.0, sx.GAMMA_V2, 0.0])
        pw = racer.win(x[0] * (Fho.d_sum() + x[1] * Fho.d_gap()) + x[2] * fho, Fho.T)
        b = sx.paired_boot(pw, base, won)
        print(f"  +{nm:32s} dBrier={b['d_brier']:+.4f}  P_better={b['p_better']:.2f}")
    print("\nresidual gain == control gain (control has NO clutch) => it's rating "
          "recalibration, not clutch. Clutch has zero game-prediction value.")


if __name__ == "__main__":
    main()
