"""Does clutch predict games specifically in CLOSE ones? No (false positive).

Last plausible predictive angle: clutch should matter most in tight games,
where big points decide it.  Holdout projected toss-ups (v2 says 45-55%)
DO show corr(team clutch, win-above-v2) = +0.22 (n=158) — tantalizing.

But it's noise:
  * REPLICATION: the same test on the 1,237 TRAIN toss-ups gives -0.03
    (CI includes 0).  A real effect shows in the 8x-larger sample; this
    one only appears in the 158-game holdout subset.
  * The actually-close games (final margin <=2) show corr = +0.001.
  * The predictive clutch x closeness feature fits a zero loading (train
    has no signal to learn).
  * Skill control corr(eta, resid) ~ 0 in the band => not a v2-mispricing
    confound, just a chance fluctuation.

So clutch x closeness is null, like every other form.  Clutch is real,
stable, and descriptive; it has zero game-prediction value in any slice.

Run: python model/clutch_closeness_test.py
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

    clutch, zmap = {}, {}
    for r in csv.DictReader((ROOT / "data" / "clutch_players.csv").open()):
        u = uuid(r["name"], float(r["value"]))
        if u:
            clutch[u] = float(r["clutch"]); zmap[u] = float(r["z"])
    clutch_hi = {u: (c if abs(zmap[u]) >= 1.5 else 0.0) for u, c in clutch.items()}

    games = sx.load_games()

    def diff(g, t):
        c = [t.get(u, 0.0) for u in g["us"]]
        return (c[0] + c[1]) - (c[2] + c[3])

    def analyze(gs, label):
        F = sx.Frame(gs, players, chem)
        ref = racer.win(F.eta_v2(), F.T)
        resid = F.won.astype(float) - ref
        cl = np.array([diff(g, clutch_hi) for g in gs])
        eta = F.eta_v2()
        margin = np.array([abs(g["s1"] - g["s2"]) for g in gs])
        tos = np.abs(ref - 0.5) < 0.10
        r = np.corrcoef(cl[tos], resid[tos])[0, 1]
        rng = np.random.default_rng(1)
        idx = np.where(tos)[0]
        bs = [np.corrcoef(cl[idx[s]], resid[idx[s]])[0, 1]
              for s in rng.integers(0, len(idx), (2000, len(idx)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rs = np.corrcoef(eta[tos], resid[tos])[0, 1]
        close = margin <= 2
        rc = np.corrcoef(cl[close], resid[close])[0, 1]
        print(f"{label}: projected toss-up n={tos.sum():4d}  corr(clutch,resid)={r:+.3f} "
              f"CI[{lo:+.2f},{hi:+.2f}]  skill-control={rs:+.3f}  | actually-close(<=2) corr={rc:+.3f}")

    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    hold = [g for g in games if g["date"] >= sx.SPLIT
            and all(u in players and players[u]["games"] >= sx.MIN_TRAIN_GAMES for u in g["us"])]
    print("Clutch in close games — replication test (train has 8x the games):\n")
    analyze(train, "TRAIN")
    analyze(hold, "HOLD ")
    print("\nHoldout toss-up +0.22 does NOT replicate in train (-0.03) => noise, not a real effect.")


if __name__ == "__main__":
    main()
