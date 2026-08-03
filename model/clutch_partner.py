"""Is it the player, or the pairing?

The team-outcome estimator credits both partners with the same number from a
game, so an individual is identified ONLY by partner rotation. That is fine
in principle and fatal in practice if a player's record is dominated by one
partner: within a fixed pairing the two are mathematically indistinguishable
(the standing house rule on actor vs partner effects).

It matters here because the two headline names are each other's partner.
Ben Johns and Anna Leigh Waters play mixed together in 500 games — 44% of his
doubles record and 46% of hers. Their two "independent" 5-sigma results share
nearly half their data.

So split every player's games by partner and re-measure. A player whose
signal survives with the co-star removed is a player; one whose signal lives
only in the shared games is half of a pair.

This needs a per-GAME null baseline (the saved per-player sums cannot be
subset), so it re-runs the no-clutch simulation and stores the mean/sd of
each game's side-0 leverage covariance across replicates.

Run:  python model/clutch_partner.py [--reps 15]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

import clutch_leverage as cl      # noqa: E402
import clutch_null as cn          # noqa: E402
from clutch_leverage import names  # noqa: E402

warnings.filterwarnings("ignore")
LEV_SD = 0.0659


def per_game(gid, side, lev, won, ng):
    """Side-0 leverage/outcome covariance and leverage SS, per game."""
    o0 = np.where(side == 0, won, 1.0 - won).astype(np.float64)
    n = np.bincount(gid, minlength=ng).astype(np.float64)
    sl = np.bincount(gid, weights=lev, minlength=ng)
    so = np.bincount(gid, weights=o0, minlength=ng)
    sll = np.bincount(gid, weights=lev * lev, minlength=ng)
    soo = np.bincount(gid, weights=o0 * o0, minlength=ng)
    slo = np.bincount(gid, weights=lev * o0, minlength=ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        Sxx = sll - sl * sl / n
        Syy = soo - so * so / n
        Sxy = slo - sl * so / n
    bad = ~((n >= 4) & (Sxx > 0) & (Syy > 0))
    Sxx[bad] = Sxy[bad] = Syy[bad] = 0.0
    n[bad] = 0
    return Sxx, Syy, Sxy, n


def main(reps=15, seed=8080):
    d = cl.load()
    F = cl.Frame(d)
    npl, ng = F.npl, int(d["gidx"].max()) + 1
    games, gok = cn.rosters(d, F)
    rb, gdisc = [None] * ng, np.empty(ng, dtype="<U8")
    for (g, disc, T, s0, s1) in games:
        rb[g] = (s0, s1)
        gdisc[g] = disc
    keep = gok[d["gidx"]]

    lev_sd = float(d["lev_raw"].std())
    tabs = {}
    for T in (11, 15):
        tabs[("doubles", T)] = {k: v / lev_sd for k, v in
                                cl.doubles_leverage(d["k_doubles"], T).items()}
        tabs[("singles", T)] = {k: v / lev_sd for k, v in
                                cl.singles_leverage(d["k_singles"], T).items()}
    model = cn.fit_serve_model(d, F, npl)

    oXX, oYY, oXY, oN = per_game(d["gidx"][keep], d["side"][keep],
                                 d["lev"][keep], d["won"][keep].astype(float), ng)

    rng = np.random.default_rng(seed)
    nXY = np.zeros((reps, ng))
    nXX = np.zeros((reps, ng))
    for r in range(reps):
        srv, rcv, gid, lev, won = cn.sim_replicate(games, None, rng, tabs, model)
        side = np.array([0 if s in set(rb[g][0]) else 1
                         for s, g in zip(srv, gid)], dtype=np.int8)
        a, b, c, _ = per_game(gid, side, lev, won, ng)
        nXX[r], nXY[r] = a, c
        print(f"  replicate {r+1}/{reps}", flush=True)
    base_rate = np.divide(nXY.mean(axis=0), nXX.mean(axis=0),
                          out=np.zeros(ng), where=nXX.mean(axis=0) > 0)

    nm, _ = names()
    idx = {u: i for i, u in enumerate(F.uuids)}

    def player_games(p):
        """(game, sign, partner) for every doubles game p played."""
        out = []
        for g, r in enumerate(rb):
            if r is None or gdisc[g] != "doubles" or oN[g] == 0:
                continue
            for s in (0, 1):
                if p in r[s]:
                    other = [x for x in r[s] if x != p]
                    out.append((g, 1.0 if s == 0 else -1.0,
                                other[0] if other else -1))
        return out

    def measure(rows):
        """CWPA (games of win prob), its noise sd, and z, for a set of games."""
        U = sum(sg * (oXY[g] - base_rate[g] * oXX[g]) for g, sg, _ in rows)
        V = sum(oXX[g] * oYY[g] / (oN[g] - 1.0) for g, _, _ in rows if oN[g] > 1)
        cw = U * LEV_SD
        sd = np.sqrt(V) * LEV_SD
        return cw, sd, (cw / sd if sd > 0 else np.nan), len(rows)

    print("\n" + "=" * 74)
    print("IS IT THE PLAYER OR THE PAIRING?")
    print("=" * 74)

    for who, excl in (("Ben Johns", "Anna Leigh Waters"),
                      ("Anna Leigh Waters", "Ben Johns"),
                      ("Gabriel Tardio", "Ben Johns")):
        u = [k for k, v in nm.items() if v == who]
        if not u or u[0] not in idx:
            continue
        p = idx[u[0]]
        e = idx[[k for k, v in nm.items() if v == excl][0]]
        rows = player_games(p)
        with_ = [r for r in rows if r[2] == e]
        without = [r for r in rows if r[2] != e]
        print(f"\n{who}")
        for lab, rr in (("all doubles", rows),
                        (f"WITH {excl}", with_),
                        (f"WITHOUT {excl}", without)):
            cw, sd, z, n = measure(rr)
            print(f"   {lab:<28}{n:>5} games   CWPA {cw:>+7.2f}"
                  f"   +/-{sd:>5.2f}   z {z:>+5.2f}")
        # partner-by-partner, for the main partners
        cnt = Counter(r[2] for r in rows)
        print(f"   {'-- by partner --':<28}")
        for q, n in cnt.most_common(4):
            cw, sd, z, _ = measure([r for r in rows if r[2] == q])
            print(f"   {nm.get(F.uuids[q], '?'):<28}{n:>5} games   "
                  f"CWPA {cw:>+7.2f}   +/-{sd:>5.2f}   z {z:>+5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    main(**vars(ap.parse_args()))
