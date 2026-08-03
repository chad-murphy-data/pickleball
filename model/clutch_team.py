"""Doubles clutch without the attribution fiction.

The objection is correct: a doubles rally is contested by four players, and
crediting it to whoever happened to serve or return is close to arbitrary.
`clutch_leverage.py` attributes each rally to its server (and, separately,
its receiver), which is defensible only if the first ball dominates the
rally. It doesn't.

So attribute nothing. For every rally in a game, ask whether the SIDE won it,
and let the leverage/outcome covariance be a property of the side. Within a
game the cell is the whole game (both partners are on court for every rally),
and the two sides' statistics are exact negatives of each other, so a game
yields one number: how much this team's rally wins clustered on the big
points rather than the small ones.

    U_side = sum_r (Lz_r - Lbar_g) * (o_r - obar_g)

with o_r = 1 if this side won rally r. Both partners receive the same number
from that game, so individuals are identified ONLY by partner rotation across
games — the same channel the v2 rating model uses, and the same reason the
house rule says actor and partner effects are not separable within a pairing.
That is a real limitation of doubles, not of this estimator: it is what
"the rally is a team outcome" actually implies.

Singles is unaffected — there the side is one person, and this statistic
reduces to the exact per-player version.

The null is the same real-schedule no-clutch bootstrap, recomputed for this
statistic (the saved per-channel sums cannot be recombined into it).

Run:  python model/clutch_team.py --reps 30
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

import clutch_leverage as cl       # noqa: E402
import clutch_null as cn           # noqa: E402
from clutch_leverage import eb, tau_ci, boot_corr, names, v2_values  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning)
MIN_GAMES = 60


def team_stats(gid, side, lev, won, rosters_by_game, npl, ng, gsel=None):
    """Per-player U, SSL, V from the side-level leverage covariance.

    `rosters_by_game[g] = (players_side0, players_side1)`.
    """
    # within-game moments over ALL rallies, from side 0's perspective
    o0 = np.where(side == 0, won, 1.0 - won).astype(np.float64)
    n = np.bincount(gid, minlength=ng).astype(np.float64)
    sl_ = np.bincount(gid, weights=lev, minlength=ng)
    so = np.bincount(gid, weights=o0, minlength=ng)
    sll = np.bincount(gid, weights=lev * lev, minlength=ng)
    soo = np.bincount(gid, weights=o0 * o0, minlength=ng)
    slo = np.bincount(gid, weights=lev * o0, minlength=ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        Sxx = sll - sl_ * sl_ / n
        Syy = soo - so * so / n
        Sxy = slo - sl_ * so / n
    use = (n >= 4) & (Sxx > 0) & (Syy > 0)
    if gsel is not None:
        use &= gsel
    V = np.zeros_like(Sxx)
    V[use] = Sxx[use] * Syy[use] / (n[use] - 1.0)

    U = np.zeros(npl)
    SSL = np.zeros(npl)
    VV = np.zeros(npl)
    NG = np.zeros(npl)
    for g in np.where(use)[0]:
        r = rosters_by_game[g]
        if r is None:
            continue
        for s in (0, 1):
            sign = 1.0 if s == 0 else -1.0
            for p in r[s]:
                U[p] += sign * Sxy[g]
                SSL[p] += Sxx[g]
                VV[p] += V[g]
                NG[p] += 1
    return {"U": U, "SSL": SSL, "V": VV, "games": NG}


def main(reps=30, seed=515, min_games=MIN_GAMES):
    d = cl.load()
    F = cl.Frame(d)
    npl, ng = F.npl, int(d["gidx"].max()) + 1
    games, gok = cn.rosters(d, F)
    rb = [None] * ng
    gdisc = np.empty(ng, dtype="<U8")
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

    arms = {"doubles": gdisc == "doubles", "singles": gdisc == "singles",
            "all": np.ones(ng, bool)}
    year = np.array([s[:4] for s in d["date"]])
    gyear = np.empty(ng, dtype="<U4")
    gyear[d["gidx"]] = year
    arms["dbl_pre26"] = (gdisc == "doubles") & (gyear != "2026")
    arms["dbl_y26"] = (gdisc == "doubles") & (gyear == "2026")
    arms["sgl_pre26"] = (gdisc == "singles") & (gyear != "2026")
    arms["sgl_y26"] = (gdisc == "singles") & (gyear == "2026")

    obs = {a: team_stats(d["gidx"][keep], d["side"][keep], d["lev"][keep],
                         d["won"][keep].astype(float), rb, npl, ng, m)
           for a, m in arms.items()}

    rng = np.random.default_rng(seed)
    nulls = {a: [] for a in arms}
    for r in range(reps):
        srv, rcv, gid, lev, won = cn.sim_replicate(games, None, rng, tabs, model)
        # side of the serving player, recovered from the roster
        side = np.array([0 if s in set(rb[g][0]) else 1
                         for s, g in zip(srv, gid)], dtype=np.int8)
        for a, m in arms.items():
            nulls[a].append(team_stats(gid, side, lev, won, rb, npl, ng, m))
        print(f"  replicate {r+1}/{reps}", flush=True)

    def corrected(arm, mg):
        o = obs[arm]
        ok = (o["SSL"] > 0) & (o["V"] > 0) & (o["games"] >= mg)
        b = np.where(ok, o["U"] / np.where(ok, o["SSL"], 1), np.nan)
        br = np.array([np.where(ok, x["U"] / np.where(ok & (x["SSL"] > 0),
                                                     x["SSL"], 1), np.nan)
                       for x in nulls[arm]])
        bm, sdm = np.nanmean(br, axis=0), np.nanstd(br, axis=0, ddof=1)
        se = np.sqrt((np.sqrt(o["V"]) / np.where(ok, o["SSL"], 1)) ** 2
                     + sdm ** 2 / reps)
        return b - bm, se, ok

    print()
    print("=" * 74)
    print("DOUBLES CLUTCH WITHOUT ATTRIBUTION — rally credited to all four")
    print("=" * 74)
    print(f"{'arm':<14}{'players':>8}{'var(z)':>9}{'tau':>10}{'tau 95% CI':>22}")
    for arm, mg in (("doubles", min_games), ("singles", min_games),
                    ("all", min_games)):
        b, se, ok = corrected(arm, mg)
        mu, tau, post, psd, shr = eb(b[ok], se[ok])
        lo, hi = tau_ci(b[ok], se[ok])
        print(f"{arm:<14}{ok.sum():>8}{np.var((b/se)[ok]):>9.3f}{tau:>10.5f}"
              f"   [{lo:.5f}, {hi:.5f}]")

    print()
    print("cross-era reliability (disjoint games)")
    print(f"{'arm':<14}{'n':>6}{'r':>9}{'95% CI':>20}{'null floor':>16}")
    for arm, mg in (("dbl", 30), ("sgl", 30)):
        b1, s1, o1 = corrected(f"{arm}_pre26", mg)
        b2, s2, o2 = corrected(f"{arm}_y26", mg)
        m = o1 & o2
        if m.sum() < 15:
            print(f"{arm:<14}{m.sum():>6}   too few")
            continue
        r = float(np.corrcoef(b1[m], b2[m])[0, 1])
        lo, hi = boot_corr(b1[m], b2[m])
        fl = []
        for i in range(min(reps, 8)):
            def one(a):
                o = nulls[a][i]
                ok = (o["SSL"] > 0) & (o["V"] > 0) & (o["games"] >= mg)
                bb = np.where(ok, o["U"] / np.where(ok, o["SSL"], 1), np.nan)
                oth = [nulls[a][k] for k in range(reps) if k != i]
                brr = np.array([np.where(ok, x["U"] / np.where(ok & (x["SSL"] > 0),
                                                               x["SSL"], 1), np.nan)
                                for x in oth])
                return bb - np.nanmean(brr, axis=0), ok
            y1, k1 = one(f"{arm}_pre26")
            y2, k2 = one(f"{arm}_y26")
            mm = k1 & k2
            if mm.sum() > 15:
                fl.append(float(np.corrcoef(y1[mm], y2[mm])[0, 1]))
        print(f"{arm:<14}{m.sum():>6}{r:>9.3f}   [{lo:+.3f},{hi:+.3f}]"
              f"{np.mean(fl):>+10.3f}±{np.std(fl):.3f}")

    # write the doubles table
    nm, gdr = names()
    b, se, ok = corrected("doubles", min_games)
    mu, tau, post, psd, shr = eb(b[ok], se[ok])
    uu = F.uuids[ok]
    with open(DATA / "clutch_team_doubles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "name", "gender", "games", "b_adjusted",
                    "se", "z", "clutch_shrunk", "shrunk_sd"])
        for i in np.argsort(-post):
            w.writerow([uu[i], nm.get(uu[i], uu[i][:8]), gdr.get(uu[i], ""),
                        int(obs["doubles"]["games"][ok][i]),
                        f"{b[ok][i]:.6f}", f"{se[ok][i]:.6f}",
                        f"{(b/se)[ok][i]:.3f}", f"{post[i]:.6f}",
                        f"{psd[i]:.6f}"])
    print("\nwrote data/clutch_team_doubles.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    main(**vars(ap.parse_args()))
