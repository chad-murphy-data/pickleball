"""Per-player mechanical null on the REAL schedule.

`clutch_mechanical.py` establishes that the leverage-covariance statistic is
biased under a no-clutch generator, and that the bias tracks the player's own
rally-win rate. That is enough to invalidate an uncorrected leaderboard but
not enough to correct one: the size of a given player's bias depends on their
own rate, their partners' rates, their opponents' rates, how many games they
played and how long those games ran.

So we bootstrap the actual season. Every real game is replayed with its real
four players (real two, in singles) under the real side-out rules, with each
player serving at their own archive-wide rally-win rate and NO leverage
dependence anywhere. Across replicates that gives, per player and per channel,
the null mean and null sd of the raw score U — which is what the observed U
has to beat.

The corrected statistic is then

    b_adj = (U_obs - E_null[U]) / SSL_obs
    se    = sd_null[U] / SSL_obs

which needs no distributional assumption at all: it is a Monte Carlo test
against a generator that shares the real data's schedule, rosters, ability
distribution, scoring rules and stopping rules, and differs from it in exactly
one respect — nobody is clutch.

Run:  python model/clutch_null.py [--reps 30]
Writes data/clutch_null.npz.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))

import clutch_leverage as cl  # noqa: E402

CAP_EXTRA = 20


def rosters(d, F):
    """Per game: (discipline, target, side0 players, side1 players).

    Sides are read off the servers; the receiver list is the other side. A
    handful of games have a mid-game substitution or a log glitch that yields
    the wrong number of distinct servers — those are dropped from the null
    (and, for comparability, the caller drops them from the observed side
    too, via the returned game mask)."""
    ng = int(d["gidx"].max()) + 1
    seen = [defaultdict(set) for _ in range(ng)]
    meta = [None] * ng
    for i in range(len(d["gidx"])):
        g = d["gidx"][i]
        seen[g][int(d["side"][i])].add(F.srv_code[i])
        seen[g][1 - int(d["side"][i])].add(F.rcv_code[i])
        if meta[g] is None:
            meta[g] = (d["disc"][i], int(d["target"][i]))
    want = {"doubles": 2, "singles": 1}
    out, ok = [], np.zeros(ng, dtype=bool)
    for g in range(ng):
        if meta[g] is None:
            continue
        disc, T = meta[g]
        w = want[disc]
        s0, s1 = sorted(seen[g][0]), sorted(seen[g][1])
        if len(s0) != w or len(s1) != w:
            continue
        ok[g] = True
        out.append((g, disc, T, np.array(s0), np.array(s1)))
    return out, ok


def serve_rates(d, F, npl):
    """Archive-wide serve rally win rate per player, lightly shrunk toward the
    league mean so a 30-rally player cannot drive the simulation."""
    w = np.bincount(F.srv_code, weights=d["won"].astype(float), minlength=npl)
    n = np.bincount(F.srv_code, minlength=npl).astype(float)
    prior_n, prior_p = 100.0, float(d["won"].mean())
    return (w + prior_n * prior_p) / (n + prior_n)


def fit_serve_model(d, F, npl, l2=25.0, iters=60, step=1.0):
    """logit P(server wins rally) = a[server] - r[receiver], ridge-penalised.

    The flat null gives each player one archive-wide rate, so simulated games
    ignore who is on the other side of the net. This restores that: a strong
    server against a strong returner is priced correctly, which matters
    because the mechanical bias is a curved function of the rally-win rate,
    so its average over a realistic spread of per-game rates is not the same
    as its value at the average rate."""
    s, rc = F.srv_code, F.rcv_code
    y = d["won"].astype(np.float64)
    a = np.zeros(npl)
    r = np.zeros(npl)
    ns = np.bincount(s, minlength=npl).astype(float)
    nr = np.bincount(rc, minlength=npl).astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a[s] - r[rc])))
        resid = y - p
        ga = np.bincount(s, weights=resid, minlength=npl) - l2 * a
        gr = -np.bincount(rc, weights=resid, minlength=npl) - l2 * r
        # diagonal Newton scaling
        w = p * (1 - p)
        ha = np.bincount(s, weights=w, minlength=npl) + l2
        hr = np.bincount(rc, weights=w, minlength=npl) + l2
        a += step * ga / ha
        r += step * gr / hr
        a -= a.mean()          # a - r is only identified up to a shift
    return a, r


def sim_replicate(games, rates, rng, tabs, model=None):
    """One synthetic season. Returns (srv, rcv, gid, lev, won) arrays."""
    srv, rcv, gid, lev, won = [], [], [], [], []
    for (g, disc, T, s0, s1) in games:
        team = (s0, s1)
        cap = T + CAP_EXTRA
        tab = tabs[(disc, T)]
        sc = [0, 0]
        s = int(rng.integers(2))
        doubles = disc == "doubles"
        sn = 2 if doubles else 0
        who = 0
        rpos = [int(rng.integers(2)), int(rng.integers(2))]
        for _ in range(400):
            a, b = sc[s], sc[1 - s]
            if (a >= T and a - b >= 2) or (b >= T and b - a >= 2):
                break
            if a >= cap or b >= cap:
                break
            server = team[s][who] if doubles else team[s][0]
            if doubles:
                receiver = team[1 - s][rpos[1 - s]]
            else:
                receiver = team[1 - s][0]
            if model is None:
                pr = rates[server]
            else:
                pr = 1.0 / (1.0 + math.exp(-(model[0][server] - model[1][receiver])))
            w = 1 if rng.random() < pr else 0
            srv.append(server); rcv.append(receiver); gid.append(g)
            lev.append(tab[(a, b, sn)]); won.append(w)
            if w:
                sc[s] += 1
                if doubles:
                    rpos[1 - s] ^= 1
            elif doubles:
                if sn == 1:
                    sn, who = 2, 1 - who
                else:
                    s, sn, who = 1 - s, 1, 0
            else:
                s = 1 - s
    return (np.array(srv), np.array(rcv), np.array(gid),
            np.array(lev, dtype=float), np.array(won, dtype=float))


def channel_U(who, gid, lev, y, npl, ng):
    """Per-player U, SSL, permutation-variance V and rally count."""
    cell = who.astype(np.int64) * ng + gid
    st = cl.player_stats(who, cell, lev, y, npl)
    return st["U"], st["SSL"], st["V"], st["n"]


def _by_year(d, F, keep, npl, ng):
    """Observed U/SSL split by era and by discipline, for the reliability
    tests — the same corrections apply to each slice."""
    out = {}
    year = np.array([s[:4] for s in d["date"]])
    slices = {"pre26": keep & (year != "2026"), "y26": keep & (year == "2026"),
              "dbl": keep & (d["disc"] == "doubles"),
              "sgl": keep & (d["disc"] == "singles")}
    for nm, m in slices.items():
        for ch in ("S", "R"):
            who = (F.srv_code if ch == "S" else F.rcv_code)[m]
            y = d["won"][m].astype(float)
            if ch == "R":
                y = 1.0 - y
            u, ssl, v, n = channel_U(who, d["gidx"][m], d["lev"][m], y, npl, ng)
            out[f"U_{nm}_{ch}"] = u
            out[f"SSL_{nm}_{ch}"] = ssl
            out[f"V_{nm}_{ch}"] = v
            out[f"n_{nm}_{ch}"] = n
    return out


def main(reps=30, seed=4242, model=False, out="clutch_null"):
    d = cl.load()
    F = cl.Frame(d)
    npl = F.npl
    games, gok = rosters(d, F)
    ng = int(d["gidx"].max()) + 1
    print(f"{len(games):,} of {ng:,} games have a clean roster "
          f"({100*len(games)/ng:.1f}%)")

    rates = serve_rates(d, F, npl)
    sm = fit_serve_model(d, F, npl) if model else None
    if sm is not None:
        pr = 1.0 / (1.0 + np.exp(-(sm[0][F.srv_code] - sm[1][F.rcv_code])))
        print(f"  serve model: mean p {pr.mean():.4f} (obs {d['won'].mean():.4f}), "
              f"sd of fitted rally p {pr.std():.4f}")
    lev_sd = float(d["lev_raw"].std())
    tabs = {}
    for T in (11, 15):
        tabs[("doubles", T)] = {k: v / lev_sd
                                for k, v in cl.doubles_leverage(d["k_doubles"], T).items()}
        tabs[("singles", T)] = {k: v / lev_sd
                                for k, v in cl.singles_leverage(d["k_singles"], T).items()}

    # --- observed, restricted to the same games -----------------------
    keep = gok[d["gidx"]]
    obs = {}
    for ch in ("S", "R"):
        who = (F.srv_code if ch == "S" else F.rcv_code)[keep]
        y = d["won"][keep].astype(float)
        if ch == "R":
            y = 1.0 - y
        obs[ch] = channel_U(who, d["gidx"][keep], d["lev"][keep], y, npl, ng)
    np.savez_compressed(ROOT / "data" / "clutch_obs_year.npz",
                        uuids=F.uuids, **_by_year(d, F, keep, npl, ng))

    # --- null replicates ----------------------------------------------
    # game-level slice membership, so the null can be split the same ways
    year = np.array([s[:4] for s in d["date"]])
    gyear = np.empty(ng, dtype="<U4")
    gdisc = np.empty(ng, dtype="<U8")
    gyear[d["gidx"]] = year
    gdisc[d["gidx"]] = d["disc"]
    gslice = {"all": np.ones(ng, bool), "pre26": gyear != "2026",
              "y26": gyear == "2026", "dbl": gdisc == "doubles",
              "sgl": gdisc == "singles"}

    rng = np.random.default_rng(seed)
    acc = {(sl, ch): {k: np.zeros((reps, npl)) for k in ("u", "ssl", "v")}
           for sl in gslice for ch in ("S", "R")}
    for r in range(reps):
        srv, rcv, gid, lev, won = sim_replicate(games, rates, rng, tabs, sm)
        for sl, gm in gslice.items():
            m = gm[gid]
            for ch, who, y in (("S", srv, won), ("R", rcv, 1.0 - won)):
                u, ssl, v, _ = channel_U(who[m], gid[m], lev[m], y[m], npl, ng)
                a = acc[(sl, ch)]
                a["u"][r], a["ssl"][r], a["v"][r] = u, ssl, v
        print(f"  replicate {r+1}/{reps}  ({len(won):,} rallies)", flush=True)

    outname = out
    out = {"uuids": F.uuids, "rates": rates, "reps": reps}
    for ch in ("S", "R"):
        u_obs, ssl_obs, v_obs, n_obs = obs[ch]
        out[f"U_obs_{ch}"] = u_obs
        out[f"SSL_obs_{ch}"] = ssl_obs
        out[f"V_obs_{ch}"] = v_obs
        out[f"n_obs_{ch}"] = n_obs
    # keep every replicate: the channels (and eras) are drawn from the SAME
    # simulated season, so their nulls covary and a combined statistic's sd
    # cannot be recovered from per-channel sds alone.
    for (sl, ch), a in acc.items():
        out[f"U_rep_{sl}_{ch}"] = a["u"]
        out[f"SSL_rep_{sl}_{ch}"] = a["ssl"]
        out[f"V_rep_{sl}_{ch}"] = a["v"]
    np.savez_compressed(ROOT / "data" / f"{outname}.npz", **out)
    print(f"wrote data/{outname}.npz  (reps={reps})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--model", action="store_true",
                    help="opponent-aware serve model instead of flat rates")
    ap.add_argument("--out", default="clutch_null")
    main(**vars(ap.parse_args()))
