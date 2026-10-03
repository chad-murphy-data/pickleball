"""Market prices: what the backward induction settles on.

Plain language. In the auction (`auction_sim.py`) owners fill one slot at a
time against a cheat sheet, and a human who plans whole rosters beats them
(the "buy #20-26" team). This script asks what is left once nobody overpays:
owner 19 stops chasing the players that human beats him with, then 18, then
17, and so on to the fixed point. Each round every owner asks for its best
$1M roster at the current prices (exhaustive over every 3M+3W combination of
a candidate set, vectorised); rosters are handed out in a random order; a
priced player on an above-average roster is marked up and on a below-average
roster marked down; unsold priced players fall; prices are clipped to
[floor, $850k] (cap minus the cheapest legal completion, the auction's
first-buy maximum). The fixed point: every roster an owner can buy is worth
the same, except for the players the cap stops from being priced -- the
rationed players are the difference makers. A player who lands on a roster
holding a capped player is priced by the best FREE roster that would hold
him (shadow pricing), so the capped player's premium is not smeared onto
teammates.

`--rule demand` is the textbook excess-demand version (over-demanded up,
unsold down, everything else untouched). With identical owners the demand
signal is one roster, so it leaves the middle of the list where the cheat
sheet put it; kept as a check that the rationed set does not depend on the
rule. `--noise` perturbs owner beliefs (fraction of the value spread), as in
`draft_sim.py`.

Swept, not picked: the step scale `--c` (4 and 8 give the same picture), the
candidate depth `--k`, the seed. Values and the tie model come from
`draft_sim.py` (board `mlp2026`, true engine, reference team = doubles ranks
10/30/50 per gender).

    python value_cap/market_eq.py --seed 1            # one run -> cache/market_eq_<rule>_seed<S>.json
    python value_cap/market_eq.py --report            # every cached run -> market_eq.md + market_prices.csv
    python value_cap/market_eq.py --selftest          # brute-force check of the roster solver
    python value_cap/market_eq.py --at-list           # the list-price reference rosters (no run needed)
    python value_cap/market_eq.py --reanalyse         # redo the analysis on every cached run's prices
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--c", type=float, default=8.0, help="equalise step scale: price x exp(eta*c*(roster value - average))")
ap.add_argument("--eta0", type=float, default=0.10, help="initial step; eta = eta0 / (1 + round/tau)")
ap.add_argument("--tau", type=float, default=50.0)
ap.add_argument("--k", type=int, default=20, help="candidate depth per gender (top-k by value + singles/cheap/floor extras)")
ap.add_argument("--noise", type=float, default=0.0, help="owner belief noise, fraction of the value spread")
ap.add_argument("--rule", choices=["equalize", "demand"], default="equalize")
ap.add_argument("--no-shadow", action="store_true", help="equalize rule: leave players on a capped roster unpriced (the leak the report documents)")
ap.add_argument("--board", default="mlp2026")
ap.add_argument("--stars", type=int, default=24, help="top-N by list price to profile (indifference prices)")
ap.add_argument("--report", action="store_true")
ap.add_argument("--reanalyse", action="store_true", help="re-run the analysis (stars, leagues) on every cached run's prices and rewrite the caches")
ap.add_argument("--at-list", action="store_true", help="print the list-price reference rosters (best, best without Waters, Waters at the maximum, the star chain)")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--out", default=str(HERE / "market_eq.md"))
A = ap.parse_args()

import draft_sim as D  # noqa: E402
import fast_tie as FT  # noqa: E402
from phase2_pricing import FLOOR, NAME, POOL, pid_named, prices_tagged  # noqa: E402
from sitelib.race import race_dist  # noqa: E402

D.set_board(A.board)
E, REF = D.TRUE_ENGINE, D.REFERENCE
G = D.GENDER
BOARD = list(D.BOARD); N = len(BOARD)
IDX = {u: i for i, u in enumerate(BOARD)}
WATERS, BRIGHT, JOHNS = (pid_named(n) for n in ["Anna Leigh Waters", "Anna Bright", "Ben Johns"])
_lp = prices_tagged(POOL, 1.0, WATERS, "joint")
LP = {u: _lp.get(u, FLOOR) for u in BOARD}
POOLSET = {u for u in BOARD if LP[u] > FLOOR}
ISPOOL = np.array([u in POOLSET for u in BOARD])
GEN = np.array([G[u] for u in BOARD])
CAP = 1_000_000.0
MAXBUY = CAP - 5 * FLOOR
GAM = E.gamma
K_DB = FT.K_DB
GRID = np.frombuffer(E.f.g, dtype=np.float64).reshape(FT.N_SD, FT.N_ETA)
V_TRUE = np.array([E.v[u] for u in BOARD]); U2 = np.array([E.u2[u] for u in BOARD]); S_TRUE = np.array([E.s[u] for u in BOARD])
LPA = np.array([LP[u] for u in BOARD])
nm = lambda i: NAME[BOARD[i]]  # noqa: E731

_dbtab = [0.0] * 10001
for _k in range(1, 10000):
    _dbtab[_k] = race_dist(_k / 10000.0, 21)["p_win"]
_dbtab[10000] = 1.0
DBTAB = np.array(_dbtab)
_wr, _mr, _x1, _x2, DB_REF = E.lineup(REF)
S_REF = {"WD": E.S(*_wr, "WD"), "MD": E.S(*_mr, "MD"), "X1": E.S(*_x1, "MXD"), "X2": E.S(*_x2, "MXD")}
U_REF = {"WD": E.U(*_wr), "MD": E.U(*_mr), "X1": E.U(*_x1), "X2": E.U(*_x2)}


# ---------------------------------------------------------------- roster solver
def fgrid(eta, sd):
    """Bilinear lookup on the cached game-probability grid (mirrors fast_tie.GameProb)."""
    x = np.clip((eta - FT.ETA_LO) / FT.ETA_STEP, 0.0, FT.N_ETA - 1.0)
    y = np.clip((sd - FT.SD_LO) / FT.SD_STEP, 0.0, FT.N_SD - 1.0)
    j = np.minimum(x.astype(np.int64), FT.N_ETA - 2); i = np.minimum(y.astype(np.int64), FT.N_SD - 2)
    fx = x - j; fy = y - i
    top = GRID[i, j] * (1 - fx) + GRID[i, j + 1] * fx
    bot = GRID[i + 1, j] * (1 - fx) + GRID[i + 1, j + 1] * fx
    return top * (1 - fy) + bot * fy


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def candidates(g, v, s, price, avail, K):
    idx = [i for i in range(N) if GEN[i] == g and avail[i]]
    pool = [i for i in idx if ISPOOL[i]]; fl = [i for i in idx if not ISPOOL[i]]
    top = sorted(pool, key=lambda i: (-v[i], i))[:K]
    sg = sorted(pool, key=lambda i: (-s[i], i))[:3]
    cheap = sorted(pool, key=lambda i: (price[i], -v[i], i))[:3]
    flv = sorted(fl, key=lambda i: (-v[i], i))[:2]
    fls = sorted(fl, key=lambda i: (-s[i], i))[:2]
    return sorted(set(top) | set(sg) | set(cheap) | set(flv) | set(fls))


class Side:
    """All pairs of one gender's candidates + per-pair third-slot menus (Pareto: cost asc, singles desc)."""

    def __init__(self, g, C, v, s, u2, price, must=None, force_third=None, W=6):
        self.g = g; self.C = C
        pairs = [(a, b) for a, b in itertools.combinations(C, 2) if must is None or must in (a, b)]
        if not pairs:
            self.n = 0; return
        gam = GAM["MD" if g == "M" else "WD"]
        Aa = np.array([p[0] for p in pairs]); Bb = np.array([p[1] for p in pairs])
        hiA = v[Aa] >= v[Bb]
        self.hi = np.where(hiA, Aa, Bb); self.lo = np.where(hiA, Bb, Aa)
        self.vhi = v[self.hi]; self.vlo = v[self.lo]
        self.u2hi = u2[self.hi]; self.u2lo = u2[self.lo]
        self.S = self.vhi + self.vlo + gam * (self.vhi - self.vlo)
        self.U = self.u2hi + self.u2lo
        self.cost = price[Aa] + price[Bb]
        self.n = len(pairs)
        mc = np.full((self.n, W), np.inf); ms = np.zeros((self.n, W)); mw = np.full((self.n, W), -1, dtype=np.int64)
        for p in range(self.n):
            a, b = int(self.hi[p]), int(self.lo[p]); vlo = v[b]
            if force_third is not None:
                el = [force_third] if (force_third not in (a, b) and v[force_third] <= vlo) else []
            else:
                el = [c for c in C if c != a and c != b and v[c] <= vlo]
            el.sort(key=lambda c: (price[c], -s[c], c))
            best = -1e9; k = 0
            sa, sb = s[a], s[b]
            for c in el:
                if s[c] > best and k < W:
                    best = s[c]
                    three = sorted([sa, sb, s[c]], reverse=True)
                    mc[p, k] = price[c]; ms[p, k] = three[0] + three[1]; mw[p, k] = c; k += 1
        self.mc, self.ms, self.mw = mc, ms, mw


def solve(v, s, price, avail, budget=CAP, must=None, force_third=None, K=None, exclude=()):
    """Best roster (3M+3F) under budget, by tie probability vs the reference team.
    must = player index that must be in a playing pair; force_third = {gender: idx} third slot forced.
    Returns (value, roster indices, cost) or None."""
    K = A.k if K is None else K
    av = avail.copy()
    for e in exclude:
        av[e] = False
    CM = candidates("M", v, s, price, av, K); CF = candidates("F", v, s, price, av, K)
    if must is not None:
        if GEN[must] == "M" and must not in CM: CM.append(must)
        if GEN[must] == "F" and must not in CF: CF.append(must)
    ft = force_third or {}
    for g, c in ft.items():
        if g == "M" and c not in CM: CM.append(c)
        if g == "F" and c not in CF: CF.append(c)
    M = Side("M", CM, v, s, U2, price, must if (must is not None and GEN[must] == "M") else None, ft.get("M"))
    F = Side("F", CF, v, s, U2, price, must if (must is not None and GEN[must] == "F") else None, ft.get("F"))
    if M.n == 0 or F.n == 0:
        return None
    base = M.cost[:, None] + F.cost[None, :]
    best_db = np.full((M.n, F.n), -np.inf); arg = np.full((M.n, F.n), -1, dtype=np.int64)
    WM, WF = M.mc.shape[1], F.mc.shape[1]
    for wm in range(WM):
        for wf in range(WF):
            tot = base + M.mc[:, wm][:, None] + F.mc[:, wf][None, :]
            db = (M.ms[:, wm][:, None] + F.ms[:, wf][None, :]) / 4.0
            ok = (tot <= budget + 1e-6) & (db > best_db)
            best_db = np.where(ok, db, best_db); arg = np.where(ok, wm * WF + wf, arg)
    feas = np.isfinite(best_db)
    if not feas.any():
        return None
    pWD = fgrid(F.S - S_REF["WD"], np.sqrt(F.U + U_REF["WD"]))[None, :]
    pMD = fgrid(M.S - S_REF["MD"], np.sqrt(M.U + U_REF["MD"]))[:, None]
    gx = GAM["MXD"]
    w1 = F.vhi[None, :]; w2 = F.vlo[None, :]; m1 = M.vhi[:, None]; m2 = M.vlo[:, None]
    pv = lambda w, m: w + m + gx * np.abs(w - m)  # noqa: E731
    A11, A22, B12, B21 = pv(w1, m1), pv(w2, m2), pv(w1, m2), pv(w2, m1)
    useA = (A11 + A22) >= (B12 + B21)
    Sx1 = np.where(useA, A11, B12); Sx2 = np.where(useA, A22, B21)
    Ux1 = F.u2hi[None, :] + np.where(useA, M.u2hi[:, None], M.u2lo[:, None])
    Ux2 = F.u2lo[None, :] + np.where(useA, M.u2lo[:, None], M.u2hi[:, None])
    p3 = fgrid(Sx1 - S_REF["X1"], np.sqrt(Ux1 + U_REF["X1"])); p4 = fgrid(Sx2 - S_REF["X2"], np.sqrt(Ux2 + U_REF["X2"]))
    d0 = np.ones_like(p3); d1 = np.zeros_like(p3); d2 = np.zeros_like(p3); d3 = np.zeros_like(p3); d4 = np.zeros_like(p3)
    for p in (pWD * np.ones_like(p3), pMD * np.ones_like(p3), p3, p4):
        q = 1.0 - p
        d0, d1, d2, d3, d4 = d0 * q, d1 * q + d0 * p, d2 * q + d1 * p, d3 * q + d2 * p, d4 + d3 * p
    dbx = np.where(feas, best_db, DB_REF)
    pdb = DBTAB[np.rint(sig(K_DB * (dbx - DB_REF)) * 10000).astype(np.int64)]
    val = np.where(feas, d3 + d4 + d2 * pdb, -np.inf)
    k = int(np.argmax(val)); pm, pf = divmod(k, F.n)
    a = int(arg[pm, pf]); wm, wf = divmod(a, WF)
    roster = [int(M.hi[pm]), int(M.lo[pm]), int(M.mw[pm, wm]), int(F.hi[pf]), int(F.lo[pf]), int(F.mw[pf, wf])]
    cost = float(base[pm, pf] + M.mc[pm, wm] + F.mc[pf, wf])
    return float(val[pm, pf]), roster, cost


def solve_with(v, s, price, avail, u, budget=CAP, K=None, exclude=()):
    """Best roster containing u (u in a playing pair, or u as the third slot)."""
    outs = [solve(v, s, price, avail, budget, must=u, K=K, exclude=exclude),
            solve(v, s, price, avail, budget, force_third={GEN[u]: u}, K=K, exclude=exclude)]
    outs = [o for o in outs if o is not None and u in o[1]]
    return max(outs, key=lambda o: o[0]) if outs else None


def true_tie(roster):
    return E.tie(tuple(BOARD[i] for i in roster), REF)


def selftest():
    """Brute force on a small candidate set: list prices, then two perturbed price vectors."""
    rng = random.Random(0)
    avail = np.ones(N, dtype=bool)
    for trial in range(3):
        price = LPA * np.exp(np.array([rng.gauss(0, 0.3) for _ in range(N)])) if trial else LPA.copy()
        price = np.where(ISPOOL, np.clip(price, FLOOR, MAXBUY), FLOOR)
        Kt = 7
        CM = candidates("M", V_TRUE, S_TRUE, price, avail, Kt); CF = candidates("F", V_TRUE, S_TRUE, price, avail, Kt)
        best = (-1, None)
        for tm in itertools.combinations(CM, 3):
            cm = price[list(tm)].sum()
            if cm > CAP: continue
            for tf in itertools.combinations(CF, 3):
                if cm + price[list(tf)].sum() > CAP + 1e-6: continue
                t = true_tie(list(tm) + list(tf))
                if t > best[0]: best = (t, list(tm) + list(tf))
        got = solve(V_TRUE, S_TRUE, price, avail, K=Kt)
        print(f"selftest {trial}: brute {best[0]:.6f} {[nm(i) for i in best[1]]}\n            solver {got[0]:.6f} {[nm(i) for i in got[1]]} cost ${got[2]/1e3:.0f}k  true {true_tie(got[1]):.6f}")
        assert abs(best[0] - got[0]) < 2e-4, (best[0], got[0])
    print("selftest OK")


def beliefs(rng, noise):
    if noise <= 0:
        return V_TRUE, S_TRUE
    v = V_TRUE.copy(); s = S_TRUE.copy()
    for i in range(N):
        g = GEN[i]
        v[i] += rng.gauss(0, noise * D.SPREAD_V[g]); s[i] += rng.gauss(0, noise * D.SPREAD_S[g])
    return v, s


def league_win(rosters):
    n = len(rosters)
    P = [[0.5 if i == j else E.tie(tuple(BOARD[x] for x in rosters[i]), tuple(BOARD[x] for x in rosters[j])) for j in range(n)] for i in range(n)]
    return [sum(P[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]


# ---------------------------------------------------------------- price adjustment
def run():
    rng = random.Random(A.seed)
    price = np.where(ISPOOL, LPA, FLOOR).astype(float)
    avg = np.zeros(N); navg = 0; hist = []; t0 = time.time()
    iW, iB, iJ = IDX[WATERS], IDX[BRIGHT], IDX[JOHNS]
    for r in range(A.rounds):
        eta = A.eta0 / (1.0 + r / A.tau)
        order = list(range(D.N_TEAMS)); rng.shuffle(order)
        avail = np.ones(N, dtype=bool); demand = np.zeros(N); rosters = []; full = None
        for _ in order:
            v, s = beliefs(rng, A.noise)
            if A.rule == "demand":
                if A.noise > 0 or full is None:
                    full = solve(v, s, price, np.ones(N, dtype=bool))
                demand[full[1]] += 1
            got = solve(v, s, price, avail)
            rosters.append(got[1]); avail[got[1]] = False
        q = np.array([true_tie(ro) for ro in rosters])
        sold = np.zeros(N, dtype=bool)
        for ro in rosters: sold[ro] = True
        capped = np.array([any(ISPOOL[i] and price[i] >= MAXBUY - 1 for i in ro) for ro in rosters])
        free = ~capped
        qbar = float(q[free].mean() if free.any() else q.mean())
        mult = np.ones(N)
        if A.rule == "demand":
            up = ISPOOL & (demand >= 2)
            mult = np.where(up, np.exp(eta * np.minimum(demand - 1, 3) / 3.0), mult)
        else:
            for ro, qk, cp in zip(rosters, q, capped):
                if cp:
                    Rc = [i for i in ro if ISPOOL[i] and price[i] >= MAXBUY - 1]
                    for i in ro:
                        if i in Rc:
                            mult[i] *= math.exp(eta)
                        elif ISPOOL[i] and not A.no_shadow:
                            w = solve_with(V_TRUE, S_TRUE, price, np.ones(N, dtype=bool), i, exclude=Rc)
                            if w is not None: mult[i] *= math.exp(eta * A.c * (w[0] - qbar))
                    continue
                for i in ro: mult[i] *= math.exp(eta * A.c * (qk - qbar))
        mult = np.where(ISPOOL & ~sold, math.exp(-eta), mult)
        price = np.where(ISPOOL, np.clip(price * mult, FLOOR, MAXBUY), FLOOR)
        if r >= A.rounds * 2 // 3:
            avg += price; navg += 1
        if r % 10 == 0 or r == A.rounds - 1:
            spend = sum(float(price[ro].sum()) for ro in rosters)
            atcap = int((ISPOOL & (price >= MAXBUY - 1)).sum())
            rec = dict(r=r, eta=eta, sum_pool=float(price[ISPOOL].sum()), spend=spend, unsold=int((ISPOOL & ~sold).sum()), atcap=atcap,
                       W=float(price[iW]), B=float(price[iB]), J=float(price[iJ]), q_min=float(q.min()), q_max=float(q.max()),
                       q_free_mean=qbar, q_free_sd=float(q[free].std()) if free.any() else None, q_capped=[float(x) for x in q[capped]])
            hist.append(rec)
            print(f"r{r:3d} eta {eta:.3f} sum ${rec['sum_pool']/1e6:.2f}M spend ${spend/1e6:.2f}M unsold {rec['unsold']} atcap {atcap} "
                  f"W ${rec['W']/1e3:.0f}k B ${rec['B']/1e3:.0f}k J ${rec['J']/1e3:.0f}k  free rosters {qbar:.3f}+-{rec['q_free_sd'] or 0:.3f} "
                  f"[{q.min():.3f},{q.max():.3f}] capped {[round(x, 3) for x in rec['q_capped']]}  {time.time()-t0:.0f}s", flush=True)
    return avg / max(navg, 1), hist


def leagues_at(pbar, seed, reps=5, seasons=200):
    """Sequential allocation at the given prices in `reps` random orders: each owner in turn takes the best
    roster still available. Returns the league records and the mean tie value of the rosters that hold no
    rationed player (the market's 'average team')."""
    rng = random.Random(seed + 100)
    R = [i for i in range(N) if ISPOOL[i] and pbar[i] >= MAXBUY - 1000]
    leagues = []; free_q = []
    for rep in range(reps):
        order = list(range(D.N_TEAMS)); rng.shuffle(order)
        av = np.ones(N, dtype=bool); rosters = []
        for _ in order:
            got = solve(V_TRUE, S_TRUE, pbar, av); rosters.append(got[1]); av[got[1]] = False
        ex = league_win(rosters)
        q = [true_tie(ro) for ro in rosters]
        free_q += [qq for qq, ro in zip(q, rosters) if not any(i in R for i in ro)]
        spend = [float(pbar[ro].sum()) for ro in rosters]
        wt = [t for t, ro in enumerate(rosters) if IDX[WATERS] in ro]
        rs = [tuple(BOARD[x] for x in ro) for ro in rosters]
        _exp, _mw, ttl = D.season(rs, seasons, rng)
        leagues.append(dict(spread=(max(ex) - min(ex)) * 100, top=max(ex), waters_team=(ex[wt[0]] if wt else None), waters_title=(ttl[wt[0]] if wt else None),
                            fav_title=max(ttl), n10=sum(1 for t in ttl if t >= 0.10), runner=sorted(ttl)[-2], q=q, spend=spend,
                            rosters=[[(nm(i), round(float(pbar[i]) / 1e3)) for i in ro] for ro in rosters], ex=ex, ttl=list(ttl)))
        print(f"league rep {rep}: win% {min(ex)*100:.1f}-{max(ex)*100:.1f} (spread {(max(ex)-min(ex))*100:.1f}), Waters team {(ex[wt[0]] if wt else float('nan'))*100:.1f}% / title {(ttl[wt[0]] if wt else float('nan'))*100:.0f}%, "
              f"fav title {max(ttl)*100:.0f}%, runner {sorted(ttl)[-2]*100:.0f}%, teams>=10% {sum(1 for t in ttl if t>=0.10)}, spend ${min(spend)/1e3:.0f}k-${max(spend)/1e3:.0f}k", flush=True)
    return leagues, (float(np.mean(free_q)) if free_q else float("nan")), (float(np.std(free_q)) if free_q else float("nan"))


def _bisect_price(i, target, pbar, exclude):
    """Highest price for player i at which the best roster holding i is still worth >= target."""
    p2 = pbar.copy(); lo, hi = FLOOR, 3 * MAXBUY
    for _ in range(22):
        mid = (lo + hi) / 2; p2[i] = mid
        w = solve_with(V_TRUE, S_TRUE, p2, np.ones(N, dtype=bool), i, exclude=exclude)
        if w is not None and w[0] >= target: lo = mid
        else: hi = mid
    return lo


def analyse(pbar):
    """At the time-averaged prices: rationed set, best roster with/without it, five leagues (sequential
    allocation), and for each star two indifference prices -- against the best roster an owner can build
    without the rationed players (available to one owner) and against the AVERAGE roster the leagues hand out
    (what the market actually equalises to)."""
    avail = np.ones(N, dtype=bool)
    out = {}
    top = sorted(range(N), key=lambda i: -LPA[i])[:A.stars]
    best_all = solve(V_TRUE, S_TRUE, pbar, avail)
    R = [i for i in range(N) if ISPOOL[i] and pbar[i] >= MAXBUY - 1000]
    out["rationed"] = [nm(i) for i in R]
    out["best_overall"] = dict(value=best_all[0], cost=best_all[2], roster=[(nm(i), round(float(pbar[i]) / 1e3)) for i in best_all[1]])
    best_noR = solve(V_TRUE, S_TRUE, pbar, avail, exclude=R)
    out["best_without_rationed"] = dict(value=best_noR[0], cost=best_noR[2], roster=[(nm(i), round(float(pbar[i]) / 1e3)) for i in best_noR[1]])
    print(f"rationed (at cap): {out['rationed']};  best overall {best_all[0]:.3f} {out['best_overall']['roster']};  best without them {best_noR[0]:.3f} {out['best_without_rationed']['roster']}", flush=True)
    leagues, q_avg, q_avg_sd = leagues_at(pbar, A.seed)
    out["leagues"] = leagues; out["q_avg"] = q_avg; out["q_avg_sd"] = q_avg_sd
    print(f"average free roster at these prices: {q_avg:.3f} +- {q_avg_sd:.3f}", flush=True)
    rows = []
    for i in top:
        X = [j for j in R if j != i]
        without = solve(V_TRUE, S_TRUE, pbar, avail, exclude=X + [i])
        with_eq = solve_with(V_TRUE, S_TRUE, pbar, avail, i, exclude=X)
        p2 = pbar.copy(); p2[i] = MAXBUY
        withcap = solve_with(V_TRUE, S_TRUE, p2, avail, i, exclude=X)
        indiff = _bisect_price(i, without[0], pbar, X)
        indiff_avg = _bisect_price(i, q_avg, pbar, X)
        rows.append(dict(name=nm(i), list=float(LPA[i]), eq=float(pbar[i]), indiff=indiff, indiff_avg=indiff_avg, without=without[0],
                         with_eq=(with_eq[0] if with_eq else None),
                         with_eq_roster=[(nm(j), round(float(pbar[j]) / 1e3)) for j in with_eq[1]] if with_eq else None,
                         with_at_cap=(withcap[0] if withcap else None)))
        print(f"{nm(i):24s} list ${LPA[i]/1e3:4.0f}k  market ${pbar[i]/1e3:4.0f}k  indiff ${indiff/1e3:5.0f}k  avg-team ${indiff_avg/1e3:5.0f}k  with@market {with_eq[0] if with_eq else float('nan'):.3f} / without {without[0]:.3f} / with@cap {withcap[0] if withcap else float('nan'):.3f}", flush=True)
    out["stars"] = rows
    return out


def at_list():
    """The list-price reference rosters (the numbers auction.md quotes): the best $1M roster, the best without
    Waters, the best with her at the first-buy maximum, and what the four-star chain would cost."""
    lp = np.where(ISPOOL, LPA, FLOOR).astype(float); avail = np.ones(N, dtype=bool); iw = IDX[WATERS]
    show = lambda got, pr: f"{got[0]:.3f} ${got[2]/1e3:.0f}k " + ", ".join(f"{nm(j)} ${pr[j]/1e3:.0f}k" for j in got[1])  # noqa: E731
    print("best roster at list prices:      ", show(solve(V_TRUE, S_TRUE, lp, avail), lp))
    print("best without Waters:             ", show(solve(V_TRUE, S_TRUE, lp, avail, exclude=[iw]), lp))
    lp2 = lp.copy(); lp2[iw] = MAXBUY
    print(f"best with Waters at ${MAXBUY/1e3:.0f}k:      ", show(solve_with(V_TRUE, S_TRUE, lp2, avail, iw), lp2))
    chain = ["Anna Leigh Waters", "Ben Johns", "Anna Bright", "JW Johnson"]
    print(f"chain {' + '.join(chain)}: ${sum(lp[IDX[pid_named(n)]] for n in chain)/1e6:.2f}M at list prices")
    caches = sorted(CACHE.glob("market_eq_equalize_seed*_c*_n0.json"))
    for f in caches:
        r = json.load(open(f))
        if r["args"].get("no_shadow"): continue
        print(f"  ${sum(r['prices'][n] for n in chain)/1e6:.2f}M at the market prices of {f.name}")


def reanalyse():
    """Re-run analyse() on every cached run's time-averaged prices (the run itself is not repeated)."""
    global A
    for f in sorted(CACHE.glob("market_eq_*.json")):
        r = json.load(open(f)); a = r["args"]
        A.seed = a["seed"]
        pbar = np.array([r["prices"].get(NAME[u], FLOOR) for u in BOARD], dtype=float)
        print(f"== {f.name}", flush=True)
        res = analyse(pbar)
        r.update(res)
        json.dump(r, open(f, "w"), indent=1)


# ---------------------------------------------------------------- report
def _tiers(prices):
    out = {}
    for g in "FM":
        rows = sorted([(LP[u], prices[NAME[u]], NAME[u]) for u in POOLSET if G[u] == g], key=lambda t: -t[0])
        seg = {}
        for a, b, t in ((0, 5, "#1-5"), (5, 15, "#6-15"), (15, 30, "#16-30"), (30, 60, "#31-60")):
            s_ = rows[a:b]
            seg[t] = (sum(x[1] for x in s_) / sum(x[0] for x in s_), sum(x[1] for x in s_) / len(s_), sum(x[0] for x in s_) / len(s_))
        seg["floor"] = sum(1 for x in rows if x[1] <= FLOOR + 500)
        out[g] = seg
    return out


def report():
    runs = []
    for f in sorted(CACHE.glob("market_eq_*.json")):
        runs.append(json.load(open(f)))
    if not runs:
        sys.exit("no cached runs; run with --seed first")
    eq_runs = [r for r in runs if r["args"]["rule"] == "equalize" and not r["args"].get("no_shadow")]
    main = eq_runs or runs
    L = []
    L.append("# Market prices -- what the backward induction settles on\n")
    L.append("Every owner plans whole $1M rosters (exhaustive 3M+3W search), rosters are handed out in a random order, "
             "priced players on above-average rosters are marked up and on below-average rosters marked down, unsold "
             "priced players fall, prices clipped to [$30k floor, $850k first-buy maximum]; prices are the average over the last third "
             "of the rounds. Board `mlp2026`, 20 teams, true values, identical owners unless `noise` is set. "
             "The hand-written read is in `auction.md` (\"The market limit\"). Built by `market_eq.py`.\n")
    L.append("## Runs\n")
    L.append("| rule | seed | c | rounds | noise | roster-value sd at the end (free rosters) | rationed (at $850k with excess demand) | priced pool total | priced players at the floor | Waters' team win% / title% | second-best team | runner-up title | teams >= 10% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        a = r["args"]; h = r["hist"][-1]; lg = r["leagues"]
        ex_sorted = sorted(lg[0]["ex"], reverse=True)
        n_floor = sum(1 for p in r["prices"].values() if p <= FLOOR + 500)
        L.append(f"| {a['rule']}{' (no shadow)' if a.get('no_shadow') else ''} | {a['seed']} | {a['c']:g} | {a['rounds']} | {a['noise']:g} | "
                 f"{h['q_free_sd'] if h['q_free_sd'] is not None else float('nan'):.3f} | {', '.join(r['rationed']) or 'none'} | ${sum(r['prices'].values())/1e6:.2f}M (list $20.00M) | {n_floor} | "
                 f"{lg[0]['waters_team']*100:.1f}% / {min(x['waters_title'] for x in lg)*100:.0f}-{max(x['waters_title'] for x in lg)*100:.0f}% | {ex_sorted[1]*100:.1f}% | "
                 f"{min(x['runner'] for x in lg)*100:.0f}-{max(x['runner'] for x in lg)*100:.0f}% | {min(x['n10'] for x in lg)}-{max(x['n10'] for x in lg)} |")
    L.append("")
    L.append("## Market price vs list by tier (list rank within gender)\n")
    L.append("| run | women #1-5 | #6-15 | #16-30 | #31-60 | at floor | men #1-5 | #6-15 | #16-30 | #31-60 | at floor |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        a = r["args"]; t = _tiers(r["prices"])
        cells = []
        for g in "FM":
            for k in ("#1-5", "#6-15", "#16-30", "#31-60"):
                cells.append(f"{t[g][k][0]*100:.0f}% (${t[g][k][1]/1e3:.0f}k vs ${t[g][k][2]/1e3:.0f}k)")
            cells.append(str(t[g]["floor"]))
        L.append(f"| {a['rule']} seed {a['seed']} c {a['c']:g}{' no shadow' if a.get('no_shadow') else ''} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"## The top {A.stars} by list price (mean over the {len(main)} main run{'s' if len(main) != 1 else ''}, range in brackets)\n")
    qa = [r.get("q_avg") for r in main if r.get("q_avg") is not None]
    L.append("Two benchmarks. Indifference (best) = the price at which the best roster holding the player equals the best roster an owner can build without the rationed players -- "
             "a roster only one owner gets. Indifference (average) = the price at which the best roster holding the player equals the AVERAGE roster the leagues below hand out"
             + (f" ({np.mean(qa):.3f} tie probability vs the reference team)" if qa else "") + " -- the benchmark the market actually equalises to. "
             "with@market / without = the best roster with the player at market prices and the best without the rationed players; "
             "with@cap = the best roster holding the player if he cost $850k. The market column is a time average of oscillating prices, and a roster that is affordable in most "
             "rounds is not always affordable at the averaged prices, so the two differ by a few percent either way (market above for the top dozen, below for the next); together they bracket the star's price.\n")
    L.append("| player | list | market | market / list | indifference (average) | (average) / list | indifference (best) | with@market | without | with@cap |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    names = [s["name"] for s in main[0]["stars"]]
    star_rows = {}
    for r in main:
        for s in r["stars"]:
            star_rows.setdefault(s["name"], []).append(s)
    for n_ in names:
        ss = star_rows[n_]
        mk = [s["eq"] for s in ss]; ind = [s["indiff"] for s in ss]; inda = [s.get("indiff_avg") for s in ss if s.get("indiff_avg") is not None]
        we = [s["with_eq"] for s in ss if s["with_eq"] is not None]; wo = [s["without"] for s in ss]; wc = [s["with_at_cap"] for s in ss if s["with_at_cap"] is not None]
        lst = ss[0]["list"]
        rng_ = f" [{min(mk)/1e3:.0f}-{max(mk)/1e3:.0f}]" if len(ss) > 1 and max(mk) - min(mk) > 500 else ""
        rnga = f" [{min(inda)/1e3:.0f}-{max(inda)/1e3:.0f}]" if len(inda) > 1 and max(inda) - min(inda) > 500 else ""
        L.append(f"| {n_} | ${lst/1e3:.0f}k | ${np.mean(mk)/1e3:.0f}k{rng_} | {np.mean(mk)/lst*100:.0f}% | "
                 + (f"${np.mean(inda)/1e3:.0f}k{rnga} | {np.mean(inda)/lst*100:.0f}% | " if inda else "-- | -- | ")
                 + f"${np.mean(ind)/1e3:.0f}k | {np.mean(we) if we else float('nan'):.3f} | {np.mean(wo):.3f} | {np.mean(wc) if wc else float('nan'):.3f} |")
    L.append("")
    L.append("## The Waters roster and the best roster without her, at market prices\n")
    for r in main:
        a = r["args"]
        L.append(f"- seed {a['seed']}, c {a['c']:g}: best overall {r['best_overall']['value']:.3f} = " + ", ".join(f"{n} ${p}k" for n, p in r["best_overall"]["roster"]) +
                 f"; best without the rationed {r['best_without_rationed']['value']:.3f} = " + ", ".join(f"{n} ${p}k" for n, p in r["best_without_rationed"]["roster"]) + ".")
    L.append("")
    L.append("## Leagues at market prices (five random allocation orders per run, 200 seasons each)\n")
    for r in runs:
        a = r["args"]
        for k, lg in enumerate(r["leagues"]):
            ex = sorted(lg["ex"], reverse=True)
            L.append(f"- {a['rule']} seed {a['seed']} c {a['c']:g} rep {k}: win% {ex[0]*100:.1f} / {ex[1]*100:.1f} / ... / {ex[-1]*100:.1f}, "
                     f"Waters' team {lg['waters_team']*100:.1f}% / title {lg['waters_title']*100:.0f}%, runner-up {lg['runner']*100:.0f}%, teams >= 10%: {lg['n10']}, "
                     f"spend ${min(lg['spend'])/1e3:.0f}k-${max(lg['spend'])/1e3:.0f}k.")
    L.append("")
    L.append("## Convergence (every 10th round of each run)\n")
    for r in runs:
        a = r["args"]
        L.append(f"- {a['rule']} seed {a['seed']} c {a['c']:g}: " + "; ".join(
            f"r{h['r']} free {h['q_free_mean']:.3f}+-{(h['q_free_sd'] or 0):.3f} capped {[round(x,3) for x in h['q_capped']]} sum ${h['sum_pool']/1e6:.2f}M unsold {h['unsold']} W ${h['W']/1e3:.0f}k B ${h['B']/1e3:.0f}k J ${h['J']/1e3:.0f}k"
            for h in r["hist"][::3]) + ".")
    Path(A.out).write_text("\n".join(L) + "\n")
    # CSV: market price per priced player, averaged over the main runs
    rows = []
    inda_by = {}
    for r in main:
        for s_ in r["stars"]:
            if s_.get("indiff_avg") is not None: inda_by.setdefault(s_["name"], []).append(s_["indiff_avg"])
    for u in POOLSET:
        ps = [r["prices"][NAME[u]] for r in main]
        ia = inda_by.get(NAME[u])
        rows.append(dict(player=NAME[u], gender=G[u], list_price=round(LP[u]), market_price=round(float(np.mean(ps))),
                         market_min=round(min(ps)), market_max=round(max(ps)), ratio=round(float(np.mean(ps)) / LP[u], 3),
                         runs_at_floor=sum(1 for p in ps if p <= FLOOR + 500), n_runs=len(ps),
                         avg_team_price=(round(float(np.mean(ia))) if ia else ""), avg_team_ratio=(round(float(np.mean(ia)) / LP[u], 3) if ia else "")))
    rows.sort(key=lambda d: (-d["list_price"], d["player"]))
    with (HERE / "market_prices.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {A.out} and {HERE / 'market_prices.csv'} from {len(runs)} run(s) ({len(main)} main)")


if __name__ == "__main__":
    if A.selftest:
        selftest()
    elif A.at_list:
        at_list()
    elif A.reanalyse:
        reanalyse()
    elif A.report:
        report()
    else:
        t0 = time.time()
        got = solve(V_TRUE, S_TRUE, np.where(ISPOOL, LPA, FLOOR).astype(float), np.ones(N, dtype=bool))
        print(f"list-price best roster {got[0]:.4f} ${got[2]/1e3:.0f}k {[nm(i) for i in got[1]]}  ({time.time()-t0:.2f}s per solve)")
        pbar, hist = run()
        res = analyse(pbar)
        res["hist"] = hist
        res["prices"] = {NAME[BOARD[i]]: float(pbar[i]) for i in range(N) if ISPOOL[i]}
        res["args"] = dict(rule=A.rule, seed=A.seed, rounds=A.rounds, c=A.c, eta0=A.eta0, tau=A.tau, k=A.k, noise=A.noise, no_shadow=A.no_shadow, board=A.board)
        CACHE.mkdir(exist_ok=True)
        f = CACHE / f"market_eq_{A.rule}{'_noshadow' if A.no_shadow else ''}_seed{A.seed}_c{A.c:g}_n{A.noise:g}.json"
        json.dump(res, open(f, "w"), indent=1)
        print(f"wrote {f}")
