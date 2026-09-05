"""value_cap/strategic_auction.py -- auction agents with strategic dials,
self-play to a symmetric equilibrium in the dial space, the size of the best
remaining deviation (exploitability), validated on the exactly-solved toy.

Why. `auction_sim.py` owners bid truthfully (ceiling = the price at which
they are indifferent, against a greedy one-slot-at-a-time projection);
`market_eq.py` shows a whole-roster planner beats them; `toy_auction.py`
solves small rooms exactly and finds that truthful planners are NOT the
equilibrium there. A full 20-team equilibrium is out of reach (see
toy_auction.md), so this is the next honest instrument: a small family of
strategies (dials), every team playing the same dial vector (a symmetric
profile), coordinate ascent on single-team deviations until no dial change
helps -- a symmetric equilibrium in the dial space -- and the best remaining
single-dial deviation reported as the exploitability. It is validated first
on the toy, where the exact equilibrium is known: the search passes a cell
if it lands on the equilibrium ROSTERS (prices are a selection there,
toy_auction.md finding 1).

Dials (theta):
  a_star / a_good / a_depth  multiplier on the indifference ceiling by the
                             tier of the player on the block (list rank per
                             gender: 1-5 star, 6-30 good, 31+ depth; the toy
                             has no depth sales)
  a_good0  the good-tier multiplier used INSTEAD of a_good by a team that
           holds no star (role-conditional bidding: the toy's exact
           equilibrium has the no-star team out-bidding the star teams for
           every good player, which no symmetric unconditional multiplier
           can express)
  expect   list | inflate | rivals | learned  expected prices for the
           projection: the cheat sheet; the cheat sheet scaled by money-left
           / value-left (auction_sim's `inflated`); the cheat sheet capped at
           what the richest rival could still pay for that player; the prices
           the room actually paid last time (fictitious play: an all-learned
           profile is iterated to a price fixed point, a deviator expects the
           incumbent room's prices)
  plan     greedy | planner  greedy one-slot-at-a-time fill (auction_sim's
           projection) or the exhaustive completion of the roster held so
           far (market_eq's solver generalised to a partly filled roster)
  nom      snake | want | cheap | drain | dear  (real board only; the toy
           fixes the sale order) auction_sim's rule: the player I would
           pick right now in a snake draft, projected under the auction's
           own scarcity / the dearest player in my planned roster /
           the cheapest / the dearest available player NOT in my plan (make
           the rivals spend) / the dearest available player, full stop
  bid      shortlist | all  (real board only) bid only on players you would
           shortlist for a snake pick (auction_sim's rule: top 30 by believed
           value per gender, top 6 singles, 5 cheapest) / bid on everyone

Mechanism (unchanged from auction_sim / the toy's `secondprice`): every
eligible team names a ceiling, the highest wins at the second-highest plus
one increment; the nominator opens at the floor; nobody can bid past budget
minus the cheapest legal completion; rotation nomination. Payoff = in-league
win% (mean tie probability against the other teams; on the toy the fixed
Waters team is one of them). Self-play: R auctions per evaluation, the
deviating team's slot rotating with the seed, gain = deviator's win% minus
what the same slot earned with everyone on the incumbent profile (paired).

    python value_cap/strategic_auction.py --world toy --validate            # the toy ladder vs the exact equilibria
    python value_cap/strategic_auction.py --world toy --teams 3 --board 1,1,3,3 --unit 50000
    python value_cap/strategic_auction.py --world real --time              # time one 20-team auction
    python value_cap/strategic_auction.py --world real --seeds 4 --jobs 4  # the 20-team self-play
"""
from __future__ import annotations

import argparse
import itertools
import os
import json
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--world", choices=["toy", "real"], default="toy")
ap.add_argument("--teams", type=int, default=3, help="toy: strategic teams")
ap.add_argument("--board", default="1,1,3,3", help="toy: SM,SF,GM,GF on the block")
ap.add_argument("--unit", type=int, default=50_000, help="toy: money unit")
ap.add_argument("--order", default="stars_first")
ap.add_argument("--validate", action="store_true", help="toy: run the ladder of cells against the exact solver")
ap.add_argument("--exact", choices=["secondprice", "english", "both"], default="both", help="toy: which exact conventions to solve")
ap.add_argument("--seeds", type=int, default=8, help="real: auctions per evaluation (the deviator's slot is spread over the nomination order: slot = 7*seed mod 20)")
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--rounds", type=int, default=6, help="coordinate-ascent rounds")
ap.add_argument("--start", default="truthful", help="starting profile: truthful | planner | a dial spec like a_star=1.25,plan=planner")
ap.add_argument("--tol", type=float, default=None, help="minimum gain to adopt a deviation (default: toy 1e-9, real 0.0025)")
ap.add_argument("--se-mult", type=float, default=2.0, help="real: a deviation is adopted only if gain > se-mult * its standard error over seeds (the toy's seeds are the exhaustive deviator slots, so its mean is exact and this is 0 there)")
ap.add_argument("--K", type=int, default=10, help="real: planner candidate depth per gender")
ap.add_argument("--plan-top", type=int, default=40, help="real: planner ceilings only for players ranked in the top N by list (else greedy)")
ap.add_argument("--noise", type=float, default=0.0, help="real: owner belief noise (fraction of the value spread), fixed per owner per auction")
ap.add_argument("--time", action="store_true", help="real: time one auction on the starting profile and stop")
ap.add_argument("--fix", default="", help="real: comma list of dials frozen at their start value (a league RULE, e.g. --fix nom with --start nom=dear = nominations in list-price order); they are not searched")
ap.add_argument("--room", action="store_true", help="real: report the room at the starting profile over the seeds (all clones) and stop -- no self-play")
ap.add_argument("--seasons", type=int, default=200, help="real: seasons for the title share in the final report")
ap.add_argument("--tag", default="", help="cache/report suffix")
ap.add_argument("--quiet", action="store_true")
A = ap.parse_args()

EPS = 1e-9
FIELDS = ("a_star", "a_good", "a_good0", "a_depth", "expect", "plan", "nom", "bid")
DIALS = {
    "a_star": [0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
    "a_good": [0.6, 0.8, 1.0, 1.25, 1.5],
    "a_good0": [0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
    "a_depth": [0.6, 0.8, 1.0, 1.25, 1.5],
    "expect": ["list", "inflate", "rivals", "learned"],
    "plan": ["greedy", "planner"],
    "nom": ["snake", "want", "cheap", "drain", "dear"],
    "bid": ["shortlist", "all"],
}
STARTS = {
    "truthful": dict(a_star=1.0, a_good=1.0, a_depth=1.0, a_good0=1.0, expect="list", plan="greedy", nom="snake", bid="shortlist"),
    "planner": dict(a_star=1.0, a_good=1.0, a_depth=1.0, a_good0=1.0, expect="list", plan="planner", nom="want", bid="all"),
}


def theta_of(spec):
    if spec in STARTS:
        return tuple(STARTS[spec][f] for f in FIELDS)
    d = dict(STARTS["truthful"])
    for kv in spec.split(","):
        k, v = kv.split("=")
        d[k] = float(v) if k.startswith("a_") else v
    return tuple(d[f] for f in FIELDS)


def th(theta, f):
    return theta[FIELDS.index(f)]


def fmt_theta(theta):
    return " ".join(f"{f}={th(theta, f):g}" if f.startswith("a_") else f"{f}={th(theta, f)}" for f in FIELDS)


def with_dial(theta, f, v):
    t = list(theta)
    t[FIELDS.index(f)] = v
    return tuple(t)


# =============================================================== the toy world
class ToyWorld:
    """toy_auction.py's typed game played by dial agents: fixed sale order,
    sealed ceilings, second price + 1, rotation tie-break (as the exact solver
    and its truthful-planner benchmark)."""
    name = "toy"
    dials = [f for f in FIELDS if f not in ("nom", "bid")]

    def __init__(self, teams, board, unit, order, waters=True):
        import importlib
        argv = sys.argv
        sys.argv = ["toy_auction.py", "--teams", str(teams), "--board", board, "--unit", str(unit),
                    "--order", order, "--quiet"] + ([] if waters else ["--no-waters"])
        import toy_auction as T
        importlib.reload(T)
        sys.argv = argv
        self.T = T
        self.n = T.n
        self.label = f"{teams} teams, board {board}, unit ${unit/1e3:.0f}k, {order}"
        # type index (SM, GM, SF, GF) in descending value: the greedy projection's pick order
        self.type_keys = [("S", "M"), ("G", "M"), ("S", "F"), ("G", "F")]
        self.value_order = sorted(range(4), key=lambda t: -T.TYPES[self.type_keys[t][0] + self.type_keys[t][1]]["v"])
        self.list_units = [T.LISTU["SM"], T.LISTU["GM"], T.LISTU["SF"], T.LISTU["GF"]]
        self.seeds = list(range(self.n))  # deterministic game: a "seed" is the deviator's slot

    def slot(self, sd):
        return sd % self.n

    # -- projections
    def objective(self, nh):
        T = self.T
        return T.tie(T.roster_ids(nh), T.REF_ROSTER)

    def greedy_completion(self, h, b, remaining, prices):
        T = self.T
        h = list(h); rem = list(remaining)
        while sum(h) < 6:
            pick = None
            for t in self.value_order:
                if rem[t] <= 0:
                    continue
                g = self.type_keys[t][1]
                if T.gender_count(h, g) >= 3:
                    continue
                if b - prices[t] < T.FLOORC * (6 - sum(h) - 1):
                    continue
                pick = t
                break
            if pick is None:
                break
            h[pick] += 1; rem[pick] -= 1; b -= prices[pick]
        return self.objective(tuple(h))

    def completion(self, theta, h, b, remaining, prices):
        if th(theta, "plan") == "planner":
            return self.T.best_completion(h, b, remaining, prices, self.objective)
        return self.greedy_completion(h, b, remaining, prices)

    def expected(self, theta, i, holds, budgets, remaining, learned=None):
        T = self.T
        mode = th(theta, "expect")
        lp = self.list_units
        if mode == "list" or (mode == "learned" and learned is None):
            return lp
        if mode == "learned":
            return list(learned)
        if mode == "inflate":
            money = sum(budgets[j] for j in range(self.n) if sum(holds[j]) < 6)
            value = 0
            for g, ts in (("M", (0, 1)), ("F", (2, 3))):
                slots = sum(3 - T.gender_count(holds[j], g) for j in range(self.n))
                items = sorted([lp[t] for t in ts for _ in range(remaining[t])], reverse=True)[:slots]
                value += sum(items) + T.FLOORC * (slots - len(items))
            rho = money / value if value > 0 else 1.0
            return [max(T.FLOORC, int(round(rho * p))) for p in lp]
        # rivals: the most any other team could still pay for a player of that gender
        out = []
        for t in range(4):
            g = self.type_keys[t][1]
            best = 0
            for j in range(self.n):
                if j == i or T.gender_count(holds[j], g) >= 3:
                    continue
                best = max(best, budgets[j] - T.FLOORC * (6 - sum(holds[j]) - 1))
            out.append(max(T.FLOORC, min(lp[t], best)))
        return out

    def ceiling(self, theta, i, item, holds, budgets, remaining, learned=None):
        T = self.T
        h, b = holds[i], budgets[i]
        prices = self.expected(theta, i, holds, budgets, remaining, learned)
        without = self.completion(theta, h, b, remaining, prices)
        hi = b - T.FLOORC * (6 - sum(h) - 1)
        c = 0
        for bid in range(T.FLOORC, hi + 1):
            if self.completion(theta, T.add(h, item), b - bid, remaining, prices) >= without - EPS:
                c = bid
            else:
                break
        if item[0] == "S":
            a = th(theta, "a_star")
        else:
            a = th(theta, "a_good") if (h[0] + h[2]) > 0 else th(theta, "a_good0")
        if c > 0:
            c = max(T.FLOORC, min(hi, int(math.floor(a * c + 0.5))))
        return c

    def run(self, thetas, seed, learned=None):
        T = self.T
        n = self.n
        holds = [(0, 0, 0, 0)] * n
        budgets = [T.BUDGET] * n
        remaining = [T.BSM, T.BGM, T.BSF, T.BGF]
        path = []
        for k, item in enumerate(T.ITEMS):
            remaining[T.HI[item]] -= 1
            g = item[1]
            ceil = []
            for i in range(n):
                if not T.can_bid(holds[i], budgets[i], g, T.FLOORC):
                    ceil.append(0)
                    continue
                ceil.append(self.ceiling(thetas[i], i, item, holds, budgets, remaining, learned))
            order = sorted(range(n), key=lambda i: (-ceil[i], (i - k) % n))
            w = order[0]
            if ceil[w] < T.FLOORC:
                path.append((item, "unsold", None, 0, tuple(ceil)))
                continue
            second = ceil[order[1]] if n > 1 else 0
            p = max(T.FLOORC, min(ceil[w], second + 1))
            budgets[w] -= p
            holds[w] = T.add(holds[w], item)
            path.append((item, "win", w, p, tuple(ceil)))
        holds = tuple(holds)
        # realised price per type (mean over sold; unsold/none -> list), the next room's expectation
        paid = {t: [] for t in range(4)}
        for item, kind, who, p, *_ in path:
            if kind == "win":
                paid[T.HI[item]].append(p)
        prices = tuple(int(round(statistics.mean(paid[t]))) if paid[t] else self.list_units[t] for t in range(4))
        return dict(rosters=holds, pay=list(T.payoff(holds)), path=path, budgets=tuple(budgets), prices=prices)

    # -- the exact solver, for validation
    def exact(self, mech):
        T = self.T
        T.SOLVE = T.solve_sale_sp if mech == "secondprice" else T.solve_sale
        T.VMEMO.clear(); T.STATS["sale_states"] = 0
        t0 = time.time()
        path, holds, budgets = T.play()
        return dict(rosters=tuple(holds), pay=list(T.payoff(tuple(holds))), budgets=budgets, secs=time.time() - t0,
                    states=len(T.VMEMO), sales=[(it, kind, who, p) for (k, it, kind, who, p, *rest) in path])

    def describe(self, res):
        T = self.T
        rows = []
        for i, h in enumerate(res["rosters"]):
            rows.append(f"T{i} {T.fmt_roster(h)} spent {T.BUDGET - res['budgets'][i]}/{T.BUDGET} win {100*res['pay'][i]:.1f}%")
        sales = res.get("sales") or [(it, kind, who, p) for (it, kind, who, p, *_) in res["path"]]
        line = "  ".join(f"{it[0]}{it[1]}->{'--' if kind != 'win' else 'T%d' % who}@{p}" for it, kind, who, p in sales)
        return line, rows


# ============================================================== the real world
class RealWorld:
    """The real 2026 board (draft_sim.set_board('mlp2026'), list prices as the
    cheat sheet, market_eq's values and tie model) auctioned by dial agents.
    Rotation nomination, the nominator opens at the floor, sealed ceilings,
    second price + INC. 20 teams; payoff = mean tie probability vs the other
    19 (a symmetric profile earns 50% by construction)."""
    name = "real"
    dials = list(FIELDS)
    INC = 5_000

    def __init__(self, K, plan_top, noise):
        argv = sys.argv
        sys.argv = ["market_eq.py"]
        import market_eq as M
        import draft_sim as D
        sys.argv = argv
        self.M, self.D = M, D
        self.n = D.N_TEAMS
        self.N = M.N
        self.K = K
        self.plan_top = plan_top
        self.noise = noise
        self.E = M.E
        self.GEN = M.GEN
        self.LP = M.LPA.copy()
        self.floor_price = np.where(M.ISPOOL, M.LPA, M.FLOOR).astype(float)
        self.name_of = lambda i: M.NAME[M.BOARD[i]]
        # tier by list rank within gender among the priced pool
        self.rank = np.full(self.N, 999, dtype=int)
        for g in ("M", "F"):
            idx = sorted((i for i in range(self.N) if M.GEN[i] == g and M.ISPOOL[i]), key=lambda i: (-M.LPA[i], i))
            for r, i in enumerate(idx):
                self.rank[i] = r + 1
        self.tier = np.where(self.rank <= 5, 0, np.where(self.rank <= 30, 1, 2))
        self.label = f"{self.n} teams, board mlp2026 ({self.N} players), K={K}, planner for list rank<={plan_top}, noise {noise:g}"
        self.seeds = list(range(A.seeds))
        self.waters = M.IDX[M.WATERS]
        self.trace = set(os.environ.get("SA_TRACE", "").split(",")) - {""}
        self.pids = list(M.BOARD)

    def slot(self, sd):
        """The deviator's slot for seed sd, spread over the nomination order (slot 0 nominates first)."""
        return (7 * sd) % self.n

    # -- values (beliefs)
    def beliefs(self, rng):
        return self.M.beliefs(rng, self.noise)

    # -- the completion solver (market_eq.solve generalised to a partly filled roster)
    def _triples(self, g, fixed, v, s, price, avail):
        M = self.M
        C = [c for c in M.candidates(g, v, s, price, avail, self.K) if c not in fixed]
        need = 3 - len(fixed)
        if need == 0:
            combos = [tuple(fixed)]
        else:
            combos = [tuple(fixed) + c for c in itertools.combinations(C, need)]
            if not combos:
                return None
        arr = np.array(combos, dtype=np.int64)
        vv = v[arr]
        order = np.argsort(-vv, axis=1, kind="stable")
        idx = np.take_along_axis(arr, order, 1)
        vs = np.take_along_axis(vv, order, 1)
        gam = M.GAM["MD" if g == "M" else "WD"]
        vhi, vlo = vs[:, 0], vs[:, 1]
        u2hi, u2lo = M.U2[idx[:, 0]], M.U2[idx[:, 1]]
        ss = np.sort(s[arr], axis=1)
        fixed_cost = float(price[list(fixed)].sum()) if fixed else 0.0
        return dict(idx=arr, vhi=vhi, vlo=vlo, u2hi=u2hi, u2lo=u2lo, S=vhi + vlo + gam * (vhi - vlo), U=u2hi + u2lo,
                    s2=ss[:, 2] + ss[:, 1], cost=price[arr].sum(1) - fixed_cost)

    def complete(self, fixed, budget, v, s, price, avail):
        """Best completion of `fixed` (indices held) under `budget` at `price`, by tie prob vs the reference team."""
        M = self.M
        fm = [i for i in fixed if self.GEN[i] == "M"]; ff = [i for i in fixed if self.GEN[i] == "F"]
        Mt = self._triples("M", fm, v, s, price, avail); Ft = self._triples("F", ff, v, s, price, avail)
        if Mt is None or Ft is None:
            return None
        tot = Mt["cost"][:, None] + Ft["cost"][None, :]
        feas = tot <= budget + 1e-6
        if not feas.any():
            return None
        pWD = M.fgrid(Ft["S"] - M.S_REF["WD"], np.sqrt(Ft["U"] + M.U_REF["WD"]))[None, :]
        pMD = M.fgrid(Mt["S"] - M.S_REF["MD"], np.sqrt(Mt["U"] + M.U_REF["MD"]))[:, None]
        gx = M.GAM["MXD"]
        w1 = Ft["vhi"][None, :]; w2 = Ft["vlo"][None, :]; m1 = Mt["vhi"][:, None]; m2 = Mt["vlo"][:, None]
        pv = lambda w, m: w + m + gx * np.abs(w - m)  # noqa: E731
        A11, A22, B12, B21 = pv(w1, m1), pv(w2, m2), pv(w1, m2), pv(w2, m1)
        useA = (A11 + A22) >= (B12 + B21)
        Sx1 = np.where(useA, A11, B12); Sx2 = np.where(useA, A22, B21)
        Ux1 = Ft["u2hi"][None, :] + np.where(useA, Mt["u2hi"][:, None], Mt["u2lo"][:, None])
        Ux2 = Ft["u2lo"][None, :] + np.where(useA, Mt["u2lo"][:, None], Mt["u2hi"][:, None])
        p3 = M.fgrid(Sx1 - M.S_REF["X1"], np.sqrt(Ux1 + M.U_REF["X1"])); p4 = M.fgrid(Sx2 - M.S_REF["X2"], np.sqrt(Ux2 + M.U_REF["X2"]))
        d0 = np.ones_like(p3); d1 = np.zeros_like(p3); d2 = np.zeros_like(p3); d3 = np.zeros_like(p3); d4 = np.zeros_like(p3)
        for p in (pWD * np.ones_like(p3), pMD * np.ones_like(p3), p3, p4):
            q = 1.0 - p
            d0, d1, d2, d3, d4 = d0 * q, d1 * q + d0 * p, d2 * q + d1 * p, d3 * q + d2 * p, d4 + d3 * p
        db = (Mt["s2"][:, None] + Ft["s2"][None, :]) / 4.0
        pdb = M.DBTAB[np.rint(M.sig(M.K_DB * (db - M.DB_REF)) * 10000).astype(np.int64)]
        val = np.where(feas, d3 + d4 + d2 * pdb, -np.inf)
        k = int(np.argmax(val)); pm, pf = divmod(k, Ft["cost"].shape[0])
        roster = [int(x) for x in Mt["idx"][pm]] + [int(x) for x in Ft["idx"][pf]]
        return float(val[pm, pf]), roster, float(tot[pm, pf])

    # -- the greedy projection (auction_sim.project on indices)
    def completion_cost(self, price, by_g, exclude, need):
        tot = 0.0
        for g in ("M", "F"):
            k = need[g]
            if k <= 0:
                continue
            got = 0
            for u in by_g[g]:
                if u in exclude:
                    continue
                tot += price[u]; got += 1
                if got == k:
                    break
            if got < k:
                return float("inf")
        return tot

    def greedy(self, roster, budget, price, need, order, by_g, exclude, gaps=None, by_price=None):
        proj = list(roster); left = budget; na = dict(need); taken = set(roster) | set(exclude)
        j = 0
        while na["M"] > 0 or na["F"] > 0:
            gone = set()
            if gaps is not None and j < len(gaps):
                gone = set([u for u in by_price if u not in taken][:gaps[j]])
            j += 1
            pick = None
            for u in order:
                if u in taken or u in gone or na[self.GEN[u]] <= 0:
                    continue
                n2 = dict(na); n2[self.GEN[u]] -= 1
                if price[u] + self.completion_cost(price, by_g, taken | {u}, n2) <= left + 1e-6:
                    pick = u
                    break
            if pick is None:
                return None
            proj.append(pick); taken.add(pick); left -= price[pick]; na[self.GEN[pick]] -= 1
        return proj

    def score(self, eng, roster):
        return eng.tie(tuple(self.pids[i] for i in roster), self.M.REF)

    # -- per-owner state helpers
    def value(self, ctx, theta, o, fixed, budget, exclude, use_planner):
        """Best projected tie prob for owner o holding `fixed` with `budget`, x excluded. Memoised per sale."""
        key = (o if ctx["distinct"] else 0, tuple(sorted(fixed)), int(round(budget / 1000.0)), tuple(sorted(exclude)), use_planner, th(theta, "expect"))
        r = ctx["memo"].get(key)
        if r is not None:
            return r
        price = ctx["prices"][th(theta, "expect")][o] if th(theta, "expect") == "rivals" else ctx["prices"][th(theta, "expect")]
        if use_planner:
            av = ctx["avail"].copy()
            for e in exclude:
                av[e] = False
            out = self.complete(fixed, budget, ctx["v"][o], ctx["s"][o], price, av)
            r = out[0] if out else -1.0
        else:
            need = {"M": 3 - sum(1 for i in fixed if self.GEN[i] == "M"), "F": 3 - sum(1 for i in fixed if self.GEN[i] == "F")}
            by_g = ctx["by_g"][th(theta, "expect")][o] if th(theta, "expect") == "rivals" else ctx["by_g"][th(theta, "expect")]
            proj = self.greedy(list(fixed), budget, price, need, ctx["order"][o], by_g, set(exclude))
            r = self.score(ctx["eng"][o], proj) if proj else -1.0
        ctx["memo"][key] = r
        return r

    def ceiling(self, ctx, theta, o, x, roster, spent, need):
        M = self.M
        gx = self.GEN[x]
        if need[gx] <= 0:
            return 0.0
        budget = M.CAP - spent
        need_after = dict(need); need_after[gx] -= 1
        mode = th(theta, "expect")
        price = ctx["prices"][mode][o] if mode == "rivals" else ctx["prices"][mode]
        by_g = ctx["by_g"][mode][o] if mode == "rivals" else ctx["by_g"][mode]
        top = budget - self.completion_cost(price, by_g, set(roster) | {x}, need_after)
        if top < M.FLOOR:
            return 0.0
        use_planner = th(theta, "plan") == "planner" and self.rank[x] <= self.plan_top
        base = self.value(ctx, theta, o, roster, budget, (x,), use_planner)
        withx = list(roster) + [x]

        def f(p):
            return self.value(ctx, theta, o, withx, budget - p, (x,), use_planner)
        if f(top) >= base - EPS:
            c = top
        elif f(M.FLOOR) < base - EPS:
            return 0.0
        else:
            lo, hi = M.FLOOR, top
            for _ in range(5 if use_planner else 8):
                mid = (lo + hi) / 2
                if f(mid) >= base - EPS:
                    lo = mid
                else:
                    hi = mid
            c = lo
        if self.tier[x] == 0:
            a = th(theta, "a_star")
        elif self.tier[x] == 1:
            a = th(theta, "a_good") if any(self.tier[u] == 0 for u in roster) else th(theta, "a_good0")
        else:
            a = th(theta, "a_depth")
        c = min(top, a * c)
        return c if c >= M.FLOOR else 0.0

    def expected_prices(self, mode, avail, spent, need, o=None, learned=None):
        M = self.M
        lp = self.floor_price
        if mode == "list" or (mode == "learned" and learned is None):
            return lp
        if mode == "learned":
            return np.asarray(learned, dtype=float)
        if mode == "inflate":
            money = sum(max(0.0, M.CAP - spent[j]) for j in range(self.n) if need[j]["M"] + need[j]["F"] > 0)
            value = 0.0
            for g in ("M", "F"):
                k = sum(nd[g] for nd in need)
                tops = np.sort(lp[avail & (self.GEN == g)])[::-1][:k]
                value += float(tops.sum())
            rho = money / value if value > 0 else 1.0
            return np.where(lp <= M.FLOOR, M.FLOOR, np.maximum(M.FLOOR, rho * lp))
        # rivals: cap at the most any other owner could still pay for a player of that gender
        out = lp.copy()
        for g in ("M", "F"):
            best = 0.0
            for j in range(self.n):
                if j == o or need[j][g] <= 0:
                    continue
                slots = need[j]["M"] + need[j]["F"]
                best = max(best, M.CAP - spent[j] - M.FLOOR * (slots - 1))
            m = self.GEN == g
            out[m] = np.maximum(M.FLOOR, np.minimum(lp[m], best))
        return out

    def plan_of(self, ctx, theta, o, roster, spent, need):
        """The owner's planned completion at expected prices (new players only)."""
        M = self.M
        mode = th(theta, "expect")
        price = ctx["prices"][mode][o] if mode == "rivals" else ctx["prices"][mode]
        by_g = ctx["by_g"][mode][o] if mode == "rivals" else ctx["by_g"][mode]
        budget = M.CAP - spent
        if th(theta, "plan") == "planner":
            out = self.complete(list(roster), budget, ctx["v"][o], ctx["s"][o], price, ctx["avail"])
            proj = out[1] if out else None
        else:
            proj = self.greedy(list(roster), budget, price, need, ctx["order"][o], by_g, set())
        if proj is None:
            return None, price
        return [u for u in proj if u not in set(roster)], price

    def snake_pick(self, ctx, theta, o, roster, spent, need):
        """auction_sim.nominate: shortlist as a snake pick would, project each
        candidate with the greedy fill under the auction's scarcity (before I
        nominate again every other active team nominates once), best projected
        tie prob wins; ties to the dearer candidate."""
        M = self.M; D = self.D
        mode = th(theta, "expect")
        price = ctx["prices"][mode][o] if mode == "rivals" else ctx["prices"][mode]
        by_g = ctx["by_g"][mode][o] if mode == "rivals" else ctx["by_g"][mode]
        budget = M.CAP - spent
        order = ctx["order"][o]
        by_price = sorted(order, key=lambda u: (-price[u], u))
        others = sum(1 for k in range(self.n) if k != o and ctx["need"][k]["M"] + ctx["need"][k]["F"] > 0)
        gaps = [(j + 1) * others for j in range(D.ROUNDS)]
        cands = set(); cost = {}
        for g in ("M", "F"):
            if need[g] <= 0:
                continue
            need_after = dict(need); need_after[g] -= 1
            pool_g = []
            for u in order:
                if self.GEN[u] != g:
                    continue
                top = budget - self.completion_cost(price, by_g, set(roster) | {u}, need_after)
                if top + 1e-6 >= M.FLOOR:
                    pool_g.append(u); cost[u] = min(price[u], top)
            cands.update(pool_g[:D.CAND_TOP])
            cands.update(sorted(pool_g, key=lambda u: (-ctx["s"][o][u], u))[:D.CAND_SINGLES])
            cands.update(sorted(pool_g, key=lambda u: (price[u], u))[:D.CAND_CHEAP])
        best = None
        for x in sorted(cands):
            need_after = dict(need); need_after[self.GEN[x]] -= 1
            proj = self.greedy(list(roster) + [x], budget - cost[x], price, need_after, order, by_g, set(),
                               gaps=gaps, by_price=[u for u in by_price if u != x])
            if proj is None:
                continue
            key = (round(self.score(ctx["eng"][o], proj), 6), price[x], x)
            if best is None or key > best[0]:
                best = (key, x)
        return best[1] if best else None

    def nominate(self, ctx, theta, o, roster, spent, need):
        M = self.M
        if th(theta, "nom") == "snake":
            x = self.snake_pick(ctx, theta, o, roster, spent, need)
            if x is not None:
                return x
        plan, price = self.plan_of(ctx, theta, o, roster, spent, need)
        avail_idx = np.flatnonzero(ctx["avail"])
        anyone = {g: any(nd[g] > 0 for nd in ctx["need"]) for g in ("M", "F")}
        pol = th(theta, "nom")
        if plan:
            if pol == "want":
                return max(plan, key=lambda u: (price[u], -self.rank[u], u))
            if pol == "cheap":
                return min(plan, key=lambda u: (price[u], self.rank[u], u))
        if pol in ("drain", "dear") or plan:
            pool = [u for u in avail_idx if anyone[self.GEN[u]] and (pol != "drain" or u not in set(plan or []))]
            if pool:
                return max(pool, key=lambda u: (price[u], -self.rank[u], u))
        # nothing planned/affordable: the cheapest available player of a gender I still need
        pool = [u for u in avail_idx if need[self.GEN[u]] > 0]
        return min(pool, key=lambda u: (price[u], u)) if pool else None

    def run(self, thetas, seed, learned=None):
        M = self.M
        n = self.n
        rng = random.Random(seed)
        avail = np.ones(self.N, dtype=bool)
        rosters = [[] for _ in range(n)]
        spent = [0.0] * n
        need = [{"M": 3, "F": 3} for _ in range(n)]
        v = []; s = []; eng = []
        distinct = self.noise > 0
        for _ in range(n):
            vv, ss = self.beliefs(rng)
            v.append(vv); s.append(ss)
            if distinct:
                dbl = dict(M.D.DOUBLES); sg = dict(M.D.SINGLES)
                for i in range(self.N):
                    dbl[self.pids[i]] = dict(dbl[self.pids[i]], v=float(vv[i])); sg[self.pids[i]] = float(ss[i])
                eng.append(M.FT.FastTie(dbl, sg, self.E.gamma))
            else:
                eng.append(self.E)
        sales = []
        unsold = 0
        turn = 0
        while any(nd["M"] + nd["F"] > 0 for nd in need):
            t = turn % n; turn += 1
            if need[t]["M"] + need[t]["F"] <= 0:
                continue
            modes = {th(x, "expect") for x in thetas}
            prices = {}; by_g = {}
            for mode in modes:
                if mode == "rivals":
                    prices[mode] = [self.expected_prices(mode, avail, spent, need, o) for o in range(n)]
                    by_g[mode] = [{g: sorted((int(u) for u in np.flatnonzero(avail) if self.GEN[u] == g), key=lambda u: (prices[mode][o][u], u)) for g in ("M", "F")} for o in range(n)]
                else:
                    prices[mode] = self.expected_prices(mode, avail, spent, need, learned=learned)
                    by_g[mode] = {g: sorted((int(u) for u in np.flatnonzero(avail) if self.GEN[u] == g), key=lambda u: (prices[mode][u], u)) for g in ("M", "F")}
            order = [sorted((int(u) for u in np.flatnonzero(avail)), key=lambda u: (-v[o][u], u)) for o in range(n)] if distinct \
                else [sorted((int(u) for u in np.flatnonzero(avail)), key=lambda u: (-v[0][u], u))] * n
            ctx = dict(avail=avail, prices=prices, by_g=by_g, order=order, v=v, s=s, eng=eng, memo={}, distinct=distinct, need=need)
            x = self.nominate(ctx, thetas[t], t, rosters[t], spent[t], need[t])
            if x is None:
                raise RuntimeError("nothing to nominate")
            gx = self.GEN[x]
            bids = []
            short = {}
            for o in range(n):
                if need[o][gx] <= 0:
                    continue
                if o != t and th(thetas[o], "bid") == "shortlist":
                    key = o if distinct else 0
                    if key not in short:
                        gord = [u for u in order[o] if self.GEN[u] == gx]
                        short[key] = set(gord[:self.D.CAND_TOP]) | set(sorted(gord, key=lambda u: (-s[o][u], u))[:self.D.CAND_SINGLES])
                    mode = th(thetas[o], "expect")
                    bg = by_g[mode][o] if mode == "rivals" else by_g[mode]
                    if x not in short[key] and x not in set(bg[gx][:self.D.CAND_CHEAP]):
                        continue
                b = self.ceiling(ctx, thetas[o], o, x, rosters[o], spent[o], need[o])
                if o == t:
                    need_after = dict(need[o]); need_after[gx] -= 1
                    mode = th(thetas[o], "expect")
                    price = prices[mode][o] if mode == "rivals" else prices[mode]
                    bg = by_g[mode][o] if mode == "rivals" else by_g[mode]
                    if M.FLOOR + self.completion_cost(price, bg, set(rosters[o]) | {x}, need_after) <= M.CAP - spent[o] + 1e-6:
                        b = max(b, M.FLOOR)
                if b >= M.FLOOR:
                    bids.append((b, rng.random(), o))
            if not bids:
                avail[x] = False; unsold += 1
                sales.append((x, None, 0.0, t))
                continue
            bids.sort(key=lambda b: (-b[0], b[1]))
            if self.trace and self.name_of(x) in self.trace:
                print(f"TRACE sale {len(sales)} {self.name_of(x)} (list ${self.LP[x]/1e3:.0f}k) nominated by T{t}: " +
                      " ".join(f"T{o}:{b/1e3:.0f}k" for b, _, o in bids), flush=True)
            b1, _, w = bids[0]
            b2 = bids[1][0] if len(bids) > 1 else M.FLOOR
            paid = min(b1, max(M.FLOOR, b2 + self.INC))
            paid = min(b1, round(paid / 1000.0) * 1000.0)
            rosters[w].append(x); spent[w] += paid; need[w][gx] -= 1; avail[x] = False
            sales.append((x, w, paid, t))
        ros = [tuple(self.pids[i] for i in r) for r in rosters]
        P = [[0.5 if i == j else self.E.tie(ros[i], ros[j]) for j in range(n)] for i in range(n)]
        pay = [sum(P[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
        prices = self.floor_price.copy()
        for x, w, paid, t in sales:
            prices[x] = paid if w is not None else M.FLOOR
        return dict(rosters=[tuple(r) for r in rosters], pay=pay, spent=spent, sales=sales, unsold=unsold, P=P, prices=tuple(prices))

    def describe(self, res):
        M = self.M
        rows = []
        for i, r in enumerate(res["rosters"]):
            paid = {x: p for x, w, p, t in res["sales"] if w == i}
            names = ", ".join(f"{self.name_of(u)} ${paid.get(u, 0)/1e3:.0f}k" for u in sorted(r, key=lambda u: -paid.get(u, 0)))
            rows.append(f"T{i} win {100*res['pay'][i]:.1f}% spent ${res['spent'][i]/1e3:.0f}k: {names}")
        sold = [(x, p) for x, w, p, t in res["sales"] if w is not None]
        line = f"{len(sold)} sold, {res['unsold']} unsold"
        return line, rows

    def room_stats(self, results, rng):
        """Prices vs list by rank bucket, Waters, spread, titles -- over a set of all-clone runs."""
        M = self.M; D = self.D
        buckets = {"#1-5": (1, 5), "#6-15": (6, 15), "#16-30": (16, 30), "#31-60": (31, 60)}
        ratio = {b: [] for b in buckets}
        wp, wwin, wtitle, spread, best, second, n10, unspent, floor_n = [], [], [], [], [], [], [], [], []
        for res in results:
            paid = {x: p for x, w, p, t in res["sales"] if w is not None}
            for b, (lo, hi) in buckets.items():
                for x, p in paid.items():
                    if lo <= self.rank[x] <= hi:
                        ratio[b].append(p / self.LP[x])
            team_of = {x: w for x, w, p, t in res["sales"] if w is not None}
            ros = [tuple(self.pids[i] for i in r) for r in res["rosters"]]
            exp, _, ttl = D.season(ros, A.seasons, rng)
            if self.waters in team_of:
                wp.append(paid[self.waters]); wwin.append(exp[team_of[self.waters]]); wtitle.append(ttl[team_of[self.waters]])
            e = sorted(exp, reverse=True)
            spread.append(100 * statistics.pstdev(exp)); best.append(e[0]); second.append(e[1])
            n10.append(sum(1 for t in ttl if t >= 0.10))
            unspent.append(sum(M.CAP - s for s in res["spent"]))
            floor_n.append(sum(1 for x, p in paid.items() if self.rank[x] <= 60 and p <= M.FLOOR + 1e-6))
        m = lambda xs: statistics.mean(xs) if xs else float("nan")  # noqa: E731
        return dict(ratio={b: m(r) for b, r in ratio.items()}, waters_price=m(wp), waters_win=m(wwin), waters_title=m(wtitle),
                    spread=m(spread), best=m(best), second=m(second), n10=m(n10), unspent=m(unspent), floor_priced=m(floor_n))


# ================================================================ self-play
WORLD = None


def _job(args):
    thetas, seed, learned = args
    return WORLD.run(list(thetas), seed, learned)


class SelfPlay:
    def __init__(self, world, jobs, tol, log, se_mult=0.0):
        self.w = world
        self.jobs = jobs
        self.tol = tol
        self.se_mult = se_mult
        self.log = log
        self.runs = {}      # (thetas tuple, seed, learned prices or None) -> result
        self.learned = {}   # (all-clone theta, seed) -> the price vector the incumbent room settles on
        self.evals = []     # records
        self.fp_iters = []  # fixed-point iteration counts (info)

    def _run_many(self, keys):
        todo = [k for k in keys if k not in self.runs]
        if not todo:
            return
        if self.jobs > 1 and len(todo) > 1:
            with mp.get_context("fork").Pool(self.jobs) as pool:
                outs = pool.map(_job, todo, chunksize=1)
        else:
            outs = [_job(k) for k in todo]
        for k, o in zip(todo, outs):
            self.runs[k] = o

    def learned_for(self, theta, sd):
        """What an owner who 'expects last time's prices' expects under incumbent theta at seed sd."""
        return self.learned.get((theta, sd))

    def settle(self, theta, seeds, max_iter=8):
        """Incumbent (all-clone) runs. If the profile expects learned prices, iterate
        prices to a fixed point (fictitious play on prices); else one run."""
        n = self.w.n
        need = [sd for sd in seeds if (theta, sd) not in self.learned]
        if not need:
            return
        if th(theta, "expect") != "learned":
            keys = [(self.profile(theta), sd, None) for sd in need]
            self._run_many(keys)
            for sd in need:
                self.learned[(theta, sd)] = self.runs[(self.profile(theta), sd, None)]["prices"]
            return
        cur = {sd: None for sd in need}
        seen = {sd: set() for sd in need}
        for it in range(max_iter):
            keys = [(self.profile(theta), sd, cur[sd]) for sd in need]
            self._run_many(keys)
            done = True
            for sd in need:
                nxt = self.runs[(self.profile(theta), sd, cur[sd])]["prices"]
                if cur[sd] is not None and (nxt == cur[sd] or nxt in seen[sd]):
                    continue
                seen[sd].add(nxt); cur[sd] = nxt; done = False
            if done:
                break
        self.fp_iters.append(it + 1)
        for sd in need:
            self.learned[(theta, sd)] = cur[sd]
            # the canonical incumbent run is the one AT the settled prices
            self._run_many([(self.profile(theta), sd, cur[sd])])

    def base_key(self, theta, sd):
        return (self.profile(theta), sd, self.learned_for(theta, sd) if th(theta, "expect") == "learned" else None)

    def dev_key(self, theta, slot, dev, sd):
        return (self.profile(theta, slot, dev), sd, self.learned_for(theta, sd))

    def batch(self, keys):
        self._run_many(keys)

    def profile(self, theta, slot=None, dev=None):
        return tuple(dev if (slot is not None and i == slot) else theta for i in range(self.w.n))

    def gain(self, theta, dev):
        """Paired: deviator at world.slot(seed) vs the same slot under the incumbent profile."""
        self.settle(theta, self.w.seeds)
        self.batch([self.dev_key(theta, self.w.slot(sd), dev, sd) for sd in self.w.seeds])
        diffs = []
        for sd in self.w.seeds:
            slot = self.w.slot(sd)
            diffs.append(self.runs[self.dev_key(theta, slot, dev, sd)]["pay"][slot] - self.runs[self.base_key(theta, sd)]["pay"][slot])
        mean = statistics.mean(diffs)
        se = statistics.pstdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        return mean, se, diffs

    def ascend(self, theta, rounds):
        hist = [theta]
        table = None
        for r in range(rounds):
            # every single-dial deviation, all runs batched together
            devs = [(f, v) for f in self.w.dials for v in DIALS[f] if v != th(theta, f)]
            t0 = time.time()
            self.settle(theta, self.w.seeds)
            keys = []
            for f, v in devs:
                for sd in self.w.seeds:
                    keys.append(self.dev_key(theta, self.w.slot(sd), with_dial(theta, f, v), sd))
            self.batch(keys)
            table = []
            for f, v in devs:
                mean, se, diffs = self.gain(theta, with_dial(theta, f, v))
                table.append((mean, se, f, v))
                self.evals.append(dict(round=r, theta=theta, dial=f, value=v, gain=mean, se=se))
            table.sort(key=lambda t: -t[0])
            self.log(f"round {r}: profile [{fmt_theta(theta)}]  {len(keys)} runs, {time.time() - t0:.0f}s")
            for mean, se, f, v in table[:8]:
                self.log(f"    {f}={v:<8} gain {100*mean:+.2f}pp (se {100*se:.2f})")
            best = table[0]
            if best[0] > self.tol and best[0] > self.se_mult * best[1]:
                theta = with_dial(theta, best[2], best[3])
                if theta in hist:
                    self.log(f"  -> {best[2]}={best[3]} adopted; profile seen before: CYCLE, stopping")
                    hist.append(theta)
                    return theta, hist, table, "cycle"
                self.log(f"  -> adopt {best[2]}={best[3]}")
                hist.append(theta)
            else:
                self.log("  -> no improving single-dial deviation: symmetric equilibrium in the dial space")
                return theta, hist, table, "converged"
        return theta, hist, table, "max rounds"


# ===================================================================== main
def log_to(path):
    fh = open(path, "a")

    def log(msg):
        print(msg, flush=True)
        fh.write(msg + "\n"); fh.flush()
    return log


def toy_cell(teams, board, unit, order, exact, starts, rounds, log, known=None):
    global WORLD
    w = ToyWorld(teams, board, unit, order)
    WORLD = w
    T = w.T
    log(f"\n=== toy cell: {w.label} ===")
    log("  sale order: " + " ".join(a + b for a, b in T.ITEMS))
    out = dict(cell=w.label, teams=teams, board=board, unit=unit, exact={}, searches=[])
    targets = {}
    if known:
        targets["known secondprice"] = known
        log(f"  exact (known, from a previous solve): {[T.fmt_roster(h) for h in known]}")
    for mech in ([] if not exact else ["secondprice", "english"] if exact == "both" else [exact]):
        res = w.exact(mech)
        line, rows = w.describe(res)
        log(f"  exact {mech}: {res['states']:,} states, {res['secs']:.0f}s")
        log("    " + line)
        for rw in rows:
            log("    " + rw)
        out["exact"][mech] = dict(rosters=res["rosters"], pay=res["pay"], sales=[(a + b, k, who, p) for (a, b), k, who, p in res["sales"]])
        targets[mech] = res["rosters"]
    for start in starts:
        theta0 = theta_of(start)
        sp = SelfPlay(w, 1, 1e-9, log)
        log(f"  self-play from {start}: [{fmt_theta(theta0)}]")
        theta, hist, table, status = sp.ascend(theta0, rounds)
        sp.settle(theta, w.seeds)
        res = sp.runs[sp.base_key(theta, w.seeds[0])]
        line, rows = w.describe(res)
        log(f"  result ({status}): [{fmt_theta(theta)}]")
        log("    " + line)
        for rw in rows:
            log("    " + rw)
        verdict = {}
        for name, target in targets.items():
            same = sorted(res["rosters"]) == sorted(target)
            verdict[name] = same
            log(f"    rosters vs exact {name}: {'MATCH' if same else 'DIFFER'}")
        # the truthful benchmark for reference: same rosters as toy_auction's bench?
        out["searches"].append(dict(start=start, theta=theta, status=status, hist=hist, rosters=res["rosters"], pay=res["pay"],
                                    sales=[(a + b, k, who, p) for (a, b), k, who, p, c in res["path"]], verdict=verdict,
                                    exploit=[(m, se, f, v) for m, se, f, v in table[:5]] if table else []))
    return out


def main():
    global WORLD
    CACHE.mkdir(exist_ok=True)
    tag = f"_{A.tag}" if A.tag else ""
    if A.world == "toy":
        log = log_to(CACHE / f"strategic_toy{tag}.log")
        starts = ["truthful", "planner"] if A.start == "truthful" else [A.start]
        cells = []
        if A.validate:
            ladder = [(2, "1,1,4,4", 50_000, None), (2, "2,2,4,4", 50_000, None), (3, "1,1,3,3", 100_000, None),
                      (3, "1,1,2,2", 50_000, None),
                      (4, "1,1,3,3", 100_000, ((0, 0, 0, 2), (1, 0, 0, 1), (0, 0, 1, 0), (0, 3, 0, 0)))]
            for teams, board, unit, known in ladder:
                # a cell with a known exact solution skips the (expensive) solve
                cells.append(toy_cell(teams, board, unit, A.order, None if known else A.exact, starts, A.rounds, log, known=known))
        else:
            cells.append(toy_cell(A.teams, A.board, A.unit, A.order, A.exact, starts, A.rounds, log))
        (CACHE / f"strategic_toy{tag}.json").write_text(json.dumps(cells, indent=1, default=str))
        return
    # -- real
    log = log_to(CACHE / f"strategic_real{tag}.log")
    w = RealWorld(A.K, A.plan_top, A.noise)
    WORLD = w
    log(f"=== real world: {w.label}; seeds {A.seeds}, jobs {A.jobs} ===")
    theta0 = theta_of(A.start)
    if A.fix:
        fixed = [f for f in A.fix.split(",") if f]
        w.dials = [f for f in w.dials if f not in fixed]
        log(f"fixed by rule: {', '.join(f'{f}={th(theta0, f)}' for f in fixed)}; searched dials: {w.dials}")
    if A.room:
        sp = SelfPlay(w, A.jobs, 0.0, log, A.se_mult)
        sp.settle(theta0, w.seeds)
        results = [sp.runs[sp.base_key(theta0, sd)] for sd in w.seeds]
        stats = w.room_stats(results, random.Random(0))
        log(f"room on [{fmt_theta(theta0)}] (all clones, {len(results)} seeds): " + json.dumps(stats))
        line, rows = w.describe(results[0])
        log("  seed 0: " + line)
        for rw in rows:
            log("  " + rw)
        (CACHE / f"strategic_room{tag}.json").write_text(json.dumps(dict(label=w.label, theta=theta0, stats=stats, seeds=A.seeds), indent=1, default=str))
        return
    if A.time:
        t0 = time.time()
        res = w.run([theta0] * w.n, 0)
        log(f"one auction on [{fmt_theta(theta0)}]: {time.time() - t0:.0f}s")
        line, rows = w.describe(res)
        log("  " + line)
        for rw in rows:
            log("  " + rw)
        return
    tol = A.tol if A.tol is not None else 0.0025
    sp = SelfPlay(w, A.jobs, tol, log, A.se_mult)
    theta, hist, table, status = sp.ascend(theta0, A.rounds)
    log(f"\nRESULT ({status}): [{fmt_theta(theta)}]  path: " + " -> ".join(fmt_theta(t) for t in hist))
    rng = random.Random(0)
    sp.settle(theta, w.seeds)
    results = [sp.runs[sp.base_key(theta, sd)] for sd in w.seeds]
    stats = w.room_stats(results, rng)
    log("room at the equilibrium profile (all clones, %d seeds): " % len(results) + json.dumps(stats))
    line, rows = w.describe(results[0])
    log("  seed 0: " + line)
    for rw in rows:
        log("  " + rw)
    # the starting profile's room, for contrast
    sp.settle(theta0, w.seeds)
    stats0 = w.room_stats([sp.runs[sp.base_key(theta0, sd)] for sd in w.seeds], rng)
    log("room at the starting profile: " + json.dumps(stats0))
    (CACHE / f"strategic_real{tag}.json").write_text(json.dumps(dict(
        label=w.label, start=theta0, theta=theta, status=status, hist=hist, evals=sp.evals, table=table,
        stats=stats, stats0=stats0, seeds=A.seeds), indent=1, default=str))


if __name__ == "__main__":
    main()
