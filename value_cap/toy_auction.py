"""value_cap/toy_auction.py -- an EXACTLY SOLVED toy auction: typed players
(star / good / floor per gender), a few strategic teams, one deterministic
Waters team, money on a grid, fixed sale order, an English auction played
as an alternating-raise game, solved by backward induction over the whole
auction (subgame-perfect equilibrium).

    python value_cap/toy_auction.py                        # 3 teams, board 2,2,6,6, stars first
    python value_cap/toy_auction.py --teams 4 --board 2,2,8,8
    python value_cap/toy_auction.py --order goods_first --unit 25000
    python value_cap/toy_auction.py --no-waters             # no fixed favourite in the league

Why: the full 20-team auction cannot be solved (auction.md, "The market
limit"; the strategy-search plan). This toy CAN be, and it is the check
for any heuristic strategy search: run the search on the same toy and see
whether it finds the exact equilibrium. It also answers, exactly and for a
small league, whether planners chase a fixed 66% team with two stars.

The game.
  - Types: per gender, star = the top five pool players by value (women:
    #2-6, Waters excluded), good = #6-30, floor = the $30k fill-ins the
    market-limit rosters use. Each type carries the tier's mean doubles
    value, mean singles value, mean uncertainty and mean list price. The
    tie model is the production FastTie on these representative players
    (same weakest-link pairing, DreamBreaker foursome, race DP).
  - Board: --board SM,SF,GM,GF counts of star men, star women, good men,
    good women. Floor players are an unlimited outside option at the
    floor price, never auctioned; every team ends with 3M+3W.
  - Waters team: fixed, Waters + five floor players, not in the auction
    (she sells at the first-buy maximum at sale 1 in every simulation we
    have, so nothing is lost by treating her as environment).
  - Money: --unit dollars per grid step (budget $1M / unit; floor = 1 unit);
    a bid is feasible only if the team can still fill its remaining slots
    with floor players.
  - Sale order: fixed and swept (--order stars_first | goods_first |
    alternate). Nomination is not a choice here (that is one strategic
    dimension deliberately dropped to keep the state small).
  - One sale: opens at the floor price with no high bidder; teams move in
    rotation (opener = sale index mod teams); a mover either passes or
    bids (takes the opening price, or raises the standing price by one
    unit); the standing high bidder wins when every other team has
    passed in a row; the item is unsold if everyone passes at the open.
    A team raises only when strictly better off (--tie aggressive: when
    weakly better off). Perfect information, finite, so the equilibrium
    outcome is unique given the tie rule.
  - Payoff: in-league win% = mean tie probability against every other
    team (the other strategic teams and the Waters team). Unspent money
    is worth nothing.

  - --mech secondprice replaces the alternating-raise stage by a sealed
    second-price stage in which every eligible team bids its indifference
    price against what happens if it abstains (computed with equilibrium
    continuation values, by induction on the bidder set); highest ceiling
    wins at the second-highest + 1. Complete-information English auctions
    have a continuum of equilibria; the alternating-raise strict rule
    selects the LOWEST price (a loser never raises just to run the winner
    up unless that is strictly better for it later), the second-price
    convention selects the truthful one. Report both; a room sits between.

Benchmarks on the same toy (--bench): truthful planners -- each bidder's
ceiling is the price at which "this player + best completion at expected
type prices" equals "best completion without", objective = tie vs a
reference roster (two goods and a floor per gender); the winner pays the
second-highest ceiling + 1 unit. Expectations = list-price types
(--expect list) or the equilibrium's own realised prices (--expect eq).
"""
from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
import threading
import functools
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--teams", type=int, default=3, help="strategic teams (plus the Waters team)")
ap.add_argument("--board", default="2,2,6,6", help="SM,SF,GM,GF counts on the board")
ap.add_argument("--order", default="stars_first", choices=["stars_first", "goods_first", "alternate"])
ap.add_argument("--unit", type=float, default=50_000, help="money grid step in dollars")
ap.add_argument("--mech", default="english", choices=["english", "secondprice"], help="stage mechanism (see docstring)")
ap.add_argument("--tie", default="strict", choices=["strict", "aggressive"], help="raise when strictly / weakly better off")
ap.add_argument("--no-waters", action="store_true", help="drop the fixed Waters team")
ap.add_argument("--bench", action="store_true", help="also run the truthful-planner benchmarks")
ap.add_argument("--expect", default="both", choices=["list", "eq", "both"])
ap.add_argument("--star-top", type=int, default=5, help="players per gender in the star tier")
ap.add_argument("--good-to", type=int, default=30, help="the good tier runs from star-top+1 to this rank")
ap.add_argument("--quiet", action="store_true")
A = ap.parse_args()
print = functools.partial(print, flush=True)  # noqa: A001  (stdout may be a pipe)

_argv = sys.argv
sys.argv = [sys.argv[0]]
import market_eq as M  # noqa: E402
from fast_tie import FastTie  # noqa: E402
sys.argv = _argv

E = M.E
NAME = M.NAME
FLOOR_D = M.FLOOR
CAP_D = M.CAP
UNIT = A.unit
BUDGET = int(round(CAP_D / UNIT))
FLOORC = max(1, int(round(FLOOR_D / UNIT)))
EPS = 1e-12

# ------------------------------------------------------------- the types
WATERS = M.pid_named("Anna Leigh Waters")
FLOOR_NAMES = {"M": ["Jonathan Truong", "Gabriel Joseph", "Martin Emmrich"],
               "F": ["Lina Padegimaite", "Genie Erokhina", "Alexa Schull"]}


def tier_stats(us):
    v = statistics.mean(E.v[u] for u in us)
    s = statistics.mean(E.s[u] for u in us)
    sd = math.sqrt(statistics.mean(E.u2[u] for u in us))
    lp = statistics.mean(M.LP[u] for u in us)
    return {"v": v, "s": s, "sd": sd, "list": lp, "n": len(us), "names": [NAME[u] for u in us]}


TYPES = {}
for g in "MF":
    pool = sorted((u for u in M.POOLSET if M.G[u] == g and u != WATERS), key=lambda u: -E.v[u])
    TYPES["S" + g] = tier_stats(pool[:A.star_top])
    TYPES["G" + g] = tier_stats(pool[A.star_top:A.good_to])
    TYPES["F" + g] = tier_stats([M.pid_named(n) for n in FLOOR_NAMES[g]])
TYPES["W"] = {"v": E.v[WATERS], "s": E.s[WATERS], "sd": math.sqrt(E.u2[WATERS]), "list": M.LP[WATERS], "n": 1, "names": ["Anna Leigh Waters"]}

_dbl = {t: {"v": d["v"], "sd": d["sd"], "gender": ("F" if t == "W" else t[1])} for t, d in TYPES.items()}
_sgl = {t: d["s"] for t, d in TYPES.items()}
TE = FastTie(_dbl, _sgl, E.gamma)  # typed engine on the production tie model

LISTU = {t: max(1, int(round(d["list"] / UNIT))) for t, d in TYPES.items()}  # list price per type in units

# ------------------------------------------------------------- board & rosters
BSM, BSF, BGM, BGF = (int(x) for x in A.board.split(","))
n = A.teams


def interleave(tier, nf, nm):
    """F, M, F, M, ... then whichever gender has more."""
    out = [x for _ in range(min(nf, nm)) for x in ((tier, "F"), (tier, "M"))]
    out += [(tier, "F")] * (nf - nm) if nf > nm else [(tier, "M")] * (nm - nf)
    return out


stars, goods = interleave("S", BSF, BSM), interleave("G", BGF, BGM)
if A.order == "stars_first":
    ITEMS = stars + goods
elif A.order == "goods_first":
    ITEMS = goods + stars
else:  # alternate: one star, then a share of the goods, then the next star
    per = max(1, len(goods) // max(1, len(stars)))
    ITEMS = []
    gi = 0
    for st in stars:
        ITEMS.append(st)
        ITEMS += goods[gi:gi + per]
        gi += per
    ITEMS += goods[gi:]
assert len(ITEMS) == BSM + BSF + BGM + BGF
NS = len(ITEMS)

# holdings per team: (sM, gM, sF, gF)
HI = {("S", "M"): 0, ("G", "M"): 1, ("S", "F"): 2, ("G", "F"): 3}


def roster_ids(h):
    sm, gm, sf, gf = h
    return tuple(["SM"] * sm + ["GM"] * gm + ["FM"] * (3 - sm - gm) + ["SF"] * sf + ["GF"] * gf + ["FF"] * (3 - sf - gf))


WATERS_ROSTER = ("W", "FF", "FF", "FM", "FM", "FM")
REF_ROSTER = ("GM", "GM", "FM", "GF", "GF", "FF")

_tie_cache = {}


def tie(ra, rb):
    k = (ra, rb)
    r = _tie_cache.get(k)
    if r is None:
        r = TE.tie(ra, rb)
        _tie_cache[k] = r
    return r


_pay_cache = {}


def payoff(holds):
    """In-league win% per strategic team (mean tie vs every other team incl. Waters')."""
    r = _pay_cache.get(holds)
    if r is not None:
        return r
    ros = [roster_ids(h) for h in holds]
    opp = [] if A.no_waters else [WATERS_ROSTER]
    out = []
    for i in range(n):
        others = [ros[j] for j in range(n) if j != i] + opp
        out.append(sum(tie(ros[i], o) for o in others) / len(others))
    r = tuple(out)
    _pay_cache[holds] = r
    return r


def waters_win(holds):
    ros = [roster_ids(h) for h in holds]
    return sum(tie(WATERS_ROSTER, o) for o in ros) / len(ros)


def held_total(h):
    return sum(h)


def gender_count(h, g):
    return (h[0] + h[1]) if g == "M" else (h[2] + h[3])


def can_bid(h, budget, g, bid):
    if gender_count(h, g) >= 3:
        return False
    remaining_after = 6 - held_total(h) - 1
    return budget - bid >= FLOORC * remaining_after


def add(h, item):
    lst = list(h)
    lst[HI[item]] += 1
    return tuple(lst)


# ------------------------------------------------------------- the exact solve
sys.setrecursionlimit(1_000_000)
VMEMO = {}
STATS = {"sale_states": 0}
T0 = time.time()


def V(k, holds, budgets):
    """Equilibrium payoff vector from the start of sale k."""
    if k == NS:
        return payoff(holds)
    key = (k, holds, budgets)
    r = VMEMO.get(key)
    if r is not None:
        return r
    r = SOLVE(k, holds, budgets)[0]
    VMEMO[key] = r
    if len(VMEMO) % 200_000 == 0:
        print(f"  ... {len(VMEMO):,} between-sale states, {STATS['sale_states']:,} in-sale, {time.time() - T0:.0f}s", file=sys.stderr, flush=True)
    return r


def solve_sale(k, holds, budgets, trace=False):
    """Alternating-raise English auction for item k, backward induction. Returns (value vector, outcome)."""
    item = ITEMS[k]
    g = item[1]
    memo = {}
    opener = k % n

    def f(p, h, j, passes):
        key = (p, h, j, passes)
        r = memo.get(key)
        if r is not None:
            return r
        if h is not None and passes >= n - 1:
            nb = list(budgets); nb[h] -= p
            nh = list(holds); nh[h] = add(holds[h], item)
            r = (V(k + 1, tuple(nh), tuple(nb)), ("win", h, p))
        elif h is None and passes >= n:
            r = (V(k + 1, holds, budgets), ("unsold", None, 0))
        else:
            nxt = (j + 1) % n
            if j == h:
                r = f(p, h, nxt, passes)
            else:
                v_pass, o_pass = f(p, h, nxt, passes + 1)
                bid = p if h is None else p + 1
                if can_bid(holds[j], budgets[j], g, bid):
                    v_bid, o_bid = f(bid, j, nxt, 0)
                    better = (v_bid[j] > v_pass[j] + EPS) if A.tie == "strict" else (v_bid[j] >= v_pass[j] - EPS)
                    r = (v_bid, o_bid) if better else (v_pass, o_pass)
                else:
                    r = (v_pass, o_pass)
        memo[key] = r
        return r

    out = f(FLOORC, None, opener, 0)
    STATS["sale_states"] += len(memo)
    return out


def solve_sale_sp(k, holds, budgets):
    """Second-price stage: every eligible team bids its indifference price against what happens if it
    abstains (that outcome is the same rule on the other bidders, by induction on the bidder set); the
    highest ceiling wins and pays the second-highest + 1 unit (at least the floor, at most its own
    ceiling); ties by rotation from the opener. Returns (value vector, ("win"|"unsold", who, price, ceilings))."""
    item = ITEMS[k]
    g = item[1]
    elig = frozenset(i for i in range(n) if can_bid(holds[i], budgets[i], g, FLOORC))
    memo = {}

    def outcome(S):
        r = memo.get(S)
        if r is not None:
            return r
        if not S:
            r = (V(k + 1, holds, budgets), ("unsold", None, 0, {}))
        else:
            ceilings = {}
            for i in S:
                base = outcome(S - {i})[0]
                hi = budgets[i] - FLOORC * (6 - held_total(holds[i]) - 1)
                c = 0
                nh = list(holds); nh[i] = add(holds[i], item); nh = tuple(nh)
                for b in range(FLOORC, hi + 1):
                    nb = list(budgets); nb[i] -= b
                    if V(k + 1, nh, tuple(nb))[i] >= base[i] - EPS:
                        c = b
                ceilings[i] = c
            order = sorted(S, key=lambda i: (-ceilings[i], (i - k) % n))
            w = order[0]
            if ceilings[w] < FLOORC:
                r = (V(k + 1, holds, budgets), ("unsold", None, 0, ceilings))
            else:
                second = ceilings[order[1]] if len(order) > 1 else 0
                p = max(FLOORC, min(ceilings[w], second + 1))
                nb = list(budgets); nb[w] -= p
                nh = list(holds); nh[w] = add(holds[w], item)
                r = (V(k + 1, tuple(nh), tuple(nb)), ("win", w, p, ceilings))
        memo[S] = r
        return r

    out = outcome(elig)
    STATS["sale_states"] += len(memo)
    return out


SOLVE = solve_sale_sp if A.mech == "secondprice" else solve_sale


def play():
    """Walk the equilibrium path from the root."""
    holds = tuple((0, 0, 0, 0) for _ in range(n))
    budgets = tuple(BUDGET for _ in range(n))
    path = []
    for k in range(NS):
        val, outcome = SOLVE(k, holds, budgets)
        kind, who, p = outcome[:3]
        path.append((k, ITEMS[k], kind, who, p) + ((outcome[3],) if len(outcome) > 3 else ()))
        if kind == "win":
            nb = list(budgets); nb[who] -= p
            nh = list(holds); nh[who] = add(holds[who], ITEMS[k])
            holds, budgets = tuple(nh), tuple(nb)
    return path, holds, budgets


# ------------------------------------------------------------- truthful-planner benchmark
def best_completion(h, budget, remaining, prices, objective):
    """Max objective over feasible completions of h from `remaining` type counts at `prices` (units)."""
    sm, gm, sf, gf = h
    best = -1.0
    for asm in range(0, min(remaining[0], 3 - sm - gm) + 1):
        for agm in range(0, min(remaining[1], 3 - sm - gm - asm) + 1):
            for asf in range(0, min(remaining[2], 3 - sf - gf) + 1):
                for agf in range(0, min(remaining[3], 3 - sf - gf - asf) + 1):
                    nh = (sm + asm, gm + agm, sf + asf, gf + agf)
                    cost = asm * prices[0] + agm * prices[1] + asf * prices[2] + agf * prices[3] + FLOORC * (6 - sum(nh))
                    if cost > budget:
                        continue
                    val = objective(nh)
                    if val > best:
                        best = val
    return best


def bench(prices_by_type, label):
    """Truthful planners: ceiling = indifference price vs the reference roster; second price + 1."""
    holds = [(0, 0, 0, 0)] * n
    budgets = [BUDGET] * n
    remaining = [BSM, BGM, BSF, BGF]

    def objective(nh):
        return tie(roster_ids(nh), REF_ROSTER)

    path = []
    for k, item in enumerate(ITEMS):
        g = item[1]
        remaining[HI[item]] -= 1
        ceilings = []
        for i in range(n):
            if not can_bid(holds[i], budgets[i], g, FLOORC):
                ceilings.append(0)
                continue
            without = best_completion(holds[i], budgets[i], remaining, prices_by_type, objective)
            c = 0
            hi = budgets[i] - FLOORC * (6 - held_total(holds[i]) - 1)
            for b in range(FLOORC, hi + 1):
                withp = best_completion(add(holds[i], item), budgets[i] - b, remaining, prices_by_type, objective)
                if withp >= without - EPS:
                    c = b
                else:
                    break
            ceilings.append(c)
        order = sorted(range(n), key=lambda i: (-ceilings[i], (i - k) % n))
        w = order[0]
        if ceilings[w] < FLOORC:
            path.append((k, item, "unsold", None, 0))
            continue
        second = ceilings[order[1]] if n > 1 else 0
        p = max(FLOORC, min(ceilings[w], second + 1))
        budgets[w] -= p
        holds[w] = add(holds[w], item)
        path.append((k, item, "win", w, p))
    return path, tuple(holds), tuple(budgets)


# ------------------------------------------------------------- reporting
def fmt_roster(h):
    sm, gm, sf, gf = h
    return f"{sm}S+{gm}G+{3-sm-gm}F men / {sf}S+{gf}G+{3-sf-gf}F women"


def describe(path, holds, budgets, label):
    print(f"\n== {label} ==")
    line = []
    for row in path:
        k, item, kind, who, p = row[:5]
        tag = item[0] + item[1]
        line.append(f"{tag}->{'--' if kind != 'win' else 'T%d' % who}@{p if kind == 'win' else 0}")
    print("  sales: " + "  ".join(line))
    if len(path[0]) > 5:
        print("  ceilings: " + "  ".join(item[0] + item[1] + ":" + "/".join(str(row[5].get(i, "-")) for i in range(n)) for row in path for item in [row[1]]))
    pay = payoff(holds)
    for i in range(n):
        spent = BUDGET - budgets[i]
        stars = holds[i][0] + holds[i][2]
        print(f"  T{i}: {fmt_roster(holds[i])}  spent {spent}/{BUDGET} (${spent*UNIT/1e3:.0f}k)  win {pay[i]*100:.1f}%  stars {stars}")
    if not A.no_waters:
        print(f"  Waters team: win {waters_win(holds)*100:.1f}%")
    by_type = {}
    for row in path:
        k, item, kind, who, p = row[:5]
        if kind == "win":
            by_type.setdefault(item[0] + item[1], []).append(p)
    print("  prices (units): " + "  ".join(f"{t} {min(v)}-{max(v)} (list {LISTU[t]})" for t, v in by_type.items()))
    return by_type


def main():
    print(f"toy auction: {n} strategic teams{' + Waters team' if not A.no_waters else ''}, board SM/SF/GM/GF = {BSM}/{BSF}/{BGM}/{BGF}, "
          f"order {A.order}, unit ${UNIT/1e3:.0f}k (budget {BUDGET}, floor {FLOORC}), tie rule {A.tie}")
    for t, d in TYPES.items():
        print(f"  type {t}: v {d['v']:+.2f}  s {d['s']:+.2f}  sd {d['sd']:.2f}  list ${d['list']/1e3:.0f}k ({LISTU[t]} u)  n={d['n']}"
              + ("" if A.quiet else f"  [{', '.join(d['names'][:5])}{'...' if d['n'] > 5 else ''}]"))
    print("  sale order: " + " ".join(a + b for a, b in ITEMS))
    t0 = time.time()
    path, holds, budgets = play()
    dt = time.time() - t0
    print(f"\nexact solve: {len(VMEMO):,} between-sale states, {STATS['sale_states']:,} in-sale states, {len(_pay_cache):,} terminal rosters, {dt:.1f}s")
    eq_prices = describe(path, holds, budgets, "EQUILIBRIUM (subgame-perfect; %s)" % ("second-price stages, indifference bids" if A.mech == "secondprice" else "alternating-raise English stages, %s raises" % A.tie))
    if A.bench:
        if A.expect in ("list", "both"):
            lp = [LISTU["SM"], LISTU["GM"], LISTU["SF"], LISTU["GF"]]
            bp, bh, bb = bench(lp, "list")
            describe(bp, bh, bb, f"BENCHMARK truthful planners, expectations = list prices {lp}")
        if A.expect in ("eq", "both"):
            def mean_or(t, default):
                v = eq_prices.get(t)
                return int(round(statistics.mean(v))) if v else default
            ep = [mean_or("SM", LISTU["SM"]), mean_or("GM", LISTU["GM"]), mean_or("SF", LISTU["SF"]), mean_or("GF", LISTU["GF"])]
            bp, bh, bb = bench(ep, "eq")
            describe(bp, bh, bb, f"BENCHMARK truthful planners, expectations = equilibrium prices {ep}")


if __name__ == "__main__":
    threading.stack_size(512 * 1024 * 1024)
    th = threading.Thread(target=main)
    th.start()
    th.join()
