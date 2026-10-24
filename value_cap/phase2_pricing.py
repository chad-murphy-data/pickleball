"""value_cap/phase2_pricing.py -- Phase 2: turn player value into $1M-cap
prices, and the instruments that test whether a price list is fair.

    python value_cap/phase2_pricing.py --quick          # implied gender split + indifference pairs
    python value_cap/phase2_pricing.py                  # + must-buy sweeps for Waters/Bright/Johns (~10 min)
    python value_cap/phase2_pricing.py --must-buy "Anna Leigh Waters" --modes joint
    python value_cap/phase2_pricing.py --value total    # HANDOFF-era basis (Phase 1 V_total pool)

Value basis (--value, see pool.py): default "phi" = the context-averaged,
self-consistent pool from shapley_value.py; "total" = Phase 1's
replacement-context V_total, top 60 per gender, kept so the 2026-09-04
morning numbers (indifference pairs A/B) stay reproducible. "shapley" is
accepted as an alias for "phi".

Price formula, one league pool (mode="joint"):

    price_i = floor + (v_i^alpha / sum_j v_j^alpha) * (20 * cap - N * floor)

with j running over all 120 priced players; players outside the pool cost
the floor. mode="split" is HANDOFF.md's older convention -- each gender's
60 share $10M -- kept for comparison. Value is defined per player relative
to the SAME-GENDER replacement (doubles #60), so it is cross-gender
offset-free and a joint pool makes the gender split an OUTPUT of the data
instead of a 50/50 input.

Instruments:
  - prices(pool, alpha, mode)          joint or split price list
  - find_crossover(anchor, block_fn)   sweep a challenger spec over ranks
                                       until P(anchor wins) crosses 0.5
  - alpha_reconciling(a, b, mode)      bisection: alpha at which two rosters
                                       cost the same
  - best_roster(prices, opp, ...)      real (tie_win_prob) roster optimizer
                                       under the cap, with must-include /
                                       exclude constraints
  - must_buy(pid, alpha, mode)         P(best roster WITH pid beats best
                                       roster WITHOUT pid), each side
                                       best-responding to the other. 0.5 =
                                       fairly priced, >0.5 = bargain,
                                       None = pid cannot be rostered at all.

Conventions carried over from HANDOFF.md so results are comparable:
ranks are DOUBLES ranks (data/v2_players.csv order within gender),
replacement = doubles #60, floor = $30,000 (cosmetic, see the write-up
§6), 20 teams, $1M cap.
"""
from __future__ import annotations

import sys
from itertools import combinations

from phase1_value_model import load_doubles, load_singles, tie_win_prob
from pool import load_pool

FLOOR = 30_000
if "--floor" in sys.argv:
    FLOOR = int(sys.argv[sys.argv.index("--floor") + 1])
N_TEAMS = 20
TEAM_CAP = 1_000_000
LEAGUE_TOTAL = N_TEAMS * TEAM_CAP
REPL_RANK = 60
BUDGET_STEP = 10_000
K_CAND = 25          # candidate triples per gender per budget split (by proxy V)

DOUBLES = load_doubles()
SINGLES = load_singles()
VALUE_SOURCE = "phi"
if "--value" in sys.argv:
    VALUE_SOURCE = sys.argv[sys.argv.index("--value") + 1]
    if VALUE_SOURCE == "shapley":
        VALUE_SOURCE = "phi"
POOL = load_pool(VALUE_SOURCE)           # gender -> [(pid, name, value)], 60 each
V_OF = {pid: v for g in POOL for pid, _, v in POOL[g]}
NAME = {u: DOUBLES[u]["name"] for u in DOUBLES}
RANKED = {g: sorted((u for u in DOUBLES if DOUBLES[u]["gender"] == g),
                    key=lambda u: -DOUBLES[u]["v"]) for g in ("M", "F")}


def rank(g, r):
    """player_id at doubles rank r (1-based) within gender g."""
    return RANKED[g][r - 1]


def roster_from_ranks(men_ranks, women_ranks):
    return tuple(rank("M", r) for r in men_ranks) + tuple(rank("F", r) for r in women_ranks)


def label(roster):
    return " / ".join(f"{NAME[u]}(#{RANKED[DOUBLES[u]['gender']].index(u)+1}{DOUBLES[u]['gender']})"
                      for u in roster)


# ---------------------------------------------------------------- pricing
def prices(pool, alpha, mode="joint", floor=FLOOR):
    """pid -> price. mode='joint': one pool of 120 sharing $20M.
    mode='split': each gender's 60 share $10M (HANDOFF.md convention)."""
    out = {}
    if mode == "joint":
        groups = [pool["M"] + pool["F"]]
        budgets = [LEAGUE_TOTAL]
    else:
        groups = [pool["M"], pool["F"]]
        budgets = [LEAGUE_TOTAL // 2, LEAGUE_TOTAL // 2]
    for grp, budget in zip(groups, budgets):
        w = {pid: max(v, 0.0) ** alpha for pid, _, v in grp}
        tot = sum(w.values())
        rem = budget - len(grp) * floor
        for pid in w:
            out[pid] = floor + w[pid] / tot * rem
    return out


def prices_tagged(pool, alpha, tag_pid, mode="joint", floor=FLOOR, cap=TEAM_CAP):
    """prices() with one player FRANCHISE-TAGGED: their price is set directly
    to the most a team can pay and still field a legal roster (cap minus the
    cheapest 5-player completion), instead of coming off the curve. The gap
    between their curve price and the tag is redistributed over the rest of
    the pool in proportion to their value-weighted share, so the pool still
    sums to 20 x cap. Motivation (2026-09-05): a player worth more than a
    team's share of the cap cannot be priced fairly by any proportional
    rule, and bending alpha to fit her discounts every other star and
    surcharges every role player (~10-15% either way at alpha 0.845 vs 1)."""
    price = prices(pool, alpha, mode, floor)
    g_tag = DOUBLES[tag_pid]["gender"]
    others = [u for u in price if u != tag_pid]
    weights = {u: price[u] - floor for u in others}
    wtot = sum(weights.values())
    for _ in range(5):                    # fixed point: tag depends on the redistributed prices
        same = sorted(price[u] for u in price if u != tag_pid and DOUBLES[u]["gender"] == g_tag)[:2]
        other = sorted(price[u] for u in price if DOUBLES[u]["gender"] != g_tag)[:3]
        tag = cap - sum(same) - sum(other)
        surplus = prices(pool, alpha, mode, floor)[tag_pid] - tag
        new = {u: floor + weights[u] + surplus * weights[u] / wtot for u in others}
        new[tag_pid] = tag
        if all(abs(new[u] - price[u]) < 1 for u in price):
            price = new
            break
        price = new
    return price


def cost(roster, price):
    return sum(price.get(u, FLOOR) for u in roster)


def women_share(price):
    return sum(price[u] for u, _, _ in POOL["F"]) / LEAGUE_TOTAL


# ------------------------------------------------------- crossover search
def win(a, b):
    return tie_win_prob(a, b, DOUBLES, SINGLES)


def find_crossover(anchor, block_fn, lo=2, hi=REPL_RANK - 2):
    """Sweep k = lo..hi, challenger = block_fn(k); return the k whose
    P(anchor beats challenger) is closest to 0.5, plus the bracket."""
    rows = []
    for k in range(lo, hi + 1):
        rows.append((k, win(anchor, block_fn(k))))
    best = min(rows, key=lambda t: abs(t[1] - 0.5))
    return best[0], best[1], rows


def alpha_reconciling(a, b, mode, lo=0.2, hi=3.0):
    """alpha at which cost(a) == cost(b). Bisection on the sign of the
    difference; returns None if no sign change on [lo, hi]."""
    f = lambda al: cost(a, prices(POOL, al, mode)) - cost(b, prices(POOL, al, mode))
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (lo + hi) / 2


# ------------------------------------------------------ roster optimizer
def _triples(g):
    ids = [pid for pid, _, _ in POOL[g]]
    return [(c, V_OF[c[0]] + V_OF[c[1]] + V_OF[c[2]]) for c in combinations(ids, 3)]


TRIPLES = {g: sorted(_triples(g), key=lambda t: -t[1]) for g in ("M", "F")}


def candidates(g, price, budget, k, must=None, exclude=()):
    out = []
    for ids, v in TRIPLES[g]:
        if must is not None and must not in ids:
            continue
        if any(u in ids for u in exclude):
            continue
        if price[ids[0]] + price[ids[1]] + price[ids[2]] <= budget:
            out.append(ids)
            if len(out) == k:
                break
    return out


def best_roster(price, opp, must=None, exclude=(), k=None, cap=TEAM_CAP):
    """Best roster under `cap` at these prices, scored by REAL tie_win_prob
    against `opp`. Enumerates budget splits; per split takes the top-k
    triples per gender by proxy value and scores all k*k combinations.
    With `must`, the cheapest legal roster containing `must` is always a
    candidate too, so the search is infeasible only when no legal roster
    with that player exists (not when the budget grid missed it)."""
    k = K_CAND if k is None else k
    g_must = DOUBLES[must]["gender"] if must else None
    best = (-1.0, None)
    for b_men in range(0, cap + 1, BUDGET_STEP):
        cm = candidates("M", price, b_men, k, must if g_must == "M" else None, exclude)
        cw = candidates("F", price, cap - b_men, k, must if g_must == "F" else None, exclude)
        for m in cm:
            for w in cw:
                p = win(m + w, opp)
                if p > best[0]:
                    best = (p, m + w)
    if must is not None:
        cheap = {g: sorted((u for u, _, _ in POOL[g] if u != must and u not in exclude),
                           key=lambda u: price[u]) for g in ("M", "F")}
        og = "F" if g_must == "M" else "M"
        r = tuple(sorted((must,) + tuple(cheap[g_must][:2]))) + tuple(cheap[og][:3])
        r = tuple(u for u in r if DOUBLES[u]["gender"] == "M") + \
            tuple(u for u in r if DOUBLES[u]["gender"] == "F")
        if cost(r, price) <= cap:
            p = win(r, opp)
            if p > best[0]:
                best = (p, r)
    return best


def must_buy(pid, alpha, mode, rounds=2):
    """P(best roster WITH pid beats best roster WITHOUT pid) after
    `rounds` of alternating best response. >0.5 = pid is underpriced
    (a must-buy at these prices); 0.5 = fairly priced."""
    price = prices(POOL, alpha, mode)
    repl = roster_from_ranks([REPL_RANK] * 3, [REPL_RANK] * 3)
    _, without = best_roster(price, repl, exclude=(pid,))
    _, with_ = best_roster(price, without, must=pid)
    if with_ is None:                  # pid + 5 cheapest teammates > cap
        return None, None, without, price
    for _ in range(rounds - 1):
        _, without = best_roster(price, with_, exclude=(pid,))
        _, with_ = best_roster(price, without, must=pid)
    return win(with_, without), with_, without, price


def crossover_alpha(pid, mode, grid):
    rows = []
    for a in grid:
        p, w, wo, price = must_buy(pid, a, mode)
        rows.append((a, p, w, wo, price))
    # linear interpolation of the first 0.5 crossing
    xo = None
    for (a0, p0, *_), (a1, p1, *_) in zip(rows, rows[1:]):
        if p0 is None or p1 is None:
            continue
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p0 != p1:
            xo = a0 + (0.5 - p0) * (a1 - a0) / (p1 - p0)
            break
    return xo, rows


# ---------------------------------------------------------------- report
def pid_named(name):
    return next(u for u in DOUBLES if DOUBLES[u]["name"] == name)


def block(k):
    """HANDOFF.md challenger shape: ranks k, k+1 + replacement, both genders."""
    return roster_from_ranks([k, k + 1, REPL_RANK], [k, k + 1, REPL_RANK])


def block3(k):
    """Alternative shape: three contiguous ranks, no replacement filler."""
    return roster_from_ranks([k, k + 1, k + 2], [k, k + 1, k + 2])


PAIRS = [
    # name, anchor ranks (men, women), challenger shape, first challenger rank
    ("A  #3M/#2F anchor + 2 scrubs", ([3, 79, 80], [2, 79, 80]), block, 4),
    ("B  #10 anchor + 2 scrubs",     ([10, 79, 80], [10, 79, 80]), block, 11),
    ("C  #1 anchor + 2 scrubs",      ([1, 79, 80], [1, 79, 80]),  block, 2),
    # (#2+#3+repl) vs any (k,k+1,repl) never ties: the anchor pair is
    # stronger than every lower pair and the third slot is DB-only, so
    # "two stars vs depth" shapes carry no price information. Dropped.
    ("F  #1 + 2 replacements",       ([1, 60, 60], [1, 60, 60]),  block3, 2),
]


def main(quick=False):
    print(f"value basis for pricing: {VALUE_SOURCE}  (pool = top 60 per gender on that basis, see pool.py)\n")
    print("=== gender split implied by a joint $20M pool ===")
    for a in (0.6, 0.76, 0.88, 1.0, 1.2):
        pj = prices(POOL, a, "joint")
        print(f"alpha={a:<5} women's share {100*women_share(pj):.1f}%   "
              f"Waters ${pj[pid_named('Anna Leigh Waters')]:,.0f}  "
              f"Bright ${pj[pid_named('Anna Bright')]:,.0f}  "
              f"Johns ${pj[pid_named('Ben Johns')]:,.0f}  "
              f"#60M ${pj[POOL['M'][-1][0]]:,.0f}  #60F ${pj[POOL['F'][-1][0]]:,.0f}")
    print()

    print("=== indifference pairs: rank sweep, then the alpha that price-matches them ===")
    print("(same pair -> two alphas: split = HANDOFF.md's two $10M sub-pools, joint = one $20M pool)")
    pair_results = []
    for name, (mr, wr), shape, lo in PAIRS:
        anchor = roster_from_ranks(mr, wr)
        k, p, _ = find_crossover(anchor, shape, lo=lo)
        chal = shape(k)
        a_split = alpha_reconciling(anchor, chal, "split")
        a_joint = alpha_reconciling(anchor, chal, "joint")
        fmt = lambda x: f"{x:.3f}" if x is not None else "none"
        print(f"{name}: ties at k={k} (P={p:.3f})  alpha_split={fmt(a_split)}  alpha_joint={fmt(a_joint)}")
        print(f"    anchor:     {label(anchor)}")
        print(f"    challenger: {label(chal)}")
        pair_results.append((name, k, p, a_split, a_joint))
    print()

    if quick:
        return

    grid = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
    print("=== must-buy test: P(best roster WITH player beats best roster WITHOUT) ===")
    print("(crossover = alpha at which the player is fairly priced; below it they are a bargain)")
    for who in ("Anna Leigh Waters", "Anna Bright", "Ben Johns"):
        pid = pid_named(who)
        for mode in ("split", "joint"):
            xo, rows = crossover_alpha(pid, mode, grid)
            line = "  ".join(f"a={a}:{'INFEAS' if p is None else f'{p:.3f}'}" for a, p, *_ in rows)
            print(f"{who:18s} {mode:5s}  crossover={xo if xo is None else round(xo, 3)}   {line}")
            feas = [r for r in rows if r[1] is not None]
            if not feas:
                continue
            a_show = min(feas, key=lambda r: abs(r[1] - 0.5))
            a, p, w, wo, price = a_show
            print(f"    at alpha={a}: with    ${cost(w, price):,.0f}  {label(w)}")
            print(f"                 without ${cost(wo, price):,.0f}  {label(wo)}")
    print()

    print("=== cheapest legal roster (3M+3F from the priced pool) vs a $500k min-spend rule ===")
    for mode in ("split", "joint"):
        for a in (0.6, 0.8, 1.0, 1.2):
            price = prices(POOL, a, mode)
            cheapest = sum(sorted(price[u] for u, _, _ in POOL["M"])[:3]) + \
                sum(sorted(price[u] for u, _, _ in POOL["F"])[:3])
            print(f"{mode:5s} alpha={a}: ${cheapest:,.0f}  {'BINDS' if cheapest < 500_000 else 'redundant'}")


def must_buy_report(who, modes, grid):
    pid = pid_named(who)
    for mode in modes:
        xo, rows = crossover_alpha(pid, mode, grid)
        line = "  ".join(f"a={a}:{'INFEAS' if p is None else f'{p:.3f}'}" for a, p, *_ in rows)
        print(f"{who:18s} {VALUE_SOURCE:7s} {mode:5s} floor={FLOOR}  crossover={xo if xo is None else round(xo, 3)}   {line}")
        feas = [r for r in rows if r[1] is not None]
        if not feas:
            continue
        a, p, w, wo, price = min(feas, key=lambda r: abs(r[1] - 0.5))
        print(f"    at alpha={a}: with    ${cost(w, price):,.0f}  {label(w)}")
        print(f"                 without ${cost(wo, price):,.0f}  {label(wo)}")
        sys.stdout.flush()


if __name__ == "__main__":
    if "--must-buy" in sys.argv:
        who = sys.argv[sys.argv.index("--must-buy") + 1]
        modes = ("joint", "split")
        if "--modes" in sys.argv:
            modes = tuple(sys.argv[sys.argv.index("--modes") + 1].split(","))
        grid = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6)
        must_buy_report(who, modes, grid)
    else:
        main(quick="--quick" in sys.argv)
