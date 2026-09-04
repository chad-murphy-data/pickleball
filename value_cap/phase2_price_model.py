"""value_cap/phase2_price_model.py -- Phase 2 quick pass: turn Phase 1's V
into dollar prices under a few different star-premium settings, and check
whether any one of them quietly hands a dominant strategy to whoever finds
it, using a real search instead of hand-picked example rosters.

    python value_cap/phase2_price_model.py

Two things bundled here (per project discussion, 2026-09-02, before the
real sample-roster fit which needs Chad's picks):

  (1) A price list for a spread of star-premium settings (alpha), so
      there are actual dollar numbers to look at tonight.
  (2) A real search for the best $1,000,000 roster under EACH price list
      -- not three cherry-picked archetypes -- using the actual joint
      tie-win model (dyad pairing, weakest link, DB lineup selection),
      not just adding up V's, plus two named strategy variants (stars
      first / no-superstar) to see whether the answer to "which strategy
      wins" changes as alpha changes.

Formula (from the original project brief), fit separately for men and
women because a roster needs exactly 3 of each regardless of how talent
happens to be distributed within either market (see phase0_bench_value.md
for why the two markets look very different):

    price_i = floor + (V_i^alpha / sum_j V_j^alpha) * (pool - N*floor)

Explicit placeholder assumptions -- inputs to tomorrow's real fit, not
conclusions:
  - N_TEAMS = 20 (matches Phase 1).
  - floor = $30,000, the illustrative number from the original brief.
    Not fit. Left fixed tonight so the alpha sweep isn't also chasing a
    moving floor.
  - The $1,000,000 team cap splits 50/50 into men's and women's league
    sub-pools ($10M each across 20 teams) -- arbitrary, pending a reason
    to do otherwise.
  - alpha is SWEPT (0.5, 1.0, 1.5, 2.0, 3.0), never picked, per the
    project's working rule.
  - Priced pool = each gender's top 60 by V (Phase 1's replacement line)
    -- the players a 20-team league would actually roster on this
    ranking.

The roster search enumerates every 3-of-60 combination per gender exactly
(34,220 each -- cheap), gets each combo's price and proxy value (sum of
its 3 players' V, since V's were computed one-at-a-time against a
replacement roster rather than jointly), then finds the men's/women's
budget split that maximizes combined proxy value under the $1M cap. The
winning roster is then VALIDATED with the real joint model
(tie_win_prob, from phase1_value_model.py) rather than trusted on the
proxy alone.
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

from phase1_value_model import ROOT, load_doubles, load_singles, tie_win_prob

FLOOR = 30_000
N_TEAMS = 20
TEAM_CAP = 1_000_000
SUBPOOL = N_TEAMS * TEAM_CAP // 2      # $10M per gender across the league
ALPHAS = (0.5, 1.0, 1.5, 2.0, 3.0)
BUDGET_STEP = 5_000                     # split-search granularity


def load_pool():
    """gender -> [(player_id, full_name, V_total)] top 60 by V, from
    Phase 1's output (value_cap/player_value.csv)."""
    rows = list(csv.DictReader((ROOT / "value_cap" / "player_value.csv").open()))
    pool = {"M": [], "F": []}
    for r in rows:
        pool[r["gender"]].append((r["player_id"], r["full_name"], float(r["V_total"])))
    for g in pool:
        pool[g].sort(key=lambda t: -t[2])
        pool[g] = pool[g][:N_TEAMS * 3]
    return pool


def prices_for_alpha(pool, alpha):
    """player_id -> price, for one gender's pool, at one alpha."""
    n = len(pool)
    weights = [max(v, 0.0) ** alpha for _, _, v in pool]
    total_w = sum(weights)
    remaining = SUBPOOL - n * FLOOR
    return {pid: FLOOR + (w / total_w) * remaining
            for (pid, _, _), w in zip(pool, weights)}


def all_triples(pool, prices):
    """Every 3-of-60 combo: (player_ids, total_price, total_V)."""
    out = []
    for combo in combinations(pool, 3):
        ids = tuple(c[0] for c in combo)
        price = sum(prices[i] for i in ids)
        v = sum(c[2] for c in combo)
        out.append((ids, price, v))
    return out


def best_frontier(triples):
    """Sorted by price; running max V achievable at or under that price.
    Returns (sorted_prices, running_max_v, running_max_triple)."""
    triples = sorted(triples, key=lambda t: t[1])
    prices, best_v, best_ids = [], [], []
    cur_best, cur_ids = float("-inf"), None
    for ids, price, v in triples:
        if v > cur_best:
            cur_best, cur_ids = v, ids
        prices.append(price)
        best_v.append(cur_best)
        best_ids.append(cur_ids)
    return prices, best_v, best_ids


def lookup(frontier, budget):
    """Best (v, ids) achievable at total price <= budget."""
    prices, best_v, best_ids = frontier
    lo, hi = 0, len(prices) - 1
    if not prices or prices[0] > budget:
        return float("-inf"), None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if prices[mid] <= budget:
            lo = mid
        else:
            hi = mid - 1
    return best_v[lo], best_ids[lo]


def optimize(men_triples, women_triples, cap=TEAM_CAP, exclude_men=(), exclude_women=()):
    """Best combined roster under `cap`, optionally with some triples
    pre-excluded (used for the no-single-superstar variant below)."""
    men_f = best_frontier([t for t in men_triples if t[0] not in exclude_men])
    women_f = best_frontier([t for t in women_triples if t[0] not in exclude_women])
    best = (float("-inf"), None, None, None)
    for b_men in range(0, cap + 1, BUDGET_STEP):
        v_m, ids_m = lookup(men_f, b_men)
        v_w, ids_w = lookup(women_f, cap - b_men)
        if ids_m is None or ids_w is None:
            continue
        if v_m + v_w > best[0]:
            best = (v_m + v_w, ids_m, ids_w, (b_men, cap - b_men))
    return best


def real_win_prob(ids_m, ids_w, doubles, singles):
    roster = list(ids_m) + list(ids_w)
    replacement = {g: sorted((u for u in doubles if doubles[u]["gender"] == g),
                              key=lambda u: -doubles[u]["v"])[N_TEAMS * 3 - 1]
                   for g in ("M", "F")}
    opp = [replacement["F"]] * 3 + [replacement["M"]] * 3
    return tie_win_prob(roster, opp, doubles, singles)


def name_of(pool, pid):
    for g in pool:
        for u, name, _ in pool[g]:
            if u == pid:
                return name
    return pid


def main():
    pool = load_pool()
    doubles = load_doubles()
    singles = load_singles()

    print(f"pool: top {len(pool['M'])} men, top {len(pool['F'])} women "
          f"(replacement line = Phase 1's rank {N_TEAMS*3})\n")

    marquee_m = ["Ben Johns", "Christopher Haworth"]
    marquee_f = ["Anna Leigh Waters", "Anna Bright"]

    print("=== price by alpha, marquee names ($) ===")
    header = "player".ljust(25) + "".join(f"a={a:<10}" for a in ALPHAS)
    print(header)
    prices_by_alpha = {a: {"M": prices_for_alpha(pool["M"], a),
                            "F": prices_for_alpha(pool["F"], a)} for a in ALPHAS}
    for g, names in (("M", marquee_m), ("F", marquee_f)):
        for name in names:
            pid = next(u for u, n, _ in pool[g] if n == name)
            row = name.ljust(25)
            for a in ALPHAS:
                row += f"${prices_by_alpha[a][g][pid]:>9,.0f} "
            print(row)
    print()

    print("=== top-heaviness: share of the $10M subpool going to the top 5 ===")
    for g in ("M", "F"):
        row = g.ljust(25)
        for a in ALPHAS:
            top5 = sum(sorted(prices_by_alpha[a][g].values())[-5:])
            row += f"{100*top5/SUBPOOL:>9.1f}% "
        print(row)
    print()

    print("=== single most expensive player's price vs the $1M team cap ===")
    for g in ("M", "F"):
        row = g.ljust(25)
        for a in ALPHAS:
            top1 = max(prices_by_alpha[a][g].values())
            flag = " *OVER CAP*" if top1 > TEAM_CAP else ""
            row += f"${top1:>10,.0f}{flag} "
        print(row)
    print("(*OVER CAP* = this alpha prices one player above what any team could ever pay)\n")

    print("=== best achievable $1M roster per alpha (free search, real win prob vs replacement) ===")
    for alpha in ALPHAS:
        pm, pf = prices_by_alpha[alpha]["M"], prices_by_alpha[alpha]["F"]
        men_t = all_triples(pool["M"], pm)
        women_t = all_triples(pool["F"], pf)

        free_v, free_m, free_w, split = optimize(men_t, women_t)
        p_win = real_win_prob(free_m, free_w, doubles, singles)
        cost = sum(pm[i] for i in free_m) + sum(pf[i] for i in free_w)
        print(f"alpha={alpha}: cost=${cost:,.0f}  P(win tie)={p_win:.3f}")
        print(f"  men:   {', '.join(name_of(pool,i) for i in free_m)}")
        print(f"  women: {', '.join(name_of(pool,i) for i in free_w)}")

        # stars-first: force the single priciest man+woman in, cheapest legal fill
        star_m = max(pool["M"], key=lambda t: pm[t[0]])[0]
        star_w = max(pool["F"], key=lambda t: pf[t[0]])[0]
        cheap_m = sorted(pool["M"], key=lambda t: pm[t[0]])[:2]
        cheap_w = sorted(pool["F"], key=lambda t: pf[t[0]])[:2]
        stars_m = (star_m, cheap_m[0][0], cheap_m[1][0])
        stars_w = (star_w, cheap_w[0][0], cheap_w[1][0])
        stars_cost = sum(pm[i] for i in stars_m) + sum(pf[i] for i in stars_w)
        if stars_cost > TEAM_CAP:
            print(f"  [stars-first]     INFEASIBLE at this alpha -- the priciest "
                  f"man+woman alone plus 4 floor players costs ${stars_cost:,.0f}, "
                  f"over the ${TEAM_CAP:,} cap")
        else:
            stars_p = real_win_prob(stars_m, stars_w, doubles, singles)
            print(f"  [stars-first]     cost=${stars_cost:,.0f}  P(win tie)={stars_p:.3f}")

        # no-superstar: exclude the top 3 priciest per gender, re-optimize
        top3_m = {t[0] for t in sorted(pool["M"], key=lambda t: -pm[t[0]])[:3]}
        top3_f = {t[0] for t in sorted(pool["F"], key=lambda t: -pf[t[0]])[:3]}
        bal_v, bal_m, bal_w, _ = optimize(men_t, women_t,
                                           exclude_men={t[0] for t in men_t if set(t[0]) & top3_m},
                                           exclude_women={t[0] for t in women_t if set(t[0]) & top3_f})
        bal_cost = sum(pm[i] for i in bal_m) + sum(pf[i] for i in bal_w)
        bal_p = real_win_prob(bal_m, bal_w, doubles, singles)
        print(f"  [no top-3 stars]  cost=${bal_cost:,.0f}  P(win tie)={bal_p:.3f}")
        print()


if __name__ == "__main__":
    main()
