"""value_cap/draft_strategies.py -- hypothetical $1M rosters under different
draft strategies, priced off the Phase 2 list, and how they fare against
each other.

    python value_cap/draft_strategies.py                 # ~10 min, writes draft_strategies.md
    python value_cap/draft_strategies.py --alpha 0.845 --rounds 2 --k 20 --step 10000
    python value_cap/draft_strategies.py --alpha 1.0 --tag "Anna Leigh Waters" \
        --out value_cap/draft_strategies_tag.md      # proportional prices + franchise tag

Setup (user framing, 2026-09-05): a team drafts ANY six players from the
priced pool (3 men + 3 women, no exclusivity between teams -- every team
may want the same player, this is about price, not scarcity), four
start. The tie model already plays the top 2 men + top 2 women by doubles
value in the four doubles games and the top 2+2 by singles value in the
DreamBreaker, so "bench" = the third man and third woman by doubles value,
and the DB foursome may pull a bench player in.

Each strategy is a CONSTRAINT on the roster; the roster shown is the best
legal roster satisfying it, scored with the real tie model against the
other strategies' rosters (the "field"), and rebuilt for --rounds rounds
so each build is a best response to the others. Prices: alpha (default
0.845 = the fair-and-rosterable-Waters edge from phase2_pricing.md), one
joint $20M pool, $30k floor. Nothing here is tuned to make a strategy
win; the point is to see which ones the price list leaves balanced.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

from phase1_value_model import singles_of
from phase2_pricing import (DOUBLES, NAME, POOL, REPL_RANK, SINGLES, TEAM_CAP,
                            V_OF, cost, pid_named, prices, prices_tagged,
                            roster_from_ranks, win)

HERE = Path(__file__).resolve().parent
TOL = 1e-3      # dollars; a franchise tag is defined so the cheapest legal roster costs EXACTLY the cap


# ------------------------------------------------------------ helpers
def triples(g):
    ids = [pid for pid, _, _ in POOL[g]]
    out = [(c, V_OF[c[0]] + V_OF[c[1]] + V_OF[c[2]]) for c in combinations(ids, 3)]
    return sorted(out, key=lambda t: -t[1])


TRIPLES = {g: triples(g) for g in ("M", "F")}
DV = lambda u: DOUBLES[u]["v"]
SV = lambda u: singles_of(u, DOUBLES, SINGLES)


def starters(triple):
    """top 2 by doubles value within one gender's triple; third = bench."""
    s = sorted(triple, key=lambda u: -DV(u))
    return s[:2], s[2]


def specialists(g, singles_top=8, phi_rank_from=20):
    """singles-elite players who are NOT front-line doubles players:
    singles rank <= singles_top within the pool, phi rank >= phi_rank_from."""
    ids = [u for u, _, _ in POOL[g]]
    by_s = sorted(ids, key=lambda u: -SV(u))
    return {u for i, u in enumerate(by_s[:singles_top]) if ids.index(u) + 1 >= phi_rank_from}


SPECIALISTS = {g: specialists(g) for g in ("M", "F")}


# ---------------------------------------------------------- strategies
def strategy_specs(price):
    """name -> dict(desc, pred{g: fn(triple)->bool}, share=(lo,hi) men's
    share of spend, cap). Predicates see prices via closure."""
    anchor = 0.35 * TEAM_CAP        # price-based so the strategy means the same thing on every list
    waters = pid_named("Anna Leigh Waters")

    def under(x):
        return lambda t: all(price[u] <= x for u in t)

    def bench_specialist(g):
        def f(t):
            _, bench = starters(t)
            return bench in SPECIALISTS[g]
        return f

    def balanced(t):
        st, bench = starters(t)
        return all(price[u] <= 0.28 * TEAM_CAP for u in st) and price[bench] <= 80_000

    return {
        "Quant (no constraint)": dict(
            desc="whatever the optimizer likes best at these prices",
            pred={}, share=(0, 1), cap=TEAM_CAP),
        "Superstar: Waters": dict(
            desc="must roster Anna Leigh Waters, fill around her",
            pred={"F": lambda t: waters in t}, share=(0, 1), cap=TEAM_CAP),
        "Two anchors": dict(
            desc="a man AND a woman each costing >= 35% of cap, fill the rest",
            pred={"M": lambda t: any(price[u] >= anchor for u in t),
                  "F": lambda t: any(price[u] >= anchor for u in t)},
            share=(0, 1), cap=TEAM_CAP),
        "Balanced four": dict(
            desc="no starter over 28% of cap, bench at <= $80k each",
            pred={"M": balanced, "F": balanced}, share=(0, 1), cap=TEAM_CAP),
        "Deep six": dict(
            desc="nobody over 20% of cap; six real contributors",
            pred={"M": under(0.20 * TEAM_CAP), "F": under(0.20 * TEAM_CAP)},
            share=(0, 1), cap=TEAM_CAP),
        "Women first": dict(
            desc=">= 65% of spend on the women",
            pred={}, share=(0, 0.35), cap=TEAM_CAP),
        "Men first": dict(
            desc=">= 55% of spend on the men",
            pred={}, share=(0.55, 1), cap=TEAM_CAP),
        "DreamBreaker specialist": dict(
            desc="a singles-elite, doubles-middling player on the bench (either gender)",
            pred={}, share=(0, 1), cap=TEAM_CAP, any_of={"M": bench_specialist("M"),
                                                          "F": bench_specialist("F")}),
        "Cheapskate ($500k)": dict(
            desc="best roster on half the cap (the min-spend floor); reference only, not in the field",
            pred={}, share=(0, 1), cap=500_000, reference=True),
    }


def candidates(g, price, budget, k, pred):
    out = []
    for ids, _ in TRIPLES[g]:
        if pred is not None and not pred(ids):
            continue
        if price[ids[0]] + price[ids[1]] + price[ids[2]] <= budget + TOL:
            out.append(ids)
            if len(out) == k:
                break
    return out


def build(price, opps, spec, k, step):
    """best roster satisfying spec, scored by mean tie win prob vs opps."""
    cap = spec["cap"]
    lo, hi = spec["share"]
    any_of = spec.get("any_of")
    best = (-1.0, None)
    # budget grid + the two extreme splits (one gender at its cheapest legal
    # triple, the other taking the rest) so a must-include star is never
    # lost to grid resolution (phase2_pricing.py's cheapest-completion fix)
    cheapest = {g: min(cost(t, price) for t, _ in TRIPLES[g]
                       if spec["pred"].get(g) is None or spec["pred"][g](t))
                for g in ("M", "F")}
    splits = sorted(set(range(0, cap + 1, step)) | {cheapest["M"], cap - cheapest["F"]})
    for b_men in splits:
        if b_men < 0 or b_men > cap + TOL:
            continue
        cm = candidates("M", price, b_men, k, spec["pred"].get("M"))
        cw = candidates("F", price, cap - b_men, k, spec["pred"].get("F"))
        if any_of:   # at least one gender's triple satisfies its any_of predicate
            cm_ok = candidates("M", price, b_men, k, any_of["M"])
            cw_ok = candidates("F", price, cap - b_men, k, any_of["F"])
            pairs = [(m, w) for m in cm_ok for w in cw] + [(m, w) for m in cm for w in cw_ok]
        else:
            pairs = [(m, w) for m in cm for w in cw]
        for m, w in pairs:
            c_m, c_w = cost(m, price), cost(w, price)
            if c_m + c_w > cap + TOL:
                continue
            share = c_m / (c_m + c_w)
            if not (lo <= share <= hi):
                continue
            p = sum(win(m + w, o) for o in opps) / len(opps)
            if p > best[0]:
                best = (p, m + w)
    return best


# --------------------------------------------------------------- report
def describe(roster, price):
    men = [u for u in roster if DOUBLES[u]["gender"] == "M"]
    women = [u for u in roster if DOUBLES[u]["gender"] == "F"]
    sm, bm = starters(men)
    sw, bw = starters(women)
    db = sorted(women, key=lambda u: -SV(u))[:2] + sorted(men, key=lambda u: -SV(u))[:2]
    fmt = lambda u: f"{NAME[u]} (${price[u]/1000:,.0f}k)"
    return {
        "starters": [fmt(u) for u in sw + sm],
        "bench": [fmt(u) for u in (bw, bm)],
        "db": [NAME[u] for u in db],
        "db_uses_bench": [NAME[u] for u in db if u in (bw, bm)],
        "cost": cost(roster, price),
        "women_share": cost(women, price) / cost(roster, price),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.845)
    ap.add_argument("--rounds", type=int, default=2, help="best-response rounds vs the field")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--step", type=int, default=10_000)
    ap.add_argument("--tag", default=None, help="franchise-tag this player (see phase2_pricing.prices_tagged)")
    ap.add_argument("--out", default=str(HERE / "draft_strategies.md"))
    a = ap.parse_args()

    if a.tag:
        price = prices_tagged(POOL, a.alpha, pid_named(a.tag), "joint")
        print(f"tag: {a.tag} at ${price[pid_named(a.tag)]:,.0f}", file=sys.stderr)
    else:
        price = prices(POOL, a.alpha, "joint")
    specs = strategy_specs(price)
    repl = roster_from_ranks([REPL_RANK] * 3, [REPL_RANK] * 3)
    print(f"specialists: M={[NAME[u] for u in SPECIALISTS['M']]}  F={[NAME[u] for u in SPECIALISTS['F']]}",
          file=sys.stderr)

    rosters = {}
    for name, spec in specs.items():                 # round 0: vs replacement
        p, r = build(price, [repl], spec, a.k, a.step)
        if r is None:
            print(f"round 0  {name:26s} INFEASIBLE at these prices", file=sys.stderr)
            continue
        rosters[name] = r
        print(f"round 0  {name:26s} P(beat repl)={p:.3f}  ${cost(r, price):,.0f}", file=sys.stderr)
    for rd in range(1, a.rounds + 1):                # rounds 1..: vs the field
        churn = 0
        for name, spec in specs.items():
            if name not in rosters:
                continue
            field = [rosters[o] for o in rosters if o != name and not specs[o].get("reference")]
            p, r = build(price, field, spec, a.k, a.step)
            churn += r != rosters[name]
            rosters[name] = r
            print(f"round {rd}  {name:26s} P(beat field)={p:.3f}  ${cost(r, price):,.0f}", file=sys.stderr)
        print(f"round {rd}: {churn} rosters changed", file=sys.stderr)

    names = [n for n in specs if n in rosters]
    comp = [n for n in names if not specs[n].get("reference")]
    M = {(x, y): win(rosters[x], rosters[y]) for x in names for y in names if x != y}
    vs_field = {x: sum(M[(x, y)] for y in comp if y != x) / (len(comp) - (x in comp)) for x in names}
    vs_repl = {x: win(rosters[x], repl) for x in names}

    lines = []
    w = lines.append
    w("# Draft strategies under the $1M cap -- hypothetical rosters\n")
    tag_note = (f", **{a.tag} franchise-tagged at ${price[pid_named(a.tag)]:,.0f}** "
                f"(most a team can pay and still field a legal roster; the surplus is "
                f"spread over the rest of the pool)" if a.tag else "")
    w(f"Prices: alpha = {a.alpha}, one joint $20M pool, $30k floor{tag_note} "
      f"(`phase2_pricing.py`). Any 6 from the priced pool (3M+3W), four start "
      f"(top 2 per gender by doubles value), DreamBreaker foursome picked "
      f"separately by singles value. Each roster is the best legal roster for "
      f"its strategy against the other strategies' rosters after {a.rounds} "
      f"best-response rounds (k={a.k} candidate triples per gender per budget "
      f"split, ${a.step:,} budget grid). No exclusivity between teams.\n")
    w("## Rosters\n")
    for x in sorted(names, key=lambda n: -vs_field[n]):
        d = describe(rosters[x], price)
        w(f"### {x}  --  {specs[x]['desc']}\n")
        w(f"- **vs field {100*vs_field[x]:.1f}%** | vs replacement {100*vs_repl[x]:.1f}% | "
          f"spend ${d['cost']:,.0f} ({100*d['women_share']:.0f}% on women)")
        w(f"- Starters: {', '.join(d['starters'])}")
        w(f"- Bench: {', '.join(d['bench'])}")
        db = ", ".join(d["db"])
        if d["db_uses_bench"]:
            db += f"  (bench in the DB: {', '.join(d['db_uses_bench'])})"
        w(f"- DreamBreaker four: {db}\n")
    w("## Head to head: P(row beats column), one tie\n")
    w("(\"mean\" and \"vs field\" exclude the $500k reference roster.)\n")
    short = {x: x.split(":")[0].split(" (")[0] for x in names}
    order = sorted(names, key=lambda n: -vs_field[n])
    w("| | " + " | ".join(short[y] for y in order) + " | mean |")
    w("|---|" + "---|" * (len(order) + 1))
    for x in order:
        cells = ["--" if x == y else f"{100*M[(x, y)]:.0f}" for y in order]
        w(f"| **{short[x]}** | " + " | ".join(cells) + f" | {100*vs_field[x]:.1f} |")
    w("")
    text = "\n".join(lines)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
