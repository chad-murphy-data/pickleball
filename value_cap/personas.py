"""value_cap/personas.py -- owner personas in the 20-team draft: what
happens to a team that drafts like a person instead of a quant, and what
their presence does to the league.

    python value_cap/personas.py                       # full grid -> personas.md (~10 min)
    python value_cap/personas.py --counts 1 20 --drafts 8 --seasons 100   # quick look

The personas (user's words, 2026-09-04) and how each is implemented as an
Owner subclass (draft_sim.Owner: beliefs -> projection -> best projected tie
probability vs the reference roster). Every persona has a STRENGTH that is
swept, not picked:

  overvalues men      believes men's gaps are (1+k)x as wide as they are:
                      men's doubles + singles beliefs stretched around the
                      men's mean; k = 0.5, 1.0. Mirror: overvalues women.
  cheapskate $500k    quant with a $500k cap (the min-spend line).
  marketing guy       believes big names are better than they are: +k x
                      (gender spread) x fame, fame = 1 at #1 in doubles,
                      falling to 0 at #31; k = 0.5, 1.0.
  real teams          wants players who were 2026 MLP teammates: objective
                      = projected tie prob + lam x (largest same-franchise
                      group on the roster - 1)/5; lam = 0.05, 0.15, 0.5 (a full
                      real six is worth +lam of win prob to them). Franchise =
                      a player's modal 2026 MLP team (games.csv x
                      mlp_matchups_2026.csv).
  bargains first      rounds 1-3 only look at players priced <= T, then
                      draft like a quant; T = $120k, $250k.

Each cell: k persona owners (1, 5, or all 20) at random draft slots among
quants, everyone at 10% belief noise, snake draft on the mlp2026 board with
the shipped tag list; 20 drafts x 200 seasons. Reads: the persona's teams'
win% / title% / spend vs the quants' in the same league, parity spread,
Waters' team, the persona's favourite picks, top-30 players left undrafted.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import draft_sim as D  # noqa: E402
from phase2_pricing import DOUBLES, FLOOR, NAME, POOL, pid_named, prices_tagged  # noqa: E402
sys.argv = _argv

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


# ------------------------------------------------------------ 2026 franchises
def franchises_2026():
    """pid -> modal 2026 MLP franchise (from games.csv joined to the matchup
    table on match_id; side resolved from winner_side + scores)."""
    mm = {r["match_id"]: r for r in csv.DictReader((DATA / "mlp_matchups_2026.csv").open())}
    seen = defaultdict(Counter)
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["tour"] != "MLP" or not g["date"].startswith("2026") or g["is_dreambreaker"] == "True":
            continue
        m = mm.get(g["match_id"])
        if not m or m["winner_side"] not in ("1", "2"):
            continue
        t1_won = int(g["t1_score"]) > int(g["t2_score"])
        t1_is_one = (m["winner_side"] == "1") == t1_won
        for u in (g["t1_p1"], g["t1_p2"]):
            seen[u.lower()][m["team_one"] if t1_is_one else m["team_two"]] += 1
        for u in (g["t2_p1"], g["t2_p2"]):
            seen[u.lower()][m["team_two"] if t1_is_one else m["team_one"]] += 1
    return {u: c.most_common(1)[0][0] for u, c in seen.items()}


FRANCHISE = franchises_2026()


# ------------------------------------------------------------------ personas
class Quant(D.Owner):
    name = "quant"


class OvervaluesGender(D.Owner):
    """Stretch one gender's believed gaps by (1+k) around that gender's mean."""
    gender = "M"
    name = "overvalues men"

    def __init__(self, noise, rng, gamma, k):
        super().__init__(noise, rng, gamma)
        self.k = k
        us = [u for u in D.BOARD if D.GENDER[u] == self.gender]
        mv = statistics.mean(self.dbl[u]["v"] for u in us)
        ms = statistics.mean(self.sgl[u] for u in us)
        for u in us:
            self.dbl[u]["v"] = mv + (1 + k) * (self.dbl[u]["v"] - mv)
            self.sgl[u] = ms + (1 + k) * (self.sgl[u] - ms)
        self.rebuild()


class OvervaluesMen(OvervaluesGender):
    gender, name = "M", "overvalues men"


class OvervaluesWomen(OvervaluesGender):
    gender, name = "F", "overvalues women"


class Cheapskate(D.Owner):
    name = "cheapskate $500k"

    def __init__(self, noise, rng, gamma, cap=500_000):
        super().__init__(noise, rng, gamma)
        self.cap = cap


def fame_table():
    """fame = 1 at #1 doubles (true value, per gender, on the board) -> 0 at #31."""
    out = {}
    for g in ("M", "F"):
        order = sorted((u for u in D.BOARD if D.GENDER[u] == g), key=lambda u: -DOUBLES[u]["v"])
        for i, u in enumerate(order):
            out[u] = max(0.0, 1.0 - i / 30.0)
    return out


FAME = fame_table()


class Marketing(D.Owner):
    name = "marketing guy (big names)"

    def __init__(self, noise, rng, gamma, k):
        super().__init__(noise, rng, gamma)
        self.k = k
        for u in D.BOARD:
            self.dbl[u]["v"] += k * D.SPREAD_V[D.GENDER[u]] * FAME[u]
        self.rebuild()


class RealTeams(D.Owner):
    name = "wants real teams"

    def __init__(self, noise, rng, gamma, lam):
        super().__init__(noise, rng, gamma)
        self.lam = lam

    def score(self, proj):
        fr = Counter(FRANCHISE.get(u) for u in proj if FRANCHISE.get(u))
        biggest = max(fr.values()) if fr else 1
        return self.engine.tie(tuple(proj), D.REFERENCE) + self.lam * (biggest - 1) / 5.0


class BargainsFirst(D.Owner):
    name = "bargains first"

    def __init__(self, noise, rng, gamma, threshold, rounds=3):
        super().__init__(noise, rng, gamma)
        self.threshold = threshold
        self.rounds = rounds

    def filter_cands(self, cands, roster, avail, price):
        if len(roster) >= self.rounds:
            return cands
        cheap = {u for u in cands if price[u] <= self.threshold}
        return cheap or cands


PERSONAS = [
    ("overvalues men", OvervaluesMen, "k", [0.5, 1.0]),
    ("overvalues women", OvervaluesWomen, "k", [0.5, 1.0]),
    ("cheapskate $500k", Cheapskate, "cap", [500_000]),
    ("marketing guy (big names)", Marketing, "k", [0.5, 1.0]),
    ("wants real teams", RealTeams, "lam", [0.05, 0.15, 0.5]),
    ("bargains first", BargainsFirst, "threshold", [120_000, 250_000]),
]


# ---------------------------------------------------------------- experiment
def factory_for(cls, kw, count, noise):
    def make(rng):
        slots = set(rng.sample(range(D.N_TEAMS), count))
        return [cls(noise, rng, None, **kw) if t in slots else Quant(noise, rng, None)
                for t in range(D.N_TEAMS)]
    return make


def fmt_strength(key, val):
    if key in ("cap", "threshold"):
        return f"${val/1e3:,.0f}k"
    return f"{key} = {val:g}"


def run(args):
    D.set_board(args.board)
    waters = pid_named("Anna Leigh Waters")
    price = prices_tagged(POOL, 1.0, waters, "joint")
    price = {u: price.get(u, FLOOR) for u in D.BOARD}
    stars = [waters, pid_named("Anna Bright"), pid_named("Ben Johns")]
    rows = []
    base = D.run_variant(price, "snake", args.noise, args.drafts, args.seasons, args.seed, stars,
                         owner_factory=factory_for(Quant, {}, 0, args.noise))
    rows.append(("quant baseline", "", 0, base))
    print(f"baseline: spread {100*base['spread']:.1f}, max {100*base['max_exp']:.1f}%, {base['secs']:.0f}s", file=sys.stderr)
    for label, cls, key, vals in PERSONAS:
        if args.only and label not in args.only:
            continue
        for val in vals:
            for count in args.counts:
                r = D.run_variant(price, "snake", args.noise, args.drafts, args.seasons, args.seed, stars,
                                  owner_factory=factory_for(cls, {key: val}, count, args.noise))
                rows.append((label, fmt_strength(key, val), count, r))
                k = r["kinds"].get(label)
                print(f"{label} {fmt_strength(key, val)} x{count}: persona win {100*statistics.mean(k['exp']):.1f}% "
                      f"title {100*statistics.mean(k['title']):.1f}% spend ${statistics.mean(k['spend'])/1e3:.0f}k; "
                      f"spread {100*r['spread']:.1f}; {r['secs']:.0f}s", file=sys.stderr)
    return price, stars, rows


def render(price, stars, rows, args, out):
    L = ["# Owner personas in the draft", "",
         f"Shipped tag list (alpha 1, joint pool, Waters tagged at ${price[stars[0]]/1e3:,.0f}k), snake draft on the "
         f"`{args.board}` board, every owner at {args.noise:.0%} belief noise, {args.drafts} drafts x {args.seasons} "
         f"seasons per cell, seed {args.seed}. Each cell puts k persona owners at random draft slots among quants. "
         f"Persona definitions and strengths are in the docstring of `personas.py` (strengths are swept, not picked). "
         f"Parity = 50% win, 5% title. Built by `personas.py`.", "",
         "## The grid", "",
         "| persona | strength | how many of 20 | persona teams: win% | title% | spend | quant teams: win% | title% | "
         "parity spread | Waters' team win% | Bright's team | Johns' team | top-30 undrafted (any draft) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, strength, count, r in rows:
        k = r["kinds"].get(label if count else "quant")
        q = r["kinds"].get("quant")
        und = [u for u, c in r["undrafted"].items() if D.POOL_RANK[u] <= 30]
        st = {u: statistics.mean(r["stars"][u]["exp"]) if r["stars"][u]["exp"] else float("nan") for u in stars}
        L.append(f"| {label} | {strength} | {count if count else '--'} | "
                 f"{100*statistics.mean(k['exp']):.1f}% | {100*statistics.mean(k['title']):.1f}% | "
                 f"${statistics.mean(k['spend'])/1e3:,.0f}k | "
                 + (f"{100*statistics.mean(q['exp']):.1f}% | {100*statistics.mean(q['title']):.1f}% | " if q and count else "-- | -- | ")
                 + f"{100*r['spread']:.1f} pts | {100*st[stars[0]]:.1f}% | {100*st[stars[1]]:.1f}% | {100*st[stars[2]]:.1f}% | "
                 + (", ".join(f"{NAME[u]} #{D.POOL_RANK[u]}{D.GENDER[u]}" for u in sorted(und, key=lambda u: D.POOL_RANK[u])[:4])
                    + (f" (+{len(und)-4})" if len(und) > 4 else "") if und else "none") + " |")
    L += ["", "## Who each persona drafts", "",
          "Most-drafted players per persona (share of that persona's rosters that carried them), one row per strength "
          "at the k = 1 cell (the persona among 19 quants) and the all-20 cell.", ""]
    for label, strength, count, r in rows:
        if not count or count not in (1, D.N_TEAMS):
            continue
        k = r["kinds"][label]
        top = sorted(k["picks"].items(), key=lambda kv: (-kv[1], price[kv[0]]))[:8]
        L.append(f"- **{label}**, {strength}, x{count}: " + ", ".join(
            f"{NAME[u]} ${price[u]/1e3:,.0f}k ({100*c/k['n']:.0f}%)" for u, c in top))
    L.append("")
    Path(out).write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--drafts", type=int, default=20)
    ap.add_argument("--seasons", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--board", default="mlp2026")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE / "personas.md"))
    args = ap.parse_args(_argv[1:])
    price, stars, rows = run(args)
    render(price, stars, rows, args, args.out)


if __name__ == "__main__":
    main()
