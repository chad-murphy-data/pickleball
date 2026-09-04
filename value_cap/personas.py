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
import pickle
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


_FAME = {}


def fame():
    """fame_table() for the CURRENT board (set_board rebinds D.BOARD, so this
    is computed lazily and cached per board rather than at import)."""
    key = frozenset(D.BOARD)
    if key not in _FAME:
        _FAME.clear()
        _FAME[key] = fame_table()
    return _FAME[key]


class Marketing(D.Owner):
    name = "marketing guy (big names)"

    def __init__(self, noise, rng, gamma, k):
        super().__init__(noise, rng, gamma)
        self.k = k
        fm = fame()
        for u in D.BOARD:
            self.dbl[u]["v"] += k * D.SPREAD_V[D.GENDER[u]] * fm.get(u, 0.0)
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


READS = """## What this says (hand-written against the seed-1 grid; re-check the numbers above if the grid is re-run)

- **Overvaluing one gender costs nothing.** Stretch the men's or women's gaps to double their real size and the persona's teams still win 47-50% with a normal title shot; the quants around them do not move. The price list already carries the ranking, so a lopsided belief about which gender matters changes a pick or two at the margin, not the roster.
- **The marketing owner comes out slightly AHEAD (51-52% win, ~6% title).** At these prices the big names are fairly priced, so preferring them is free -- and the fame table is built from real doubles rank, so a fame bias is partly a bias toward the truth. Read it as "chasing names at fair prices does not hurt you", not as "fame beats analysis".
- **The $500k cheapskate is the persona that breaks the league.** Alone, the team wins 21% with no title shot. Five of them push the parity spread from 4.4 to 13.7 points and hand the other fifteen a 58% win rate; all twenty, and Waters, Johns, Bright and the top of the list are never drafted (no one can afford them). This is the case for the $500k min-spend rule: it is not about fairness to the cheap team, it is that unspent money makes the whole league worse.
- **Loyalty is cheap in small doses and expensive in large ones.** A light preference for 2026 teammates (lam 0.05) is free; at lam 0.15 it costs ~1.5 points of win rate; at lam 0.5 (a full real six worth half a win of belief) the team drops to 42% / 2% title and leaves $110k unspent because the teammates it wants do not fill a legal roster efficiently.
- **Bargains first is the worst strategy that looks sensible.** Spending rounds 1-3 on <=$120k players wins 24% with no title shot, and only $600k gets spent: in a 20-team snake the whole top 60 is gone before round 4, so the money has nothing left to buy. Raising the threshold to $250k gets 39%. The lesson is that in a draft, waiting is the expensive move -- the stars are gone, not overpriced.
- **Nothing here dents the pick-1 team.** Waters' team wins 63-68% in every cell it is drafted; the only cells that move it are the ones where personas leave the league lopsided (five cheapskates 68.5%).
- **No persona creates a second contender** (title-odds table above): the favourite sits at 33-37%, the runner-up at 7-9%, and exactly one team has a 10%+ title shot in every cell where she is drafted. The only cells with two or more 10%+ teams are the ones that leave her undrafted (twenty cheapskates: 11.6% favourite, 14 effective contenders; twenty bargain hunters at $250k: 15% / 12% / 10%). The personas spread title odds across the pack; none of them lifts a slot-2 or slot-3 team, which is what the EPL-style "two or three teams can win" shape needs. See `dials.md` for what does move it.
"""



def concentration(r):
    """Title-odds concentration per draft, averaged over drafts: how many teams
    can really win. Uses the per-slot title% the season sim recorded."""
    n_d = len(r["slot_title"][0])
    out = {"top": [], "second": [], "third": [], "n10": [], "n5": [], "eff": [], "gini": [], "gap": []}
    for d in range(n_d):
        p = sorted((r["slot_title"][t][d] for t in range(D.N_TEAMS)), reverse=True)
        tot = sum(p) or 1.0
        p = [x / tot for x in p]
        out["top"].append(p[0]); out["second"].append(p[1]); out["third"].append(p[2])
        out["n10"].append(sum(1 for x in p if x >= 0.10))
        out["n5"].append(sum(1 for x in p if x >= 0.05))
        out["eff"].append(1.0 / sum(x * x for x in p))
        out["gap"].append(p[0] - p[1])
        n = len(p)
        srt = sorted(p)
        out["gini"].append(sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(srt)) / (n * sum(srt)))
    return {k: statistics.mean(v) for k, v in out.items()}


def render_concentration(rows):
    L = ["", "## Who can actually win: title-odds concentration", "",
         "Per draft, every team's title odds from the season sim, then averaged over the drafts. "
         "'Effective contenders' = 1 / sum(title odds squared): 20 means twenty equal teams, 1 means one "
         "certain champion. Gini is over the twenty teams' title odds (0 = parity). Contenders = teams with "
         "at least a 10% (or 5%) title shot. 'Runner-up favourite' = the title odds of the second-best team, "
         "i.e. how real the chase is. For scale, a bookmaker's pre-season English Premier League board "
         "(favourite ~55%, two challengers at ~20% and ~15%) is about 2.7 effective contenders, 3 teams at 10%+.", "",
         "| persona | strength | how many | favourite | runner-up favourite | third | gap 1st-2nd | teams >= 10% | "
         "teams >= 5% | effective contenders | Gini |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, strength, count, r in rows:
        c = concentration(r)
        L.append(f"| {label} | {strength} | {count if count else '--'} | {100*c['top']:.1f}% | {100*c['second']:.1f}% | "
                 f"{100*c['third']:.1f}% | {100*c['gap']:.1f} pts | {c['n10']:.1f} | {c['n5']:.1f} | {c['eff']:.1f} | "
                 f"{c['gini']:.2f} |")
    return L


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
        st = {u: (f"{100*statistics.mean(r['stars'][u]['exp']):.1f}%" if r["stars"][u]["exp"] else "not drafted")
              for u in stars}
        L.append(f"| {label} | {strength} | {count if count else '--'} | "
                 f"{100*statistics.mean(k['exp']):.1f}% | {100*statistics.mean(k['title']):.1f}% | "
                 f"${statistics.mean(k['spend'])/1e3:,.0f}k | "
                 + (f"{100*statistics.mean(q['exp']):.1f}% | {100*statistics.mean(q['title']):.1f}% | " if q and count else "-- | -- | ")
                 + f"{100*r['spread']:.1f} pts | {st[stars[0]]} | {st[stars[1]]} | {st[stars[2]]} | "
                 + (", ".join(f"{NAME[u]} #{D.POOL_RANK[u]}{D.GENDER[u]}" for u in sorted(und, key=lambda u: D.POOL_RANK[u])[:4])
                    + (f" (+{len(und)-4})" if len(und) > 4 else "") if und else "none") + " |")
    L += ["", "## Who each persona drafts", "",
          "Players the persona carries more often than the quants in the same league (share of the persona's "
          "rosters minus share of the quants' rosters, percentage points), k = 1 cells. The all-20 cells are "
          "uninformative here (every drafted player is on exactly one roster per draft).", ""]
    for label, strength, count, r in rows:
        if count != 1:
            continue
        k = r["kinds"][label]
        q = r["kinds"]["quant"]
        diff = {u: 100 * (c / k["n"] - q["picks"].get(u, 0) / q["n"]) for u, c in k["picks"].items()}
        top = sorted(diff.items(), key=lambda kv: (-kv[1], price[kv[0]]))[:8]
        L.append(f"- **{label}**, {strength}: " + ", ".join(
            f"{NAME[u]} ${price[u]/1e3:,.0f}k (+{d:.0f}pp)" for u, d in top if d > 0))
    L += render_concentration(rows)
    L += ["", READS]
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
    ap.add_argument("--rerender", action="store_true",
                    help="skip the grid; render from cache/personas_rows.pkl (same args)")
    args = ap.parse_args(_argv[1:])
    cache = HERE / "cache" / "personas_rows.pkl"
    if args.rerender:
        price, stars, rows = pickle.loads(cache.read_bytes())
    else:
        price, stars, rows = run(args)
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps((price, stars, rows)))
    render(price, stars, rows, args, args.out)


if __name__ == "__main__":
    main()
