"""value_cap/draft_sim.py -- 20 teams draft from the priced pool with
scarcity, then play a season. Varies the draft (snake vs fixed order) and
the owners' information (perfect vs noisy beliefs).

    python value_cap/draft_sim.py                                   # tag list, all variants
    python value_cap/draft_sim.py --alpha 0.845                     # the old list
    python value_cap/draft_sim.py --noise 0 0.1 0.25 --drafts 40 --seasons 200
    python value_cap/draft_sim.py --formats snake linear --out value_cap/draft_sim_tag.md

Setup. 20 teams, $1M cap, 6 rounds, 3 men + 3 women each, one pick per
turn, every player available to exactly one team. Prices = the Phase 2
list (alpha, joint pool, optional franchise tag). Owners must always be
able to fill their remaining slots with the cheapest players left, so
nobody is ever stranded.

Owner behaviour (deliberately dumb-but-consistent; nothing is tuned):
  - each owner believes every player's doubles and singles value with
    their own error, eps ~ N(0, noise x that gender's pool spread), drawn
    once per draft and held for all six picks (consistent beliefs);
  - at each turn the owner projects a final roster for every affordable
    candidate: the candidate plus a greedy fill of the remaining slots
    with the best-believed players still available that keep the roster
    completable, then takes the candidate whose projected roster has the
    highest believed tie probability against a REFERENCE roster;
  - the reference = doubles ranks (10, 30, 50) per gender, which is what a
    median snake slot expects to end up with (slot k in a 20-team snake
    sees ranks ~k, ~40-k, ~40+k). A stronger reference tilts owners toward
    stars, a weaker one toward balance; it is a knob, not a fact.
  - noise 0 => every owner has the true values => the draft is deterministic.

Season = double round robin (38 ties), top 4 playoff, scored with the TRUE
values (fast_tie.FastTie on the production numbers). Reads, per variant:
parity spread, win% and title% by draft slot, where each star went and
how their team did ("stuck" = below parity), who went undrafted (info,
not a test -- user call 2026-09-04), the blueprint mix (how much of a
team's spend sits on its top player), and mean spend.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_tie import FastTie  # noqa: E402
from phase2_pricing import (DOUBLES, FLOOR, NAME, POOL, SINGLES, TEAM_CAP,  # noqa: E402
                            pid_named, prices, prices_tagged, rank)
import phase1_value_model as p1  # noqa: E402

HERE = Path(__file__).resolve().parent
N_TEAMS = 20
ROUNDS = 6
PER_GENDER = 3
REF_RANKS = (10, 30, 50)
CAND_TOP = 30          # candidates per pick: top-N believed doubles value per needed gender ...
CAND_SINGLES = 6       # ... plus top-N believed singles value (DreamBreaker specialists)
CAND_CHEAP = 5         # ... plus the cheapest few (a legal fallback always exists)
EXTRA_PER_GENDER = 60  # floor-priced free agents on the board beyond the priced 60


def order_for(fmt, rnd):
    base = list(range(N_TEAMS))
    if fmt == "snake" and rnd % 2 == 1:
        return base[::-1]
    return base


class Owner:
    def __init__(self, noise, rng, gamma):
        self.noise = noise
        eng_gamma = TRUE_ENGINE.gamma if gamma is None else gamma
        if noise <= 0:
            self.engine = TRUE_ENGINE if gamma is None else FastTie(DOUBLES, SINGLES, gamma)
            self.v = {u: DOUBLES[u]["v"] for u in BOARD}
            self.s = {u: TRUE_ENGINE.s[u] for u in BOARD}
            return
        dbl = {}
        sgl = {}
        for g in ("M", "F"):
            sd_v = SPREAD_V[g]
            sd_s = SPREAD_S[g]
            for u in BOARD:
                if GENDER[u] != g:
                    continue
                d = dict(DOUBLES[u])
                d["v"] = d["v"] + rng.gauss(0, noise * sd_v)
                dbl[u] = d
                sgl[u] = TRUE_ENGINE.s[u] + rng.gauss(0, noise * sd_s)
        # players outside the pool keep true values (never drafted anyway)
        for u in DOUBLES:
            if u not in dbl:
                dbl[u] = DOUBLES[u]
        merged_singles = dict(SINGLES)
        merged_singles.update(sgl)
        self.engine = FastTie(dbl, merged_singles, eng_gamma)
        self.v = {u: dbl[u]["v"] for u in BOARD}
        self.s = {u: sgl[u] for u in BOARD}

    def choose(self, roster, spent, avail, price, need, gaps):
        """roster: list of pids; need: {g: slots left}; gaps[j] = how many
        picks the OTHER teams make before this owner's (j+1)-th next turn.
        When projecting the fill for that turn the owner assumes the
        gaps[j] highest-PRICED players still on the board are gone (price =
        the public consensus of value; the owner's private error only
        shapes their own choice). Without this every owner defers the
        stars, because "I can take them next round" looks free. Returns pid."""
        by_g = {g: sorted((u for u in avail if GENDER[u] == g), key=lambda u: price[u]) for g in ("M", "F")}
        by_price = sorted(avail, key=lambda u: -price[u])

        def completion_cost(exclude, need_after):
            tot = 0.0
            for g in ("M", "F"):
                k = need_after[g]
                if k <= 0:
                    continue
                got = 0
                for u in by_g[g]:
                    if u in exclude:
                        continue
                    tot += price[u]
                    got += 1
                    if got == k:
                        break
                if got < k:
                    return float("inf")
            return tot

        budget = TEAM_CAP - spent
        cands = set()
        for g in ("M", "F"):
            if need[g] <= 0:
                continue
            need_after = dict(need)
            need_after[g] -= 1
            # affordable = can still complete the roster at the cheapest prices
            pool_g = [u for u in avail if GENDER[u] == g
                      and price[u] + completion_cost({u}, need_after) <= budget + 1e-6]
            cands.update(sorted(pool_g, key=lambda u: -self.v[u])[:CAND_TOP])
            cands.update(sorted(pool_g, key=lambda u: -self.s[u])[:CAND_SINGLES])
            cands.update(by_g[g][:CAND_CHEAP])
        best, best_p = None, -1.0
        for x in cands:
            gx = GENDER[x]
            need_after = dict(need)
            need_after[gx] -= 1
            if price[x] + completion_cost({x}, need_after) > budget + 1e-6:
                continue
            # greedy projection of the final roster
            proj = list(roster) + [x]
            left = budget - price[x]
            na = dict(need_after)
            taken = {x}
            j = 0
            while na["M"] > 0 or na["F"] > 0:
                pick = None
                gone = set(by_price[:gaps[j] + 1]) - {x} if j < len(gaps) else set()
                gone = set(list(gone)[:gaps[j]]) if j < len(gaps) else gone
                j += 1
                for u in sorted((u for u in avail if u not in taken and u not in gone
                                 and na[GENDER[u]] > 0), key=lambda u: -self.v[u]):
                    n2 = dict(na)
                    n2[GENDER[u]] -= 1
                    if price[u] + completion_cost(taken | {u}, n2) <= left + 1e-6:
                        pick = u
                        break
                if pick is None:      # cannot happen if feasibility held, but be safe
                    break
                proj.append(pick)
                taken.add(pick)
                left -= price[pick]
                na[GENDER[pick]] -= 1
            if na["M"] > 0 or na["F"] > 0:
                continue
            p = self.engine.tie(tuple(proj), REFERENCE)
            if p > best_p:
                best, best_p = x, p
        if best is None:
            raise RuntimeError("no feasible pick")
        return best


def run_draft(price, fmt, noise, rng, gamma=None):
    owners = [Owner(noise, rng, gamma) for _ in range(N_TEAMS)]
    avail = set(BOARD)
    rosters = [[] for _ in range(N_TEAMS)]
    spent = [0.0] * N_TEAMS
    need = [{"M": PER_GENDER, "F": PER_GENDER} for _ in range(N_TEAMS)]
    picks = {}          # pid -> (team, round)
    seq = [(rnd, team) for rnd in range(ROUNDS) for team in order_for(fmt, rnd)]
    for i, (rnd, team) in enumerate(seq):
        mine = [k for k in range(i + 1, len(seq)) if seq[k][1] == team]
        gaps = [k - i - 1 - j for j, k in enumerate(mine)]
        if True:
            x = owners[team].choose(rosters[team], spent[team], avail, price, need[team], gaps)
            rosters[team].append(x)
            spent[team] += price[x]
            need[team][GENDER[x]] -= 1
            avail.discard(x)
            picks[x] = (team, rnd + 1)
    return [tuple(r) for r in rosters], spent, picks, avail


def season(rosters, seasons, rng):
    """True-engine double round robin + top-4 playoff. Returns per-team
    (expected win%, mean wins, title share)."""
    n = len(rosters)
    P = [[0.5 if i == j else TRUE_ENGINE.tie(rosters[i], rosters[j]) for j in range(n)] for i in range(n)]
    exp = [sum(P[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    titles = [0] * n
    wins_tot = [0.0] * n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for _ in range(seasons):
        wins = [0] * n
        for i, j in pairs:
            for _leg in range(2):
                if rng.random() < P[i][j]:
                    wins[i] += 1
                else:
                    wins[j] += 1
        order = sorted(range(n), key=lambda t: (-wins[t], rng.random()))
        s1, s2, s3, s4 = order[:4]
        f1 = s1 if rng.random() < P[s1][s4] else s4
        f2 = s2 if rng.random() < P[s2][s3] else s3
        champ = f1 if rng.random() < P[f1][f2] else f2
        titles[champ] += 1
        for t in range(n):
            wins_tot[t] += wins[t]
    return exp, [w / seasons for w in wins_tot], [t / seasons for t in titles]


def blueprint(roster, price):
    tot = sum(price[u] for u in roster)
    top = max(price[u] for u in roster) / tot
    if top >= 0.55:
        return "superstar (top player >=55% of spend)"
    if top >= 0.40:
        return "star-led (40-55%)"
    if top >= 0.28:
        return "anchor (28-40%)"
    return "balanced (<28%)"


def run_variant(price, fmt, noise, drafts, seasons, seed, stars, gamma=None):
    rng = random.Random(seed)
    slot_exp = [[] for _ in range(N_TEAMS)]
    slot_title = [[] for _ in range(N_TEAMS)]
    star_rows = {u: {"rounds": [], "undrafted": 0, "exp": [], "title": [], "slot": []} for u in stars}
    undrafted = {}
    shapes = {}
    spends = []
    spreads = []
    maxes = []
    floor_taken = []
    t0 = time.time()
    for d in range(drafts):
        rosters, spent, picks, left = run_draft(price, fmt, noise, rng, gamma)
        exp, mw, ttl = season(rosters, seasons, rng)
        spreads.append(statistics.pstdev(exp))
        maxes.append(max(exp))
        spends.extend(spent)
        for t in range(N_TEAMS):
            slot_exp[t].append(exp[t])
            slot_title[t].append(ttl[t])
            b = blueprint(rosters[t], price)
            shapes[b] = shapes.get(b, 0) + 1
        for u in stars:
            if u in picks:
                team, rnd = picks[u]
                star_rows[u]["rounds"].append(rnd)
                star_rows[u]["exp"].append(exp[team])
                star_rows[u]["title"].append(ttl[team])
                star_rows[u]["slot"].append(team + 1)
            else:
                star_rows[u]["undrafted"] += 1
        for u in left:
            if u in POOL_SET:
                undrafted[u] = undrafted.get(u, 0) + 1
        floor_taken.append(sum(1 for u in picks if u not in POOL_SET))
    return dict(fmt=fmt, noise=noise, drafts=drafts, seasons=seasons, secs=time.time() - t0,
                slot_exp=slot_exp, slot_title=slot_title, stars=star_rows, undrafted=undrafted,
                shapes=shapes, spend=statistics.mean(spends), spread=statistics.mean(spreads),
                max_exp=statistics.mean(maxes), floor_taken=statistics.mean(floor_taken))


def render(results, price, header, stars, out):
    L = ["# Draft simulation -- 20 teams, scarcity, varied draft and information", "",
         header, "",
         f"20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final "
         f"roster for each affordable candidate (candidate + greedy fill of the best-believed "
         f"players still available) and take the one whose projection has the highest believed "
         f"tie probability against a reference roster of doubles ranks {REF_RANKS} per gender. "
         f"Noise = sd of each owner's belief error as a fraction of the gender's pool spread "
         f"(men {SPREAD_V['M']:.3f}, women {SPREAD_V['F']:.3f} logit), fixed per owner per draft. "
         f"Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. "
         f"Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.", ""]
    L += ["## Summary", "",
          "| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | "
          "mean spend | blueprint mix |", "|---|---|---|---|---|---|---|"]
    for r in results:
        mix = ", ".join(f"{k.split(' (')[0]} {100*v/sum(r['shapes'].values()):.0f}%"
                        for k, v in sorted(r["shapes"].items(), key=lambda kv: -kv[1]))
        L.append(f"| {r['fmt']} | {r['noise']:.0%} | {r['drafts']} | {100*r['spread']:.1f} pts | "
                 f"{100*r['max_exp']:.1f}% | ${r['spend']/1e3:,.0f}k | {mix} |")
    L.append("")
    for r in results:
        L += [f"## {r['fmt']} draft, owner noise {r['noise']:.0%} ({r['drafts']} draft(s) x "
              f"{r['seasons']} seasons, {r['secs']:.0f} s)", ""]
        # undrafted first (user call: info, up top, never buried)
        und = sorted(r["undrafted"].items(), key=lambda kv: (-kv[1], -PHI[kv[0]]))
        notable = [(u, c) for u, c in und if POOL_RANK[u] <= 30]
        L.append(f"**Undrafted priced players (info, not a test):** the board is the priced 60+60 "
                 f"plus {len(EXTRA_PIDS)//2} free agents per gender at the ${FLOOR/1e3:.0f}k floor; teams "
                 f"took {r['floor_taken']:.1f} floor players per draft, leaving {len(und)} distinct "
                 f"priced players unpicked in at least one draft. Inside the top 30 of their gender: "
                 + (", ".join(f"{NAME[u]} (#{POOL_RANK[u]}{GENDER[u]}, ${price[u]/1e3:,.0f}k, "
                              f"{100*c/r['drafts']:.0f}% of drafts)" for u, c in notable)
                    if notable else "none") + ". All: "
                 + (", ".join(f"{NAME[u]} #{POOL_RANK[u]}{GENDER[u]}" for u, c in und) if und else "none") + ".")
        L.append("")
        L += ["Stars (top 6 per gender by phi): where they went and how their team did.", "",
              "| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |",
              "|---|---|---|---|---|---|---|---|"]
        for u in stars:
            s = r["stars"][u]
            if s["rounds"]:
                exp = statistics.mean(s["exp"])
                L.append(f"| {NAME[u]} (#{POOL_RANK[u]}{GENDER[u]}) | ${price[u]/1e3:,.0f}k | "
                         f"{statistics.mean(s['rounds']):.1f} | {100*s['undrafted']/r['drafts']:.0f}% | "
                         f"{statistics.mean(s['slot']):.1f} | {100*exp:.1f}% | "
                         f"{100*statistics.mean(s['title']):.1f}% | {'YES' if exp < 0.49 else 'no'} |")
            else:
                L.append(f"| {NAME[u]} (#{POOL_RANK[u]}{GENDER[u]}) | ${price[u]/1e3:,.0f}k | -- | 100% | -- | -- | -- | -- |")
        L.append("")
        L += ["By draft slot (mean over drafts):", "",
              "| slot | " + " | ".join(str(t + 1) for t in range(N_TEAMS)) + " |",
              "|---|" + "---|" * N_TEAMS,
              "| win% | " + " | ".join(f"{100*statistics.mean(r['slot_exp'][t]):.0f}" for t in range(N_TEAMS)) + " |",
              "| title% | " + " | ".join(f"{100*statistics.mean(r['slot_title'][t]):.0f}" for t in range(N_TEAMS)) + " |",
              ""]
    Path(out).write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tag", default="Anna Leigh Waters", help="'' for no tag")
    ap.add_argument("--formats", nargs="+", default=["snake", "linear"])
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.1, 0.25])
    ap.add_argument("--drafts", type=int, default=30, help="per noisy variant; noise 0 runs once")
    ap.add_argument("--seasons", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.tag:
        price = prices_tagged(POOL, a.alpha, pid_named(a.tag), "joint")
        price = {u: price.get(u, FLOOR) for u in BOARD}
        header = (f"Prices: alpha = {a.alpha}, one joint $20M pool, **{a.tag} franchise-tagged at "
                  f"${price[pid_named(a.tag)]:,.0f}** (`phase2_pricing.prices_tagged`).")
    else:
        price = prices(POOL, a.alpha, "joint")
        price = {u: price.get(u, FLOOR) for u in BOARD}
        header = f"Prices: alpha = {a.alpha}, one joint $20M pool, $30k floor (`phase2_pricing.prices`)."
    stars = [u for g in ("F", "M") for u, _, _ in POOL[g][:6]]
    results = []
    for fmt in a.formats:
        for noise in a.noise:
            drafts = 1 if noise == 0 else a.drafts
            r = run_variant(price, fmt, noise, drafts, a.seasons, a.seed, stars)
            print(f"{fmt} noise {noise}: {drafts} drafts in {r['secs']:.0f}s; spread {100*r['spread']:.1f} pts, "
                  f"max {100*r['max_exp']:.1f}%", file=sys.stderr)
            results.append(r)
    out = a.out or str(HERE / ("draft_sim_tag.md" if a.tag else f"draft_sim_a{a.alpha}.md"))
    render(results, price, header, stars, out)


# ------------------------------------------------------------ module state
TRUE_ENGINE = FastTie(DOUBLES, SINGLES)
POOL_PIDS = [u for g in ("M", "F") for u, _, _ in POOL[g]]
# the priced pool is exactly 20 x 3 per gender, so on its own it is always fully
# drafted; the board adds the next players by phi at the FLOOR price (free agents
# at the minimum) so "a priced player went undrafted" can actually happen.
import csv as _csv
PHI = {}
_EXTRA = {"M": [], "F": []}
for _r in _csv.DictReader((HERE / "player_value_shapley.csv").open()):
    PHI[_r["player_id"]] = float(_r["phi"])
    if _r["in_pool"] != "1" and _r["player_id"] in DOUBLES:
        _EXTRA[_r["gender"]].append(_r["player_id"])
for _g in _EXTRA:
    _EXTRA[_g].sort(key=lambda u: -PHI[u])
    # pad to 60 per gender with the next tracked players by doubles value, so a
    # team can ALWAYS complete at the floor no matter what the other 19 took
    # (60 priced + 60 floor >= the 60 slots per gender): feasibility never
    # decays as the board empties, and the cheapest completion is a constant.
    _have = set(POOL_PIDS) | set(_EXTRA[_g])
    for _u in sorted((u for u in DOUBLES if DOUBLES[u]["gender"] == _g and u not in _have),
                     key=lambda u: -DOUBLES[u]["v"]):
        if len(_EXTRA[_g]) >= EXTRA_PER_GENDER:
            break
        _EXTRA[_g].append(_u)
        PHI.setdefault(_u, float("-inf"))
EXTRA_PIDS = [u for g in ("M", "F") for u in _EXTRA[g]]
BOARD = POOL_PIDS + EXTRA_PIDS
POOL_RANK = {u: i + 1 for g in ("M", "F") for i, (u, _, _) in enumerate(POOL[g])}
for _g in _EXTRA:
    for _i, _u in enumerate(_EXTRA[_g]):
        POOL_RANK[_u] = len(POOL[_g]) + _i + 1
GENDER = {u: DOUBLES[u]["gender"] for u in DOUBLES}
SPREAD_V = {g: statistics.pstdev([DOUBLES[u]["v"] for u, _, _ in POOL[g]]) for g in ("M", "F")}
SPREAD_S = {g: statistics.pstdev([TRUE_ENGINE.s[u] for u, _, _ in POOL[g]]) for g in ("M", "F")}
POOL_SET = set(POOL_PIDS)
REFERENCE = tuple(rank("M", r) for r in REF_RANKS) + tuple(rank("F", r) for r in REF_RANKS)

if __name__ == "__main__":
    main()
