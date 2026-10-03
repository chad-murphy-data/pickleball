"""fan_auction.py -- the first-ever MLP auction, run by NEW OWNERS -- people who bought a
team, follow the sport, have no analytics -- and have never
seen a price sheet (fan_owner_spec.md).

Every earlier room (auction_sim, strategic_auction, market_eq) hands its
owners OUR list as the starting expectation, so every price it reports is
conditional on the room believing the sheet. Here nobody has one. Each
owner knows (fan_view.py): the rules and cap arithmetic, 2026 records and
game counts, MLP usage, and ONE ordinal picture per gender -- a single
joint draw from the v2 posterior, so owner 1 has Fahey 10th, owner 2 14th,
owner 3 6th, and none of them knows her distribution. Same for singles.
Nobody knows a dollar value, a tie probability or how to compare a man to a
woman. What they DO have is a philosophy -- a roster SHAPE with a plan for
spreading $1M over its six roles -- and eyes: once three players from the
same band (own ordering) have sold, they use the going rate.

Personas (roles = gender, own-rank band, budget share; `A` = either gender):
  star      one top-3 of either gender at 85%, five floor slots
  two       a top-5 woman AND a top-5 man at 40% each, four cheap
  four      2W + 2M from own top-15 at 22% each, bench W/M at 6%
  six       six from own top-30 at ~17% each
  singles   `four`, but one of the six must be a real singles player (own
            singles top-10, active on the singles tour), any gender --
            the bench slot pays up to 6% for one if the starters missed
  risk      2W + 2M top-15 at 20% + a real bench (own top-40) at 10% each;
            the only persona that reads usage (avoids <75% appearances)

Bid rule per owner, per player on the block (spec, signed off 2026-09-05):
  1 role match  -> the most expensive unfilled role x qualifies for
  2 plan money  -> share x $1M + savings carried from roles filled under plan
  3 the room    -> >=3 same-gender same-band sales tonight: median x (1+premium)
                   caps the bid if below plan money
  4 scarcity    -> acceptable players left for the role <= rivals still
                   needing that gender: full plan money
  5 hard cap    -> never above budget - floor x (slots left - 1)
Payment: second-highest + $5k. Nomination: rotation; the nominator names the
top target (own ordering) of its dearest unfilled role. A role nobody left
on the board can fill degrades to "anyone of that gender" (the owner keeps
the money). Payoffs are scored on the TRUE tie model (draft_sim.season).

    python value_cap/fan_auction.py                       # default mix, 8 seeds
    python value_cap/fan_auction.py --mix star=20         # an all-star&scrubs room
    python value_cap/fan_auction.py --describe            # rosters, seed 0
    python value_cap/fan_auction.py --grid                # the sweep -> fan_auction.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fan_view as F  # noqa: E402

CAP = 1_000_000.0
FLOOR = 30_000.0
INC = 5_000.0
N_TEAMS = 20
SLOTS = {"F": 3, "M": 3}

# ------------------------------------------------------------------ personas
# role = (gender, band, share); band = (lo, hi) in the owner's OWN ordering,
# None = a floor slot (anyone), "S" = a real singles player (own singles top-K)
PERSONAS = {
    "star":    [("A", (1, 3), .85)] + [("A", None, .03)] * 5,
    "two":     [("F", (1, 5), .40), ("M", (1, 5), .40)] + [("A", None, .05)] * 4,
    "four":    [("F", (1, 15), .22), ("F", (1, 15), .22), ("M", (1, 15), .22), ("M", (1, 15), .22),
                ("F", None, .06), ("M", None, .06)],
    "six":     [("F", (1, 30), .17), ("F", (1, 30), .17), ("F", (1, 30), .17),
                ("M", (1, 30), .17), ("M", (1, 30), .17), ("M", (1, 30), .15)],
    "singles": [("F", (1, 15), .22), ("F", (1, 15), .22), ("M", (1, 15), .22), ("M", (1, 15), .22),
                ("A", "S", .06), ("A", None, .06)],
    "risk":    [("F", (1, 15), .20), ("F", (1, 15), .20), ("M", (1, 15), .20), ("M", (1, 15), .20),
                ("F", (1, 40), .10), ("M", (1, 40), .10)],
}
DEFAULT_MIX = "star=2,two=3,four=6,six=4,singles=3,risk=2"
LONG = {"star": "star & scrubs", "two": "two stars", "four": "four starters", "six": "balanced six",
        "singles": "singles-minded", "risk": "risk-averse"}


def band_of(r):
    return 1 if r <= 5 else 2 if r <= 15 else 3 if r <= 30 else 4


# ------------------------------------------------------------------ the world
class World:
    def __init__(self, sgl_top=10, usage_min=0.75):
        import draft_sim as D  # noqa: E402
        D.set_board("mlp2026")
        self.D = D
        self.pids = list(D.BOARD)
        self.idx = {u: i for i, u in enumerate(self.pids)}
        self.gender = {u: F.GENDER[u] for u in self.pids}
        self.name = {u: F.NAME[u] for u in self.pids}
        # the shipped list, for comparison only (no owner sees it)
        import csv
        self.lp = {u: FLOOR for u in self.pids}
        for r in csv.DictReader((HERE / "price_list.csv").open()):
            if r["player_id"].lower() in self.lp:
                self.lp[r["player_id"].lower()] = float(r["price"])
        self.lrank = {}
        for g in ("F", "M"):
            for i, u in enumerate(sorted((u for u in self.pids if self.gender[u] == g), key=lambda u: -self.lp[u])):
                self.lrank[u] = i + 1
        self.waters = next(u for u in self.pids if self.name[u] == "Anna Leigh Waters")
        # what a new owner knows
        s26 = F.singles_records("2026"); sc = F.singles_records(None)
        self.singles_active = {u for u in self.pids if s26.get(u, [0, 0])[1] >= 10 or sc.get(u, [0, 0])[1] >= 100}
        use = F.mlp_usage()
        self.low_usage = {u for u in self.pids if u in use and use[u][1] and use[u][0] / use[u][1] < usage_min}
        self.sgl_top = sgl_top

    def season(self, rosters, seasons, rng):
        return self.D.season(rosters, seasons, rng)


# ------------------------------------------------------------------ an owner
class Owner:
    def __init__(self, W, persona, rng, sd_mult, premium, jitter=0.1):
        self.W = W; self.persona = persona; self.premium = premium
        shares = np.array([s for _, _, s in PERSONAS[persona]])
        if jitter > 0:   # no two owners hold the same plan to the dollar
            shares = shares * np.clip(1 + jitter * rng.standard_normal(len(shares)), 0.5, 1.5)
            shares = shares / shares.sum()
        self.roles = [dict(g=g, band=b, share=float(sh), pid=None, paid=None, degraded=False)
                      for (g, b, _), sh in zip(PERSONAS[persona], shares)]
        self.rank = {}; self.srank = {}
        for g in ("F", "M"):
            for i, u in enumerate(F.draw_order(rng, F.DOUBLES_POST, g, W.pids, sd_mult)):
                self.rank[u] = i + 1
            act = [u for u in W.pids if u in W.singles_active]
            for i, u in enumerate(F.draw_order(rng, F.SINGLES_POST, g, act, sd_mult)):
                self.srank[u] = i + 1
        self.roster = []; self.spent = 0.0; self.savings = 0.0
        self.need = dict(SLOTS)
        # reputation multipliers (pid -> x) learned from results across seasons
        # (owner_learning.py's W channel); None = a first-ever auction, nobody has a record
        self.rep = None
        # acceptable sets per role (static; availability handled outside)
        self.acc = [self._acceptable_set(r) for r in self.roles]

    # -- what qualifies for a role
    def _ok(self, role, u):
        g = self.W.gender[u]
        if role["g"] not in ("A", g):
            return False
        b = role["band"]
        if role["degraded"]:
            return True          # must fill the slot: takes what is left, usage included
        if self.persona == "risk" and u in self.W.low_usage:
            return False
        if b is None:
            return True
        if b == "S":
            return u in self.srank and self.srank[u] <= self.W.sgl_top
        return b[0] <= self.rank[u] <= b[1]

    def _acceptable_set(self, role):
        return {u for u in self.W.pids if self._ok(role, u)}

    def targeted(self, role):
        return role["band"] is not None

    def gender_feasible(self, role, g):
        """Can this role take a player of gender g and leave the fixed-gender
        roles fillable?"""
        if self.need[g] <= 0:
            return False
        fixed = sum(1 for r in self.roles if r["pid"] is None and r is not role and r["g"] == g)
        return fixed <= self.need[g] - 1

    def open_roles(self):
        return [r for r in self.roles if r["pid"] is None]

    def degrade(self, avail):
        """A targeted role with nobody acceptable left on the board becomes
        'anyone of that gender' (money kept)."""
        for i, r in enumerate(self.roles):
            if r["pid"] is None and self.targeted(r) and not r["degraded"]:
                if not any(self.gender_feasible(r, self.W.gender[u]) for u in self.acc[i] & avail):
                    r["degraded"] = True
                    self.acc[i] = self._acceptable_set(r)

    def match(self, x, avail):
        g = self.W.gender[x]
        cands = [(r["share"], i) for i, r in enumerate(self.roles)
                 if r["pid"] is None and x in self.acc[i] and self.gender_feasible(r, g)]
        if not cands:
            return None
        cands.sort(key=lambda c: (-c[0], c[1]))
        return cands[0][1]

    def plan_money(self, role):
        """Role share x $1M plus an even split of the savings pot over the open
        targeted roles (a star's money chases stars first); once no targeted
        role is open the pot spreads over the floor slots."""
        open_t = [r for r in self.roles if r["pid"] is None and self.targeted(r)]
        pool = open_t if open_t else [r for r in self.roles if r["pid"] is None]
        extra = self.savings / len(pool) if role in pool and pool else 0.0
        return role["share"] * CAP + extra

    def hard_cap(self):
        slots_left = self.need["F"] + self.need["M"]
        return CAP - self.spent - FLOOR * (slots_left - 1)

    def ceiling(self, x, avail, sold, rivals_need):
        """sold: list of (pid, paid) tonight; rivals_need[g] = other owners with
        an open targeted role that can take gender g."""
        i = self.match(x, avail)
        if i is None:
            return None, None
        role = self.roles[i]
        g = self.W.gender[x]
        plan = self.plan_money(role)
        c = plan
        if self.targeted(role):
            # the room: what have players like x (own band, same gender) gone for tonight?
            b = band_of(self.rank[x]) if role["band"] != "S" or role["degraded"] else "S"
            same = [p for u, p in sold if self.W.gender[u] == g and
                    ((band_of(self.rank[u]) == b) if b != "S" else (u in self.srank and self.srank[u] <= self.W.sgl_top))]
            if len(same) >= 3:
                rate = statistics.median(same)
                if rate < plan:
                    c = min(plan, rate * (1 + self.premium))
            if not role["degraded"]:
                left = sum(1 for u in self.acc[i] & avail if u != x and self.gender_feasible(role, self.W.gender[u]))
                if left <= rivals_need[g]:
                    c = plan
        else:
            # a floor slot: only bid on the best of what is left (own ordering)
            ahead = sum(1 for u in avail if u != x and self.W.gender[u] == g and self.rank[u] < self.rank[x])
            if ahead >= 2 * self.need[g] + 2:
                return None, None
        if self.rep is not None:
            c *= self.rep.get(x, 1.0)     # "the team with x won": pay x times the plan
        c = max(c, FLOOR)
        return min(c, self.hard_cap()), i

    def nominate(self, avail, rng):
        """Top target (own ordering) of the dearest unfilled role."""
        roles = sorted(((r["share"], i) for i, r in enumerate(self.roles) if r["pid"] is None), key=lambda c: (-c[0], c[1]))
        for _, i in roles:
            r = self.roles[i]
            cands = [u for u in self.acc[i] & avail if self.gender_feasible(r, self.W.gender[u])]
            if not cands:
                continue
            key = (lambda u: self.srank[u]) if r["band"] == "S" and not r["degraded"] else (lambda u: self.rank[u])
            best = {}
            for u in cands:
                g = self.W.gender[u]
                if g not in best or key(u) < key(best[g]):
                    best[g] = u
            # a new owner cannot compare across genders: take the better own-rank, coin flip on ties
            # (sorted by pid first so the rng stream does not depend on set-iteration order)
            pick = sorted(best.values(), key=lambda u: (key(u), u))
            if len(pick) > 1 and key(pick[0]) == key(pick[1]) and rng.random() < 0.5:
                return pick[1]
            return pick[0]
        return None

    def buy(self, x, paid, i):
        r = self.roles[i]
        r["pid"] = x; r["paid"] = paid
        self.savings = max(0.0, self.savings + r["share"] * CAP - paid)
        self.roster.append(x); self.spent += paid; self.need[self.W.gender[x]] -= 1

    def plan_met(self):
        return all(not self.targeted(r) or not r["degraded"] for r in self.roles)


# ------------------------------------------------------------------ the room
def parse_mix(s):
    mix = []
    for part in s.split(","):
        k, v = part.split("=")
        mix += [k] * int(v)
    assert len(mix) == N_TEAMS, f"mix has {len(mix)} seats, need {N_TEAMS}"
    return mix


def run_auction(W, mix, seed, sd_mult=1.0, premium=0.1, trace=(), jitter=0.1, owners=None, memory=()):
    """One night. owners: pre-built Owner objects (owner_learning.py carries
    them across seasons; seat order is theirs); memory: (pid, paid) sales the
    room remembers from before tonight -- they seed the going rate."""
    rng = np.random.default_rng(seed)
    if owners is None:
        seats = list(mix); rng.shuffle(seats)
        owners = [Owner(W, p, rng, sd_mult, premium, jitter) for p in seats]
    else:
        seats = [o.persona for o in owners]
    avail = set(W.pids)
    sales = []          # (pid, owner idx or None, paid, nominator)
    sold = list(memory)  # (pid, paid) -- public record; tonight's sales are appended
    turn = 0
    while any(o.need["F"] + o.need["M"] > 0 for o in owners):
        t = turn % N_TEAMS; turn += 1
        nom = owners[t]
        if nom.need["F"] + nom.need["M"] <= 0:
            continue
        for o in owners:
            o.degrade(avail)
        x = nom.nominate(avail, rng)
        if x is None:   # every role of the nominator is boxed in: name the best of what is left
            cands = [u for u in avail if nom.need[W.gender[u]] > 0]
            if not cands:
                raise RuntimeError("nothing to nominate")
            cands.sort(key=lambda u: (nom.rank[u], u))
            tied = [u for u in cands if nom.rank[u] == nom.rank[cands[0]]]
            x = tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]
        gx = W.gender[x]
        rivals_need = {g: sum(1 for o in owners if any(r["pid"] is None and o.targeted(r) and r["g"] in ("A", g)
                                                        and o.gender_feasible(r, g) for r in o.roles)) for g in ("F", "M")}
        bids = []
        for k, o in enumerate(owners):
            if o.need[gx] <= 0:
                continue
            rn = dict(rivals_need); rn[gx] -= 1 if any(r["pid"] is None and o.targeted(r) and r["g"] in ("A", gx)
                                                        and o.gender_feasible(r, gx) for r in o.roles) else 0
            c, i = o.ceiling(x, avail, sold, rn)
            if k == t and o.hard_cap() >= FLOOR:
                if c is None:
                    # the nominator opens at the floor for anyone it can legally take
                    i = next((j for j, r in enumerate(o.roles) if r["pid"] is None and o.gender_feasible(r, gx)), None)
                    c = FLOOR if i is not None else None
                else:
                    c = max(c, FLOOR)
            if c is not None and c >= FLOOR:
                bids.append((c, rng.random(), k, i))
        if not bids:
            avail.discard(x); sales.append((x, None, 0.0, t))
            continue
        bids.sort(key=lambda b: (-b[0], b[1]))
        b1, _, w, i = bids[0]
        b2 = bids[1][0] if len(bids) > 1 else FLOOR
        paid = min(b1, max(FLOOR, b2 + INC))
        paid = min(b1, round(paid / 1000.0) * 1000.0)
        if W.name[x] in trace:
            print(f"TRACE sale {len(sales)+1} {W.name[x]} (list ${W.lp[x]/1e3:.0f}k) nominated by T{t} [{seats[t]}]: " +
                  " ".join(f"T{k}[{seats[k]}]:{b/1e3:.0f}k" for b, _, k, _ in bids) + f" -> T{w} pays ${paid/1e3:.0f}k", flush=True)
        owners[w].buy(x, paid, i)
        avail.discard(x)
        sales.append((x, w, paid, t)); sold.append((x, paid))
    return dict(owners=owners, seats=seats, sales=sales)


def score(W, res, seasons, rng):
    ros = [tuple(o.roster) for o in res["owners"]]
    exp, _, ttl = W.season(ros, seasons, rng)
    res["exp"] = exp; res["ttl"] = ttl
    return res


def describe(W, res):
    rows = []
    for k, o in enumerate(res["owners"]):
        paid = {r["pid"]: r["paid"] for r in o.roles if r["pid"] is not None}
        names = ", ".join(f"{W.name[u]} ${paid[u]/1e3:.0f}k" for u in sorted(o.roster, key=lambda u: -paid[u]))
        rows.append(f"T{k:2d} {o.persona:8s} win {100*res['exp'][k]:.1f}% title {100*res['ttl'][k]:.0f}% spent ${o.spent/1e3:.0f}k"
                    f"{'' if o.plan_met() else ' (plan missed)'}: {names}")
    return rows


def room_stats(W, results):
    buckets = {"#1-5": (1, 5), "#6-15": (6, 15), "#16-30": (16, 30), "#31-60": (31, 60)}
    ratio = {b: [] for b in buckets}
    wp, wbuyer, wwin, wtitle, spread, best, second, n10, unspent, top30_unsold, plan_missed = [], Counter(), [], [], [], [], [], [], [], [], []
    per = defaultdict(lambda: dict(win=[], ttl=[], spent=[]))
    bestp = Counter()
    for res in results:
        paid = {x: p for x, w, p, t in res["sales"] if w is not None}
        for b, (lo, hi) in buckets.items():
            for x, p in paid.items():
                if lo <= W.lrank[x] <= hi:
                    ratio[b].append(p / W.lp[x])
        team_of = {x: w for x, w, p, t in res["sales"] if w is not None}
        if W.waters in team_of:
            k = team_of[W.waters]
            wp.append(paid[W.waters]); wbuyer[res["seats"][k]] += 1
            wwin.append(res["exp"][k]); wtitle.append(res["ttl"][k])
        e = sorted(res["exp"], reverse=True)
        spread.append(100 * statistics.pstdev(res["exp"])); best.append(e[0]); second.append(e[1])
        n10.append(sum(1 for t in res["ttl"] if t >= 0.10))
        unspent.append(sum(CAP - o.spent for o in res["owners"]))
        top30_unsold.append(sum(1 for u in W.pids if W.lrank[u] <= 30 and u not in team_of))
        plan_missed.append(sum(1 for o in res["owners"] if not o.plan_met()))
        for k, o in enumerate(res["owners"]):
            per[o.persona]["win"].append(res["exp"][k]); per[o.persona]["ttl"].append(res["ttl"][k]); per[o.persona]["spent"].append(o.spent)
        bestp[res["seats"][int(np.argmax(res["exp"]))]] += 1
    m = lambda xs: statistics.mean(xs) if xs else float("nan")  # noqa: E731
    return dict(ratio={b: m(r) for b, r in ratio.items()}, waters_price=m(wp), waters_buyer=wbuyer, waters_win=m(wwin),
                waters_title=m(wtitle), spread=m(spread), best=m(best), second=m(second), n10=m(n10), unspent=m(unspent),
                top30_unsold=m(top30_unsold), plan_missed=m(plan_missed),
                per={p: {k: m(v) for k, v in d.items()} for p, d in per.items()}, best_persona=bestp, n=len(results))


def run_cell(W, mix, seeds, sd_mult, premium, seasons, trace=(), jitter=0.1):
    results = []
    for s in range(seeds):
        res = run_auction(W, mix, s, sd_mult, premium, trace, jitter)
        score(W, res, seasons, np.random.default_rng(10_000 + s))
        results.append(res)
    return results


def fmt_stats(st, mixname):
    r = st["ratio"]
    wb = " ".join(f"{k}:{v}" for k, v in st["waters_buyer"].most_common())
    line = (f"{mixname:34s} | Waters ${st['waters_price']/1e3:4.0f}k ({wb}) win {100*st['waters_win']:.0f}% title {100*st['waters_title']:.0f}% | "
            f"list% {100*r['#1-5']:.0f}/{100*r['#6-15']:.0f}/{100*r['#16-30']:.0f}/{100*r['#31-60']:.0f} | "
            f"spread {st['spread']:.1f} best {100*st['best']:.0f}% 2nd {100*st['second']:.0f}% n10 {st['n10']:.1f} | "
            f"unspent ${st['unspent']/1e6:.2f}M top30-unsold {st['top30_unsold']:.1f} plan-missed {st['plan_missed']:.1f}")
    return line


def fmt_personas(st):
    return " ".join(f"{p}:{100*d['win']:.0f}%/{100*d['ttl']:.0f}%/${d['spent']/1e3:.0f}k" for p, d in sorted(st["per"].items()))


# ------------------------------------------------------------------ the sweep
def grid(W, seeds, seasons, out):
    t0 = time.time()
    cells = []
    mixes = [("default mix (2/3/6/4/3/2)", DEFAULT_MIX)]
    for p in PERSONAS:
        mixes.append((f"all {LONG[p]}", f"{p}=20"))
    for p in PERSONAS:
        if p != "four":
            mixes.append((f"one {LONG[p]} + 19 four starters", f"{p}=1,four=19"))
    mixes.append(("no star & scrubs (0/4/7/4/3/2)", "two=4,four=7,six=4,singles=3,risk=2"))
    mixes.append(("half star & scrubs (10/2/4/2/1/1)", "star=10,two=2,four=4,six=2,singles=1,risk=1"))
    for name, mix in mixes:
        st = room_stats(W, run_cell(W, parse_mix(mix), seeds, 1.0, 0.1, seasons))
        cells.append((name, 1.0, 0.1, st))
        print(fmt_stats(st, name), file=sys.stderr, flush=True)
    for sd, pr in ((2.0, 0.1), (1.0, 0.3), (2.0, 0.3)):
        st = room_stats(W, run_cell(W, parse_mix(DEFAULT_MIX), seeds, sd, pr, seasons))
        cells.append((f"default mix, sd x{sd:g}, premium {pr:g}", sd, pr, st))
        print(fmt_stats(st, cells[-1][0]), file=sys.stderr, flush=True)
    for jit in (0.0, 0.2):
        st = room_stats(W, run_cell(W, parse_mix(DEFAULT_MIX), seeds, 1.0, 0.1, seasons, jitter=jit))
        cells.append((f"default mix, share jitter {jit:g}", 1.0, 0.1, st))
        print(fmt_stats(st, cells[-1][0]), file=sys.stderr, flush=True)
    # Waters seed by seed: what sold before her decides her price
    by_seed = []
    for sd in (1.0, 2.0):
        for sdd in range(seeds):
            res = score(W, run_auction(W, parse_mix(DEFAULT_MIX), sdd, sd, 0.1), seasons, np.random.default_rng(10_000 + sdd))
            by_seed.append((sd, sdd, waters_row(W, res)))
    # one described room
    res = score(W, run_auction(W, parse_mix(DEFAULT_MIX), 0, 1.0, 0.1), seasons, np.random.default_rng(10_000))
    write_md(W, cells, res, by_seed, seeds, seasons, out, time.time() - t0)


def waters_row(W, res):
    sales = res["sales"]
    n = next(i for i, sl in enumerate(sales, 1) if sl[0] == W.waters)
    x, k, paid, t = sales[n - 1]
    before = [f"{W.name[u]} ${p/1e3:.0f}k" for u, w, p, _ in sales[:n - 1] if w is not None and p >= 300_000]
    o = res["owners"][k]
    mates = ", ".join(f"{W.name[u]} ${r['paid']/1e3:.0f}k" for r in o.roles for u in [r["pid"]] if u and u != W.waters)
    return dict(sale=n, paid=paid, buyer=res["seats"][k], before=before, win=res["exp"][k], ttl=res["ttl"][k], mates=mates)


FINDINGS = """## What the room does (read from the tables below)

1. **Her price is the persona mix and the sale order, not the cap.** With two
   star-and-scrubs owners live she sells for $835-850k; if a top-3 MAN is
   nominated first, one of them spends its star money on him -- JW Johnson,
   Ben Johns or Hayden Patriquin at $410-460k, the two-star owners' second bid
   (an owner without a sheet has no way to price a man against Anna Leigh Waters, and men's #1
   is a four-way question so the other star owner often does not bid) --
   and she then faces one $850k bid and goes for the two-star owners' plan
   money plus $5k, $390-420k. In
   a room with no star-and-scrubs owner she is a "top-15 woman" at
   $206-265k (every all-X room bar the star one), $407-440k where two-star
   owners set the second bid. The list says $769k; the no-sheet room averages
   $570k in the default mix and only reaches the max when two star owners
   happen to meet on her.
2. **New owners invert our curve.** Default mix: #1-5 sell at 68% of list, #6-15
   at 78%, #16-30 at 113%, #31-60 at 149%. Plan money is flat within a band
   (a four-starters owner pays the same ~$220k for the 2nd-best woman and the
   14th), so stars are cheap and depth is dear -- the mirror image of every
   quant room (top 5 at 101-117%, #31-60 at 51-67%). The all-star room is the
   exception: twenty owners chasing six top-3 players put #1-5 at 155%.
3. **Paying the max for her is the WORST way to own her.** $845k leaves $30k
   a slot; the five who come with it are whoever nobody else wanted
   (61-68% win, 3-15% title, seeds 0/3/5). The star owner who gets her at
   $410k spends the other $440k on $70-180k players and wins 68-77% / 12-30%;
   the two-star owner who lands her at $265k in a four-starters room wins
   88% / 65%. The quant rooms' "Waters + cheap singles specialists = 66% /
   35%" needed the MODEL to pick the specialists; an owner with $30k slots does
   not get them.
4. **Which philosophy wins a first draft:** four starters, singles-minded
   and two stars are level (58-60% in the default mix); star-and-scrubs 52%;
   risk-averse 39% (it refuses contested players and bids 20% into a room
   bidding 22%); balanced six 29% (six $170k players lose every contested
   sale). One two-star owner in a four-starters room is the biggest single
   edge in any room we have run (88% / 65% title); one star-and-scrubs owner
   there is 68% / 30%. Homogeneous rooms miss their plans 10-19 times out of
   20 (everyone's top-15 is the same 15 women).
5. **Dead dials:** the going-rate premium (0.1 vs 0.3) changes nothing --
   plan money IS what sets prices, so the going rate never sits below it.
   Share jitter 0 / 0.1 / 0.2 moves her mean price ($794k / $570k / $565k)
   only through sale order (textbook plans nominate her first and put both
   star owners on her), not the shape of the curve. Posterior fog x2 sends
   her to the star owners more often ($792k) and makes the $30k scrubs
   worse, her team 71% -> 59%.
6. **Parity:** spread 13-17 points in every no-sheet room vs 4.3-4.6 with our
   list -- not one runaway favourite but a league of mismatched rosters,
   because owners with identical plans win random members of the same band.
   Unspent money $0.2-1.4M of $20M.

CAVEATS: one bid rule (the spec's), six hand-picked shapes, no learning
within the auction (across seasons: owner_learning.py / owner_learning.md),
truthful bids, rotation nomination;
the "real singles player" and the floor-slot pickiness rules are
build-time dials (fan_owner_spec.md, Status). The room is a probe of what
a sheetless first auction does, not a forecast of MLP's.
"""


def write_md(W, cells, res, by_seed, seeds, seasons, out, secs):
    L = []
    L.append("# The new-owner auction: no price sheet in the room\n")
    L.append(f"Generated by `value_cap/fan_auction.py --grid` ({seeds} seeds per cell, {seasons} simulated seasons per "
             f"room, {secs/60:.1f} min). Owners are records-only new owners -- people who bought a team, not analysts (fan_owner_spec.md, fan_view.py; file names keep the earlier 'fan' label): one ordinal draw "
             "per owner per gender from the v2 posterior, a roster shape with budget shares, second price + $5k, "
             "rotation nomination, $1M cap / $30k floor / $850k first-buy max. Payoffs scored on the true tie model. "
             "List prices appear ONLY in the comparison columns; no owner sees them.\n")
    L.append(FINDINGS)
    L.append("## Rooms\n")
    L.append("| room | sd | premium | Waters price (buyer) | her win / title | paid as % of list #1-5 / #6-15 / #16-30 / #31-60 | spread | best / 2nd | teams ≥10% title | unspent | top-30 unsold | plans missed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, sd, pr, st in cells:
        r = st["ratio"]
        wb = ", ".join(f"{LONG[k]} {v}/{st['n']}" for k, v in st["waters_buyer"].most_common())
        L.append(f"| {name} | {sd:g} | {pr:g} | ${st['waters_price']/1e3:.0f}k ({wb}) | {100*st['waters_win']:.0f}% / {100*st['waters_title']:.0f}% | "
                 f"{100*r['#1-5']:.0f} / {100*r['#6-15']:.0f} / {100*r['#16-30']:.0f} / {100*r['#31-60']:.0f} | {st['spread']:.1f} | "
                 f"{100*st['best']:.0f}% / {100*st['second']:.0f}% | {st['n10']:.1f} | ${st['unspent']/1e6:.2f}M | {st['top30_unsold']:.1f} | {st['plan_missed']:.1f} |")
    L.append("\nSpread = 100 × sd of expected win% across the 20 teams (snake/auction rooms with our list: 4.3–4.6; "
             "market limit ≈ 7–8 with a runaway favourite). Waters' list price is $769k (tagged); first-buy max $850k.\n")
    L.append("## Which philosophy wins (mean expected win% / title share / spent, by persona)\n")
    L.append("| room | " + " | ".join(LONG[p] for p in PERSONAS) + " | most often best |")
    L.append("|---|" + "---|" * (len(PERSONAS) + 1))
    for name, sd, pr, st in cells:
        cols = []
        for p in PERSONAS:
            d = st["per"].get(p)
            cols.append(f"{100*d['win']:.0f}% / {100*d['ttl']:.0f}% / ${d['spent']/1e3:.0f}k" if d else "–")
        bp = ", ".join(f"{LONG[k]} {v}" for k, v in st["best_persona"].most_common(2))
        L.append(f"| {name} | " + " | ".join(cols) + f" | {bp} |")
    L.append("\n## Waters, seed by seed (default mix): what sold for $300k+ before her sets her price\n")
    L.append("| fog | seed | her sale # | paid | buyer | $300k+ sales before her | her team win / title | her teammates |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sd, sdd, r in by_seed:
        L.append(f"| x{sd:g} | {sdd} | {r['sale']} | ${r['paid']/1e3:.0f}k | {LONG[r['buyer']]} | {'; '.join(r['before']) or '–'} | "
                 f"{100*r['win']:.0f}% / {100*r['ttl']:.0f}% | {r['mates']} |")
    L.append("\n## One room in full (default mix, seed 0)\n")
    L.append("```")
    L += describe(W, res)
    L.append("```")
    L.append("\nSale order of the first 15 (who nominated, what was paid, list price):\n")
    L.append("```")
    seats = res["seats"]
    for n, (x, w, paid, t) in enumerate(res["sales"][:15], 1):
        who = f"T{w} [{seats[w]}]" if w is not None else "UNSOLD"
        L.append(f"{n:2d}. {W.name[x]:24s} nominated by T{t} [{seats[t]}] -> {who} ${paid/1e3:.0f}k (list ${W.lp[x]/1e3:.0f}k)")
    L.append("```")
    Path(out).write_text("\n".join(L) + "\n")
    print(f"wrote {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sd-mult", type=float, default=1.0)
    ap.add_argument("--premium", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.1, help="sd of per-owner share jitter (0 = every owner holds the textbook plan)")
    ap.add_argument("--seasons", type=int, default=300)
    ap.add_argument("--sgl-top", type=int, default=10)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--trace", default="", help="comma-separated player names to trace bids for")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default=str(HERE / "fan_auction.md"))
    A = ap.parse_args()
    W = World(sgl_top=A.sgl_top)
    trace = tuple(s.strip() for s in A.trace.split(",") if s.strip())
    if A.grid:
        grid(W, A.seeds, A.seasons, A.out)
        return
    mix = parse_mix(A.mix)
    if A.describe:
        res = score(W, run_auction(W, mix, 0, A.sd_mult, A.premium, trace, A.jitter), A.seasons, np.random.default_rng(10_000))
        for row in describe(W, res):
            print(row)
        print(f"{sum(1 for s in res['sales'] if s[1] is not None)} sold, {sum(1 for s in res['sales'] if s[1] is None)} unsold")
    results = run_cell(W, mix, A.seeds, A.sd_mult, A.premium, A.seasons, trace, A.jitter)
    st = room_stats(W, results)
    print(fmt_stats(st, A.mix))
    print("personas win%/title%/spent:", fmt_personas(st), "| best most often:", dict(st["best_persona"]))


if __name__ == "__main__":
    main()
