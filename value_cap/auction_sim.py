"""value_cap/auction_sim.py -- the same 20 owners (quants + the personas.py
personas) buying the same board at AUCTION instead of in a snake draft.

    python value_cap/auction_sim.py                       # full grid -> auction.md
    python value_cap/auction_sim.py --counts 1 --drafts 4 --seasons 100   # quick look
    python value_cap/auction_sim.py --rerender            # re-render from cache

Mechanism (a standard fantasy auction, nothing tuned):
  - owners nominate in rotation; an owner nominates the player they would
    take right now in a snake draft at the EXPECTED prices (the snake
    owners' shortlist + projection + believed tie probability, so every
    persona's beliefs and habits carry over), under the auction's own
    scarcity: before they nominate again every other team nominates once,
    so the highest-priced players are assumed gone by then -- which is
    why the stars come up while the room still has money. Anyone the
    owner can open the bidding on (floor + cheapest completion within
    budget) is nominatable, projected at the lower of the expected price
    and the most that owner could pay; ties go to the most expensive;
  - every owner who still needs that gender works out a CEILING: the price
    at which they are indifferent between "this player plus what my
    remaining money buys at expected prices" and "what my money buys
    without them" (same greedy projection + believed tie probability vs
    the reference roster that the snake owners use; bisection to ~$3k),
    hard-capped at budget minus the cheapest legal completion of the
    remaining slots -- nobody can ever be stranded -- and at any
    self-imposed limit (the bargain hunter caps their first three buys);
  - the highest ceiling wins and pays the second-highest ceiling plus one
    increment (an ascending auction resolved analytically); ties at the
    same ceiling are broken at random; the reserve is the $30k floor;
  - EXPECTED prices are the shipped list (`expect = list`) or the list
    scaled by money-left / list-value-left each nomination (`inflated`,
    the fantasy "inflation" heuristic) -- swept, not picked.
The price list is therefore only a cheat sheet here: what a player costs is
whatever the room pays. Seasons and reads are draft_sim's (double round
robin + top-4 playoff on the TRUE values; parity = 50% / 5%), plus what an
auction adds: realized prices vs the list, unspent money, who overpaid.
"""
from __future__ import annotations

import argparse
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import draft_sim as D  # noqa: E402
import personas as P  # noqa: E402
from phase2_pricing import FLOOR, NAME, POOL, pid_named, prices_tagged  # noqa: E402
sys.argv = _argv

HERE = Path(__file__).resolve().parent
INC = 5_000       # bid increment: winner pays second-highest ceiling + INC
STEPS = 8         # bisection steps for the ceiling (~$3k resolution)


# ------------------------------------------------------------------ helpers
def completion_cost(by_g, price, exclude, need):
    """Cheapest way to fill `need` from by_g (per-gender price-sorted avail)."""
    tot = 0.0
    for g in ("M", "F"):
        k = need[g]
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


def project(owner, roster, budget, price, need, order, by_g, exclude, gaps=None, by_price=None):
    """Greedy fill of the open slots (draft_sim.Owner.choose's projection):
    best-believed player that keeps the roster completable, repeated. order =
    avail sorted by the owner's believed doubles value. With gaps (the
    nomination case) the j-th fill assumes the gaps[j] highest-PRICED players
    other than the roster are already sold -- choose()'s scarcity rule;
    without it (the ceiling case) every player is assumed available at the
    expected price. Returns the projected roster or None."""
    proj = list(roster)
    left = budget
    na = dict(need)
    taken = set(roster) | set(exclude)
    j = 0
    while na["M"] > 0 or na["F"] > 0:
        gone = set()
        if gaps is not None and j < len(gaps):
            gone = set([u for u in by_price if u not in taken][:gaps[j]])
        j += 1
        pick = None
        for u in order:
            if u in taken or u in gone or na[D.GENDER[u]] <= 0:
                continue
            n2 = dict(na)
            n2[D.GENDER[u]] -= 1
            if price[u] + completion_cost(by_g, price, taken | {u}, n2) <= left + 1e-6:
                pick = u
                break
        if pick is None:
            return None
        proj.append(pick)
        taken.add(pick)
        left -= price[pick]
        na[D.GENDER[pick]] -= 1
    return proj


def ceiling(owner, x, roster, spent, price, need, order, by_g):
    """The most this owner would pay for x right now (0 = not even the floor)."""
    gx = D.GENDER[x]
    if need[gx] <= 0:
        return 0.0
    budget = owner.cap - spent
    need_after = dict(need)
    need_after[gx] -= 1
    top = budget - completion_cost(by_g, price, {x}, need_after)
    top = min(top, owner.bid_cap(x, roster))
    if top < FLOOR:
        return 0.0
    alt = project(owner, roster, budget, price, need, order, by_g, {x})
    g = owner.score(alt) if alt else -1.0

    def f(p):
        pr = project(owner, roster + [x], budget - p, price, need_after, order, by_g, set())
        return owner.score(pr) if pr else -2.0

    if f(top) >= g:
        return top
    if f(FLOOR) < g:
        return 0.0
    lo, hi = FLOOR, top
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        if f(mid) >= g:
            lo = mid
        else:
            hi = mid
    return lo


def interested(owner, x, avail_g_by_value, avail_g_by_singles, cheap_g):
    """Only owners who would shortlist x in a snake pick bid on x (same
    candidate rule as choose(): top-N believed doubles value, top singles,
    cheapest few)."""
    return x in avail_g_by_value or x in avail_g_by_singles or x in cheap_g


def expected_prices(list_price, avail, owners, spent, need, mode):
    if mode == "list":
        return list_price
    money = sum(max(0.0, o.cap - s) for o, s, n in zip(owners, spent, need) if n["M"] + n["F"] > 0)
    value = 0.0
    for g in ("M", "F"):
        k = sum(n[g] for n in need)
        tops = sorted((list_price[u] for u in avail if D.GENDER[u] == g), reverse=True)[:k]
        value += sum(tops)
    rho = money / value if value > 0 else 1.0
    return {u: (FLOOR if list_price[u] <= FLOOR else max(FLOOR, rho * list_price[u])) for u in list_price}


def nominate(owner, roster, spent, avail, price, need, by_g, others_active):
    """The owner's snake pick, transplanted: shortlist as choose() does (top
    believed doubles value per needed gender, top singles, cheapest few;
    persona filter), project candidate + greedy fill under the auction's own
    scarcity -- before this owner nominates again every other active team
    nominates once, so the (j+1)*others_active highest-priced players are
    assumed sold before the j-th fill -- and take the best believed tie
    probability; ties go to the most expensive candidate, pid last.
    Two auction-specific rules: a candidate only has to be OPENABLE (the
    floor plus the cheapest completion fits the budget), and is projected
    at the lower of its expected price and the most this owner could pay
    for it -- an owner short of money still puts the star up, and the room
    decides. Returns None when nothing is openable."""
    budget = owner.cap - spent
    order = sorted(avail, key=lambda u: (-owner.v[u], u))
    by_price = sorted(avail, key=lambda u: (-price[u], u))
    gaps = [(j + 1) * others_active for j in range(D.ROUNDS)]
    cands = set()
    cost = {}
    for g in ("M", "F"):
        if need[g] <= 0:
            continue
        need_after = dict(need)
        need_after[g] -= 1
        pool_g = []
        for u in order:
            if D.GENDER[u] != g:
                continue
            top = budget - completion_cost(by_g, price, {u}, need_after)
            if top + 1e-6 >= FLOOR:
                pool_g.append(u)
                cost[u] = min(price[u], top)
        cands.update(pool_g[:D.CAND_TOP])
        cands.update(sorted(pool_g, key=lambda u: (-owner.s[u], u))[:D.CAND_SINGLES])
        cands.update(sorted(pool_g, key=lambda u: (price[u], u))[:D.CAND_CHEAP])
    cands = owner.filter_cands(cands, roster, avail, price)
    best = None
    for x in sorted(cands):
        if x not in cost:
            continue
        need_after = dict(need)
        need_after[D.GENDER[x]] -= 1
        proj = project(owner, roster + [x], budget - cost[x], price, need_after, order, by_g, set(),
                       gaps=gaps, by_price=[u for u in by_price if u != x])
        if proj is None:
            continue
        key = (round(owner.score(proj), 6), min(price[x], owner.bid_cap(x, roster)), x)
        if best is None or key > best[0]:
            best = (key, x)
    return best[1] if best else None


# ------------------------------------------------------------------ auction
def run_auction(list_price, noise, rng, owners=None, expect="list", inc=INC, gamma=None):
    owners = owners or [D.Owner(noise, rng, gamma) for _ in range(D.N_TEAMS)]
    avail = set(D.BOARD)
    rosters = [[] for _ in range(D.N_TEAMS)]
    spent = [0.0] * D.N_TEAMS
    need = [{"M": D.PER_GENDER, "F": D.PER_GENDER} for _ in range(D.N_TEAMS)]
    picks = {}          # pid -> (team, price paid, order of sale)
    stranded = 0
    nbids = []          # bidders per sale (info)
    sale = 0
    turn = 0
    while any(n["M"] + n["F"] > 0 for n in need):
        t = turn % D.N_TEAMS
        turn += 1
        if need[t]["M"] + need[t]["F"] <= 0:
            continue
        price = expected_prices(list_price, avail, owners, spent, need, expect)
        by_g = {g: sorted((u for u in avail if D.GENDER[u] == g), key=lambda u: (price[u], u)) for g in ("M", "F")}
        # nomination = the owner's snake pick under the auction's own scarcity (stars while the room has money)
        others_active = sum(1 for k in range(D.N_TEAMS) if k != t and need[k]["M"] + need[k]["F"] > 0)
        x = nominate(owners[t], rosters[t], spent[t], avail, price, need[t], by_g, others_active)
        if x is None:
            # stranded: nothing affordable -- league assigns the cheapest player of a needed gender at the floor
            g = "M" if need[t]["M"] > 0 else "F"
            x = by_g[g][0]
            stranded += 1
            _assign(t, x, FLOOR, rosters, spent, need, avail, picks, sale)
            sale += 1
            continue
        gx = D.GENDER[x]
        shortlist = {}
        bids = []
        for k, o in enumerate(owners):
            if need[k][gx] <= 0:
                continue
            order = sorted(avail, key=lambda u: (-o.v[u], u))
            gord = [u for u in order if D.GENDER[u] == gx]
            top_v = set(gord[:D.CAND_TOP])
            top_s = set(sorted(gord, key=lambda u: (-o.s[u], u))[:D.CAND_SINGLES])
            cheap = set(by_g[gx][:D.CAND_CHEAP])
            if k != t and not interested(o, x, top_v, top_s, cheap):
                continue
            b = ceiling(o, x, rosters[k], spent[k], price, need[k], order, by_g)
            if k == t:
                # the nomination is an opening bid at the floor if that is feasible
                need_after = dict(need[k])
                need_after[gx] -= 1
                if FLOOR + completion_cost(by_g, price, {x}, need_after) <= o.cap - spent[k] + 1e-6:
                    b = max(b, FLOOR)
            if b >= FLOOR:
                bids.append((b, rng.random(), k))
        if not bids:
            # nobody wants x even at the floor; give the nominator the cheapest legal player instead
            x = by_g[gx][0]
            stranded += 1
            _assign(t, x, FLOOR, rosters, spent, need, avail, picks, sale)
            sale += 1
            continue
        bids.sort(key=lambda b: (-b[0], b[1]))
        b1, _, winner = bids[0]
        b2 = bids[1][0] if len(bids) > 1 else FLOOR
        paid = min(b1, max(FLOOR, b2 + inc))
        paid = round(paid / 1000.0) * 1000.0
        paid = min(paid, b1)
        nbids.append(len(bids))
        _assign(winner, x, paid, rosters, spent, need, avail, picks, sale)
        sale += 1
    return [tuple(r) for r in rosters], spent, picks, avail, owners, stranded, nbids


def _assign(t, x, paid, rosters, spent, need, avail, picks, sale):
    rosters[t].append(x)
    spent[t] += paid
    need[t][D.GENDER[x]] -= 1
    avail.discard(x)
    picks[x] = (t, paid, sale)


def run_variant(list_price, noise, drafts, seasons, seed, stars, owner_factory=None, expect="list", inc=INC):
    rng = random.Random(seed)
    kinds = {}
    slot_exp = [[] for _ in range(D.N_TEAMS)]
    slot_title = [[] for _ in range(D.N_TEAMS)]
    star_rows = {u: {"rounds": [], "undrafted": 0, "exp": [], "title": [], "slot": [], "paid": [],
                     "mates_paid": [], "mates_floor": []} for u in stars}
    undrafted = {}
    shapes = {}
    paid = {}           # pid -> [price paid per draft it was sold]
    spends, spreads, maxes, floor_taken, unspent, strandeds, nb = [], [], [], [], [], [], []
    t0 = time.time()
    for d in range(drafts):
        owners = owner_factory(rng) if owner_factory else None
        rosters, spent, picks, left, owners, stranded, nbids = run_auction(list_price, noise, rng, owners, expect, inc)
        exp, mw, ttl = D.season(rosters, seasons, rng)
        for t in range(D.N_TEAMS):
            k = kinds.setdefault(owners[t].name, {"exp": [], "title": [], "spend": [], "picks": {}, "n": 0, "paid": {}})
            k["exp"].append(exp[t])
            k["title"].append(ttl[t])
            k["spend"].append(spent[t])
            k["n"] += 1
            for u in rosters[t]:
                k["picks"][u] = k["picks"].get(u, 0) + 1
                k["paid"].setdefault(u, []).append(picks[u][1])
        spreads.append(statistics.pstdev(exp))
        maxes.append(max(exp))
        spends.extend(spent)
        unspent.append(sum(owners[t].cap - spent[t] for t in range(D.N_TEAMS)))
        strandeds.append(stranded)
        nb.append(statistics.mean(nbids) if nbids else 0)
        for t in range(D.N_TEAMS):
            slot_exp[t].append(exp[t])
            slot_title[t].append(ttl[t])
            pr = {u: picks[u][1] for u in rosters[t]}
            b = D.blueprint(rosters[t], pr)
            shapes[b] = shapes.get(b, 0) + 1
        for u, (team, pp, order) in picks.items():
            paid.setdefault(u, []).append(pp)
        for u in stars:
            if u in picks:
                team, pp, order = picks[u]
                star_rows[u]["rounds"].append(order + 1)
                star_rows[u]["exp"].append(exp[team])
                star_rows[u]["title"].append(ttl[team])
                star_rows[u]["slot"].append(team + 1)
                star_rows[u]["paid"].append(pp)
                mates = [v for v in rosters[team] if v != u]
                star_rows[u]["mates_paid"].append(sum(picks[v][1] for v in mates))
                star_rows[u]["mates_floor"].append(sum(1 for v in mates if picks[v][1] <= FLOOR + 1e-6))
            else:
                star_rows[u]["undrafted"] += 1
        for u in left:
            if u in D.POOL_SET:
                undrafted[u] = undrafted.get(u, 0) + 1
        floor_taken.append(sum(1 for u in picks if u not in D.POOL_SET))
    return dict(fmt="auction", expect=expect, noise=noise, drafts=drafts, seasons=seasons, secs=time.time() - t0,
                slot_exp=slot_exp, slot_title=slot_title, stars=star_rows, undrafted=undrafted, shapes=shapes,
                spend=statistics.mean(spends), spread=statistics.mean(spreads), max_exp=statistics.mean(maxes),
                floor_taken=statistics.mean(floor_taken), kinds=kinds, paid=paid,
                unspent=statistics.mean(unspent), stranded=statistics.mean(strandeds), bidders=statistics.mean(nb))


# --------------------------------------------------------------- experiment
def run(args):
    D.set_board(args.board)
    waters = pid_named("Anna Leigh Waters")
    list_price = prices_tagged(POOL, 1.0, waters, "joint")
    list_price = {u: list_price.get(u, FLOOR) for u in D.BOARD}
    stars = [waters, pid_named("Anna Bright"), pid_named("Ben Johns")]
    rows = []
    # reference: the snake draft with the same quants (personas.md baseline settings)
    snake = D.run_variant(list_price, "snake", args.noise, args.drafts, args.seasons, args.seed, stars,
                          owner_factory=P.factory_for(P.Quant, {}, 0, args.noise))
    rows.append(("snake draft, quants", "", 0, "list", snake))
    print(f"snake: spread {100*snake['spread']:.1f}, max {100*snake['max_exp']:.1f}%, {snake['secs']:.0f}s", file=sys.stderr)
    for expect in args.expect:
        for noise in sorted({0.0, args.noise}) if args.perfect else [args.noise]:
            # an auction is random even at noise 0 (ties at the same ceiling), so always average
            r = run_variant(list_price, noise, args.drafts, args.seasons, args.seed, stars,
                            owner_factory=P.factory_for(P.Quant, {}, 0, noise), expect=expect)
            rows.append(("auction, quants", f"noise {noise:.0%}", 0, expect, r))
            w = r["stars"][waters]
            print(f"auction {expect} noise {noise:.0%}: Waters ${statistics.mean(w['paid'])/1e3:.0f}k team "
                  f"{100*statistics.mean(w['exp']):.1f}%; spread {100*r['spread']:.1f}; unspent ${r['unspent']/1e3:.0f}k; "
                  f"{r['secs']:.0f}s", file=sys.stderr)
    for label, cls, key, vals in P.PERSONAS:
        if args.only and label not in args.only:
            continue
        for val in vals:
            for count in args.counts:
                for expect in args.expect:
                    r = run_variant(list_price, args.noise, args.drafts, args.seasons, args.seed, stars,
                                    owner_factory=P.factory_for(cls, {key: val}, count, args.noise), expect=expect)
                    rows.append((label, P.fmt_strength(key, val), count, expect, r))
                    k = r["kinds"].get(label)
                    w = r["stars"][waters]
                    print(f"{label} {P.fmt_strength(key, val)} x{count} {expect}: persona win {100*statistics.mean(k['exp']):.1f}% "
                          f"title {100*statistics.mean(k['title']):.1f}% spend ${statistics.mean(k['spend'])/1e3:.0f}k; "
                          f"Waters ${statistics.mean(w['paid'])/1e3 if w['paid'] else 0:.0f}k; spread {100*r['spread']:.1f}; "
                          f"{r['secs']:.0f}s", file=sys.stderr)
    return list_price, stars, rows


def tier_ratios(r, list_price):
    """Mean realized/list price by pool-rank tier (priced players only)."""
    tiers = [("#1-5", 1, 5), ("#6-15", 6, 15), ("#16-30", 16, 30), ("#31-60", 31, 60)]
    out = {}
    for g in ("F", "M"):
        for lab, lo, hi in tiers:
            us = [u for u in r["paid"] if u in D.POOL_SET and D.GENDER[u] == g and lo <= D.POOL_RANK[u] <= hi]
            if not us:
                continue
            ratio = [statistics.mean(r["paid"][u]) / list_price[u] for u in us]
            out[(g, lab)] = (statistics.mean(ratio), len(us))
    return out


def movers(r, list_price, n=6):
    rows = []
    for u, ps in r["paid"].items():
        if u in D.POOL_SET and len(ps) >= max(1, r["drafts"] // 2) and list_price[u] > FLOOR:
            rows.append((statistics.mean(ps) / list_price[u], u, statistics.mean(ps)))
    rows.sort()
    disc = rows[:n]
    prem = rows[-n:][::-1]
    fmt = lambda x: ", ".join(f"{NAME[u]} ${p/1e3:,.0f}k vs ${list_price[u]/1e3:,.0f}k ({100*(k-1):+.0f}%)" for k, u, p in x)
    return fmt(prem), fmt(disc)


READS = """## What this says (hand-written against the seed-1 grid; re-check the numbers above if it is re-run)

- **The auction does not dent the Waters team.** She sells at the cap's maximum
  ($850k = cap minus five floor players) at sale 1 in every quant cell and every
  seed; her buyer is left with $150k for five slots and takes five floor players;
  the team still wins 67-68% of ties with a 33-39% title shot (snake: 66% / 36%).
  The snake gave her buyer floor-priced players too, so nothing is lost. The
  one price-side lever from `dials.md` (charge her more than the list) is what
  the room does on its own, and it is not enough: at $850k she is still the best
  buy in the league. Seeds 2 and 3 agree (67-69% / 32-40%).
- **But the auction makes a chase.** Runner-up title odds 11.5-16% (snake 7.3%),
  1.8-3.0 teams at 10%+ (snake 1.0), effective contenders 5.2-5.8 (snake 6.4,
  i.e. the pack is LESS equal). The 10%+ teams are almost all the same build:
  two players at $390-490k plus four floor players -- Patriquin + Rohrabacher,
  Jorja Johnson + Alshon, Todd + Staksrud, JW Johnson + Humberg, Fahey + Black.
  A man and a woman is the strong version: the tie model puts the two stars
  together in one mixed game (near-lock), and each carries a same-gender game
  with a floor partner at about even odds, so a star plays in three of the four
  games (Patriquin + Rohrabacher vs the field: WD 48%, MD 66%, MXD1 74%, MXD2
  46%, tie 62%; the same roster with Tardio in place of Rohrabacher, 45%; with
  Fahey in place of Patriquin, 59%). A snake cannot build it -- the pick-2 team
  waits until pick 39 and the top 60 are gone before round 4 -- so every snake
  team but Waters' is "one star plus depth", and nineteen of those are equals.
  Money is the only constraint at auction, and the cap rewards concentration.
  Nobody programmed the build; it falls out of owners who value rosters, not
  players (their objective is the projected roster's tie probability).
- **The room re-prices the list's middle.** Stars (#1-5) go at 101-115% of
  list, the #6-15 tier at 111-130% (the second star of a two-star build; the
  biggest single premiums are the $130-210k men -- Bhatia, Howells, Frazier,
  Huynh, Garnett -- at +40-65%), #16-30 at about list, and the #31-60 depth at
  51-67%: the $79-96k role players (Rane, Van Reek, Dunlap, Brascia, Petrei)
  sell for the floor. The room pays for a second star and for fit, and not
  for depth, because the winning build has four floor slots. The DreamBreaker
  specialists the list floors go at 3-6x (Joseph $98k, Haworth $185k on the
  rosters that want them): phi is a context average and cannot see fit; a
  room prices one context. Every cap is spent ($999-1,000k), nothing in the
  top 30 is left unsold, ~14-15 bidders per sale, nobody stranded.
- **Expectations and noise are second order** (the earlier sensitivity was an
  artefact of a nomination rule that let stars come up after the money was
  gone). Owners who anticipate inflation give Waters a slightly smaller title
  share (33-37% vs 37-39%) and the runner-up more (11.5-16%); 10% belief noise
  raises the stars' prices ~8% (Bright $654-675k vs $610-622k) and trims the
  runner-up (11.5-11.7% vs 14.8-16%): noisy owners overpay the stars, which
  squeezes the two-star budgets.
- **Personas: the auction forgives individual mistakes and punishes shared
  ones.** Alone, the $500k cheapskate wins 27-29% (snake 21%), bargains-first
  at $120k 34-36% (snake 24%) -- a bad opening no longer costs the whole top 60,
  because anyone can be bought at any time. Five cheapskates still break the
  pack (spread 11.5-12.7, the quants at 56-57%). Overvaluing a gender, chasing
  names and mild loyalty are free, as in the snake (overvalues-women k=1 alone
  is again mildly AHEAD, 53% / 7-8%: women carry mixed). Strong loyalty
  (lam 0.5) costs 1-3 points alone and, as a league norm, overpays the known
  stars (Bright $705-739k, Johns $577-598k) and gives the widest chase in the
  grid (runner-up 17-18%, 3 teams at 10%+).
- **The only leagues where Waters goes cheaper are the ones where everyone
  shares the same blind spot, and they are worse leagues.** Twenty owners who
  overvalue men (k=1) sell her at $759-779k and her team wins 74-75% with a 50%
  title share; twenty bargain hunters ($250k threshold) spend rounds 1-3 on
  mid-priced players and then let the stars go for half price (Waters
  $515-547k, Bright $440-499k, Johns $283-396k): her team 84-88%, favourite
  56-57%, parity spread 15-16, worse than anything the snake produced. Twenty
  $500k cheapskates hit their own $350k first-buy maximum on Waters (Bright
  $328-350k, Johns $260-350k) and get a 43% favourite with a runner-up at
  10-22%. The cap maximum binds in every mixed league: whatever the room, she
  costs what a team may pay.
- **For the price list**: the room's curve is more convex than phi's -- a
  premium on the second star and on fit, the floor for depth. That is not a
  reason to change the list (the list prices context-averaged value; a room
  prices fit and liquidity on the day) but it is the shape to expect in the
  league-price / surplus column once MLP publishes, and it says the list's
  $60-100k depth tier is where the surplus will look largest.
"""


def render(list_price, stars, rows, args, out):
    W, B, J = stars
    L = ["# Auction draft -- the same owners and personas, prices set by the room", "",
         f"Shipped tag list as the cheat sheet (Waters listed at ${list_price[W]/1e3:,.0f}k), `{args.board}` board, "
         f"20 teams, $1M cap, 3M+3W, {args.drafts} auctions x {args.seasons} seasons per cell, seed {args.seed}, "
         f"every owner at {args.noise:.0%} belief noise unless stated. The mechanism is in the docstring of "
         f"`auction_sim.py`: nominate your snake pick, bid up to your indifference price, pay the second-highest "
         f"ceiling + ${INC/1e3:.0f}k, ceilings hard-capped at budget minus the cheapest legal completion "
         f"(${(D.TEAM_CAP - 5*FLOOR)/1e3:,.0f}k for a first buy on this board). `expect` = the prices owners assume "
         f"for the players they have not bought yet: the list, or the list inflated by money-left / value-left. "
         f"Parity = 50% win, 5% title. Built by `auction_sim.py`.", "",
         "## Headline: auction vs snake with twenty quants", "",
         "| format | expect | owner noise | Waters paid | sold at sale # | her other five cost (at floor) | her team win% | title% | "
         "Bright paid | her team | Johns paid | his team | "
         "parity spread | favourite title | runner-up | teams >= 10% | effective contenders | mean spend | unspent per team | top-30 undrafted |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def star(r, u):
        s = r["stars"][u]
        if not s["exp"]:
            return "not sold", "--", "--"
        pd = f"${statistics.mean(s['paid'])/1e3:,.0f}k" if s.get("paid") else f"${list_price[u]/1e3:,.0f}k (list)"
        return pd, f"{100*statistics.mean(s['exp']):.1f}%", f"{100*statistics.mean(s['title']):.1f}%"

    for label, strength, count, expect, r in rows:
        if count:
            continue
        c = P.concentration(r)
        und = [u for u, cc in r["undrafted"].items() if D.POOL_RANK[u] <= 30]
        wp, we, wt = star(r, W)
        bp, be, _ = star(r, B)
        jp, je, _ = star(r, J)
        nz = strength or f"noise {r['noise']:.0%}"
        sw = r["stars"][W]
        mates = (f"${statistics.mean(sw['mates_paid'])/1e3:,.0f}k ({statistics.mean(sw['mates_floor']):.1f} of 5)"
                 if sw.get("mates_paid") else "--")
        sale_no = ((f"{statistics.mean(sw['rounds']):.1f}" if r["fmt"] == "auction" else f"pick {statistics.mean(sw['rounds']):.0f}")
                   if sw.get("rounds") else "--")
        L.append(f"| {label} | {expect if r['fmt'] == 'auction' else '--'} | {nz} | {wp} | {sale_no} | {mates} | {we} | {wt} | "
                 f"{bp} | {be} | {jp} | {je} | {100*r['spread']:.1f} pts | {100*c['top']:.1f}% | {100*c['second']:.1f}% | "
                 f"{c['n10']:.1f} | {c['eff']:.1f} | ${r['spend']/1e3:,.0f}k | "
                 f"${r.get('unspent', 0)/1e3/D.N_TEAMS:,.0f}k | "
                 + (", ".join(f"{NAME[u]} #{D.POOL_RANK[u]}{D.GENDER[u]}" for u in sorted(und, key=lambda u: D.POOL_RANK[u])[:4])
                    + (f" (+{len(und)-4})" if len(und) > 4 else "") if und else "none") + " |")
    L += ["", "## Price discovery: what the room pays vs the list", "",
          "Mean realized price / list price by pool-rank tier (priced players sold in at least one auction), "
          "quant-only auctions.", "",
          "| expect | owner noise | women #1-5 | #6-15 | #16-30 | #31-60 | men #1-5 | #6-15 | #16-30 | #31-60 | bidders per sale | stranded per auction |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, strength, count, expect, r in rows:
        if count or r["fmt"] != "auction":
            continue
        tr = tier_ratios(r, list_price)
        cells = " | ".join(f"{100*tr[(g, t)][0]:.0f}%" if (g, t) in tr else "--"
                           for g in ("F", "M") for t in ("#1-5", "#6-15", "#16-30", "#31-60"))
        L.append(f"| {expect} | {strength} | {cells} | {r['bidders']:.1f} | {r['stranded']:.1f} |")
    for label, strength, count, expect, r in rows:
        if count or r["fmt"] != "auction":
            continue
        prem, disc = movers(r, list_price)
        L += ["", f"- **{expect}, {strength}** -- biggest premiums: {prem}.", f"  Biggest discounts: {disc}."]
    L += ["", "## Personas at auction", "",
          "Same persona definitions and strengths as `personas.md`; k persona owners at random seats among quants.", "",
          "| persona | strength | how many of 20 | expect | persona teams: win% | title% | spend | quant teams: win% | title% | "
          "parity spread | Waters paid | her team win% | Bright paid | Johns paid | favourite title | runner-up | teams >= 10% | "
          "effective contenders | top-30 undrafted |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, strength, count, expect, r in rows:
        if not count:
            continue
        k = r["kinds"].get(label)
        q = r["kinds"].get("quant")
        c = P.concentration(r)
        und = [u for u, cc in r["undrafted"].items() if D.POOL_RANK[u] <= 30]
        wp, we, wt = star(r, W)
        bp, _, _ = star(r, B)
        jp, _, _ = star(r, J)
        L.append(f"| {label} | {strength} | {count} | {expect} | {100*statistics.mean(k['exp']):.1f}% | "
                 f"{100*statistics.mean(k['title']):.1f}% | ${statistics.mean(k['spend'])/1e3:,.0f}k | "
                 + (f"{100*statistics.mean(q['exp']):.1f}% | {100*statistics.mean(q['title']):.1f}% | " if q else "-- | -- | ")
                 + f"{100*r['spread']:.1f} pts | {wp} | {we} | {bp} | {jp} | {100*c['top']:.1f}% | {100*c['second']:.1f}% | "
                 f"{c['n10']:.1f} | {c['eff']:.1f} | "
                 + (", ".join(f"{NAME[u]} #{D.POOL_RANK[u]}{D.GENDER[u]}" for u in sorted(und, key=lambda u: D.POOL_RANK[u])[:4])
                    + (f" (+{len(und)-4})" if len(und) > 4 else "") if und else "none") + " |")
    L += ["", "## Who each persona buys, and at what premium", "",
          "k = 1 cells, `list` expectation: players the persona carries more often than the quants in the same room "
          "(share of persona rosters minus share of quant rosters), with what the persona paid vs the list.", ""]
    for label, strength, count, expect, r in rows:
        if count != 1 or expect != "list":
            continue
        k = r["kinds"][label]
        q = r["kinds"]["quant"]
        diff = {u: 100 * (c / k["n"] - q["picks"].get(u, 0) / q["n"]) for u, c in k["picks"].items()}
        top = sorted(diff.items(), key=lambda kv: (-kv[1], list_price[kv[0]]))[:6]
        L.append(f"- **{label}**, {strength}: " + ", ".join(
            f"{NAME[u]} (+{d:.0f}pp, paid ${statistics.mean(k['paid'][u])/1e3:,.0f}k vs ${list_price[u]/1e3:,.0f}k list)"
            for u, d in top if d > 0))
    L += ["", READS, ""]
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
    ap.add_argument("--expect", nargs="+", default=["list", "inflated"])
    ap.add_argument("--perfect", action="store_true", help="also run the quant auction at noise 0")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE / "auction.md"))
    ap.add_argument("--rerender", action="store_true")
    args = ap.parse_args(_argv[1:])
    cache = HERE / "cache" / "auction_rows.pkl"
    if args.rerender:
        list_price, stars, rows = pickle.loads(cache.read_bytes())
    else:
        list_price, stars, rows = run(args)
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps((list_price, stars, rows)))
    render(list_price, stars, rows, args, args.out)


if __name__ == "__main__":
    main()
