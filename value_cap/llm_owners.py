"""llm_owners.py -- LLM agents as auction owners (the cheap probe, 2026-09-06).

Every room in fan_auction.py / owner_learning.py runs on owners WE wrote; the
persona grids therefore restate their own inputs. This probe replaces the
hand-coded owner with a language-model agent that never sees our list: it
gets the rules, a persona in plain words, and a BLIND records-only player
table (anonymous ids shuffled within gender, 2026 doubles records by
division, singles record, MLP usage), and writes ONE valuation sheet -- the
most it would pay for each player -- plus a short strategy. The sheets are
the only thing that costs money; the auction engine below then runs them
through as many rooms as we like for free (sale order is first-order, so
that is where the variance lives).

    python value_cap/llm_owners.py --packet            # write packet.md + hidden key
    python value_cap/llm_owners.py --analyze DIR       # sheets in DIR/*.json -> report
    python value_cap/llm_owners.py --rooms DIR --seeds 200 --seasons 200

Sheet format (what an agent writes): {"persona": str, "strategy": str,
"ceilings": {"W07": 250000, ...}} -- every id present, 0 = will not bid.

Engine (deliberately the fan_auction rules, so the two rooms compare):
rotation nomination (nominator names its highest-ceiling affordable player
it still has a slot for), everyone with a slot bids its ceiling clipped to
the hard cap (budget - FLOOR x remaining other slots; $850k first buy),
second price + $5k, floor $30k; a team that runs out of ceilings fills at
the floor. Payoffs on the TRUE tie model via draft_sim.season. This is a
probe: 20 sheets, one model, one prompt; read the disagreement numbers
before reading anything downstream.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fan_view as F  # noqa: E402
import draft_sim as D  # noqa: E402

OUT = HERE / "llm_owners"
CAP, FLOOR, INC, N_TEAMS = 1_000_000.0, 30_000.0, 5_000.0, 20
SLOTS = {"F": 3, "M": 3}

RULES = """\
LEAGUE RULES (all confirmed for this exercise)
- 20 teams. Every team has exactly $1,000,000 to spend and must finish the auction with a roster of exactly
  6 players: 3 women and 3 men. Minimum price for any player is $30,000.
- A tie (one team vs another) is 4 doubles games: women's doubles, men's doubles, and two mixed doubles games
  (each of your 3 women pairs with a man in mixed; only 4 of your 6 players play in a given tie -- 2 women and
  2 men -- and the two mixed games use those same 4 players). If the games split 2-2 the tie is decided by a
  DreamBreaker: rally-scoring singles to 21 where each team rotates all 4 of its chosen players through
  4-point singles shifts. Singles ability matters ONLY in the DreamBreaker.
- Season: every team plays every other team twice, top 4 make a playoff, one champion.
- Auction: players are nominated one at a time. Each interested owner has a private maximum. The player goes
  to the highest maximum at the second-highest maximum plus $5,000. You must always keep enough money to fill
  your remaining slots at $30,000 each. Nobody sees anybody else's numbers.
- Players' true quality is whatever it is; you only have the records below. No injuries or trades this season.
"""

PERSONAS = {
    "star_scrubs": "You believe titles come from ONE superstar. Plan: spend the overwhelming majority of the budget on the single best player available (either gender) and fill the other five slots as cheaply as possible.",
    "two_stars": "You believe in two stars: the best woman and the best man you can get, roughly 40% of the budget each, and four cheap fill-ins. The two stars can pair together in mixed doubles.",
    "four_starters": "You believe four good players beat one great one: two solid women and two solid men who will actually play every tie, roughly a fifth of the budget each, plus two cheap bench players.",
    "balanced_six": "You believe depth and balance win: six good players at roughly equal prices, nobody expensive, no weak link anywhere on the roster.",
    "singles_minded": "You believe the DreamBreaker decides close ties, so at least one of your six must be a genuinely strong singles player (someone who actually plays singles and wins), even if their doubles record is ordinary. Otherwise build four solid starters and two cheap bench players.",
    "risk_averse": "You hate surprises. You want four solid starters plus a real bench (two players good enough to start), and you avoid players who missed a lot of their MLP team's matchups in 2026 -- you read low usage as injury or bench risk and discount it heavily.",
    "free": "No particular philosophy: build the team you think wins the most ties for the money.",
}
# 20 seats: 3 of each shaped persona + 2 unprompted
SEATS = [p for p in ("star_scrubs", "two_stars", "four_starters", "balanced_six", "singles_minded", "risk_averse") for _ in range(3)] + ["free", "free"]


# ------------------------------------------------------------------ the blind packet
def build_packet(seed=7):
    D.set_board("mlp2026")
    pids = list(D.BOARD)
    rng = np.random.default_rng(seed)
    key = {}
    for g, tag in (("F", "W"), ("M", "M")):
        gp = [u for u in pids if F.GENDER[u] == g]
        rng.shuffle(gp)
        for i, u in enumerate(gp):
            key[f"{tag}{i+1:02d}"] = u
    dbl = F.doubles_records(); s26 = F.singles_records("2026"); sc = F.singles_records(None); use = F.mlp_usage()

    def rec(r):
        return f"{r[0]}-{r[1]-r[0]}" if r[1] else "none"

    def pts(r):
        return f"{100*r[2]/(r[2]+r[3]):.0f}%" if r[2] + r[3] else "  -"

    lines = ["| id | 2026 same-gender doubles W-L | pts won | 2026 mixed doubles W-L | pts won | 2026 singles W-L | career singles W-L | MLP 2026 matchups played / team's matchups |",
             "|---|---|---|---|---|---|---|---|"]
    for aid in sorted(key, key=lambda a: (a[0], int(a[1:]))):
        u = key[aid]; d = dbl.get(u, {})
        own = d.get("womens" if aid[0] == "W" else "mens", [0, 0, 0, 0]); mx = d.get("mixed", [0, 0, 0, 0])
        us = use.get(u)
        lines.append(f"| {aid} | {rec(own)} | {pts(own)} | {rec(mx)} | {pts(mx)} | {rec(s26.get(u, [0, 0]))} | "
                     f"{rec(sc.get(u, [0, 0]))} | {f'{us[0]}/{us[1]}' if us else 'did not play MLP'} |")
    OUT.mkdir(exist_ok=True)
    (OUT / "packet.md").write_text(RULES + "\nPLAYER TABLE (pro doubles + singles records, 2026 season unless stated; ids are anonymous and in random order)\n"
                                   + "\n".join(lines) + "\n")
    with (OUT / "key.json").open("w") as fh:
        json.dump(key, fh, indent=0)
    print(f"packet: {len(key)} players ({sum(a[0]=='W' for a in key)} W / {sum(a[0]=='M' for a in key)} M) -> {OUT/'packet.md'}; key -> key.json")


def load_key():
    return json.load((OUT / "key.json").open())


def list_prices():
    lp = {}
    for r in csv.DictReader((HERE / "price_list.csv").open()):
        lp[r["player_id"].lower()] = float(r["price"])
    return lp


# ------------------------------------------------------------------ sheets
def load_sheets(d):
    sheets = []
    for p in sorted(Path(d).glob("*.json")):
        try:
            s = json.load(p.open())
            c = {k: float(v) for k, v in s["ceilings"].items()}
        except Exception as e:  # noqa: BLE001
            print(f"skip {p.name}: {e}"); continue
        # units guard: one owner wrote "240" meaning $240k; the rules quote
        # dollars, so anything with a max under $5k is read as thousands.
        if c and max(c.values()) < 5_000:
            c = {k: v * 1_000 for k, v in c.items()}; s["units_fixed"] = True
        s["ceilings"] = c; s["file"] = p.name; sheets.append(s)
    return sheets


def _rank(x):
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=float)
    # average ties
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]: j += 1
        if j > i: r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(d):
    key = load_key(); ids = sorted(key, key=lambda a: (a[0], int(a[1:])))
    lp = list_prices(); name = F.NAME; gender = {a: F.GENDER[key[a]] for a in ids}
    sheets = load_sheets(d)
    print(f"{len(sheets)} sheets")
    missing = [(s["file"], len([a for a in ids if a not in s["ceilings"]])) for s in sheets]
    print("missing ids per sheet:", [m for m in missing if m[1]])
    M = np.array([[s["ceilings"].get(a, 0.0) for a in ids] for s in sheets])
    # what each owner spends if it got its plan: sum of top-3 ceilings per gender
    for s, row in zip(sheets, M):
        top = {g: sorted([row[i] for i, a in enumerate(ids) if gender[a] == g], reverse=True)[:3] for g in "FM"}
        s["plan_top6"] = sum(top["F"]) + sum(top["M"])
        s["n_bid"] = int((row > 0).sum())
    print(f"\n{'file':34s} {'persona':15s} {'#priced':>7s} {'max':>6s} {'top6 sum':>9s}")
    for s, row in zip(sheets, M):
        print(f"{s['file']:34s} {s.get('persona','?'):15s} {s['n_bid']:7d} {row.max()/1e3:5.0f}k {s['plan_top6']/1e3:8.0f}k")
    # disagreement: pairwise Spearman on ceilings within gender
    def pair_rho(i, j, g):
        idx = [k for k, a in enumerate(ids) if gender[a] == g]
        return spearman(M[i, idx], M[j, idx])
    same, diff = defaultdict(list), defaultdict(list)
    for i in range(len(sheets)):
        for j in range(i + 1, len(sheets)):
            for g in "FM":
                r = pair_rho(i, j, g)
                (same if sheets[i].get("persona") == sheets[j].get("persona") else diff)[g].append(r)
    print("\nDISAGREEMENT (pairwise Spearman of ceilings within gender; 1.0 = twenty copies of one owner)")
    for g in "FM":
        print(f"  {g}: same persona median {np.median(same[g]):.2f} [{np.min(same[g]):.2f},{np.max(same[g]):.2f}]  "
              f"different persona median {np.median(diff[g]):.2f} [{np.min(diff[g]):.2f},{np.max(diff[g]):.2f}]")
    # vs our list (the contamination check AND the substantive comparison)
    print("\nVS THE SHIPPED LIST (Spearman within gender; the agents never saw it)")
    for g in "FM":
        idx = [k for k, a in enumerate(ids) if gender[a] == g]
        lpv = np.array([lp.get(key[ids[k]], FLOOR) for k in idx])
        rhos = [spearman(M[i, idx], lpv) for i in range(len(sheets))]
        mean_rho = spearman(M[:, idx].mean(0), lpv)
        print(f"  {g}: per-owner median {np.median(rhos):.2f} [{np.min(rhos):.2f},{np.max(rhos):.2f}]; consensus (mean ceiling) {mean_rho:.2f}")
    # the naive baseline: same-gender win% -- did the agents just read one column?
    dbl = F.doubles_records()
    print("\nVS RAW SAME-GENDER WIN% (did they just sort one column?)")
    for g in "FM":
        idx = [k for k, a in enumerate(ids) if gender[a] == g]
        wp = np.array([(lambda r: r[0] / r[1] if r[1] else 0.0)(dbl.get(key[ids[k]], {}).get("womens" if g == "F" else "mens", [0, 0, 0, 0])) for k in idx])
        rhos = [spearman(M[i, idx], wp) for i in range(len(sheets))]
        print(f"  {g}: per-owner median {np.median(rhos):.2f} [{np.min(rhos):.2f},{np.max(rhos):.2f}]")
    # headline players
    w_id = next(a for a in ids if name[key[a]] == "Anna Leigh Waters")
    print(f"\nANNA LEIGH WATERS = {w_id}. Ceilings by persona (k$):")
    byp = defaultdict(list)
    for s, row in zip(sheets, M):
        byp[s.get("persona", "?")].append(row[ids.index(w_id)] / 1e3)
    for p, v in byp.items():
        print(f"  {p:15s} " + " ".join(f"{x:4.0f}" for x in v))
    print(f"  all: median {np.median(M[:, ids.index(w_id)])/1e3:.0f}k, max {M[:, ids.index(w_id)].max()/1e3:.0f}k, "
          f"second-highest {np.sort(M[:, ids.index(w_id)])[-2]/1e3:.0f}k (= roughly what she sells for in a 20-seat room)")
    print("\nCONSENSUS TOP 8 PER GENDER (mean ceiling across owners) vs list rank")
    for g in "FM":
        idx = [k for k, a in enumerate(ids) if gender[a] == g]
        lrank = {a: r + 1 for r, a in enumerate(sorted([ids[k] for k in idx], key=lambda a: -lp.get(key[a], FLOOR)))}
        order = sorted(idx, key=lambda k: -M[:, k].mean())[:8]
        for k in order:
            a = ids[k]
            print(f"  {g} {name[key[a]]:24s} mean {M[:, k].mean()/1e3:4.0f}k  max {M[:, k].max()/1e3:4.0f}k  list ${lp.get(key[a], FLOOR)/1e3:.0f}k (#{lrank[a]})")
    return sheets, M, ids, key


# ------------------------------------------------------------------ the engine
class SheetOwner:
    """A sheet turned into an auction bidder.

    Sheets are sparse: most owners priced 6-20 players and wrote 0 for the
    rest.  A real room still makes every team fill 3W+3M, so an owner with an
    open slot and no positive ceiling left takes the best remaining player at
    the floor, ordered by their own sheet first and then by the raw 2026
    same-gender win% every owner saw in the packet (`fill` ordering).  Their
    own positive ceilings always come first; the fill order only decides WHICH
    zero-ceiling player they open at $30k."""

    def __init__(self, sheet, key, gender, fill):
        self.persona = sheet.get("persona", "?"); self.file = sheet["file"]
        self.c = {key[a]: v for a, v in sheet["ceilings"].items() if a in key}
        self.gender = gender; self.fill = fill
        self.need = dict(SLOTS); self.spent = 0.0; self.roster = []; self.paid = {}

    def hard_cap(self):
        left = self.need["F"] + self.need["M"]
        return CAP - self.spent - FLOOR * (left - 1) if left else 0.0

    def ceiling(self, u):
        if self.need[self.gender[u]] <= 0:
            return None
        c = min(self.c.get(u, 0.0), self.hard_cap())
        return c if c >= FLOOR else None

    def nominate(self, avail):
        cands = [u for u in avail if self.need[self.gender[u]] > 0]
        if not cands:
            return None
        cands.sort(key=lambda u: (-min(self.c.get(u, 0.0), self.hard_cap()), -self.fill.get(u, 0.0), u))
        return cands[0]

    def buy(self, u, paid):
        self.need[self.gender[u]] -= 1; self.spent += paid; self.roster.append(u); self.paid[u] = paid


def fill_order(key, gender, sheets):
    """Ordering for floor fills when an owner's own sheet says 0: the ROOM's
    consensus (mean ceiling across every sheet, i.e. what the other owners
    think the player is worth), then a shrunk 2026 same-gender win%
    ((w+5)/(n+10), so a 1-0 record does not outrank a 40-25 one).  Raw win%
    was tried first and handed the last teams 2-0 nobodies."""
    dbl = F.doubles_records(); out = {}
    cons = defaultdict(float)
    for sh in sheets:
        for a, v in sh["ceilings"].items():
            if a in key:
                cons[key[a]] += v / len(sheets)
    for u in key.values():
        div = "womens" if gender[u] == "F" else "mens"
        w, n = dbl[u][div][0], dbl[u][div][1]
        out[u] = cons[u] + 1e3 * (w + 5) / (n + 10)
    return out


def run_room(sheets, key, gender, rng, waters, fill=None):
    fill = fill if fill is not None else fill_order(key, gender, sheets)
    owners = [SheetOwner(s, key, gender, fill) for s in sheets]
    avail = set(key.values())
    sales = []; turn = 0
    while any(o.need["F"] + o.need["M"] > 0 for o in owners):
        t = turn % len(owners); turn += 1
        nom = owners[t]
        if nom.need["F"] + nom.need["M"] <= 0:
            continue
        x = nom.nominate(avail)
        bids = []
        for k, o in enumerate(owners):
            c = o.ceiling(x)
            if k == t and c is None and o.need[gender[x]] > 0 and o.hard_cap() >= FLOOR:
                c = FLOOR                      # the nominator opens at the floor
            if c is not None:
                bids.append((c, rng.random(), k))
        if not bids:
            avail.discard(x); sales.append((x, None, 0.0)); continue
        bids.sort(key=lambda b: (-b[0], b[1]))
        b1, _, w = bids[0]
        b2 = bids[1][0] if len(bids) > 1 else FLOOR
        paid = min(b1, max(FLOOR, round((b2 + INC) / 1000.0) * 1000.0))
        owners[w].buy(x, paid); avail.discard(x); sales.append((x, w, paid))
    return owners, sales


def rooms(d, seeds, seasons):
    key = load_key(); gender = {u: F.GENDER[u] for u in key.values()}
    lp = list_prices(); name = F.NAME
    sheets = load_sheets(d)
    waters = next(u for u in key.values() if name[u] == "Anna Leigh Waters")
    D.set_board("mlp2026"); fill = fill_order(key, gender, sheets)
    out = defaultdict(list); paid_by = defaultdict(list); persona_win = defaultdict(list); persona_title = defaultdict(list)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = list(range(len(sheets))); rng.shuffle(order)
        S = [sheets[i] for i in order[:N_TEAMS]]
        owners, sales = run_room(S, key, gender, rng, waters, fill)
        assert all(o.need["F"] == 0 and o.need["M"] == 0 for o in owners), "unfilled roster"
        rosters = [o.roster for o in owners]
        exp, wins, titles = D.season(rosters, seasons, rng)
        wi = next(k for k, o in enumerate(owners) if waters in o.roster)
        out["waters_paid"].append(owners[wi].paid[waters]); out["waters_win"].append(exp[wi]); out["waters_title"].append(titles[wi])
        out["waters_buyer"].append(owners[wi].persona)
        out["spread"].append((max(exp) - min(exp)) * 100)
        out["best"].append(max(exp)); out["best_is_waters"].append(exp[wi] == max(exp))
        srt = sorted(titles, reverse=True); out["runner_up_title"].append(srt[1]); out["n10"].append(sum(t >= 0.10 for t in titles))
        out["unspent"].append(sum(CAP - o.spent for o in owners))
        out["unsold_top30"].append(sum(1 for u, w, p in sales if w is None and lp.get(u, 0) >= sorted(lp.values(), reverse=True)[29]))
        for u, w, p in sales:
            if w is not None:
                paid_by[u].append(p)
        for k, o in enumerate(owners):
            persona_win[o.persona].append(exp[k]); persona_title[o.persona].append(titles[k])
    q = lambda v: f"{np.median(v):.0f}k [{np.percentile(v,10):.0f},{np.percentile(v,90):.0f}]"
    print(f"\nROOMS: {seeds} rooms x {seasons} seasons, {N_TEAMS} of {len(sheets)} sheets per room, random sale order")
    print(f"  Waters paid          {q(np.array(out['waters_paid'])/1e3)}   (list $769k; hand-coded no-sheet room $570k default mix)")
    print(f"  her team win / title {np.mean(out['waters_win'])*100:.1f}% / {np.mean(out['waters_title'])*100:.1f}%   best team is hers in {np.mean(out['best_is_waters'])*100:.0f}% of rooms")
    print(f"  best team            {np.mean(out['best'])*100:.1f}%;  parity spread {np.mean(out['spread']):.1f} pts;  runner-up title {np.mean(out['runner_up_title'])*100:.1f}%;  teams at 10%+ {np.mean(out['n10']):.1f}")
    print(f"  unspent per team     ${np.mean(out['unspent'])/N_TEAMS/1e3:.0f}k of $1,000k;  top-30 list players unsold {np.mean(out['unsold_top30']):.1f}")
    buyers = defaultdict(int)
    for b in out["waters_buyer"]:
        buyers[b] += 1
    print("  who buys her:        " + ", ".join(f"{k} {v/seeds*100:.0f}%" for k, v in sorted(buyers.items(), key=lambda kv: -kv[1])))
    print("\n  persona              win%   title%")
    for p in sorted(persona_win, key=lambda p: -np.mean(persona_win[p])):
        print(f"  {p:20s} {np.mean(persona_win[p])*100:5.1f}  {np.mean(persona_title[p])*100:5.1f}")
    print("\n  room price vs list, by list band (median paid / list):")
    band = defaultdict(list)
    for u, v in paid_by.items():
        r = sorted(lp, key=lambda z: -lp[z]).index(u) + 1 if u in lp else 999
        b = "1-5" if r <= 5 else "6-15" if r <= 15 else "16-30" if r <= 30 else "31-60" if r <= 60 else "61-120" if r <= 120 else "unpriced"
        band[b].append(np.median(v) / max(lp.get(u, FLOOR), FLOOR))
    for b in ("1-5", "6-15", "16-30", "31-60", "61-120", "unpriced"):
        if band[b]:
            print(f"    #{b:7s} {np.median(band[b])*100:4.0f}% of list (n={len(band[b])})")
    print("\n  top prices (median paid across rooms):")
    for u in sorted(paid_by, key=lambda u: -np.median(paid_by[u]))[:12]:
        print(f"    {name[u]:24s} {np.median(paid_by[u])/1e3:4.0f}k  [{np.percentile(paid_by[u],10)/1e3:.0f},{np.percentile(paid_by[u],90)/1e3:.0f}]  list ${lp.get(u, FLOOR)/1e3:.0f}k")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", action="store_true")
    ap.add_argument("--analyze", metavar="DIR")
    ap.add_argument("--rooms", metavar="DIR")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seasons", type=int, default=200)
    A = ap.parse_args()
    if A.packet:
        build_packet()
    if A.analyze:
        analyze(A.analyze)
    if A.rooms:
        rooms(A.rooms, A.seeds, A.seasons)


if __name__ == "__main__":
    main()
