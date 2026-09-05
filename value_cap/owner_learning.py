"""owner_learning.py -- do the records-only owners learn?

Runs the no-sheet room (fan_auction.py: owners who just bought a team, no
price sheet, one ordinal draw per owner per gender, a roster shape with budget
shares) for T seasons in a row. Every season is a full re-auction with fresh
money; between seasons the owners may learn through three channels, each a
switch so the sweep can say which one matters:

  P  prices   the room remembers last season's sale prices. They seed tonight's
              going rate (the >=3-sales rule fires from sale 1) and each owner
              re-anchors its budget shares: for a targeted role, new share =
              (1-lam) x plan + lam x (median paid last season for players in
              that role's gender and own-rank band) / $1M, then renormalised to
              spend the cap. lam is swept.
  S  shape    after the season each owner, with probability p, copies the
              roster shape (persona) of a random playoff team (top 4 by realised
              wins; --copy champ = the champion only). Shares reset to the
              copied persona's defaults (jittered).
  K  players  the rank draws sharpen: sd_t = sd_0 x decay^t. decay 1 = owners
              watch a season and get no better at knowing who is good.

Payoffs on the true tie model (draft_sim.season logic: double round robin +
top-4 playoff). Each season reports the same room statistics as fan_auction.md
(Waters' price and buyer, her team's win%/title share, paid/list by list-rank
bucket, parity spread, best/second team, teams at 10%+ title odds, unspent,
persona mix).

    python value_cap/owner_learning.py --grid            # the sweep -> owner_learning.md
    python value_cap/owner_learning.py --lam 0.5 --p 0.25 --decay 0.7 --years 10
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fan_auction as A  # noqa: E402
import fan_view as F  # noqa: E402

CAP, FLOOR, N_TEAMS = A.CAP, A.FLOOR, A.N_TEAMS


# ------------------------------------------------------------------ a season on the true engine
def play_season(W, rosters, seasons, rng):
    """draft_sim.season plus ONE realised standing (the season the owners
    actually watched). Returns exp, ttl, realised wins, champion."""
    n = len(rosters)
    P = [[0.5 if i == j else W.D.TRUE_ENGINE.tie(rosters[i], rosters[j]) for j in range(n)] for i in range(n)]
    exp = [sum(P[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    titles = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    realised = None
    for k in range(seasons):
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
        if k == 0:
            realised = dict(wins=wins, playoff=order[:4], champ=champ)
    return exp, [t / seasons for t in titles], realised


# ------------------------------------------------------------------ the learning owner
class State:
    """What an owner carries from one season to the next."""

    def __init__(self, persona, shares):
        self.persona = persona
        self.shares = shares          # np.array, sums to 1 (jittered persona defaults, then re-anchored)


def default_shares(persona, rng, jitter):
    shares = np.array([s for _, _, s in A.PERSONAS[persona]])
    if jitter > 0:
        shares = shares * np.clip(1 + jitter * rng.standard_normal(len(shares)), 0.5, 1.5)
        shares = shares / shares.sum()
    return shares


def build_owner(W, st, rng, sd_mult, premium):
    o = A.Owner(W, st.persona, rng, sd_mult, premium, jitter=0.0)
    for r, sh in zip(o.roles, st.shares):
        r["share"] = float(sh)
    return o


def reanchor(o, st, memory, lam):
    """P channel: pull each targeted role's share toward what players like
    that (this owner's own band, same gender) went for last season."""
    if lam <= 0 or not memory:
        return
    new = st.shares.copy()
    for i, r in enumerate(o.roles):
        if not o.targeted(r):
            continue
        b = r["band"]
        same = []
        for u, paid in memory:
            g = o.W.gender[u]
            if r["g"] not in ("A", g):
                continue
            if b == "S":
                if u in o.srank and o.srank[u] <= o.W.sgl_top:
                    same.append(paid)
            elif b[0] <= o.rank[u] <= b[1]:
                same.append(paid)
        if same:
            new[i] = (1 - lam) * new[i] + lam * statistics.median(same) / CAP
    st.shares = new / new.sum()


# ------------------------------------------------------------------ T seasons
def run_years(W, mix, seed, years, lam=0.0, p_switch=0.0, decay=1.0, sd_mult=1.0, premium=0.1,
              jitter=0.1, seasons=300, copy="playoff", verbose=False):
    rng = np.random.default_rng(seed)
    seats = list(mix); rng.shuffle(seats)
    states = [State(p, default_shares(p, rng, jitter)) for p in seats]
    memory = []
    out = []
    for t in range(years):
        sd_t = sd_mult * decay ** t
        owners = [build_owner(W, st, rng, sd_t, premium) for st in states]
        if lam > 0:
            for o, st in zip(owners, states):
                reanchor(o, st, memory, lam)
                for r, sh in zip(o.roles, st.shares):
                    r["share"] = float(sh)
        order = list(range(N_TEAMS)); rng.shuffle(order)      # seat order (nomination rotation) reshuffles
        owners = [owners[k] for k in order]; states = [states[k] for k in order]
        res = A.run_auction(W, mix, int(rng.integers(1 << 30)), sd_t, premium, (), jitter,
                            owners=owners, memory=memory if lam > 0 else ())
        ros = [tuple(o.roster) for o in owners]
        exp, ttl, real = play_season(W, ros, seasons, np.random.default_rng(10_000 + seed * 101 + t))
        res["exp"] = exp; res["ttl"] = ttl; res["real"] = real; res["year"] = t
        out.append(res)
        if verbose:
            st = A.room_stats(W, [res])
            print(f"  year {t+1}: " + A.fmt_stats(st, f"seed {seed}")[38:], file=sys.stderr, flush=True)
        # what the room remembers
        memory = [(x, paid) for x, w, paid, _ in res["sales"] if w is not None]
        # S: copy a playoff team's shape
        if p_switch > 0:
            play = real["playoff"] if copy == "playoff" else [real["champ"]]
            new_states = []
            for k, st in enumerate(states):
                if rng.random() < p_switch:
                    src = states[play[int(rng.integers(len(play)))]]
                    if src.persona != st.persona:
                        st = State(src.persona, default_shares(src.persona, rng, jitter))
                new_states.append(st)
            states = new_states
    return out


def cell(W, mix, seeds, years, **kw):
    """Per-year room statistics averaged over seeds."""
    runs = [run_years(W, mix, s, years, **kw) for s in range(seeds)]
    per_year = []
    for t in range(years):
        st = A.room_stats(W, [r[t] for r in runs])
        st["mix"] = Counter(p for r in runs for p in r[t]["seats"])
        # the realised champion's persona, and how often Waters' team won the realised title
        st["champ"] = Counter(r[t]["seats"][r[t]["real"]["champ"]] for r in runs)
        per_year.append(st)
    return per_year


# ------------------------------------------------------------------ reporting
def year_table(per_year, seeds):
    hdr = ("| season | Waters $ | buyer | her win / title | list% #1-5 / #6-15 / #16-30 / #31-60 | spread | best / 2nd | n10 | unspent | shapes in the room |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for t, st in enumerate(per_year, 1):
        r = st["ratio"]
        wb = " ".join(f"{k} {v}" for k, v in st["waters_buyer"].most_common(3))
        mix = " ".join(f"{k} {v/seeds:.1f}" for k, v in sorted(st["mix"].items(), key=lambda kv: -kv[1]))
        rows.append(f"| {t} | ${st['waters_price']/1e3:.0f}k | {wb} | {100*st['waters_win']:.0f}% / {100*st['waters_title']:.0f}% | "
                    f"{100*r['#1-5']:.0f} / {100*r['#6-15']:.0f} / {100*r['#16-30']:.0f} / {100*r['#31-60']:.0f} | {st['spread']:.1f} | "
                    f"{100*st['best']:.0f}% / {100*st['second']:.0f}% | {st['n10']:.1f} | ${st['unspent']/1e6:.2f}M | {mix} |")
    return hdr + "\n".join(rows) + "\n"


def persona_table(per_year):
    names = sorted({p for st in per_year for p in st["per"]})
    hdr = "| season | " + " | ".join(f"{A.LONG[p]} win / title / $" for p in names) + " |\n|---|" + "---|" * len(names) + "\n"
    rows = []
    for t, st in enumerate(per_year, 1):
        cells = []
        for p in names:
            d = st["per"].get(p)
            cells.append("-" if d is None else f"{100*d['win']:.0f}% / {100*d['ttl']:.0f}% / ${d['spent']/1e3:.0f}k")
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows) + "\n"


GRID = [
    # name, kwargs
    ("no learning (beliefs redrawn at the same sd, no memory, no copying)", dict()),
    ("P prices only, lam 0.5", dict(lam=0.5)),
    ("P prices only, lam 1.0", dict(lam=1.0)),
    ("S shape only, p 0.25", dict(p_switch=0.25)),
    ("S shape only, p 0.5", dict(p_switch=0.5)),
    ("S shape only, p 0.25, copy the champion (not a random playoff team)", dict(p_switch=0.25, copy="champ")),
    ("K players only, decay 0.7", dict(decay=0.7)),
    ("K players only, decay 0.85", dict(decay=0.85)),
    ("P + S (lam 0.5, p 0.25)", dict(lam=0.5, p_switch=0.25)),
    ("P + K (lam 0.5, decay 0.7)", dict(lam=0.5, decay=0.7)),
    ("all three (lam 0.5, p 0.25, decay 0.7)", dict(lam=0.5, p_switch=0.25, decay=0.7)),
    ("all three, gentler sharpening (lam 0.5, p 0.25, decay 0.85)", dict(lam=0.5, p_switch=0.25, decay=0.85)),
    ("all three, owners start at sd x2", dict(lam=0.5, p_switch=0.25, decay=0.7, sd_mult=2.0)),
    ("all three, all four-starters start", dict(lam=0.5, p_switch=0.25, decay=0.7, mix="four=20")),
    ("all three, all star & scrubs start", dict(lam=0.5, p_switch=0.25, decay=0.7, mix="star=20")),
]


def grid(W, seeds, years, seasons, out):
    t0 = time.time()
    results = []
    for name, kw in GRID:
        kw = dict(kw); mix = A.parse_mix(kw.pop("mix", A.DEFAULT_MIX))
        py = cell(W, mix, seeds, years, seasons=seasons, **kw)
        results.append((name, py))
        first, last = py[0], py[-1]
        print(f"{name:60s} Waters ${first['waters_price']/1e3:.0f}k -> ${last['waters_price']/1e3:.0f}k | her {100*first['waters_win']:.0f}%/{100*first['waters_title']:.0f}% -> "
              f"{100*last['waters_win']:.0f}%/{100*last['waters_title']:.0f}% | #1-5 {100*first['ratio']['#1-5']:.0f}% -> {100*last['ratio']['#1-5']:.0f}% | "
              f"#31-60 {100*first['ratio']['#31-60']:.0f}% -> {100*last['ratio']['#31-60']:.0f}% | spread {first['spread']:.1f} -> {last['spread']:.1f} | "
              f"unspent ${first['unspent']/1e6:.2f}M -> ${last['unspent']/1e6:.2f}M | {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    write_md(results, seeds, years, seasons, out, time.time() - t0)
    return results


def write_md(results, seeds, years, seasons, out, secs):
    L = [f"# Do the owners learn? {years} seasons of the no-sheet room\n",
         f"Generated by `value_cap/owner_learning.py --grid` ({seeds} seeds per cell, {years} seasons per run, {seasons} simulated "
         f"seasons per room for win%/title, {secs/60:.1f} min). Owners are the records-only owners of `fan_auction.py` / "
         f"`fan_owner_spec.md` (people who bought a team, no price sheet). Every season is a full re-auction with fresh $1M; "
         f"between seasons three learning channels, each a switch: **P** the room remembers last season's prices (going-rate "
         f"seed + budget shares re-anchored with weight lam), **S** owners copy a random playoff team's roster shape with "
         f"probability p, **K** rank draws sharpen (sd x decay per season). List prices appear only in the comparison columns.\n",
         "Columns as in fan_auction.md: Waters' price and buyer persona (count over seeds), her team's expected win% and title "
         "share, paid/list by list-rank bucket, parity spread (sd of team win%, pts), best and second team, teams at 10%+ title "
         "odds, unspent cap summed over 20 teams, and the persona mix in the room (mean seats per persona).\n"]
    if FINDINGS:
        L.append(FINDINGS)
    for name, py in results:
        L.append(f"\n## {name}\n")
        L.append(year_table(py, seeds))
        L.append("\nBy roster shape (win% / title / spent):\n\n" + persona_table(py))
    Path(out).write_text("\n".join(L))
    print(f"wrote {out}", file=sys.stderr)


FINDINGS = """## What ten seasons do (read from the tables below; 8 seeds, so read trends, not single cells)

0. **The control is flat.** With no learning the room drifts nowhere in ten
   seasons (Waters $578k -> $536k, curve 68/77/115/154 -> 69/78/114/150,
   spread 16.3 -> 15.5). Everything below is learning, not noise.
1. **Remembering prices (P) makes the room LESS like the list, not more.**
   Stars deflate and depth inflates: #1-5 68% -> 52% of list (lam 0.5) /
   48% (lam 1), #31-60 154% -> 179% / 182%, Waters $578k -> $314k / $235k,
   spread widens 16.3 -> 18.4 / 19.6, unspent stays ~$0.9M. Mechanism: every
   shape pays a flat amount within its band, so "what players like her went
   for" is the median of a band that depth buyers price at ~$225k; a
   star-and-scrubs owner learns that a top-3 player costs $230k and cuts its
   bid, and nobody in this owner family bids more for a player it ranks
   higher within the band. Prices converge to plan money, not to skill. The
   winners are the four-starters owners (60% -> 67%, titles 7% -> 14%): they
   get stars at depth prices. Her team under lam 1: 84% / 39%.
2. **Copying the winners (S) is the channel that un-inverts the curve.** The
   room converges to two-stars + four-starters; risk-averse and balanced
   six are extinct by season 7-10 and singles-minded fades; star-and-scrubs
   holds ~2 seats (it is copied only when it makes the playoffs, and the
   star owner WITHOUT Waters runs 40-50%). Curve: p 0.25 takes
   68/77/115/154 -> 83/90/100/131; p 0.5 -> 95/100/93/106 -- the list's
   shape, reached by a room that has never seen the list. Spread 16.3 ->
   12-13. The copied shape's edge is competed away as it spreads (two stars
   61% -> 51-53% while going from 3 to 8-14 seats), which is why the mix
   settles instead of collapsing to one shape. Waters' price is flat
   ($567-592k), her team 72-74% / 31-38%. Copying the champion alone lands
   in the same place (spread 11.1).
3. **Sharper beliefs (K) mean more agreement, which is money.** Shapes fixed,
   owners converge on the same rankings, so they compete for the same
   players: unspent $0.84M -> $0.18M (decay 0.7) / $0.62M (0.85), #1-5 68%
   -> 81% / 79%, #31-60 154% -> 133% / 131%, spread unchanged (~17). The
   star-and-scrubs owner is the loser of agreement (52% -> 34-37%): once
   everyone knows who the top 15 are, its $30k slots are the true leftovers.
   Waters goes to two-star owners more often ($407k, her team 78-79% /
   41-45%).
4. **All three together: a room of four-starters owners, Waters cheaper and
   her team stronger.** Four starters take 11-14 of 20 seats (P makes depth
   pricing pay, S copies it), two stars 6, star-and-scrubs 1-2. Waters
   $526k (decay 0.7) / $457k (0.85), her team 76-79% / 32-42% -- she gets
   CHEAPER as the room learns and her team gets STRONGER, because she goes
   to owners with $440k left to spend on real players. Curve
   68/77/115/154 -> 86/87/127/104 (0.7) / 83/84/132/110 (0.85): top and
   bottom at the list, the middle (#16-30) dear -- the learned room pays for
   "a top-15 player" and for nothing else. Spread 14.5-15.0, never the
   list's 4.5. Owners who start twice as foggy (sd x2) end in the same place
   (86/90/125/110, spread 12.5, Waters $483k).
5. **The degenerate corner, and what it teaches.** Twenty four-starters
   owners with all three channels: Waters $225k every season, her team 99% /
   89% by season 7, spread 23. In season 10 one owner holds Waters, Bright,
   Ben Johns and JW Johnson at $222-226k each. Identical plans (a flat ~$225k
   for any top-15 player) + a shared ranking (K) + prices anchored to last
   season (P) make every contested sale tie to within the jitter, and the
   owner with the fattest plan wins every one for $5k more. Not a forecast:
   it is what this owner family does when nobody pays more for #1 than for
   #15. The missing dial is a within-band rank slope -- a design call, not
   ours to pick.
6. **What never happens.** No cell reaches the list's parity (4.3-4.6); no
   cell prices Waters above the $769k list except the all-star start
   ($850k from season 1, unchanged -- fully priced on day one, and its
   #1-5 at 154-164% of list never corrects because nobody copies a shape
   that runs 50%); and every cell with S or K makes HER TEAM stronger
   (72% -> 76-79%): the cheap-star + deep-roster build is what wins, and
   copying finds it. Unspent money shrinks only through agreement (K).

CAVEATS: 8 seeds per cell; imitation copies the SHAPE only (persona;
shares reset to defaults); P anchors to the band MEDIAN (alternatives not
run: anchor to the own top target's price; asymmetric "raise if I lost");
nobody learns a within-band slope; the realised standing is ONE simulated
season; a full re-auction every season, no keepers/contracts; six fixed
shapes. The channels are switches with one swept rate each, not a fitted
model of owner behaviour.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=A.DEFAULT_MIX)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--seasons", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.0, help="P: weight on last season's prices in the budget shares")
    ap.add_argument("--p", type=float, default=0.0, help="S: probability an owner copies a playoff team's shape")
    ap.add_argument("--decay", type=float, default=1.0, help="K: rank-draw sd multiplier per season")
    ap.add_argument("--copy", default="playoff", choices=["playoff", "champ"], help="S: whose shape gets copied")
    ap.add_argument("--sd-mult", type=float, default=1.0)
    ap.add_argument("--premium", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.1)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default=str(HERE / "owner_learning.md"))
    a = ap.parse_args()
    W = A.World()
    if a.grid:
        grid(W, a.seeds, a.years, a.seasons, a.out)
        return
    py = cell(W, A.parse_mix(a.mix), a.seeds, a.years, lam=a.lam, p_switch=a.p, decay=a.decay,
              sd_mult=a.sd_mult, premium=a.premium, jitter=a.jitter, seasons=a.seasons, copy=a.copy)
    print(year_table(py, a.seeds))
    print(persona_table(py))


if __name__ == "__main__":
    main()
