"""DreamBreaker match-order simulator (2026-07-22).

Motivated by Anna Bright's essay "Women don't matter enough at MLP"
(brighterpickleball.beehiiv.com): the MLP DreamBreaker is a rally-to-21
(win by 2) singles tiebreaker with FOUR singles pairings rotating every
FOUR points, in the announced order. Because the game ends partway through
a rotation cycle, players slotted EARLY play more rallies than players
slotted late. Teams therefore stack their strongest players first — which,
league-wide, means the men — so women "see the court less" and the format
undervalues them.

This asks the quantitative question underneath that argument: holding the
roster fixed, how much can the ORDER you slot your four players into swing
the DreamBreaker result?

We keep it SAME-GENDER on purpose. Cross-gender singles values are not
identified in this project (the M/W offset is a prior convention, never a
fact — see CLAUDE.md house rules), so a mixed roster's "gaps" would be
dominated by that arbitrary offset. Within one gender every gap is a real,
identified skill gap, and the order MECHANISM (early = more volume) is
gender-blind, so the magnitude transfers directly to Anna's cross-gender
case.

Model (consistent with model/db_model.md and web/make_forecast.py):
per-rally win prob for the matchup on court = sigmoid(K_DB * (v_a - v_b)),
K_DB = 0.42 on the fit_singles value scale. Rallies are iid within a
matchup (the repo's DB model is serve-blind; k was fit at rally level).
The ONLY thing the rotation adds over the current single-p race model is
that the four matchups take turns, four rallies at a time.

Exact DP over (a, b, pos, j): scores a/b, current rotation slot pos in
0..3, rallies played so far in this slot j in 0..3. No simulation noise.

    python model/db_order_sim.py            # run the full battery
    python model/db_order_sim.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

K_DB = 0.42          # per-rally logit per unit singles-value gap (db_model.md)
TARGET = 21          # race to 21...
WIN_BY = 2           # ...win by 2
SEG = 4              # rotate every 4 points
NPOS = 4             # four matchups
CAP = 70             # deuce guard (P(reaching here) ~ 0; 0.5 fallback)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rally_ps(order_a, order_b):
    """Per-rally P(A wins) for each of the 4 rotation slots, given the two
    orderings (tuples of singles values, position i faces position i)."""
    return tuple(sigmoid(K_DB * (va - vb)) for va, vb in zip(order_a, order_b))


def db_win_prob(ps):
    """Exact P(team A wins the DreamBreaker) given per-slot rally probs
    ps = (p0, p1, p2, p3), rotating SEG points at a time from slot 0."""
    @lru_cache(maxsize=None)
    def V(a, b, pos, j):
        # terminal
        if a >= TARGET and a - b >= WIN_BY:
            return 1.0
        if b >= TARGET and b - a >= WIN_BY:
            return 0.0
        if a >= CAP or b >= CAP:
            return 0.5
        p = ps[pos]
        # advance rotation after this rally
        j2 = j + 1
        if j2 >= SEG:
            npos, nj = (pos + 1) % NPOS, 0
        else:
            npos, nj = pos, j2
        return p * V(a + 1, b, npos, nj) + (1 - p) * V(a, b + 1, npos, nj)

    out = V(0, 0, 0, 0)
    V.cache_clear()
    return out


def expected_rallies_per_slot(ps):
    """Expected number of rallies each rotation slot plays before the game
    ends (win by 2 at 21). Pure structure when ps are all 0.5."""
    counts = [0.0] * NPOS

    @lru_cache(maxsize=None)
    def reach(a, b, pos, j):
        """P(game reaches the state about to play a rally at (a,b,pos,j))."""
        return 0.0  # placeholder; filled by forward pass below

    # forward pass: probability mass flowing through each pre-rally state
    from collections import defaultdict
    mass = defaultdict(float)
    mass[(0, 0, 0, 0)] = 1.0
    # process states in increasing a+b so we never revisit
    frontier = [(0, 0, 0, 0)]
    seen = set()
    # BFS/DP by total points
    order_states = {}
    stack = [(0, 0, 0, 0)]
    # Instead do a clean sweep by total rallies played = a+b
    # gather reachable states level by level
    levels = defaultdict(list)
    levels[0].append((0, 0, 0, 0))
    visited = {(0, 0, 0, 0)}
    maxlevel = 2 * CAP
    for lvl in range(maxlevel):
        for st in levels[lvl]:
            a, b, pos, j = st
            if a >= TARGET and a - b >= WIN_BY:
                continue
            if b >= TARGET and b - a >= WIN_BY:
                continue
            if a >= CAP or b >= CAP:
                continue
            counts[pos] += mass[st]
            p = ps[pos]
            j2 = j + 1
            if j2 >= SEG:
                npos, nj = (pos + 1) % NPOS, 0
            else:
                npos, nj = pos, j2
            for na, nb, pr in ((a + 1, b, p), (a, b + 1, 1 - p)):
                nst = (na, nb, npos, nj)
                mass[nst] += mass[st] * pr
                if nst not in visited:
                    visited.add(nst)
                    levels[lvl + 1].append(nst)
    return counts


# ---------------------------------------------------------------------------
# roster loading
# ---------------------------------------------------------------------------
def load_singles():
    vals = {}
    with open(DATA / "singles_players.csv") as fh:
        for r in csv.DictReader(fh):
            vals[r["full_name"]] = (float(r["singles_value"]),
                                    int(r["singles_games"]),
                                    r["gender"])
    return vals


def v(vals, name):
    if name not in vals:
        raise SystemExit(f"no singles value for {name!r}")
    return vals[name][0]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def hr(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def exp_volume():
    hr("1. STRUCTURE: how many rallies does each rotation slot play?")
    counts = expected_rallies_per_slot((0.5, 0.5, 0.5, 0.5))
    tot = sum(counts)
    print("Between evenly matched teams (every matchup 50/50):\n")
    print(f"  {'slot':<8}{'E[rallies]':>12}{'share':>10}")
    for i, c in enumerate(counts):
        print(f"  pos {i+1:<4}{c:>12.2f}{c/tot*100:>9.1f}%")
    print(f"  {'total':<8}{tot:>12.2f}")
    r = counts[0] / counts[3]
    print(f"\n  Slot 1 plays {r:.2f}x as many rallies as slot 4 "
          f"({counts[0]:.1f} vs {counts[3]:.1f}).")
    print("  This is the whole mechanism: earlier = more court time, "
          "regardless of\n  who is in the slot. A team's biggest edges want "
          "the busiest slots.")


def exp_single_edge():
    hr("2. ISOLATED: same skill edge, different slot")
    print("Team A has ONE player who is better than his opposite; the other")
    print("three matchups are dead even. Where should the edge go?\n")
    print(f"  {'edge (singles val)':<20}" +
          "".join(f"{'pos'+str(i+1):>9}" for i in range(NPOS)) +
          f"{'best-worst':>12}")
    for delta in (0.25, 0.5, 1.0, 1.5):
        probs = []
        for slot in range(NPOS):
            ps = [0.5, 0.5, 0.5, 0.5]
            ps[slot] = sigmoid(K_DB * delta)
            probs.append(db_win_prob(tuple(ps)))
        spread = (max(probs) - min(probs)) * 100
        print(f"  +{delta:<19.2f}" +
              "".join(f"{p*100:>8.1f}%" for p in probs) +
              f"{spread:>11.1f}pp")
    print("\n  Same edge is worth more in slot 1 than slot 4 — the volume "
          "asymmetry\n  turns identical skill into different win probability "
          "purely by placement.")


def all_orderings_prob(roster_a, roster_b, fix_b=None):
    """For fixed roster value-lists, return dict {a_order_idx: winprob} and
    the B ordering used. If fix_b is None, B plays strong->weak (sorted desc,
    facing A's slots in order)."""
    b_order = fix_b if fix_b is not None else tuple(sorted(roster_b, reverse=True))
    res = {}
    for perm in set(permutations(roster_a)):
        res[perm] = db_win_prob(rally_ps(perm, b_order))
    return res, b_order


def exp_real_roster(vals):
    hr("3. REAL ROSTER: full 24-ordering spread, women's singles values")
    # A heterogeneous women's four (one star + a tail), a solid rival four.
    A = {"Anna Leigh Waters": v(vals, "Anna Leigh Waters"),
         "Kate Fahey": v(vals, "Kate Fahey"),
         "Anna Bright": v(vals, "Anna Bright"),
         "Milan Rane": v(vals, "Milan Rane")}
    B = {"Parris Todd": v(vals, "Parris Todd"),
         "Lea Jansen": v(vals, "Lea Jansen"),
         "Kaitlyn Christian": v(vals, "Kaitlyn Christian"),
         "Catherine Parenteau": v(vals, "Catherine Parenteau")}
    print("Team A:", ", ".join(f"{n} {x:.2f}" for n, x in A.items()))
    print("Team B:", ", ".join(f"{n} {x:.2f}" for n, x in B.items()))
    ra, rb = list(A.values()), list(B.values())

    # B fixed at its strongest-first ordering; vary A over all 24
    res, b_order = all_orderings_prob(ra, rb)
    best = max(res.items(), key=lambda kv: kv[1])
    worst = min(res.items(), key=lambda kv: kv[1])
    names = {x: n for n, x in {**A}.items()}
    def label(perm):
        return " > ".join(names[x] for x in perm)
    print(f"\nB fixed strongest-first. A's win prob across all 24 orderings:")
    print(f"  best  {best[1]*100:5.1f}%   {label(best[0])}")
    print(f"  worst {worst[1]*100:5.1f}%   {label(worst[0])}")
    print(f"  --> match order alone swings Team A by "
          f"{(best[1]-worst[1])*100:.1f} pp.")

    # strongest-first vs weakest-first for A specifically
    sf = db_win_prob(rally_ps(tuple(sorted(ra, reverse=True)), b_order))
    wf = db_win_prob(rally_ps(tuple(sorted(ra)), b_order))
    print(f"\n  strongest-first {sf*100:.1f}%  vs  weakest-first "
          f"{wf*100:.1f}%   (+{(sf-wf)*100:.1f} pp for playing your best "
          "early)")


def exp_equilibrium(vals):
    hr("4. GAME THEORY: does 'strongest first' survive a best-responding foe?")
    A = [v(vals, n) for n in ("Anna Leigh Waters", "Kate Fahey",
                              "Anna Bright", "Milan Rane")]
    B = [v(vals, n) for n in ("Parris Todd", "Lea Jansen",
                              "Kaitlyn Christian", "Catherine Parenteau")]
    aperms = list(set(permutations(A)))
    bperms = list(set(permutations(B)))

    # value of order when opponent is FIXED (strong-first) vs BEST-RESPONDING
    b_sf = tuple(sorted(B, reverse=True))
    fixed = [db_win_prob(rally_ps(pa, b_sf)) for pa in aperms]
    # against a best-responding B (B minimizes A's prob for each A order)
    responded = [min(db_win_prob(rally_ps(pa, pb)) for pb in bperms)
                 for pa in aperms]
    print(f"Team A's control over its own win prob:")
    print(f"  vs B fixed strong-first : {min(fixed)*100:.1f}% .. "
          f"{max(fixed)*100:.1f}%  (spread {(max(fixed)-min(fixed))*100:.1f} pp)")
    print(f"  vs B best-responding    : {min(responded)*100:.1f}% .. "
          f"{max(responded)*100:.1f}%  (spread "
          f"{(max(responded)-min(responded))*100:.1f} pp)")

    # is (strong-first, strong-first) a Nash equilibrium?
    a_sf = tuple(sorted(A, reverse=True))
    # A's best response to B strong-first
    a_best = max(db_win_prob(rally_ps(pa, b_sf)) for pa in aperms)
    a_at_sf = db_win_prob(rally_ps(a_sf, b_sf))
    # B's best response to A strong-first (B maximizes its own = minimizes A)
    b_best_for_b = min(db_win_prob(rally_ps(a_sf, pb)) for pb in bperms)
    b_at_sf = db_win_prob(rally_ps(a_sf, b_sf))
    print(f"\n  At (strong-first, strong-first): A wins {a_at_sf*100:.1f}%")
    print(f"    A's best deviation      -> {a_best*100:.1f}% "
          f"(gain {(a_best-a_at_sf)*100:+.2f} pp)")
    print(f"    B's best deviation      -> A {b_best_for_b*100:.1f}% "
          f"(B gains {(b_at_sf-b_best_for_b)*100:+.2f} pp)")
    if abs(a_best - a_at_sf) < 5e-4 and abs(b_best_for_b - b_at_sf) < 5e-4:
        print("    => strongest-first-vs-strongest-first is a Nash "
              "equilibrium (neither\n       team can profitably reorder).")
    else:
        print("    => NOT a clean equilibrium; reordering pays.")


def exp_anna(vals):
    hr("5. ANNA'S CASE: burying your weaker players late")
    print("Proxy for the cross-gender stack, kept same-gender: a roster with")
    print("two STRONG and two WEAK players. How much does hiding the weak two")
    print("in the late (low-volume) slots buy you vs an even opponent?\n")
    strong = v(vals, "Kate Fahey")          # ~1.80
    weak = v(vals, "Milan Rane")            # ~0.89
    A = [strong, strong, weak, weak]
    # opponent: four average-ish players, all equal, so A's gaps are clean
    avg = (strong + weak) / 2
    B = (avg, avg, avg, avg)
    print(f"  Team A: two strong ({strong:.2f}) + two weak ({weak:.2f})")
    print(f"  Team B: four average ({avg:.2f}) -- so A's per-slot edge is "
          f"+{strong-avg:.2f} (strong) / {weak-avg:+.2f} (weak)\n")

    configs = {
        "strong first (slots 1,2)": (strong, strong, weak, weak),
        "interleaved (S,W,S,W)   ": (strong, weak, strong, weak),
        "weak first (slots 1,2)  ": (weak, weak, strong, strong),
    }
    base = None
    for lab, order in configs.items():
        p = db_win_prob(rally_ps(order, B))
        if base is None:
            base = p
        print(f"  {lab}: {p*100:5.1f}%")
    sfp = db_win_prob(rally_ps((strong, strong, weak, weak), B))
    wfp = db_win_prob(rally_ps((weak, weak, strong, strong), B))
    print(f"\n  Playing the strong pair first is worth "
          f"{(sfp-wfp)*100:+.1f} pp over playing them last.")
    print("  That gap is exactly the incentive Anna describes: the busy early")
    print("  slots go to your strongest players. League-wide those are the")
    print("  men, so the women get the low-volume tail -- structurally")
    print("  'mattering less,' independent of any cross-gender value claim.")


# St. Louis Shock vs New Jersey 5s — the graded Gold-final rosters
# (model/prediction_midseason_final.md; NJ DB roster from db_model.md).
STL = {"Kate Fahey": "F", "Anna Bright": "F",
       "Hayden Patriquin": "M", "Gabriel Tardio": "M"}
NJ = {"Anna Leigh Waters": "F", "Jorja Johnson": "F",
      "Noe Khlif": "M", "Will Howells": "M"}


def valid_orderings(stl_names, nj_names, gender):
    """Yield (stl_order, nj_order) player-name tuples such that every slot is
    same-gender (slot i = stl_order[i] vs nj_order[i]). Dedup on the induced
    per-slot matchup so identical DPs aren't recomputed."""
    seen = set()
    for so in set(permutations(stl_names)):
        for no in set(permutations(nj_names)):
            if any(gender[s] != gender[n] for s, n in zip(so, no)):
                continue
            key = tuple(zip(so, no))
            if key in seen:
                continue
            seen.add(key)
            yield so, no


def exp_real_teams(vals):
    hr("6. REAL TEAMS: St. Louis Shock vs New Jersey 5s DreamBreaker")
    gender = {**STL, **NJ}
    def val(n):
        return v(vals, n)
    print("St. Louis Shock:  " +
          ", ".join(f"{n} {val(n):.2f}{gender[n]}" for n in STL))
    print("New Jersey 5s:    " +
          ", ".join(f"{n} {val(n):.2f}{gender[n]}" for n in NJ))
    print("\nP is per-rally P(St. Louis wins), sigmoid(0.42*gap). St. Louis is")
    print("the DB underdog here (Waters + Khlif/Howells outrate the Shock four).\n")

    stl_w = [n for n in STL if gender[n] == "F"]
    nj_w = [n for n in NJ if gender[n] == "F"]
    stl_m = [n for n in STL if gender[n] == "M"]
    nj_m = [n for n in NJ if gender[n] == "M"]

    def pair_p(a, b):
        return sigmoid(K_DB * (val(a) - val(b)))

    # ---- A. the two women pairings, two men pairings (who faces whom) ----
    print("-- Women pairings (who St. Louis's women draw) --")
    wA = [(stl_w[0], nj_w[0]), (stl_w[1], nj_w[1])]
    wB = [(stl_w[0], nj_w[1]), (stl_w[1], nj_w[0])]
    for lab, pr in (("as listed ", wA), ("reversed  ", wB)):
        print(f"  {lab}: " + " | ".join(
            f"{a} v {b}  P={pair_p(a,b)*100:.1f}%" for a, b in pr))
    print("-- Men pairings --")
    mA = [(stl_m[0], nj_m[0]), (stl_m[1], nj_m[1])]
    mB = [(stl_m[0], nj_m[1]), (stl_m[1], nj_m[0])]
    for lab, pr in (("as listed ", mA), ("reversed  ", mB)):
        print(f"  {lab}: " + " | ".join(
            f"{a} v {b}  P={pair_p(a,b)*100:.1f}%" for a, b in pr))

    # ---- B. each pairing combo at a neutral slot order (W,M,W,M) ----
    print("\n-- St. Louis DB win prob per pairing combo (slot order W,M,W,M) --")
    combos = {}
    for wl, wp in (("womenAsListed", wA), ("womenReversed", wB)):
        for ml, mp in (("menAsListed", mA), ("menReversed", mB)):
            # interleave W,M,W,M
            slots = [wp[0], mp[0], wp[1], mp[1]]
            ps = tuple(pair_p(a, b) for a, b in slots)
            p = db_win_prob(ps)
            combos[(wl, ml)] = p
            print(f"  {wl:14s} + {ml:12s}: {p*100:5.1f}%")
    best = max(combos.items(), key=lambda kv: kv[1])
    worst = min(combos.items(), key=lambda kv: kv[1])
    print(f"  pairing choice alone (fixed order) moves St. Louis "
          f"{(best[1]-worst[1])*100:.1f} pp "
          f"[{worst[1]*100:.1f}-{best[1]*100:.1f}%]")

    # ---- C. full joint: all valid same-gender orderings ----
    print("\n-- Adding slot ORDER: every valid same-gender configuration --")
    results = []
    for so, no in valid_orderings(list(STL), list(NJ), gender):
        ps = tuple(pair_p(s, n) for s, n in zip(so, no))
        results.append((db_win_prob(ps), so, no))
    results.sort(reverse=True)
    hi, so_hi, no_hi = results[0]
    lo, so_lo, no_lo = results[-1]
    def fmt(so, no):
        return "  vs  ".join(f"{s}/{n}" for s, n in zip(so, no))
    print(f"  {len(results)} distinct configurations.")
    print(f"  BEST for St. Louis  {hi*100:5.1f}%")
    print(f"      slots: {fmt(so_hi, no_hi)}")
    print(f"  WORST for St. Louis {lo*100:5.1f}%")
    print(f"      slots: {fmt(so_lo, no_lo)}")
    print(f"  Full swing from pairing + order: {(hi-lo)*100:.1f} pp.")

    # The extremes above need the OPPONENT to cooperate (NJ would never bury
    # Waters in slot 4). What survives when both order well depends on WHO
    # announces first -- because the gender-interleaving pattern must match,
    # so the announcer sets it and constrains the responder. This is exactly
    # the "who sets matchups vs who sets position" lever Anna's essay is about.
    by_stl, by_nj = {}, {}
    for p, so, no in results:
        by_stl.setdefault(so, []).append(p)
        by_nj.setdefault(no, []).append(p)
    stl_first = max(min(ps) for ps in by_stl.values())   # StL sets pattern
    nj_first = min(max(ps) for ps in by_nj.values())      # NJ sets pattern
    nj_open = min(by_nj, key=lambda k: max(by_nj[k]))
    print(f"\n  When BOTH order well, the result turns on who announces first:")
    print(f"    St. Louis announces first : {stl_first*100:.1f}% "
          "(it sets the interleave, NJ replies)")
    print(f"    New Jersey announces first: {nj_first*100:.1f}% "
          "(NJ sets it, St. Louis replies)")
    print(f"    -> announcing first (the 'button') is worth "
          f"{(stl_first-nj_first)*100:.1f} pp here.")
    print(f"  NJ's optimal opener is just strongest-first "
          f"({' , '.join(nj_open)}),")
    print(f"  which splits St. Louis's women into slots 1 and 4.")
    print(f"  So the realistic band is ~{nj_first*100:.0f}-{stl_first*100:.0f}%"
          f", not the full {lo*100:.0f}-{hi*100:.0f}% envelope.")

    # what the current forecast would say (single mean-gap p, no order)
    gap = (sum(val(n) for n in STL) - sum(val(n) for n in NJ)) / 4
    flat = db_win_prob((sigmoid(K_DB * gap),) * 4)
    print(f"  Today's forecast (single mean-gap p, order-blind): "
          f"{flat*100:.1f}%.")


def exp_relative(vals):
    hr("7. RELATIVE, NOT ABSOLUTE: order by your EDGE, not your own strength")
    print("Optimal ordering slots your biggest EDGE (highest win prob in the")
    print("matchup) first -- NOT your strongest player. They differ when your")
    print("best player is an underdog while a weaker teammate has a soft draw.\n")
    # your strongest player faces a stronger opponent; a weaker player has a
    # big edge over a weak one.
    matchups = {"your 2.0 v their 2.5": (2.0, 2.5),
                "your 1.5 v their 0.5": (1.5, 0.5),
                "your 1.2 v their 1.1": (1.2, 1.1),
                "your 1.0 v their 1.3": (1.0, 1.3)}
    p = {k: sigmoid(K_DB * (a - b)) for k, (a, b) in matchups.items()}
    for k, pv in p.items():
        print(f"    {k}:  P(you win)={pv*100:.1f}%")
    abs_order = list(matchups)                       # already strong->weak (you)
    edge_order = sorted(p, key=lambda k: -p[k])
    pa = db_win_prob(tuple(p[k] for k in abs_order))
    pe = db_win_prob(tuple(p[k] for k in edge_order))
    print(f"\n  absolute-strength-first : {pa*100:.1f}%")
    print(f"  relative (edge)-first   : {pe*100:.1f}%   (+{(pe-pa)*100:.1f} pp)")
    # confirm edge-first == brute-force optimum, here and on random sets
    best = max(permutations(list(p)), key=lambda o: db_win_prob(tuple(p[k] for k in o)))
    print(f"  brute-force optimum     : "
          f"{db_win_prob(tuple(p[k] for k in best))*100:.1f}%  "
          f"({'edge-first IS optimal' if best == tuple(edge_order) else 'differs'})")
    mism = 0
    for seed in range(500):
        ps = [0.30 + 0.40 * ((seed * (i + 1) * 0.61803398875) % 1.0)
              for i in range(NPOS)]
        edge = tuple(sorted(ps, reverse=True))
        bf = max(permutations(ps), key=db_win_prob)
        if abs(db_win_prob(edge) - db_win_prob(bf)) > 1e-12:
            mism += 1
    print(f"  edge-first optimal in {500 - mism}/500 random matchup sets "
          "(rearrangement inequality).")


def exp_nj_opener(vals):
    hr("8. NJ's opener: split the barbell, don't cluster by gender")
    gender = {**STL, **NJ}
    val = lambda n: v(vals, n)
    pp = lambda a, b: sigmoid(K_DB * (val(a) - val(b)))   # P(StL wins) a slot

    def nj_winprob(no):
        """NJ's win prob if it opens with order `no` and St. Louis best-
        responds within the forced gender pattern."""
        best = 0.0
        for so in set(permutations(list(STL))):
            if any(gender[s] != gender[n] for s, n in zip(so, no)):
                continue
            best = max(best, db_win_prob(tuple(pp(s, n) for s, n in zip(so, no))))
        return 1 - best     # StL maximizes StL; NJ gets the complement

    print("The tempting read: 'my women are relatively stronger, play them")
    print("first.' But NJ's women are a BARBELL -- Waters is the board's top")
    print("weapon, Johnson is NJ's weak link (underdog to both StL women).\n")
    print("  NJ pair edges vs St. Louis (best-response matchups):")
    print(f"    women: Waters {val('Anna Leigh Waters')-val('Kate Fahey'):+.2f} "
          f"(vs Fahey), Johnson {val('Jorja Johnson')-val('Anna Bright'):+.2f} "
          "(vs Bright)  -> net +0.18")
    print(f"    men:   Khlif +0.24, Howells +0.10                         "
          "        -> net +0.34\n")

    both_women = ("Anna Leigh Waters", "Jorja Johnson", "Noe Khlif", "Will Howells")
    split = ("Anna Leigh Waters", "Noe Khlif", "Will Howells", "Jorja Johnson")
    edge_sorted = tuple(sorted(NJ, key=val, reverse=True))
    print(f"  both women first (W,W,M,M): NJ {nj_winprob(both_women)*100:.1f}%")
    print(f"  split - Waters 1st, Johnson last (W,M,M,W): "
          f"NJ {nj_winprob(split)*100:.1f}%")
    print(f"  edge-sorted players {edge_sorted}: "
          f"NJ {nj_winprob(edge_sorted)*100:.1f}%")
    print(f"\n  Splitting the barbell is worth "
          f"{(nj_winprob(split)-nj_winprob(both_women))*100:+.1f} pp to NJ:")
    print("  lead Waters (busy slot 1), bury Johnson (dead slot 4). The right")
    print("  unit is the PLAYER's edge, not the gender pair -- 'both women")
    print("  first' only wins if BOTH women are your relatively stronger picks.")


def exp_split_roles(vals):
    hr("9. ANNA'S FIX: St. Louis sets the PAIRS, New Jersey sets the ORDER")
    val = lambda n: v(vals, n)
    pp = lambda a, b: sigmoid(K_DB * (val(a) - val(b)))    # P(StL wins a rally)
    wA = [("Kate Fahey", "Anna Leigh Waters"), ("Anna Bright", "Jorja Johnson")]
    wB = [("Kate Fahey", "Jorja Johnson"), ("Anna Bright", "Anna Leigh Waters")]
    mA = [("Hayden Patriquin", "Noe Khlif"), ("Gabriel Tardio", "Will Howells")]
    mB = [("Hayden Patriquin", "Will Howells"), ("Gabriel Tardio", "Noe Khlif")]
    sets = {"women as-is,  men as-is": wA + mA, "women as-is,  men rev ": wA + mB,
            "women rev,    men as-is": wB + mA, "women rev,    men rev ": wB + mB}
    show = lambda m: f"{m[0].split()[-1]}-{m[1].split()[-1]}({pp(*m)*100:.0f})"

    print("St. Louis fixes the four matchups; NJ then slots them adversarially")
    print("(NJ puts St. Louis's WORST matchup in the busy slot 1). St. Louis")
    print("picks the pairing with the best worst-case:\n")
    best = None
    for name, mus in sets.items():
        ps = [pp(*m) for m in mus]
        worst = min(db_win_prob(tuple(ps[i] for i in perm))
                    for perm in permutations(range(4)))
        arg = min(permutations(range(4)),
                  key=lambda perm: db_win_prob(tuple(ps[i] for i in perm)))
        if best is None or worst > best[1]:
            best = (name, worst, arg, mus)
        print(f"  {name}:  StL {worst*100:4.1f}%   "
              f"NJ orders-> {' > '.join(show(mus[i]) for i in arg)}")
    print(f"\n  St. Louis's best pairing: '{best[0]}' -> guarantees "
          f"{best[1]*100:.1f}%.")
    print("  Note the FLIP: with order power St. Louis wanted the LOPSIDED")
    print("  women pairing (Fahey-Johnson 56%); now it wants the BALANCED one")
    print("  (worst matchup 45%, not 42%) so NJ has no soft target to feature.")
    gap = (sum(val(n) for n in STL) - sum(val(n) for n in NJ)) / 4
    print(f"\n  For scale: order-blind forecast "
          f"{db_win_prob((sigmoid(K_DB*gap),)*4)*100:.1f}%, St. Louis-sets-order "
          "43.0%.\n  Setting POSITION is the stronger half of the split "
          f"(~2 pp); pairing alone\n  leaves St. Louis on the weak side, best "
          "played defensively.")


def load_rosters():
    """Reconstruct each MLP team's 2W+2M DreamBreaker roster from the
    projected lineups in data/forecasts.json (WD pair = women, MD pair =
    men). Singles values from fit_singles; players without a singles history
    imputed from doubles via the forecast's regression, intercept shrunk
    0.28->0.08 for the DreamBreaker underperformance (model/db_impute.md)."""
    sv, gen, dbl = {}, {}, {}
    with open(DATA / "singles_players.csv") as fh:
        for r in csv.DictReader(fh):
            sv[r["full_name"]] = float(r["singles_value"])
            gen[r["full_name"]] = r["gender"]
    try:
        with open(DATA / "v2_players.csv") as fh:
            for r in csv.DictReader(fh):
                dbl[r["full_name"]] = float(r["value_now_mean"])
                gen.setdefault(r["full_name"], r["gender"])
    except FileNotFoundError:
        pass

    def val(n):
        if n in sv:
            return sv[n]
        if n in dbl:
            return 0.08 + 1.14 * dbl[n]      # shrunk imputation (db_impute.md)
        return None

    fc = json.loads((DATA / "forecasts.json").read_text())["forecasts"]
    rosters = {}
    for f in fc:
        for t, pk in ((f["team1"], "t1_pair"), (f["team2"], "t2_pair")):
            wd = next(g for g in f["games"] if g["slot"] == "WD")
            md = next(g for g in f["games"] if g["slot"] == "MD")
            w = sorted(wd[pk], key=lambda n: -(val(n) or 0))
            m = sorted(md[pk], key=lambda n: -(val(n) or 0))
            if all(val(n) is not None for n in w + m):
                rosters[t] = {"W": w, "M": m}
    return rosters, val


def exp_league(vals):
    hr("10. LEAGUE-WIDE: does OPTIMAL play feature women in the top 2 slots?")
    rosters, val = load_rosters()
    teams = sorted(rosters)
    print(f"{len(teams)} MLP teams, rosters from projected lineups "
          "(forecasts.json).")
    print("Optimal ordering is edge-first, so each team's top-2 slots are the")
    print("two matchups it is most likely to win. Rank-matched same-gender")
    print("pairing (stronger-vs-stronger within gender).\n")

    def top2_women(A, B):
        ms = []
        for g in ("W", "M"):
            for i in (0, 1):
                ms.append((g, sigmoid(K_DB * (val(rosters[A][g][i])
                                              - val(rosters[B][g][i])))))
        ms.sort(key=lambda x: -x[1])
        return [g for g, _ in ms[:2]].count("W")

    from collections import Counter
    counts, fav, dog = [], [], []
    pairs = [(a, b) for a in teams for b in teams if a != b]
    for A, B in pairs:
        w = top2_women(A, B)
        counts.append(w)
        ga = sum(val(n) for g in ("W", "M") for n in rosters[A][g])
        gb = sum(val(n) for g in ("W", "M") for n in rosters[B][g])
        (fav if ga >= gb else dog).append(w)
    dist = dict(sorted(Counter(counts).items()))
    n = len(counts)
    print(f"  # women in the top-2 slots, over {n} team-vs-team orderings:")
    print(f"    0 women (both men) : {dist.get(0,0):3d}  ({dist.get(0,0)/n*100:.0f}%)")
    print(f"    1 woman            : {dist.get(1,0):3d}  ({dist.get(1,0)/n*100:.0f}%)")
    print(f"    2 women (both women): {dist.get(2,0):3d}  ({dist.get(2,0)/n*100:.0f}%)")
    print(f"    mean = {sum(counts)/n:.2f} women in top-2\n")
    print(f"  Anna's status quo (teams stack their men first): 0 women in "
          "top-2, always.")
    print(f"  Optimal edge-first play: {sum(counts)/n:.2f} on average -- women "
          f"take BOTH top\n  slots in {dist.get(2,0)/n*100:.0f}% of matchups. "
          "Optimizing (not stacking men) features\n  women far more; the "
          "ordering rule itself is gender-blind.")
    wsd = _pstdev([val(n) for t in teams for n in rosters[t]["W"]])
    msd = _pstdev([val(n) for t in teams for n in rosters[t]["M"]])
    print(f"\n  Why ~1.0 and not higher: it tracks spread, and in THIS cohort "
          f"men's\n  singles spread (sd {msd:.2f}) exceeds women's (sd {wsd:.2f})"
          ", so men's matchups\n  are marginally more lopsided. Favorites "
          f"feature {sum(fav)/len(fav):.2f} women in top-2,\n  underdogs "
          f"{sum(dog)/len(dog):.2f} -- stronger teams lean slightly more on "
          "their women.")

    # concrete example: a California team vs Dallas (the user's "LA v Dallas")
    for A, B in (("California Black Bears", "Dallas Flash"),
                 ("SoCal Hard Eights", "Dallas Flash")):
        if A in rosters and B in rosters:
            print(f"\n  Example -- {A} (as orderer) vs {B}:")
            ms = []
            for g in ("W", "M"):
                for i in (0, 1):
                    a, b = rosters[A][g][i], rosters[B][g][i]
                    ms.append((g, a, b, sigmoid(K_DB * (val(a) - val(b)))))
            ms.sort(key=lambda x: -x[3])
            for rank, (g, a, b, p) in enumerate(ms, 1):
                tag = "  <- top 2" if rank <= 2 else ""
                print(f"    slot {rank} [{g}] {a} v {b}: {p*100:.0f}%{tag}")


def _pstdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, default=None,
                    help="run only experiment N (1-10)")
    args = ap.parse_args()
    vals = load_singles()
    runs = [exp_volume, exp_single_edge,
            lambda: exp_real_roster(vals),
            lambda: exp_equilibrium(vals),
            lambda: exp_anna(vals),
            lambda: exp_real_teams(vals),
            lambda: exp_relative(vals),
            lambda: exp_nj_opener(vals),
            lambda: exp_split_roles(vals),
            lambda: exp_league(vals)]
    if args.only:
        runs[args.only - 1]()
    else:
        for r in runs:
            r()
    print()


if __name__ == "__main__":
    main()
