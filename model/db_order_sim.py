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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, default=None,
                    help="run only experiment N (1-5)")
    args = ap.parse_args()
    vals = load_singles()
    runs = [exp_volume, exp_single_edge,
            lambda: exp_real_roster(vals),
            lambda: exp_equilibrium(vals),
            lambda: exp_anna(vals)]
    if args.only:
        runs[args.only - 1]()
    else:
        for r in runs:
            r()
    print()


if __name__ == "__main__":
    main()
