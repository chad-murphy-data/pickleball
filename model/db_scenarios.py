"""DreamBreaker split-role strategy scenarios on real MLP rosters.

Rule being simulated (Anna Bright's proposal): Team 1 sets the four
same-gender singles MATCHUPS (who faces whom); Team 2 then sets the ORDER
(which matchup occupies which rotation slot). The DreamBreaker itself:
race to 21, rally scoring (every rally scores), win by 2, the four
matchups rotate every 4 points in slot order, cycling until the game ends.

Scenarios (each of the 20 current franchises plays every other, both as
Team 1 and as Team 2 -> 380 ordered matchups per scenario):

  S1  Team 1 maximins (picks the gender pairings whose worst case under
      Team 2's response is best); Team 2 sorts the four matchups by ITS
      edge, biggest edge in slot 1.
  S2  Team 1 instead picks the MOST UNBALANCED same-gender pairings it
      can (per gender, maximize |p_matchup1 - p_matchup2|); Team 2 still
      edge-sorts.
  S3  Team 1 maximins; Team 2 ignores edges and puts its MEN's matchups
      in slots 1-2, higher-PICKLE man first (women fill 3-4, higher-PICKLE
      woman first -- the spec pins only the men; documented assumption).

Model: per-rally P = sigmoid(K_RALLY * (v1 - v2)) on PICKLE singles values
(data/db_rosters.csv value_used: real singles if >= 10 games, else the
corrected imputation -0.07 + 1.14*doubles, model/db_impute.md).
K_RALLY = 0.502 -- the rally-level coefficient fitted on the 3,125
validated same-gender DreamBreaker rallies (db_impute v2), i.e. estimated
at exactly the grain simulated here. (Team-level fit was 0.42; a
sensitivity run is reported in the summary.) Rallies are iid within a
matchup; serve is not modelled (the rally-level fit absorbs it; serve
effects cancel between sides on average).

Exact computation, no simulation noise: a backward DP over
(a, b, slot, rallies-into-slot) gives P(win); an independent forward DP
gives the full final-score distribution (margins). The two are
cross-checked to 1e-9 on every configuration. --mc N runs a Monte Carlo
self-check instead.

Outputs:
  data/db_scenarios_matchups.csv   one row per (scenario, ordered pair)
  stdout summary tables

Run: python model/db_scenarios.py [--k 0.502] [--mc 0]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

K_RALLY = 0.502        # rally-level fit, db_impute v2 (sensitivity: 0.42)
TARGET, WIN_BY, SEG, NPOS = 21, 2, 4, 4
CAP = 90               # deuce guard; truncated mass ~6e-11, asserted below
ROSTERS = DATA / "db_rosters.csv"
OUT = DATA / "db_scenarios_matchups.csv"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# game DPs
#
# Rules implemented (2025 MLP Rules Guide sec. 7 + log-verified mechanics,
# see the rules report in model/db_scenarios.md):
#   * rally scoring to 21, win by 2, no cap;
#   * THE FREEZE: "a team must win while serving" -- a rally won by the
#     RECEIVING team that would be the game-winning point scores no point
#     and only transfers the serve (verified: 0 receiver-won game-winners
#     in 74 reconstructed DBs; no-point side-outs cluster at would-win
#     states). Only one team can be at would-win at a time (win-by-2), so
#     freeze transitions cannot cycle;
#   * the winner of each rally serves the next (99.3% of log transitions);
#   * rotation advances every 4 POINTS (frozen rallies do not consume a
#     rotation slot), cycling P1->P4->P1;
#   * the first server of the DB is the coin-toss serve team -- unknowable
#     pre-match, so results average the two first-server cases 50/50 (the
#     difference is tiny: serve only matters through the freeze).
# State: (a, b, pos, j, srv) with srv = which team serves the next rally.
# ---------------------------------------------------------------------------
_win_memo: dict = {}


def _would_win_a(a, b):
    return a + 1 >= TARGET and a + 1 - b >= WIN_BY


def _would_win_b(a, b):
    return b + 1 >= TARGET and b + 1 - a >= WIN_BY


def win_prob_served(ps: tuple, first_srv: int) -> float:
    """P(team A wins) by backward induction, freeze + serve modelled."""
    @lru_cache(maxsize=None)
    def V(a, b, pos, j, srv):
        if a >= TARGET and a - b >= WIN_BY:
            return 1.0
        if b >= TARGET and b - a >= WIN_BY:
            return 0.0
        if a >= CAP or b >= CAP:
            return 0.5
        p = ps[pos]
        j2, npos, nj = j + 1, pos, j + 1
        if j2 >= SEG:
            npos, nj = (pos + 1) % NPOS, 0
        # A wins the rally (prob p)
        if srv != 0 and _would_win_a(a, b):
            va = V(a, b, pos, j, 0)          # freeze: side-out only
        else:
            va = V(a + 1, b, npos, nj, 0)    # point + A serves next
        # B wins the rally (prob 1-p)
        if srv != 1 and _would_win_b(a, b):
            vb = V(a, b, pos, j, 1)          # freeze
        else:
            vb = V(a, b + 1, npos, nj, 1)
        return p * va + (1 - p) * vb

    out = V(0, 0, 0, 0, first_srv)
    V.cache_clear()
    return out


def win_prob(ps: tuple) -> float:
    """P(team A wins), averaged over who serves first (coin toss)."""
    key = tuple(round(p, 12) for p in ps)
    if key in _win_memo:
        return _win_memo[key]
    out = 0.5 * (win_prob_served(ps, 0) + win_prob_served(ps, 1))
    _win_memo[key] = out
    return out


def score_dist(ps: tuple):
    """Forward DP: {(a, b): prob} over terminal scores, averaged over the
    first server; freeze modelled. Freeze moves mass within a score level
    (a+b unchanged, serve flips to the would-win team), and since only one
    team can be at would-win, the within-level flow is a single hop:
    states where the would-win team RECEIVES resolve before states where
    it serves. Returns (finals, truncated_mass)."""
    mass = defaultdict(float)
    mass[(0, 0, 0, 0, 0)] = 0.5
    mass[(0, 0, 0, 0, 1)] = 0.5
    finals = defaultdict(float)
    truncated = 0.0
    for total in range(2 * CAP + 1):
        layer = [st for st in mass if st[0] + st[1] == total]
        # freeze-source states first: the would-win team is the receiver
        def phase(st):
            a, b, pos, j, srv = st
            if _would_win_a(a, b) and srv == 1:
                return 0
            if _would_win_b(a, b) and srv == 0:
                return 0
            return 1
        for st in sorted(layer, key=lambda s: (phase(s), s)):
            a, b, pos, j, srv = st
            m = mass.pop(st, 0.0)
            if m == 0.0:
                continue
            if (a >= TARGET and a - b >= WIN_BY) or \
               (b >= TARGET and b - a >= WIN_BY):
                finals[(a, b)] += m
                continue
            if a >= CAP or b >= CAP:
                truncated += m
                continue
            p = ps[pos]
            j2, npos, nj = j + 1, pos, j + 1
            if j2 >= SEG:
                npos, nj = (pos + 1) % NPOS, 0
            # A wins rally
            if srv != 0 and _would_win_a(a, b):
                mass[(a, b, pos, j, 0)] += m * p          # freeze
            else:
                mass[(a + 1, b, npos, nj, 0)] += m * p
            # B wins rally
            if srv != 1 and _would_win_b(a, b):
                mass[(a, b, pos, j, 1)] += m * (1 - p)    # freeze
            else:
                mass[(a, b + 1, npos, nj, 1)] += m * (1 - p)
    return dict(finals), truncated


def margins(ps: tuple):
    """(p_win_A, E[margin A-B], E[margin | A wins], E[margin | B wins])
    from the forward DP; p_win cross-checked against the backward DP."""
    finals, truncated = score_dist(ps)
    assert truncated < 1e-8, f"truncated mass {truncated}"
    pw = sum(m for (a, b), m in finals.items() if a > b)
    exp_m = sum((a - b) * m for (a, b), m in finals.items())
    wa = sum((a - b) * m for (a, b), m in finals.items() if a > b)
    wb = sum((b - a) * m for (a, b), m in finals.items() if b > a)
    mov_a = wa / pw if pw > 0 else float("nan")
    mov_b = wb / (1 - pw) if pw < 1 else float("nan")
    pw_backward = win_prob(ps)
    assert abs(pw - pw_backward) < 1e-9, (pw, pw_backward)
    return pw, exp_m, mov_a, mov_b


def mc_check(ps, n, seed):
    """Independent Monte Carlo implementation of the same rules (freeze,
    winner-serves-next, rotation on points, random first server)."""
    rng = random.Random(seed)
    wins = 0
    msum = 0
    for _ in range(n):
        a = b = 0
        pos = j = 0
        srv = rng.randrange(2)
        while not ((a >= TARGET and a - b >= WIN_BY)
                   or (b >= TARGET and b - a >= WIN_BY)):
            a_wins = rng.random() < ps[pos]
            if a_wins:
                if srv != 0 and _would_win_a(a, b):
                    srv = 0                    # freeze: side-out only
                    continue
                a += 1
                srv = 0
            else:
                if srv != 1 and _would_win_b(a, b):
                    srv = 1
                    continue
                b += 1
                srv = 1
            j += 1
            if j >= SEG:
                j, pos = 0, (pos + 1) % NPOS
        wins += a > b
        msum += a - b
    return wins / n, msum / n


# ---------------------------------------------------------------------------
# rosters & scenario policies
# ---------------------------------------------------------------------------
def load_rosters(path=ROSTERS):
    """{team_title: {"W": [(uuid, name, value) x2 desc], "M": [...]}} for
    franchise teams only."""
    teams = defaultdict(lambda: {"W": [], "M": []})
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["franchise"] != "1":
                continue
            g = "W" if r["gender"] == "F" else "M"
            teams[r["team_title"]][g].append(
                (r["player_uuid"], r["full_name"], float(r["value_used"])))
    for t, d in teams.items():
        for g in ("W", "M"):
            d[g].sort(key=lambda x: (-x[2], x[0]))
            assert len(d[g]) == 2, (t, g)
    return dict(teams)


def pairing_options(t1, t2, gender):
    """The two possible same-gender pairings, as ((p1, q1), (p2, q2)) of
    (uuid, name, value) tuples: rank-matched and crossed."""
    a, b = t1[gender]
    c, d = t2[gender]
    return [((a, c), (b, d)), ((a, d), (b, c))]


def matchup_p(k, m):
    """Team-1 rally win prob of matchup m = ((u1, n1, v1), (u2, n2, v2))."""
    return sigmoid(k * (m[0][2] - m[1][2]))


def order_edge_sort_t2(k, matchups):
    """S1/S2 Team-2 policy: sort by TEAM 2's edge, biggest first (= Team 1's
    win prob ascending). Deterministic tiebreak on player uuids."""
    return sorted(matchups, key=lambda m: (matchup_p(k, m),
                                           m[1][0], m[0][0]))


def order_men_first_t2(k, matchups, t2):
    """S3 Team-2 policy: T2's men's matchups in slots 1-2, higher-PICKLE
    T2 man first; women in 3-4, higher-PICKLE T2 woman first."""
    men_rank = {u: i for i, (u, n, v) in enumerate(t2["M"])}
    wom_rank = {u: i for i, (u, n, v) in enumerate(t2["W"])}
    men = [m for m in matchups if m[1][0] in men_rank]
    women = [m for m in matchups if m[1][0] in wom_rank]
    men.sort(key=lambda m: men_rank[m[1][0]])
    women.sort(key=lambda m: wom_rank[m[1][0]])
    return men + women


def scenario_config(scen, k, t1, t2):
    """Resolve (pairing choice, slot order) for one ordered pair under one
    scenario. Returns (t1_winprob, ordered matchups slot1..slot4)."""
    combos = []
    for wp in pairing_options(t1, t2, "W"):
        for mp in pairing_options(t1, t2, "M"):
            combos.append(list(wp) + list(mp))

    def respond(ms):
        if scen == "S3":
            return order_men_first_t2(k, ms, t2)
        return order_edge_sort_t2(k, ms)

    if scen == "S2":
        # most unbalanced pairing per gender: maximize |p1 - p2|; if tied,
        # the crossed pairing (index 1) is preferred -- deterministic
        def pick(gender):
            opts = pairing_options(t1, t2, gender)
            diffs = [abs(matchup_p(k, a) - matchup_p(k, b)) for a, b in opts]
            return opts[1] if diffs[1] >= diffs[0] - 1e-15 else opts[0]
        chosen = list(pick("W")) + list(pick("M"))
        order = respond(chosen)
        return win_prob(tuple(matchup_p(k, m) for m in order)), order

    # S1 / S3: Team 1 maximin = best combo given Team 2's response policy
    best = None
    for ms in combos:
        order = respond(ms)
        pw = win_prob(tuple(matchup_p(k, m) for m in order))
        key = (pw, tuple(m[0][0] for m in order))     # deterministic
        if best is None or key > best[0]:
            best = (key, order, pw)
    return best[2], best[1]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=K_RALLY)
    ap.add_argument("--mc", type=int, default=0,
                    help="Monte Carlo self-check with N sims per config "
                         "on a sample of configurations")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    k = args.k

    teams = load_rosters()
    names = sorted(teams)
    print(f"{len(names)} franchise teams; k = {k}")

    rows = []
    for scen in ("S1", "S2", "S3"):
        for t1n in names:
            for t2n in names:
                if t1n == t2n:
                    continue
                t1, t2 = teams[t1n], teams[t2n]
                pw, order = scenario_config(scen, k, t1, t2)
                ps = tuple(matchup_p(k, m) for m in order)
                pw2, exp_m, mov1, mov2 = margins(ps)
                genders = "".join(
                    "W" if m[0][0] in {u for u, _, _ in t1["W"]} else "M"
                    for m in order)
                rows.append({
                    "scenario": scen, "team1": t1n, "team2": t2n,
                    "t1_win_prob": f"{pw:.4f}",
                    "t2_win_prob": f"{1 - pw:.4f}",
                    "exp_margin_t1": f"{exp_m:.3f}",
                    "mov_if_t1_wins": f"{mov1:.3f}",
                    "mov_if_t2_wins": f"{mov2:.3f}",
                    "slot_order": " | ".join(
                        f"{m[0][1]} v {m[1][1]}" for m in order),
                    "slot_genders": genders,
                    "slot1_gender": genders[0],
                    "women_matches_top2": genders[:2].count("W"),
                })
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    # summary
    print(f"\n{'scenario':<10}{'T1 win% (mean)':>15}{'T1 favored':>12}"
          f"{'E[margin]':>11}{'MOV|T1':>8}{'MOV|T2':>8}"
          f"{'%W in top2':>12}{'%W slot1':>10}")
    for scen in ("S1", "S2", "S3"):
        rs = [r for r in rows if r["scenario"] == scen]
        n = len(rs)
        mp_ = sum(float(r["t1_win_prob"]) for r in rs) / n
        fav = sum(float(r["t1_win_prob"]) > 0.5 for r in rs) / n
        em = sum(float(r["exp_margin_t1"]) for r in rs) / n
        m1 = sum(float(r["mov_if_t1_wins"]) for r in rs) / n
        m2 = sum(float(r["mov_if_t2_wins"]) for r in rs) / n
        wt2 = sum(r["women_matches_top2"] for r in rs) / (2 * n)
        w1 = sum(r["slot1_gender"] == "W" for r in rs) / n
        print(f"{scen:<10}{mp_ * 100:>14.1f}%{fav * 100:>11.1f}%"
              f"{em:>11.3f}{m1:>8.2f}{m2:>8.2f}"
              f"{wt2 * 100:>11.1f}%{w1 * 100:>9.1f}%")

    if args.mc:
        print(f"\nMonte Carlo self-check ({args.mc:,} sims per config, "
              "20 sampled configs):")
        rng = random.Random(7)
        sample = rng.sample(rows, 20)
        worst = 0.0
        for r in sample:
            scen, t1n, t2n = r["scenario"], r["team1"], r["team2"]
            pw, order = scenario_config(scen, k, teams[t1n], teams[t2n])
            ps = tuple(matchup_p(k, m) for m in order)
            mp_, mm = mc_check(ps, args.mc, seed=hash((t1n, t2n, scen)) & 0xffff)
            dev = abs(mp_ - pw)
            worst = max(worst, dev / max(1e-9, (pw * (1 - pw) / args.mc) ** 0.5))
            print(f"  {scen} {t1n[:14]:14s} v {t2n[:14]:14s} "
                  f"DP {pw:.4f} MC {mp_:.4f} (dev {dev:+.4f})  "
                  f"margin DP {float(r['exp_margin_t1']):+.2f} MC {mm:+.2f}")
        print(f"  worst deviation: {worst:.2f} sigma")


if __name__ == "__main__":
    main()
