"""Opponent-skill-controlled close-game (11-9) performance.

Raw close-game win% confounds skill: a favorite "should" win most of their
2-point games. The right benchmark is the model's win prob CONDITIONAL on the
game finishing at a 2-point margin. For a race to T (win by 2), that has a
clean closed form regardless of T:

    P(win | final margin == 2) = p^2 / (p^2 + q^2),   p = sigmoid(team eta gap)

(11-9 and every deuce finish both reduce to it — the negative-binomial combs
and the shared p^9 q^9 cancel). So we score each close game against this
opponent-adjusted expectation and split by opponent strength to ask:
does the player over/under-perform in close games, and specifically against
GOOD opposition (matchups where they're not the favorite)?

Caveat: uses CURRENT v2 form values (data/v2_players.csv) as the skill gap,
so it's cleanest on recent games; --window 2026 restricts to 2026.

Run: python model/close_game_control.py [--window all|2026]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import spec_shootout as sx          # noqa: E402

PLAYERS = {
    "Max Freeman": "b03ff4de-9b8f-46a8-a530-d8f19e9d6ec9",
    "Anna Bright": "db5eef5c-99dd-49d9-bbfc-65171a366bda",
    "Christian Alshon": "79a11da1-6aeb-4719-9b1e-9918e0163cec",
    "Hayden Patriquin": "fc05556e-3e38-4a38-99c2-f855174a7c28",
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["all", "2026"], default="all")
    ap.add_argument("--strict119", action="store_true",
                    help="only games that ended exactly 11-9 (default: any 2-pt margin)")
    args = ap.parse_args()

    players, chem = sx.load_v2()
    G = sx.GAMMA_V2

    def team_val(u1, u2):
        if u1 not in players or u2 not in players:
            return None
        a, b = players[u1]["v"], players[u2]["v"]
        c = chem.get(frozenset((players[u1]["name"], players[u2]["name"])), 0.0)
        return a + b + G * abs(a - b) + c

    # collect each player's close games with p, conditional expectation, outcome
    per = {nm: [] for nm in PLAYERS}
    for g in csv.DictReader((ROOT / "data" / "games.csv").open()):
        if g["is_forfeit"] != "False" or g["is_dreambreaker"] != "False":
            continue
        if args.window == "2026" and g["date"] < "2026-01-01":
            continue
        try:
            s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        except ValueError:
            continue
        margin = abs(s1 - s2)
        is_119 = tuple(sorted((s1, s2))) == (9, 11)
        close = is_119 if args.strict119 else (margin == 2)
        if not close:
            continue
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        tvA, tvB = team_val(us[0], us[1]), team_val(us[2], us[3])
        if tvA is None or tvB is None:
            continue
        for nm, u in PLAYERS.items():
            if u not in us:
                continue
            on_A = u in us[:2]
            eta = (tvA - tvB) if on_A else (tvB - tvA)
            opp_tv = tvB if on_A else tvA
            p = sigmoid(eta)
            q = 1 - p
            c = p * p / (p * p + q * q)          # P(win | margin==2)
            won = (s1 > s2) if on_A else (s2 > s1)
            per[nm].append(dict(p=p, c=c, won=int(won), opp=opp_tv))

    label = "11-9 only" if args.strict119 else "any 2-pt margin (11-9 + deuce)"
    print(f"OPPONENT-CONTROLLED CLOSE GAMES — {label}, window={args.window}")
    print("  expected = P(win | 2-pt finish) = p^2/(p^2+q^2), p from v2 skill gap\n")

    def line(nm, rows, tag):
        if not rows:
            print(f"    {tag:16s} (none)")
            return
        n = len(rows)
        w = sum(r["won"] for r in rows)
        exp = sum(r["c"] for r in rows)
        var = sum(r["c"] * (1 - r["c"]) for r in rows)
        z = (w - exp) / math.sqrt(var) if var > 0 else 0.0
        print(f"    {tag:16s} n={n:3d}  actual {w:3d} ({w/n*100:3.0f}%)  "
              f"expected {exp:5.1f} ({exp/n*100:3.0f}%)  "
              f"delta {w-exp:+5.1f} ({(w-exp)/n*100:+4.0f} pts)  z={z:+.1f}")

    for nm in PLAYERS:
        rows = per[nm]
        print(f"  {nm}:")
        line(nm, rows, "ALL close")
        # matchup-relative opponent strength: is the player the underdog?
        line(nm, [r for r in rows if r["p"] < 0.5], "vs FAVORED opp")   # they're the underdog
        line(nm, [r for r in rows if r["p"] >= 0.5], "vs weaker opp")   # they're favored
        # absolute opponent strength (top third of opponents faced across this pool)
        allopp = sorted(x["opp"] for v in per.values() for x in v)
        if allopp:
            thr = allopp[int(len(allopp) * 2 / 3)]
            line(nm, [r for r in rows if r["opp"] >= thr], "vs strong opp*")
        print()
    print("  vs FAVORED opp = player was the per-point underdog (good opposition).")
    print("  *strong opp = opponent team value in the top third of all opponents in this pool.")


if __name__ == "__main__":
    main()
