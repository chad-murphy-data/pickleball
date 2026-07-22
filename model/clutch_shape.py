"""Where does a player's clutch come from? Decompose by leverage & game-state.

Clutch (big_points.py) is one number: covariance of within-game leverage and
serve-outcome residual. This opens it up for one player — does the positive
covariance come from the genuine crunch (9-9, deuce) or from mid-range points
on runs/comebacks? And how does it square with their close-game record?

Reuses the exact reconstruction + leverage DP from big_points, so the numbers
tie out to data/clutch_players.csv.

Run: python model/clutch_shape.py [--player "Max Freeman"]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))
import spec_shootout as sx          # noqa: E402
import big_points as bp             # noqa: E402
from sitelib.winprob import serve_probs, _table   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="Max Freeman")
    args = ap.parse_args()

    games = sx.load_games()
    players, chem = sx.load_v2()
    by_name = defaultdict(list)
    for u, d in players.items():
        by_name[d["name"]].append(u)
    pu = max(by_name[args.player], key=lambda u: players[u].get("games", 0)).lower()

    # ---- 1. close-game record (all of this player's games in the file) ----
    def margins(window):
        w = defaultdict(lambda: [0, 0])   # bucket -> [wins, losses]
        for g in games:
            if window and not (sx.RECENT <= g["date"] < sx.SPLIT):
                continue
            us = [u.lower() for u in g["us"]]
            if pu not in us:
                continue
            on_A = pu in us[:2]
            m = g["s1"] - g["s2"] if on_A else g["s2"] - g["s1"]   # signed, player POV
            am = abs(m)
            b = ("deuce/±2" if am <= 2 else "3–4" if am <= 4
                 else "5–6" if am <= 6 else "7+")
            w[b][0 if m > 0 else 1] += 1
        return w

    print(f"=== {args.player}: close-game record (games he played in) ===")
    for label, window in [("Jan–May 2026 (clutch window)", True), ("all in file", False)]:
        w = margins(window)
        tot_w = sum(v[0] for v in w.values()); tot_l = sum(v[1] for v in w.values())
        print(f"\n  {label}:  {tot_w}-{tot_l} ({tot_w/(tot_w+tot_l)*100:.0f}% of games)")
        for b in ("deuce/±2", "3–4", "5–6", "7+"):
            wins, loss = w[b]
            n = wins + loss
            if n:
                print(f"    {b:9s} {wins:2d}-{loss:<2d}  ({wins/n*100:3.0f}% win, n={n})")

    # ---- 2. rebuild his serving rallies WITH full state + leverage ----
    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    by_match = defaultdict(list)
    for g in train:
        by_match[g["match"]].append(g)
    raw = ROOT / "raw" / "match_logs"

    def eta_of(g):
        if any(u not in players for u in g["us"]):
            return None
        v = [players[u]["v"] for u in g["us"]]
        c1 = chem.get(frozenset((players[g["us"][0]]["name"],
                                 players[g["us"][1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[g["us"][2]]["name"],
                                 players[g["us"][3]]["name"])), 0.0)
        return (v[0] + v[1] + sx.GAMMA_V2 * abs(v[0] - v[1])
                - v[2] - v[3] - sx.GAMMA_V2 * abs(v[2] - v[3]) + c1 - c2)

    k_league = 0.443
    rr = []   # per Freeman serve-rally: dict(lev, levz, exp, won, a,b,side,T, my,opp)
    for m, gs in by_match.items():
        p = raw / m[:2] / f"{m}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
        if not rows:
            continue
        sides = (frozenset(u.lower() for u in gs[0]["us"][:2]),
                 frozenset(u.lower() for u in gs[0]["us"][2:]))
        if pu not in sides[0] and pu not in sides[1]:
            continue
        games_by_no = {g["gn"]: (g["s1"], g["s2"]) for g in gs}
        eta_by_no = {g["gn"]: eta_of(g) for g in gs}
        T_by_no = {g["gn"]: g["T"] for g in gs}
        seq = bp.reconstruct_states(rows, sides, games_by_no)
        if not seq:
            continue
        by_game = defaultdict(list)
        for s in seq:
            by_game[s[0]].append(s)
        for gno, rallies in by_game.items():
            e = eta_by_no.get(gno)
            if e is None:
                continue
            kA, kB = serve_probs(e, k_league)
            T = T_by_no[gno]
            V = _table(round(kA, 6), round(kB, 6), T, T + 40)
            levs = [bp.leverage_of(V, T, a, b, st, side == 0)
                    for (_, a, b, st, su, side, won) in rallies]
            levs = np.array(levs)
            if levs.std() < 1e-9:
                continue
            levz = (levs - levs.mean()) / levs.std()
            for (rly, lv, lz) in zip(rallies, levs, levz):
                _, a, b, st, su, side, won = rly
                if su != pu:
                    continue
                exp = kA if side == 0 else kB
                my, opp = (a, b) if side == 0 else (b, a)
                rr.append(dict(lev=float(lv), levz=float(lz), exp=exp,
                               won=int(won), a=a, b=b, side=side, T=T,
                               my=my, opp=opp))

    n = len(rr)
    clutch = float(np.mean([r["levz"] * (r["won"] - r["exp"]) for r in rr]))
    print(f"\n=== {args.player}: clutch shape ({n} serving rallies, "
          f"clutch={clutch:+.4f}) ===")

    # by leverage quartile
    lv = np.array([r["lev"] for r in rr])
    qs = np.quantile(lv, [0.25, 0.5, 0.75])
    def bucket(r):
        return (0 if r["lev"] <= qs[0] else 1 if r["lev"] <= qs[1]
                else 2 if r["lev"] <= qs[2] else 3)
    print("\n  by leverage quartile (Q4 = biggest points):")
    print(f"    {'bucket':6s} {'n':>4} {'win%':>6} {'exp%':>6} {'resid':>7} "
          f"{'clutch contrib':>14}")
    for q, name in [(0, "Q1 low"), (1, "Q2"), (2, "Q3"), (3, "Q4 BIG")]:
        sub = [r for r in rr if bucket(r) == q]
        if not sub:
            continue
        wr = np.mean([r["won"] for r in sub])
        ex = np.mean([r["exp"] for r in sub])
        contrib = np.mean([r["levz"] * (r["won"] - r["exp"]) for r in sub]) \
            * len(sub) / n
        print(f"    {name:6s} {len(sub):>4} {wr*100:>5.0f}% {ex*100:>5.0f}% "
              f"{(wr-ex)*100:>+6.1f}% {contrib:>+14.4f}")

    # the genuine crunch: both scores within 2 of game point (9-9+, deuce)
    crunch = [r for r in rr if r["my"] >= r["T"] - 2 and r["opp"] >= r["T"] - 2]
    near = [r for r in rr if (r["my"] >= r["T"] - 2 or r["opp"] >= r["T"] - 2)
            and not (r["my"] >= r["T"] - 2 and r["opp"] >= r["T"] - 2)]
    for label, sub in [("TRUE crunch (both ≥ game-point-2, e.g. 9-9/10-9/9-10/deuce)",
                        crunch),
                       ("one side near game point (a run in progress)", near)]:
        if sub:
            wr = np.mean([r["won"] for r in sub])
            ex = np.mean([r["exp"] for r in sub])
            print(f"\n  {label}: n={len(sub)}  win {wr*100:.0f}% vs exp "
                  f"{ex*100:.0f}%  ({(wr-ex)*100:+.1f} pts)")

    # trailing / tied / leading (his team POV, at rally start)
    print("\n  by game-state when he serves (his team's score vs opp):")
    for label, pred in [("trailing", lambda r: r["my"] < r["opp"]),
                        ("tied", lambda r: r["my"] == r["opp"]),
                        ("leading", lambda r: r["my"] > r["opp"])]:
        sub = [r for r in rr if pred(r)]
        if sub:
            wr = np.mean([r["won"] for r in sub])
            ex = np.mean([r["exp"] for r in sub])
            print(f"    {label:9s} n={len(sub):>4}  win {wr*100:3.0f}% vs exp "
                  f"{ex*100:3.0f}%  ({(wr-ex)*100:+.1f} pts)")


if __name__ == "__main__":
    main()
