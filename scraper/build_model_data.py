"""Deliverable 4: tidy modeling table from games.csv.

Filters (per the project brief):
  - modeling rows only: no DreamBreakers (already excluded), no forfeits
  - sideout_11 games only — the to-15 Challenger rounds are a different margin
    scale; kept out of the primary model (sensitivity data still in games.csv)

Output data/model_data.csv, one row per game with integer indices ready for
any mixed-model backend:
  margin           signed, team1 - team2
  a1..a4           player indices (a1,a2 = team1; a3,a4 = team2)
  dyad1, dyad2     dyad indices (unordered pair within side)
  match_idx        groups correlated games within a best-of-N match
  ctx_idx          0=mixed 1=mens 2=womens
  tour_idx         0=MLP 1=PPA
plus data/model_players.csv / data/model_dyads.csv index → uuid/name maps.

Run: python scraper/build_model_data.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CTX = {"mixed": 0, "mens": 1, "womens": 2}
TOUR = {"MLP": 0, "PPA": 1}


def main():
    import os
    min_pg = int(os.environ.get("MIN_PLAYER_GAMES", 0))
    suffix = os.environ.get("OUT_SUFFIX", "")
    date_before = os.environ.get("DATE_BEFORE", "")
    date_from = os.environ.get("DATE_FROM", "")
    games = [g for g in csv.DictReader((DATA / "games.csv").open())
             if g["is_forfeit"] == "False" and g["scoring_format"] == "sideout_11"]
    if date_before:
        games = [g for g in games if g["date"] < date_before]
        print(f"date filter < {date_before}: {len(games)} games")
    if date_from:
        games = [g for g in games if g["date"] >= date_from]
        print(f"date filter >= {date_from}: {len(games)} games")
    players = {r["player_id"]: r for r in csv.DictReader((DATA / "players.csv").open())}
    if min_pg:
        # "core pool" robustness: keep only games where all four players have
        # at least min_pg appearances in the full modeling set
        from collections import Counter
        appearances = Counter()
        for g in games:
            for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
                appearances[g[c]] += 1
        before = len(games)
        games = [g for g in games
                 if all(appearances[g[c]] >= min_pg
                        for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))]
        print(f"core-pool filter (>={min_pg} games/player): {before} -> {len(games)} games")

    pidx, didx, midx = {}, {}, {}
    dyad_meta = {}

    def p(uuid):
        return pidx.setdefault(uuid, len(pidx))

    def d(u1, u2, ctx):
        key = tuple(sorted((u1, u2)))
        if key not in didx:
            didx[key] = len(didx)
            dyad_meta[key] = ctx
        return didx[key]

    rows = []
    for g in games:
        m = midx.setdefault(g["match_id"], len(midx))
        rows.append({
            "game_id": g["game_id"],
            "margin": int(g["margin"]),
            "a1": p(g["t1_p1"]), "a2": p(g["t1_p2"]),
            "a3": p(g["t2_p1"]), "a4": p(g["t2_p2"]),
            "dyad1": d(g["t1_p1"], g["t1_p2"], g["context"]),
            "dyad2": d(g["t2_p1"], g["t2_p2"], g["context"]),
            "match_idx": m,
            "ctx_idx": CTX[g["context"]],
            "tour_idx": TOUR[g["tour"]],
            "best_of": int(g["best_of"]),
            "game_number": int(g["game_number"]),
            "event_name": g["event_name"],
            "date": g["date"],
            "stage": g["stage"],
        })

    with (DATA / f"model_data{suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with (DATA / f"model_players{suffix}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "player_id", "full_name", "gender"])
        for uuid, i in sorted(pidx.items(), key=lambda x: x[1]):
            rec = players.get(uuid, {})
            w.writerow([i, uuid, rec.get("full_name", ""), rec.get("gender", "")])

    with (DATA / f"model_dyads{suffix}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "p1_id", "p2_id", "p1_name", "p2_name", "context"])
        for key, i in sorted(didx.items(), key=lambda x: x[1]):
            u1, u2 = key
            w.writerow([i, u1, u2,
                        players.get(u1, {}).get("full_name", ""),
                        players.get(u2, {}).get("full_name", ""),
                        dyad_meta[key]])

    print(f"model_data: {len(rows)} games | {len(pidx)} players | "
          f"{len(didx)} dyads | {len(midx)} matches")


if __name__ == "__main__":
    main()
