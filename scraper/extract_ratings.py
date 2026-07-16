"""Extract per-match synced DUPR ratings from the raw/ cache.

    python scraper/extract_ratings.py    # merges into data/per_match_ratings.json
                                         # and rewrites data/platform_ratings.csv

The embedded team*Player*Rating fields on every match record are the
player's synced DUPR doubles rating as of that match (verified — see
analysis.md benchmark).  This script re-creates the two ratings artifacts
from whatever raw/ contains and MERGES with the existing JSON, so a partial
cache (e.g. an incremental CI harvest) never erases history.  Zero/absent
ratings are skipped.  Identity is by lowercased player UUID, per house
rules.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"

SIDES = ("One", "Two")
PLAYERS = ("PlayerOne", "PlayerTwo")


def ratings_from_match(m: dict, out: dict):
    for side in SIDES:
        for pn in PLAYERS:
            u = str(m.get(f"team{side}{pn}Uuid") or "").lower()
            r = m.get(f"team{side}{pn}Rating")
            if u and isinstance(r, (int, float)) and r > 0:
                out[u] = r


def main():
    pm_path = DATA / "per_match_ratings.json"
    per_match = json.loads(pm_path.read_text()) if pm_path.exists() else {}
    n_before = len(per_match)

    # MLP: one file per matchup, matches embedded
    mdir = RAW / "matchup_data"
    for f in sorted(mdir.glob("*.json")) if mdir.exists() else []:
        mu = json.loads(f.read_text()).get("data") or {}
        for m in mu.get("matches") or []:
            mid = str(m.get("matchUuid") or "").lower()
            if not mid:
                continue
            found = {}
            ratings_from_match(m, found)
            if found:
                per_match.setdefault(mid, {}).update(found)

    # PPA: match_infos_short files, one per tournament-day
    pdir = RAW / "match_infos_short"
    for f in sorted(pdir.rglob("*.json")) if pdir.exists() else []:
        for m in json.loads(f.read_text()).get("data") or []:
            mid = str(m.get("matchUuid") or m.get("uuid") or "").lower()
            if not mid:
                continue
            found = {}
            ratings_from_match(m, found)
            if found:
                per_match.setdefault(mid, {}).update(found)

    pm_path.write_text(json.dumps(per_match, indent=None, sort_keys=True))
    print(f"per_match_ratings.json: {n_before} -> {len(per_match)} matches")

    # latest snapshot per player = value at their most recent match by date
    match_date = {}
    games_csv = DATA / "games.csv"
    if games_csv.exists():
        for g in csv.DictReader(games_csv.open()):
            match_date[g["match_id"]] = g["date"]
    latest, counts = {}, {}
    for mid, players in per_match.items():
        d = match_date.get(mid, "0000-00-00")
        for u, r in players.items():
            counts[u] = counts.get(u, 0) + 1
            if u not in latest or d >= latest[u][0]:
                latest[u] = (d, r)
    with (DATA / "platform_ratings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "platform_rating_latest", "n_snapshots"])
        for u in sorted(latest):
            w.writerow([u, latest[u][1], counts[u]])
    print(f"platform_ratings.csv: {len(latest)} players")


if __name__ == "__main__":
    main()
