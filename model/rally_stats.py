"""Reusable serve/return rally stats — query once, from anywhere.

The point of the Supabase store: ask serve/return/points questions without ever
re-harvesting. This is the front door. It reads the pb_player_serve_return view
(per player, tour, year) via the Supabase REST API when SUPABASE_URL is set, and
falls back to the committed data/player_serve_rallies.csv otherwise, so it works
online or offline with the same shape.

    from model.rally_stats import serve_return, freshness

    rows = serve_return(discipline="doubles", min_points=300)   # pooled per player
    for r in sorted(rows, key=lambda r: -r["serve_pct"])[:10]:
        print(r["name"], r["serve_pct"], r["return_pct"])

Field baselines: serve% is individual; return% in doubles is team-attributed
(both partners share the side's return rallies) — a valid per-player RATE, never
summed across a team.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _names():
    names, gender = {}, {}
    with open(DATA / "players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            names[u] = r["full_name"]
            gender[u] = r.get("gender", "")
    return names, gender


def _from_supabase(discipline, tour, year):
    """View rows (player,tour,year), filtered server-side where possible."""
    import httpx
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
    params = {"select": "*"}
    if tour:
        params["tour"] = f"eq.{tour}"
    if year:
        params["yr"] = f"eq.{year}"
    r = httpx.get(f"{url}/rest/v1/pb_player_serve_return",
                  params=params,
                  headers={"apikey": key, "Authorization": f"Bearer {key}"},
                  timeout=60)
    r.raise_for_status()
    # discipline isn't in the view (doubles/singles share a player); the store
    # is doubles+singles, so callers wanting one must filter via the base table.
    return [{"player_uuid": x["player_uuid"], "tour": x["tour"],
             "year": str(x["yr"]), "serve_rallies": x["serve_rallies"],
             "serve_wins": x["serve_wins"], "return_rallies": x["return_rallies"],
             "return_wins": x["return_wins"]} for x in r.json()]


def _from_csv(discipline, tour, year):
    out = []
    with open(DATA / "player_serve_rallies.csv") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "return_rallies" not in rows[0]:
        raise SystemExit("player_serve_rallies.csv predates return columns; "
                         "refresh it (return-aware harvest_logs.py) or set "
                         "SUPABASE_URL to query the store.")
    for r in rows:
        if discipline and r["discipline"] != discipline:
            continue
        if tour and r["tour"] != tour:
            continue
        if year and r["year"] != str(year):
            continue
        out.append(r)
    return out


def serve_return(discipline="doubles", tour=None, year=None, min_points=0):
    """Per-player serve/return, pooled over the requested slice.

    Returns dicts: uuid, name, gender, serve_rallies, serve_wins, serve_pct,
    return_rallies, return_wins, return_pct, total_pct.
    """
    source = _from_supabase if os.environ.get("SUPABASE_URL") else _from_csv
    raw = source(discipline, tour, year)
    names, gender = _names()
    pl = defaultdict(lambda: [0, 0, 0, 0])
    for r in raw:
        a = pl[r["player_uuid"]]
        a[0] += int(r["serve_rallies"]); a[1] += int(r["serve_wins"])
        a[2] += int(r["return_rallies"]); a[3] += int(r["return_wins"])
    out = []
    for u, (sr, sw, rr, rw) in pl.items():
        if sw < min_points or sr < 1 or rr < 1:
            continue
        out.append({
            "uuid": u, "name": names.get(u, u[:8]), "gender": gender.get(u, ""),
            "serve_rallies": sr, "serve_wins": sw, "serve_pct": sw / sr,
            "return_rallies": rr, "return_wins": rw, "return_pct": rw / rr,
            "total_pct": (sw + rw) / (sr + rr),
        })
    return out


def freshness():
    """(rows, max_match_date) from pb_meta, or (None, None) offline."""
    if not os.environ.get("SUPABASE_URL"):
        return None, None
    import httpx
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
    r = httpx.get(f"{url}/rest/v1/pb_meta", params={"select": "key,value"},
                  headers={"apikey": key, "Authorization": f"Bearer {key}"},
                  timeout=30)
    r.raise_for_status()
    d = {x["key"]: x["value"] for x in r.json()}
    return d.get("serve_rows"), d.get("max_match_date")


if __name__ == "__main__":
    rows = serve_return(discipline="doubles", min_points=300)
    print(f"{len(rows)} players (doubles, >300 pts)")
    for r in sorted(rows, key=lambda r: -r["serve_pct"])[:10]:
        print(f"  {r['serve_pct']:.3f} serve | {r['return_pct']:.3f} return "
              f"| {r['gender']} {r['name']}")
