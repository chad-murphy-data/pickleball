"""Upsert per-match-per-player serve tallies into Supabase (pb_match_player_serve).

The droplet already holds every referee log in raw/match_logs/. This walks the
cached logs, runs the SAME validated tally() the committed summaries use, and
upserts one row per player per match into Postgres — so any serve/return/points
question becomes a SQL query from any session instead of a fresh harvest. Return
is NOT stored; it's the opposing side's serve losses, reconstructed in the
pb_player_serve_return view.

Idempotent: re-runs upsert on (match_id, player_uuid). Incremental: pass --since
to push only recent matches (the droplet's nightly top-up). No-op with a clear
message if the Supabase env is unset, so it's safe to wire unconditionally.

    SUPABASE_URL=https://<ref>.supabase.co \
    SUPABASE_SERVICE_KEY=<service-role key> \
        python scraper/upload_supabase.py [--since 2026-01-01] [--limit N] [--dry-run]

Uses the service-role key (writes bypass RLS); never commit it — it lives in the
droplet environment only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import csv
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_logs as H
from pb_api import RAW

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SERVE_TABLE = "pb_match_player_serve"
RALLY_TABLE = "pb_rally"
BATCH = 500


def _load_body(m):
    p = RAW / "match_logs" / m["match_id"][:2] / f"{m['match_id']}.json"
    if not p.exists():
        return None
    body = json.loads(p.read_text())
    rows = body.get("data") if isinstance(body, dict) else body
    return rows or None


def _serve_rows(matches):
    for m in matches:
        rows = _load_body(m)
        if not rows:
            continue
        serves, _pts, _counts = H.tally(rows, m["sides"])
        for side, players in enumerate(m["sides"]):
            for u in players:
                sn, sw = serves.get(u, (0, 0))
                yield {
                    "match_id": m["match_id"], "discipline": m["discipline"],
                    "tour": m["tour"], "match_date": m["date"],
                    "player_uuid": u, "side": side,
                    "serve_rallies": sn, "serve_wins": sw,
                }


def _rally_rows(matches):
    for m in matches:
        rows = _load_body(m)
        if not rows:
            continue
        events, _pts = H.rally_events(rows, m["sides"])
        for e in events:
            yield {
                "match_id": m["match_id"], "discipline": m["discipline"],
                "tour": m["tour"], "match_date": m["date"],
                "game_number": e["game"], "rally_number": e["rally_number"],
                "server_uuid": e["server"] or None,
                "receiver_uuid": e["receiver"] or None,
                "server_side": e["side"], "server_number": e["server_number"],
                "outcome": e["outcome"], "won": e["won"],
                "server_score": e["server_score"],
                "receiver_score": e["receiver_score"],
            }


def _player_rows():
    with open(DATA / "players.csv") as fh:
        for r in csv.DictReader(fh):
            yield {"player_uuid": r["player_id"].lower(),
                   "full_name": r["full_name"], "gender": r.get("gender", "")}


def _post(client, url, key, path, body, params=None):
    r = client.post(
        f"{url}/rest/v1/{path}",
        params=params or {},
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=body,
    )
    r.raise_for_status()


def _upsert_stream(client, url, key, table, conflict, gen, dry_run):
    rows, n, maxdate = [], 0, ""

    def flush():
        nonlocal rows, n
        if not rows:
            return
        if not dry_run:
            _post(client, url, key, table, rows, params={"on_conflict": conflict})
        n += len(rows)
        rows = []
    for row in gen:
        if row.get("match_date"):
            maxdate = max(maxdate, row["match_date"])
        rows.append(row)
        if len(rows) >= BATCH:
            flush()
    flush()
    return n, maxdate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only matches on/after YYYY-MM-DD")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-rallies", action="store_true",
                    help="skip the per-rally table (serve + players only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not (url and key):
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY unset — skipping upload.")
        return
    url = url.rstrip("/")

    matches = H.load_matches(doubles=True, singles=True)
    if args.since:
        matches = [m for m in matches if m["date"] >= args.since]
    if args.limit:
        matches = matches[:args.limit]

    t0 = time.monotonic()
    with httpx.Client(timeout=120) as client:
        np = _upsert_stream(client, url, key, "pb_player", "player_uuid",
                            _player_rows(), args.dry_run)[0]
        ns, maxdate = _upsert_stream(client, url, key, SERVE_TABLE,
                                     "match_id,player_uuid",
                                     _serve_rows(matches), args.dry_run)
        nr = 0
        if not args.no_rallies:
            nr = _upsert_stream(client, url, key, RALLY_TABLE,
                                "match_id,game_number,rally_number",
                                _rally_rows(matches), args.dry_run)[0]
        if not args.dry_run and ns:
            _post(client, url, key, "pb_meta",
                  [{"key": "serve_rows", "value": str(ns)},
                   {"key": "rally_rows", "value": str(nr)},
                   {"key": "max_match_date", "value": maxdate}],
                  params={"on_conflict": "key"})
    verb = "would upsert" if args.dry_run else "upserted"
    print(f"{verb}: {np} players, {ns} serve rows, {nr} rally rows "
          f"(through {maxdate}) in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
