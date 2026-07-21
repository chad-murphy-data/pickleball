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

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_logs as H
from pb_api import RAW

TABLE = "pb_match_player_serve"
BATCH = 500


def _rows_for(matches):
    for m in matches:
        p = RAW / "match_logs" / m["match_id"][:2] / f"{m['match_id']}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only matches on/after YYYY-MM-DD")
    ap.add_argument("--limit", type=int)
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

    rows, n, t0, maxdate = [], 0, time.monotonic(), ""
    with httpx.Client(timeout=60) as client:
        def flush():
            nonlocal rows, n
            if not rows:
                return
            if not args.dry_run:
                _post(client, url, key, TABLE, rows,
                      params={"on_conflict": "match_id,player_uuid"})
            n += len(rows)
            rows = []
        for row in _rows_for(matches):
            maxdate = max(maxdate, row["match_date"])
            rows.append(row)
            if len(rows) >= BATCH:
                flush()
        flush()

        if not args.dry_run and n:
            _post(client, url, key, "pb_meta",
                  [{"key": "serve_rows", "value": str(n)},
                   {"key": "max_match_date", "value": maxdate}],
                  params={"on_conflict": "key"})
    print(f"{'would upsert' if args.dry_run else 'upserted'} {n} rows "
          f"(through {maxdate}) in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
