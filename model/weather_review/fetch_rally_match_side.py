"""Pull pb_rally (doubles) from Supabase and aggregate to match x server-side.

Aggregation happens streaming, in this script — the agent never sees the
993k raw rally rows. Output: a compact CSV in the scratchpad that every
downstream test in this review reads.

    python model/weather_review/fetch_rally_match_side.py [outdir]

Columns per (match_id, side):
    n, w                serve rallies / serve rallies won by that side
    n_point,n_second,n_sideout   outcome split
    n1,w1,n2,w2         first-server / second-server rallies and wins
    b0..b15             serve rallies with leader score = min(max(ss,rs),15)
    c0..c15             ... of which won
Side is the pb_rally server_side integer; a companion CSV maps
(match_id, side) -> one player_uuid so callers can align with games.csv.
"""
from __future__ import annotations

import csv
import gzip
import io
import os
import ssl
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://nwgxyytowbluuykbdcfc.supabase.co/rest/v1"
KEY = os.environ.get("SUPABASE_ANON_KEY", (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53Z3h5"
    "eXRvd2JsdXV5a2JkY2ZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE1MzE1ODYsImV4cCI6MjA4"
    "NzEwNzU4Nn0.ktyO_FYxFP5xwQB0TXucnPMjMQi0HAVKGSdC0miDi4w"))
CTX = ssl.create_default_context(
    cafile=os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt")
PAGE = 1000


def get_csv(path: str, params: str, tries: int = 5):
    url = f"{BASE}/{path}?{params}"
    req = urllib.request.Request(url, headers={
        "apikey": KEY, "Authorization": f"Bearer {KEY}",
        "Accept": "text/csv", "Accept-Encoding": "gzip"})
    last = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            return list(csv.DictReader(io.StringIO(raw.decode())))
        except Exception as e:  # transient proxy/gateway hiccups
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"failed {url}: {last}")


def shard_pages(table, cols, extra, shard, sink):
    """Keyset-paginate one match_id range, feeding rows to sink().

    Ranges are gte/lt on match_id so the (match_id, game_number,
    rally_number) primary key serves both the filter and the ordering —
    a LIKE prefix filter makes the planner sort the whole table and the
    statement times out.
    """
    lo, hi = shard
    cursor = lo
    seen_last = None
    while True:
        params = (f"select={cols}&{extra}"
                  f"&match_id=gte.{cursor}&match_id=lt.{hi}"
                  f"&order=match_id.asc,game_number.asc,rally_number.asc"
                  f"&limit={PAGE}")
        rows = get_csv(table, params)
        if not rows:
            return
        ids = [r["match_id"] for r in rows]
        if len(rows) < PAGE:
            sink(rows)
            return
        last = ids[-1]
        if last == ids[0]:            # a single match overflows a page
            sink(rows)
            raise RuntimeError(f"match {last} exceeds page size")
        keep = [r for r in rows if r["match_id"] != last]
        sink(keep)
        if last == seen_last:
            raise RuntimeError("cursor stuck")
        seen_last, cursor = last, last


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- side -> a player uuid, so callers can align sides with games.csv
    print("fetching pb_match_player_serve ...", flush=True)
    side_map = {}
    lock = threading.Lock()

    def sink_side(rows):
        with lock:
            for r in rows:
                side_map.setdefault((r["match_id"], r["side"]), r["player_uuid"])

    def side_shard(sh):
        lo, hi = sh
        cursor = lo
        seen = None
        while True:
            p = (f"select=match_id,player_uuid,side&discipline=eq.doubles"
                 f"&match_id=gte.{cursor}&match_id=lt.{hi}"
                 f"&order=match_id.asc,player_uuid.asc&limit={PAGE}")
            rows = get_csv("pb_match_player_serve", p)
            if not rows:
                return
            if len(rows) < PAGE:
                sink_side(rows)
                return
            last = rows[-1]["match_id"]
            sink_side([r for r in rows if r["match_id"] != last])
            if last == seen:
                raise RuntimeError("stuck")
            seen, cursor = last, last

    hexd = "0123456789abcdef"
    bounds = [a + b for a in hexd for b in hexd] + ["g"]
    shards = list(zip(bounds, bounds[1:]))
    run_parallel(side_shard, shards)
    print(f"  {len(side_map)} (match, side) rows", flush=True)

    # ---- rallies
    print("fetching pb_rally (doubles) ...", flush=True)
    agg = {}
    counts = defaultdict(int)

    def sink(rows):
        local = {}
        for r in rows:
            key = (r["match_id"], r["server_side"])
            a = local.get(key)
            if a is None:
                a = local[key] = [0] * (9 + 32)
            won = 1 if r["won"] == "1" else 0
            a[0] += 1
            a[1] += won
            oc = r["outcome"]
            a[2 if oc == "point" else 3 if oc == "second" else 4] += 1
            sn = r["server_number"]
            if sn == "2":
                a[7] += 1
                a[8] += won
            else:
                a[5] += 1
                a[6] += won
            try:                      # a few logs carry no running score
                ss, rs = int(r["server_score"]), int(r["receiver_score"])
            except ValueError:
                continue
            b = min(max(ss, rs), 15)
            a[9 + b] += 1
            a[25 + b] += won
        with lock:
            counts["rows"] += len(rows)
            for k, v in local.items():
                cur = agg.get(k)
                if cur is None:
                    agg[k] = v
                else:
                    for i, x in enumerate(v):
                        cur[i] += x

    cols = ("match_id,game_number,rally_number,server_side,server_number,"
            "outcome,won,server_score,receiver_score")
    run_parallel(lambda sh: shard_pages("pb_rally", cols,
                                        "discipline=eq.doubles", sh, sink),
                 shards)
    print(f"  {counts['rows']} rallies -> {len(agg)} (match, side) cells",
          flush=True)

    with open(outdir / "rally_match_side.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "side", "n", "w", "n_point", "n_second",
                    "n_sideout", "n1", "w1", "n2", "w2"]
                   + [f"b{i}" for i in range(16)] + [f"c{i}" for i in range(16)])
        for (m, s), a in sorted(agg.items()):
            w.writerow([m, s] + a)
    with open(outdir / "rally_side_player.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "side", "player_uuid"])
        for (m, s), p in sorted(side_map.items()):
            w.writerow([m, s, p])
    print("wrote", outdir / "rally_match_side.csv")


def run_parallel(fn, items, workers=8):
    items = list(items)
    errs = []

    def work():
        while True:
            with ilock:
                if not items:
                    return
                it = items.pop()
            try:
                fn(it)
            except Exception as e:
                errs.append(e)

    ilock = threading.Lock()
    ts = [threading.Thread(target=work) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]


if __name__ == "__main__":
    main()
