"""Per-rally timing from the referee logs → data/rally_times.csv.

The referee logs carry `date_created` on every entry, and `log_type 12` marks
*rally underway*, so the gap to the resolving entry (14 point / 16 side-out /
23 second server) is that rally's duration.  Verified exactly against the
explicit `point_log` payload (`time_started` / `time_ended`), which carries the
same numbers but only appears on point rows — the type-12 gap covers all
rallies.

This is the ONLY measure in the stack that describes how a point was played
rather than who won it (see model/rally_duration.md), and it is not in
`pb_rally`, so it has to be extracted and persisted separately.

SELF-CONTAINED — no join required. Carries the score state, server number,
outcome, both player uuids and the timing, so it can be analysed on its own:

    match_id, game_number, rally_ord, server_score, receiver_score,
    server_number, outcome, start_utc, dur_s, clean, log_index,
    server_uuid, receiver_uuid

**Do NOT join on rally_ord.** It does not reliably align with
`pb_rally.rally_number`, because referees correct scores mid-game and
pb_rally applies correction handling (multi-point rewinds) that this naive
log walk deliberately does not replicate. Observed agreement is ~71%, and the
disagreements are real corrections, not bugs. `log_index` is the unambiguous
in-match key if a join back to the raw log is ever needed; player identity is
already here, which is the only thing a pb_rally join would have added.

`clean` is 0 when the rally sits within two log entries of a timeout, line
review or penalty — those produce multi-minute gaps that are not rally time.
Filter to clean=1 and a sane duration range for analysis; the raw rows are
kept so the exclusions stay auditable.

RESUMABLE: re-running skips match_ids already in the CSV, so this can be
stopped and restarted, and extended as the log cache grows.

    python scraper/extract_rally_times.py            # all known matches
    python scraper/extract_rally_times.py --limit 500
    python scraper/extract_rally_times.py --compress-only   # just refresh the .gz

Working file is `data/rally_times.csv` (plain, cheap to append during a
multi-hour run). The COMMITTED artifact is `data/rally_times.csv.gz` — the
plain CSV is gitignored, since at archive scale it is ~165 MB and this repo
takes nightly data commits. Resume reads whichever file exists.

Archive-scale note: the droplet already pulls these logs nightly for the
serve/return warehouse (deploy/run_logs_backfill.sh). The clean long-term home
for this is two extra columns on `pb_rally` via scraper/upload_supabase.py —
this CSV is the interim, and the pipeline that fills it is the same one.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "model"))

from pb_api import PBClient                                    # noqa: E402

# Working file is plain CSV (cheap appends during multi-hour runs); the
# COMMITTED artifact is the gzip, which is ~10x smaller — the uncompressed
# archive-scale file is ~165 MB, too large for a repo that takes nightly
# data commits. Resume reads whichever exists.
OUT = ROOT / "data" / "rally_times.csv"
OUT_GZ = ROOT / "data" / "rally_times.csv.gz"


def _open_read():
    """Existing rows, from the plain CSV if present else the gzip."""
    if OUT.exists():
        return OUT.open()
    if OUT_GZ.exists():
        return gzip.open(OUT_GZ, "rt")
    return None


def compress():
    """Refresh the committed .gz from the working CSV."""
    if not OUT.exists():
        return
    with OUT.open("rb") as a, gzip.open(OUT_GZ, "wb", compresslevel=9) as b:
        shutil.copyfileobj(a, b)
    print(f"  gzipped -> {OUT_GZ} "
          f"({OUT.stat().st_size / 1e6:.1f} MB -> {OUT_GZ.stat().st_size / 1e6:.1f} MB)")
FIELDS = ["match_id", "game_number", "rally_ord", "server_score",
          "receiver_score", "server_number", "outcome", "start_utc",
          "dur_s", "clean", "log_index", "server_uuid", "receiver_uuid"]
RALLY_START, POINT, SIDEOUT, SECOND = 12, 14, 16, 23
OUTCOME = {POINT: "point", SIDEOUT: "sideout", SECOND: "second"}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rows_for(logs):
    """Per-rally timing rows for one match, or [] if the log is unusable."""
    L = logs.get("data") if isinstance(logs, dict) else logs
    if not L:
        return []
    L = sorted(L, key=lambda r: r.get("log_index", 0))
    dirty = set()
    for r in L:
        if r.get("timeout_log") or r.get("line_review_log") or r.get("penalty_log"):
            for d in (-2, -1, 0, 1, 2):
                dirty.add(r.get("log_index", 0) + d)
    out, start, sidx, ordn = [], None, None, {}
    for r in L:
        lt = r.get("log_type")
        if lt == RALLY_START:
            start, sidx = r.get("date_created"), r.get("log_index")
            continue
        if lt not in OUTCOME:
            continue
        # EMIT EVERY resolving entry, even without a preceding type-12 start
        # marker. pb_rally is built from exactly these 14/16/23 rows, so
        # skipping any of them desynchronises rally_ord from
        # pb_rally.rally_number -- that bug cost 28% of the join.
        s = r.get("start_score_current_game_string")
        gn = r.get("game_number")
        ordn[gn] = ordn.get(gn, 0) + 1
        try:
            a, b, n = (int(x) for x in s.split("-")) if s else (None, None, None)
        except ValueError:
            a = b = n = None
        dur, clean = "", 0
        if start and r.get("date_created"):
            dur = int(round((_ts(r["date_created"]) - _ts(start)).total_seconds()))
            clean = 0 if (sidx in dirty or r.get("log_index") in dirty) else 1
        out.append({"match_id": r.get("match_uuid"), "game_number": gn,
                    "rally_ord": ordn[gn], "server_score": a,
                    "receiver_score": b, "server_number": n,
                    "outcome": OUTCOME[lt], "start_utc": start or "",
                    "dur_s": dur, "clean": clean,
                    "log_index": r.get("log_index"),
                    "server_uuid": (r.get("server_uuid") or "").lower(),
                    "receiver_uuid": (r.get("receiver_uuid") or "").lower()})
        start = None
    return out


def known_match_ids():
    """Every match id we know about, newest cache first."""
    ids = []
    cache = ROOT / "model" / "_clutch_srm_cache.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        ids = sorted({r["match_id"] for r in blob["rallies"]})
    logs_dir = ROOT / "raw" / "match_logs"
    if logs_dir.exists():
        have = {p.stem for sub in logs_dir.iterdir() if sub.is_dir()
                for p in sub.glob("*.json")}
        ids = sorted(set(ids) | have)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compress-only", action="store_true",
                    help="just refresh the committed .gz from the working CSV")
    args = ap.parse_args()
    if args.compress_only:
        compress()
        return

    done = set()
    fh0 = _open_read()
    if fh0 is not None:
        with fh0 as f:
            done = {r["match_id"] for r in csv.DictReader(f)}
    # seed the working CSV from the committed gzip if only the gzip is present
    if not OUT.exists() and OUT_GZ.exists():
        with gzip.open(OUT_GZ, "rb") as a, OUT.open("wb") as b:
            shutil.copyfileobj(a, b)
    ids = [m for m in known_match_ids() if m not in done]
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(done)} matches already in {OUT.name}; {len(ids)} to do")

    new = not OUT.exists()
    fh = OUT.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    c = PBClient()
    ok = bad = nrows = 0
    for i, mid in enumerate(ids, 1):
        try:
            rows = rows_for(c.match_logs(mid))
        except Exception:
            bad += 1
            continue
        if not rows:
            bad += 1
            continue
        for r in rows:
            r["match_id"] = r["match_id"] or mid
        w.writerows(rows)
        ok += 1
        nrows += len(rows)
        if i % 100 == 0:
            fh.flush()
            print(f"  {i}/{len(ids)}  ok={ok} bad={bad} rows={nrows}", flush=True)
    fh.close()
    print(f"done: {ok} matches, {bad} unusable, {nrows} rally rows -> {OUT}")
    compress()


if __name__ == "__main__":
    main()
