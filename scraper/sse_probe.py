"""Tier-2 live SSE capture (rte.pbgql.co) — decoded from the site bundle.

The pickleball.com client opens ONE Server-Sent-Events connection to
`rte.pbgql.co/live-scoring` and subscribes to matches/matchups by UUID.
Events arrive named by the match UUID (score updates) or `matchup_<uuid>`
(matchup/DreamBreaker state); the payload is JSON in the SSE `data:` field.
When the client asks for `withLogs`, per-rally `log_index` events also arrive
— that's the rally-resolution feed Tier 1 (polling) can't see.

Protocol (all client-side, NO server secret involved):
  URL     GET https://rte.pbgql.co/live-scoring?opts=slice,karma
          (append `withLogs` to opts only when subscribing to ONE match)
  Auth    PB-RTE-TOKEN: base64(JSON{ua, origin, fingerprint})
          The browser computes `fingerprint` via FingerprintJS; the server
          accepts any well-formed token (verified: a random hex fingerprint
          returns 200 + text/event-stream). It's a bot-shaping tag, not a
          credential — same "unauthenticated public feed" posture as the BFF.
  Subs    X-Request-Matches:            base64(comma-joined match uuids)
          X-Request-Matchups:           base64(comma-joined matchup uuids)
          X-Request-Tiebreaker-Matches: base64(comma-joined tiebreaker uuids)

Usage:
    python scraper/sse_probe.py                     # auto-discover today's live matches, stream
    python scraper/sse_probe.py --matches UUID,UUID # subscribe to explicit uuids
    python scraper/sse_probe.py --duration 600      # stop after 10 min (default 3600)
    python scraper/sse_probe.py --with-logs         # request per-rally logs (single match only)

Output: live/sse-YYYYMMDD.jsonl, one line per SSE event:
    {"ts": "...", "event": "<match_uuid>|matchup_<uuid>|message",
     "id": ..., "data": {parsed JSON} | "<raw>"}

This is a DISCOVERY/ground-truth tool: run it during a live match to capture
the real event shapes, then fold the parser into live_poller as Tier 2.
Polite: a single long-lived connection, exactly like one browser tab.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_api import UA
from live_poller import Poller, ACTIVE, mlp_match_state, ppa_match_state, TOUR_TZ

log = logging.getLogger("sse")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "live"

RTE_URL = "https://rte.pbgql.co/live-scoring"
ORIGIN = "https://pickleball.com"


def build_token(fingerprint: str = "0" * 32) -> str:
    """Mirror the client's PB-RTE-TOKEN: base64(JSON{ua, origin, fingerprint})."""
    payload = {"ua": UA, "origin": ORIGIN, "fingerprint": fingerprint}
    return base64.b64encode(json.dumps(payload).encode()).decode()


def b64_csv(uuids) -> str:
    return base64.b64encode(",".join(uuids).encode()).decode()


def collect_live_uuids(today: str):
    """Reuse the Tier-1 discovery to find match + matchup UUIDs to subscribe to.

    Returns (match_uuids, matchup_uuids). Subscribes to scheduled AND live
    matches so we're already connected when the first point is played.
    """
    p = Poller()
    targets = p.discover(today)
    match_uuids, matchup_uuids = set(), set()
    for tl, div in targets["mlp"]:
        q = (f"teamLeagueId={tl['uuid']}&organizationId={tl['organizationUuid']}"
             f"&divisionId={div['divisionUuid']}&seasonId={div['seasonUuid']}"
             f"&districtId={div['districtUuid']}&date={today}")
        if div.get("matchupGroupUuid"):
            q += f"&matchupGroupUuid={div['matchupGroupUuid']}"
        mus = p.c._get_json(
            f"/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?{q}").get("data") or []
        for mu in mus:
            if mu.get("matchupStatus") in (
                    "IN_PROGRESS_MATCHUP_STATUS",
                    "SCHEDULED_WAITING_FOR_HOME_TEAM_LINEUP",
                    "SCHEDULED_WAITING_FOR_AWAY_TEAM_LINEUP",
                    "SCHEDULED_MATCHUP_STATUS"):
                matchup_uuids.add(mu["uuid"].lower())
                detail = p.c._get_json(
                    f"/api/v2/results/getResultsMatchupData?matchupId={mu['uuid']}"
                ).get("data") or {}
                for m in detail.get("matches") or []:
                    if m.get("matchUuid"):
                        match_uuids.add(m["matchUuid"].lower())
    for tid, title, doubles in targets["ppa"]:
        ms = p.c._get_json(
            "/api/v1/results/getMatchInfosShort"
            f"?eventIds={','.join(doubles)}&date={today}").get("data") or []
        for m in ms:
            st = ppa_match_state(m)
            if st["status"] in ACTIVE and m.get("match_uuid"):
                match_uuids.add(m["match_uuid"].lower())
    return sorted(match_uuids), sorted(matchup_uuids)


def parse_sse(raw: str) -> dict:
    """Parse one SSE event block (matches the client's _parseEventChunk)."""
    ev = {"id": None, "retry": None, "data": "", "event": "message"}
    for line in raw.splitlines():
        line = line.rstrip()
        i = line.find(":")
        if i <= 0:
            continue
        field, val = line[:i], line[i + 1:].lstrip()
        if field not in ev:
            continue
        if field == "data":
            ev["data"] += val
        else:
            ev[field] = val
    return ev


def stream(match_uuids, matchup_uuids, duration, with_logs):
    OUT.mkdir(exist_ok=True)
    opts = ["slice", "karma"]
    if with_logs and len(match_uuids) == 1:
        opts.insert(0, "withLogs")
    elif with_logs:
        log.warning("--with-logs needs exactly one match; ignoring for %d", len(match_uuids))
    headers = {
        "User-Agent": UA,
        "Accept": "text/event-stream",
        "Origin": ORIGIN,
        "PB-RTE-TOKEN": build_token(),
    }
    if match_uuids:
        headers["X-Request-Matches"] = b64_csv(match_uuids)
    if matchup_uuids:
        headers["X-Request-Matchups"] = b64_csv(matchup_uuids)
    url = f"{RTE_URL}?opts={','.join(opts)}"
    day = dt.datetime.now(TOUR_TZ).strftime("%Y%m%d")
    out_path = OUT / f"sse-{day}.jsonl"
    log.info("subscribing: %d matches, %d matchups → %s",
             len(match_uuids), len(matchup_uuids), out_path)

    deadline = time.monotonic() + duration
    seen_types, n_events = {}, 0
    chunk = ""
    with httpx.Client(timeout=httpx.Timeout(15, read=None)) as c:
        with c.stream("GET", url, headers=headers) as r:
            log.info("connected: %s %s", r.status_code, r.headers.get("content-type"))
            r.raise_for_status()
            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    log.info("duration reached — closing")
                    break
                if line.strip():
                    chunk += line + "\n"
                    continue
                if not chunk.strip():
                    continue
                ev = parse_sse(chunk)
                chunk = ""
                try:
                    data = json.loads(ev["data"]) if ev["data"] else None
                except json.JSONDecodeError:
                    data = ev["data"]
                rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       "event": ev["event"], "id": ev["id"], "data": data}
                with out_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                n_events += 1
                seen_types[ev["event"]] = seen_types.get(ev["event"], 0) + 1
                log.info("event #%d: %s", n_events, ev["event"])
    log.info("done: %d events; types=%s", n_events, seen_types)
    return n_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", help="comma-joined match uuids (skip auto-discovery)")
    ap.add_argument("--matchups", help="comma-joined matchup uuids")
    ap.add_argument("--duration", type=int, default=3600, help="seconds to stream")
    ap.add_argument("--with-logs", action="store_true", help="request per-rally logs (1 match)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.matches:
        match_uuids = [u.strip().lower() for u in args.matches.split(",") if u.strip()]
        matchup_uuids = [u.strip().lower() for u in (args.matchups or "").split(",") if u.strip()]
    else:
        today = dt.datetime.now(TOUR_TZ).date().isoformat()
        match_uuids, matchup_uuids = collect_live_uuids(today)
        if not match_uuids and not matchup_uuids:
            log.info("no live-relevant matches today — nothing to subscribe to")
            return
    stream(match_uuids, matchup_uuids, args.duration, args.with_logs)


if __name__ == "__main__":
    main()
