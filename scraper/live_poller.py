"""Tier-1 live score poller (see design_handoff.md / live win probability).

Polls the same open BFF endpoints the harvester uses, but for TODAY's live
matches, and appends a JSONL event whenever any score/server state changes.
This is the fallback/ground-truth feed for live win-probability charts; the
Tier-2 SSE listener (rte.pbgql.co, still undiscovered) would layer on top
and reconcile against this.

Usage:
    python scraper/live_poller.py                # poll until day's play ends
    python scraper/live_poller.py --once         # one sweep, print, exit
    python scraper/live_poller.py --interval 20  # seconds between sweeps
    python scraper/live_poller.py --date 2026-07-16  # override event date

Output: live/events-YYYYMMDD.jsonl, one line per observed state change:
    {"ts": "...", "tour": "MLP"|"PPA", "match_uuid": ..., "matchup_uuid": ...,
     "status": int, "players": {...}, "scores": [[s1,s2] per game],
     "current_game": int, "server": ..., "server_from_team": ...}

Notes:
  - Must run on a persistent machine during event windows (a VPS, a Pi —
    not an ephemeral CI container). deploy/ has the systemd kit that runs
    this unattended (daily 09:15 PT timer + end-of-day commit & push).
  - Read-only, polite: one sweep hits a handful of endpoints; default 25 s
    interval. Do not lower the interval below ~15 s.
  - State diffing is in-memory; restarting mid-day re-emits current state
    once (downstream consumers should dedupe on (match_uuid, scores)).
  - The API's date params are event-local dates. Default "today" is taken
    in US/Pacific (westernmost tour venue), so the date never rolls over
    while any US venue still has live play — a UTC machine clock flips at
    5 PM Pacific, mid-evening, and would discover the wrong day's matchups.
    Use --date to override.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_api import PBClient
from harvest import is_mlp_league, is_ppa_tournament

log = logging.getLogger("live")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "live"
TOUR_TZ = ZoneInfo("America/Los_Angeles")

GAME_ORDINALS = ["One", "Two", "Three", "Four", "Five"]
SNAKE_ORDINALS = ["one", "two", "three", "four", "five"]

# matchStatus: 1 scheduled, 2 in progress, 4 completed (observed values)
ACTIVE = {1, 2}


def mlp_match_state(m):
    scores = []
    for o in GAME_ORDINALS:
        s1, s2 = m.get(f"teamOneGame{o}Score") or 0, m.get(f"teamTwoGame{o}Score") or 0
        if s1 or s2 or not scores:
            scores.append([s1, s2])
    return {
        "status": m.get("matchStatus"),
        "scores": scores,
        "current_game": m.get("currentGame"),
        "server": m.get("currentServingNumber"),
        "server_from_team": m.get("serverFromTeam"),
        "players": {
            "t1": [m.get("teamOnePlayerOneName"), m.get("teamOnePlayerTwoName")],
            "t2": [m.get("teamTwoPlayerOneName"), m.get("teamTwoPlayerTwoName")],
        },
        "abbrev": m.get("matchAbbreviation"),
    }


def ppa_match_state(m):
    scores = []
    for o in SNAKE_ORDINALS:
        s1 = m.get(f"team_one_game_{o}_score") or 0
        s2 = m.get(f"team_two_game_{o}_score") or 0
        if s1 or s2 or not scores:
            scores.append([s1, s2])
    return {
        "status": m.get("match_status"),
        "scores": scores,
        "current_game": None,
        "server": m.get("current_serving_number"),
        "server_from_team": m.get("server_from_team"),
        "players": {
            "t1": [m.get("team_one_player_one_name"), m.get("team_one_player_two_name")],
            "t2": [m.get("team_two_player_one_name"), m.get("team_two_player_two_name")],
        },
        "event": m.get("event_title"),
    }


class Poller:
    def __init__(self):
        self.c = PBClient()
        self.state = {}       # match_uuid -> last serialized state
        self.targets = None   # discovered once per day
        OUT.mkdir(exist_ok=True)

    def discover(self, today):
        """Find today's live-relevant MLP divisions and PPA doubles events.
        Uses direct (uncached) fetches — live data must never come from cache."""
        targets = {"mlp": [], "ppa": []}
        tls = self.c._get_json(
            f"/api/v2/results/getTeamLeaguesResultsOnDate?date={today}").get("data") or []
        for tl in tls:
            if is_mlp_league(tl):
                for div in tl.get("divisions") or []:
                    targets["mlp"].append((tl, div))
        ts = self.c._get_json(
            f"/api/v1/results/getTournamentsOnDate?date={today}").get("data") or []
        for t in ts:
            if not is_ppa_tournament(t):
                continue
            tid = t["TournamentID"]
            groups = self.c._get_json(
                "/api/v1/results/getListActiveEventsFlatGroup"
                f"?tournamentId={tid}&date={today}")
            groups = groups if isinstance(groups, list) else groups.get("data") or []
            pro = [g for g in groups if "pro" in g["group_title"].lower()
                   and "senior" not in g["group_title"].lower()
                   and "junior" not in g["group_title"].lower()]
            if not pro:
                continue
            ev = self.c._get_json(
                "/api/v1/results/getTournamentEventsShort"
                f"?tournamentId={tid}&formatId={pro[0]['format_id']}"
                f"&playerGroupId={pro[0]['player_group_id']}"
                f"&bracketLevelId={pro[0]['bracket_level_id']}&date={today}"
            ).get("data") or []
            doubles = [e["uuid"] for e in ev if "doubles" in e["title"].lower()]
            if doubles:
                targets["ppa"].append((tid, t["Title"], doubles))
        log.info("targets: %d MLP divisions, %d PPA tournaments",
                 len(targets["mlp"]), len(targets["ppa"]))
        return targets

    def sweep(self, today):
        events, n_active = [], 0
        for tl, div in self.targets["mlp"]:
            q = (f"teamLeagueId={tl['uuid']}&organizationId={tl['organizationUuid']}"
                 f"&divisionId={div['divisionUuid']}&seasonId={div['seasonUuid']}"
                 f"&districtId={div['districtUuid']}&date={today}")
            if div.get("matchupGroupUuid"):
                q += f"&matchupGroupUuid={div['matchupGroupUuid']}"
            mus = self.c._get_json(
                f"/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?{q}").get("data") or []
            for mu in mus:
                if mu.get("matchupStatus") not in (
                        "IN_PROGRESS_MATCHUP_STATUS",
                        "SCHEDULED_WAITING_FOR_HOME_TEAM_LINEUP",
                        "SCHEDULED_WAITING_FOR_AWAY_TEAM_LINEUP",
                        "SCHEDULED_MATCHUP_STATUS"):
                    continue
                detail = self.c._get_json(
                    f"/api/v2/results/getResultsMatchupData?matchupId={mu['uuid']}"
                ).get("data") or {}
                for m in detail.get("matches") or []:
                    st = mlp_match_state(m)
                    if st["status"] in ACTIVE:
                        n_active += 1
                    events += self.emit("MLP", (m.get("matchUuid") or "").lower(),
                                        mu["uuid"], st)
        for tid, title, doubles in self.targets["ppa"]:
            ms = self.c._get_json(
                "/api/v1/results/getMatchInfosShort"
                f"?eventIds={','.join(doubles)}&date={today}").get("data") or []
            for m in ms:
                st = ppa_match_state(m)
                if st["status"] in ACTIVE:
                    n_active += 1
                events += self.emit("PPA", (m.get("match_uuid") or "").lower(), None, st)
        return events, n_active

    def emit(self, tour, muid, matchup_uuid, st):
        key = json.dumps(st["scores"]) + str(st.get("server")) + str(st["status"])
        if self.state.get(muid) == key:
            return []
        self.state[muid] = key
        ev = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "tour": tour, "match_uuid": muid, "matchup_uuid": matchup_uuid, **st}
        day = self.today.replace("-", "")
        with (OUT / f"events-{day}.jsonl").open("a") as fh:
            fh.write(json.dumps(ev) + "\n")
        return [ev]

    def run(self, interval, once=False, date=None):
        today = date or dt.datetime.now(TOUR_TZ).date().isoformat()
        self.today = today
        self.targets = self.discover(today)
        if not self.targets["mlp"] and not self.targets["ppa"]:
            log.info("no live-relevant events today — exiting")
            return
        idle_sweeps = 0
        while True:
            try:
                events, n_active = self.sweep(today)
                for ev in events:
                    log.info("event: %s %s %s", ev["tour"],
                             ev["players"]["t1"], ev["scores"])
                idle_sweeps = 0 if n_active else idle_sweeps + 1
            except Exception:
                log.exception("sweep failed; continuing")
            if once:
                break
            # stop after ~30 min with zero scheduled/live matches remaining
            if idle_sweeps * interval > 1800:
                log.info("no active matches for 30 min — day over, exiting")
                break
            time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--date", help="event date YYYY-MM-DD (default: today in US/Pacific)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.interval < 15:
        raise SystemExit("interval below 15s is impolite; refusing")
    if args.date:
        dt.date.fromisoformat(args.date)
    Poller().run(args.interval, once=args.once, date=args.date)


if __name__ == "__main__":
    main()
