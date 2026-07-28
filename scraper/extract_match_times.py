"""Extract per-match start/end times (and court) into data/match_times.csv.

    python scraper/extract_match_times.py

Walks the same BFF endpoints as harvest.py for every date in games.csv,
through PBClient's raw/ cache — so on a machine with a warm raw/ (droplet,
Actions) it is nearly instant and offline; on a cold machine it refetches
at the polite 1 req/s (~25-40 min for the full 2024-26 sweep).

Emitted per match:
    match_id, event_id, tour, date, court, tz_abbrev,
    planned_start_local, start_local, completed_local   (venue-LOCAL — the
        BFF's localDateMatch* carry a fake Z suffix, see recon.md),
    g1_end_utc..g5_end_utc                              (true UTC),
    court_indoor                                        (MLP field, often null)

This is the missing piece for the HOUR-level weather join:
data/event_weather_hourly.csv is keyed by venue-local time, so
start_local's hour joins directly.
"""
from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import is_mlp_league, is_ppa_tournament  # noqa: E402
from pb_api import MATCHUP_FINAL, PBClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("extract_match_times")

PPA_FIELDS = {  # snake_case in getMatchInfosShort
    "planned_start_local": "local_date_match_planned_start",
    "start_local": "local_date_match_start",
    "completed_local": "local_date_match_completed",
    "g1_end_utc": "game_one_end_date", "g2_end_utc": "game_two_end_date",
    "g3_end_utc": "game_three_end_date", "g4_end_utc": "game_four_end_date",
    "g5_end_utc": "game_five_end_date",
}
MLP_FIELDS = {  # camelCase in getResultsMatchupData.matches
    "planned_start_local": "localDateMatchPlannedStart",
    "start_local": "localDateMatchStart",
    "completed_local": "localDateMatchCompleted",
    "g1_end_utc": "gameOneEndDate", "g2_end_utc": "gameTwoEndDate",
    "g3_end_utc": "gameThreeEndDate", "g4_end_utc": "gameFourEndDate",
    "g5_end_utc": "gameFiveEndDate",
}


def game_dates() -> dict[str, set[str]]:
    """tour -> set of ISO dates with games."""
    days = defaultdict(set)
    with open(ROOT / "data/games.csv") as f:
        for r in csv.DictReader(f):
            if r["date"]:
                days[r["tour"]].add(r["date"])
    return days


def ppa_day(c: PBClient, d: date, rows: list[dict]):
    for t in c.tournaments_on_date(d):
        if not is_ppa_tournament(t):
            continue
        tid = t["TournamentID"]
        try:
            groups = c.events_flat_group(tid, d)
        except PermissionError:
            continue
        pro = [g for g in groups
               if "pro" in g["group_title"].lower()
               and "senior" not in g["group_title"].lower()
               and "junior" not in g["group_title"].lower()]
        if not pro:
            continue
        try:
            events = c.tournament_events_short(tid, pro[0], d)
        except PermissionError:
            continue
        doubles = [e["uuid"] for e in events if "doubles" in e["title"].lower()]
        if not doubles:
            continue
        try:
            matches = c.match_infos_short(tid, doubles, d)
        except PermissionError:
            continue
        for m in matches:
            mid = (m.get("match_uuid") or "").lower()
            if not mid:
                continue
            row = {"match_id": mid, "event_id": tid.lower(), "tour": "PPA",
                   "date": str(d), "court": m.get("court_title") or "",
                   "tz_abbrev": m.get("timezone_abbreviation") or "",
                   "court_indoor": ""}
            for out, key in PPA_FIELDS.items():
                row[out] = m.get(key) or ""
            rows.append(row)


def mlp_day(c: PBClient, d: date, rows: list[dict]):
    for tl in c.team_leagues_on_date(d):
        if not is_mlp_league(tl):
            continue
        for div in tl.get("divisions") or []:
            try:
                matchups = c.tl_matchups_short(tl, div, d)
            except PermissionError:
                continue
            for mu in matchups:
                completed = mu.get("matchupStatus") in MATCHUP_FINAL
                try:
                    data = c.matchup_data(mu["uuid"], volatile=not completed)
                except PermissionError:
                    continue
                for m in data.get("matches") or []:
                    mid = (m.get("matchUuid") or "").lower()
                    if not mid:
                        continue
                    ind = m.get("courtIndoor")
                    row = {"match_id": mid,
                           "event_id": (div.get("matchupGroupUuid") or "").lower(),
                           "tour": "MLP", "date": str(d),
                           "court": m.get("courtTitle") or "",
                           "tz_abbrev": "",
                           "court_indoor": "" if ind is None else str(ind)}
                    for out, key in MLP_FIELDS.items():
                        row[out] = m.get(key) or ""
                    rows.append(row)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    days = game_dates()
    rows: list[dict] = []
    c = PBClient()
    all_days = sorted({(d, tour) for tour, ds in days.items() for d in ds})
    for i, (d, tour) in enumerate(all_days):
        dd = date.fromisoformat(d)
        (ppa_day if tour == "PPA" else mlp_day)(c, dd, rows)
        if (i + 1) % 25 == 0:
            log.info("%d/%d event-days swept, %d matches", i + 1,
                     len(all_days), len(rows))

    # de-dupe (a match can appear on consecutive days' sweeps): keep the row
    # with the most fields filled
    best: dict[str, dict] = {}
    for r in rows:
        filled = sum(1 for v in r.values() if v)
        if r["match_id"] not in best or filled > best[r["match_id"]][0]:
            best[r["match_id"]] = (filled, r)

    out = ROOT / "data/match_times.csv"
    cols = ["match_id", "event_id", "tour", "date", "court", "tz_abbrev",
            "planned_start_local", "start_local", "completed_local",
            "g1_end_utc", "g2_end_utc", "g3_end_utc", "g4_end_utc",
            "g5_end_utc", "court_indoor"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for _, r in sorted(best.values(), key=lambda kv: kv[1]["match_id"]):
            w.writerow(r)
    log.info("wrote %s (%d matches)", out, len(best))


if __name__ == "__main__":
    main()
