"""Harvest 2026 MLP + PPA pro-doubles results into the raw/ cache.

Usage:
    python scraper/harvest.py [--start 2026-01-01] [--end today] [--refresh-all]

Idempotent and resumable: every response is cached under raw/; immutable
(older than the volatile window) cache entries are never refetched, so
re-runs only hit the network for new/recent dates.

Selection rules (documented in recon.md):
  MLP:  team leagues with organizationSlug == "major-league-pickleball"
        and title starting "MLP " (excludes "Junior MLP …", MLP Australia).
  PPA:  tournaments whose contact emails end in @ppatour.com or whose title
        contains the standalone word "PPA", excluding the Australia/Asia
        franchise tours and college tour; then only bracket groups titled
        "Pro Events" (not Senior Pro / Junior), and within those only
        events whose title contains "Doubles".
Amateur draws are never even fetched: they live in non-Pro groups.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_api import RAW, PBClient

log = logging.getLogger("harvest")

SEASON_START = date(2026, 1, 1)

PPA_EXCLUDE = re.compile(r"PPA\s+TOUR\s+AUSTRALIA|PPA\s+Asia|College Pickleball", re.I)
PPA_WORD = re.compile(r"\bPPA\b")
MLP_ORG_SLUG = "major-league-pickleball"

# matchupStatus values that are final — anything else is treated as
# still-mutable so the matchup detail is refetched on the next run.
MATCHUP_FINAL = {"COMPLETED_MATCHUP_STATUS", "BYE_MATCHUP_STATUS"}


def is_ppa_tournament(t: dict) -> bool:
    title = t.get("Title") or ""
    if PPA_EXCLUDE.search(title):
        return False
    emails = " ".join(str(t.get(k) or "") for k in t if str(k).startswith("ContactForm"))
    return "ppatour.com" in emails or bool(PPA_WORD.search(title))


def is_mlp_league(tl: dict) -> bool:
    # org slug is the reliable key: city stops are titled "MLP <City>" but the
    # Grand Rapids stop is "Edward Jones Mid-Season Tournament". Junior events
    # share the org, so exclude those by title. (MLP Australia is a different
    # org, ppa-tour-australia.)
    return (
        tl.get("organizationSlug") == MLP_ORG_SLUG
        and "junior" not in str(tl.get("title") or "").lower()
    )


def harvest_ppa_day(c: PBClient, d: date, manifest: dict):
    for t in c.tournaments_on_date(d):
        if not is_ppa_tournament(t):
            continue
        tid = t["TournamentID"]
        title = t["Title"]
        try:
            groups = c.events_flat_group(tid, d)
        except PermissionError as e:
            log.warning("flat_group denied for %s (%s): %s", title, d, e)
            continue
        pro = [g for g in groups
               if "pro" in g["group_title"].lower()
               and "senior" not in g["group_title"].lower()
               and "junior" not in g["group_title"].lower()]
        if not pro:
            continue  # amateur-only tournament that day
        try:
            events = c.tournament_events_short(tid, pro[0], d)
        except PermissionError as e:
            log.warning("events_short denied for %s (%s): %s", title, d, e)
            continue
        doubles = [e for e in events if "doubles" in e["title"].lower()]
        if not doubles:
            continue
        try:
            matches = c.match_infos_short(tid, [e["uuid"] for e in doubles], d)
        except PermissionError as e:
            log.warning("match_infos denied for %s (%s): %s", title, d, e)
            continue
        manifest["ppa"].setdefault(tid, {"title": title, "days": {}})
        manifest["ppa"][tid]["days"][str(d)] = {
            "events": {e["uuid"]: e["title"] for e in doubles},
            "n_matches": len(matches),
        }
        log.info("PPA %s %s: %d doubles-pro matches (%s)",
                 d, title, len(matches), ", ".join(e["title"] for e in doubles))


def harvest_mlp_day(c: PBClient, d: date, manifest: dict):
    for tl in c.team_leagues_on_date(d):
        if not is_mlp_league(tl):
            continue
        for div in tl.get("divisions") or []:
            try:
                matchups = c.tl_matchups_short(tl, div, d)
            except PermissionError as e:
                log.warning("matchups denied for %s (%s): %s", tl["title"], d, e)
                continue
            for mu in matchups:
                muid = mu["uuid"]
                completed = mu.get("matchupStatus") in MATCHUP_FINAL
                try:
                    c.matchup_data(muid, volatile=not completed)
                except PermissionError as e:
                    log.warning("matchup_data denied %s: %s", muid, e)
                    continue
                rec = manifest["mlp"].setdefault(muid, {})
                rec.update({
                    "event": tl["title"],
                    "date": str(d),
                    "matchup_group": div.get("matchupGroupUuid"),
                    "status": mu.get("matchupStatus"),
                    "teams": f"{mu.get('teamOneTitle')} vs {mu.get('teamTwoTitle')}",
                })
            if matchups:
                log.info("MLP %s %s: %d matchups", d, tl["title"], len(matchups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=SEASON_START)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--refresh-all", action="store_true",
                    help="ignore all caches and refetch everything")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = PBClient(refresh_all=args.refresh_all)

    manifest = {"ppa": {}, "mlp": {}}
    d = args.start
    while d <= args.end:
        try:
            harvest_ppa_day(c, d, manifest)
            harvest_mlp_day(c, d, manifest)
        except Exception:
            log.exception("day %s failed; continuing", d)
        d += timedelta(days=1)

    RAW.mkdir(exist_ok=True)
    # per-run log (covers only this run's date window) — raw/ is the source of
    # truth for parsing; this is just a human-readable summary of the sweep
    (RAW / "last_run_manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("done: %d network calls, %d cache hits", c.network_calls, c.cache_hits)
    log.info("PPA tournaments harvested: %d | MLP matchups harvested: %d",
             len(manifest["ppa"]), len(manifest["mlp"]))


if __name__ == "__main__":
    main()
