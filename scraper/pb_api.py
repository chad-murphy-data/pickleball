"""Client for pickleball.com's public BFF (/api/v1|v2/results/*).

These same-origin Next.js routes proxy the authenticated api.pickleball.com
backend server-side — no token needed from us. All calls are GET.

Every response is cached to raw/ before use. Re-parsing never re-hits the
network; recent dates (still-mutable results) are refetched.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

log = logging.getLogger("pb_api")

BASE = "https://pickleball.com"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
RAW = Path(__file__).resolve().parent.parent / "raw"

MIN_INTERVAL = 0.8  # seconds between network hits — polite hobby pace
# Results for the trailing window may still change (live events, score
# corrections); anything older is treated as immutable.
VOLATILE_DAYS = 3


class PBClient:
    def __init__(self, refresh_all: bool = False):
        self.http = httpx.Client(
            base_url=BASE,
            timeout=45,
            headers={"User-Agent": UA, "Accept": "application/json"},
        )
        self.refresh_all = refresh_all
        self._last_hit = 0.0
        self.network_calls = 0
        self.cache_hits = 0

    # ---------- plumbing ----------

    def _throttle(self):
        wait = self._last_hit + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.2))
        self._last_hit = time.monotonic()

    def _get_json(self, path: str) -> dict | list:
        last_err = None
        for attempt in range(4):
            self._throttle()
            try:
                r = self.http.get(path)
            except httpx.HTTPError as e:
                last_err = e
                time.sleep(2**attempt)
                continue
            self.network_calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403, 404):
                # permanent for our purposes — do not hammer
                raise PermissionError(f"{r.status_code} for {path}: {r.text[:120]}")
            last_err = RuntimeError(f"{r.status_code} for {path}")
            time.sleep(2**attempt)
        raise RuntimeError(f"giving up on {path}: {last_err}")

    def _cached(self, cache_path: Path, path: str, volatile: bool = False):
        """Fetch path, caching the body at cache_path.

        volatile=True forces a refetch (used for dates in the mutable window).
        """
        if cache_path.exists() and not volatile and not self.refresh_all:
            self.cache_hits += 1
            return json.loads(cache_path.read_text())
        body = self._get_json(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=1))
        tmp.replace(cache_path)
        return body

    @staticmethod
    def _is_volatile(d: date) -> bool:
        return d >= date.today() - timedelta(days=VOLATILE_DAYS)

    # ---------- endpoints ----------

    def tournaments_on_date(self, d: date) -> list[dict]:
        p = RAW / "tournaments_on_date" / f"{d}.json"
        body = self._cached(p, f"/api/v1/results/getTournamentsOnDate?date={d}",
                            volatile=self._is_volatile(d))
        return body.get("data") or []

    def team_leagues_on_date(self, d: date) -> list[dict]:
        p = RAW / "team_leagues_on_date" / f"{d}.json"
        body = self._cached(p, f"/api/v2/results/getTeamLeaguesResultsOnDate?date={d}",
                            volatile=self._is_volatile(d))
        return body.get("data") or []

    def events_flat_group(self, tournament_id: str, d: date) -> list[dict]:
        p = RAW / "events_flat_group" / tournament_id / f"{d}.json"
        body = self._cached(
            p,
            f"/api/v1/results/getListActiveEventsFlatGroup?tournamentId={tournament_id}&date={d}",
            volatile=self._is_volatile(d),
        )
        return body if isinstance(body, list) else body.get("data") or []

    def tournament_events_short(self, tournament_id: str, group: dict, d: date) -> list[dict]:
        p = RAW / "tournament_events_short" / tournament_id / f"{d}.json"
        body = self._cached(
            p,
            "/api/v1/results/getTournamentEventsShort"
            f"?tournamentId={tournament_id}&formatId={group['format_id']}"
            f"&playerGroupId={group['player_group_id']}"
            f"&bracketLevelId={group['bracket_level_id']}&date={d}",
            volatile=self._is_volatile(d),
        )
        return body.get("data") or []

    def match_infos_short(self, tournament_id: str, event_ids: list[str], d: date) -> list[dict]:
        p = RAW / "match_infos_short" / tournament_id / f"{d}.json"
        ids = ",".join(event_ids)
        body = self._cached(
            p,
            f"/api/v1/results/getMatchInfosShort?eventIds={ids}&date={d}",
            volatile=self._is_volatile(d),
        )
        return body.get("data") or []

    def match_infos_short_singles(self, tournament_id: str, event_ids: list[str], d: date) -> list[dict]:
        """Same endpoint, separate cache dir: singles events were added later
        and the doubles cache is keyed by (tournament, date) only — mixing
        them would silently mask doubles refetches."""
        p = RAW / "match_infos_short_singles" / tournament_id / f"{d}.json"
        ids = ",".join(event_ids)
        body = self._cached(
            p,
            f"/api/v1/results/getMatchInfosShort?eventIds={ids}&date={d}",
            volatile=self._is_volatile(d),
        )
        return body.get("data") or []

    def tl_matchups_short(self, tl: dict, division: dict, d: date) -> list[dict]:
        p = RAW / "tl_matchups_short" / tl["uuid"] / f"{d}.json"
        q = (
            f"teamLeagueId={tl['uuid']}&organizationId={tl['organizationUuid']}"
            f"&divisionId={division['divisionUuid']}&seasonId={division['seasonUuid']}"
            f"&districtId={division['districtUuid']}&date={d}"
        )
        if division.get("matchupGroupUuid"):
            q += f"&matchupGroupUuid={division['matchupGroupUuid']}"
        body = self._cached(
            p, f"/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?{q}",
            volatile=self._is_volatile(d),
        )
        return body.get("data") or []

    def matchup_data(self, matchup_id: str, volatile: bool) -> dict:
        p = RAW / "matchup_data" / f"{matchup_id}.json"
        body = self._cached(
            p, f"/api/v2/results/getResultsMatchupData?matchupId={matchup_id}",
            volatile=volatile,
        )
        return body.get("data") or {}

    def match_logs(self, match_id: str):
        """Full referee log for a COMPLETED match (getListLogs; see recon.md).

        Completed-match logs are immutable, so the cache never expires;
        files are sharded by uuid prefix (raw/match_logs/ab/<uuid>.json,
        compact JSON — ~30k matches would be bulky pretty-printed).
        Returns the log list, [] when the API answered with no logs, or
        None when it answered 404 (cached as a sentinel so re-runs skip;
        delete the file to force a retry). Transient errors raise —
        nothing is cached, so the next run retries.
        """
        mid = match_id.lower()
        p = RAW / "match_logs" / mid[:2] / f"{mid}.json"
        if p.exists():
            self.cache_hits += 1
            body = json.loads(p.read_text())
        else:
            try:
                body = self._get_json(f"/api/v1/results/getListLogs?id={mid}")
            except PermissionError as e:
                body = {"data": None, "_error": str(e)[:80]}
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, separators=(",", ":")))
            tmp.replace(p)
        return body.get("data") if isinstance(body, dict) else body
