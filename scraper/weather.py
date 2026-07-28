"""Attach venue geography + historical weather to every event in games.csv.

Usage:
    python scraper/weather.py            # resolve venues, fetch weather
    python scraper/weather.py --refresh  # refetch weather for all events

Outputs (committed):
    data/event_geo.csv            one row per event_id: venue, lat/lon, tz,
                                  indoor/outdoor guess (heuristic + overrides)
    data/event_weather.csv        one row per (event_id, date): daily weather
    data/event_weather_hourly.csv one row per (event_id, date, hour): hourly
                                  weather, venue-local time — ready for the
                                  hour-level join once per-match start times
                                  are extracted from raw/ (localDateMatch*).

Venue resolution (no new harvest of raw/ needed — two open BFF endpoints):
  PPA: event_id == TournamentID; getTournamentsOnDate on a date the event
       played returns Latitude/Longitude/LocationVenue directly.
  MLP: event_id == matchupGroupUuid; getTeamLeaguesResultsOnDate returns a
       location object with venue lat/lon and an IANA timezone.

Weather: Open-Meteo ERA5 archive (open, keyless, ~1 req/event). Recent
dates (<7 days) may come back null — they fill in on the next run.

Indoor/outdoor: keyword heuristic on the venue name, overridable via
data/venue_overrides.csv (event_id,setting in {indoor,outdoor}). The guess
is a GUESS — any wind/weather analysis should treat "unknown" honestly.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "raw" / "weather"

BFF = "https://pickleball.com"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
    "precipitation_sum", "windspeed_10m_max", "windgusts_10m_max",
    "winddirection_10m_dominant", "cloudcover_mean",
    "relative_humidity_2m_mean",
]
HOURLY_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "cloudcover", "windspeed_10m", "windgusts_10m",
    "winddirection_10m",
]

# Setting = tour default + venue-keyword override. Domain reality: PPA tour
# stops are outdoor except the indoor-club stops (Life Time chain etc.);
# MLP plays in arenas/indoor clubs except the occasional outdoor stadium
# stop. Keywords flip the default; data/venue_overrides.csv beats both.
INDOOR_WORDS = ("life time", "lifetime", "picklr", "arena", "fieldhouse",
                "coliseum", "convention", "indoor", "expo", "dome",
                "athletic club", "racquet")
OUTDOOR_WORDS = ("park", "outdoor", "stadium", "polo fields",
                 "sports campus", "tennis garden", "country club",
                 "tennis center", "tennis centre", "resort")


def _throttled_get(client: httpx.Client, url: str, last: list[float]) -> dict:
    wait = last[0] + 1.0 - time.time()
    if wait > 0:
        time.sleep(wait)
    r = client.get(url, timeout=30)
    last[0] = time.time()
    r.raise_for_status()
    return r.json()


def _cached_json(client: httpx.Client, cache: Path, url: str,
                 last: list[float], refresh: bool = False) -> dict:
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    body = _throttled_get(client, url, last)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(body))
    return body


def load_events() -> dict[str, dict]:
    """event_id -> {name, tour, dates(sorted list of ISO dates)}."""
    events: dict[str, dict] = {}
    dates = defaultdict(set)
    with open(DATA / "games.csv") as f:
        for row in csv.DictReader(f):
            eid = row["event_id"]
            if not eid or not row["date"]:
                continue
            events.setdefault(eid, {"name": row["event_name"],
                                    "tour": row["tour"]})
            dates[eid].add(row["date"])
    for eid, meta in events.items():
        meta["dates"] = sorted(dates[eid])
    return events


def resolve_venues(client: httpx.Client, events: dict[str, dict],
                   last: list[float]) -> dict[str, dict]:
    """event_id -> geo dict. Groups events by date so each results-on-date
    payload is fetched once and can satisfy several events."""
    geo: dict[str, dict] = {}
    unresolved = dict(events)

    # try each event's dates in order until its id shows up in a payload
    max_tries = max(len(m["dates"]) for m in events.values())
    for attempt in range(max_tries):
        if not unresolved:
            break
        by_date = defaultdict(list)
        for eid, meta in unresolved.items():
            if attempt < len(meta["dates"]):
                by_date[(meta["dates"][attempt], meta["tour"])].append(eid)
        for (d, tour), eids in sorted(by_date.items()):
            if tour == "PPA":
                url = f"{BFF}/api/v1/results/getTournamentsOnDate?date={d}"
                cache = CACHE / "bff" / f"tournaments_{d}.json"
                body = _cached_json(client, cache, url, last)
                items = body if isinstance(body, list) else body.get("data", [])
                by_id = {(t.get("TournamentID") or "").lower(): t for t in items}
                for eid in eids:
                    t = by_id.get(eid.lower())
                    if not t or t.get("Latitude") is None:
                        continue
                    geo[eid] = {
                        "venue": t.get("LocationVenue") or "",
                        "city": t.get("LocationCity") or "",
                        "state": t.get("LocationState") or "",
                        "lat": t["Latitude"], "lon": t["Longitude"],
                        "iana_tz": "",  # PPA payload has offset only
                    }
                    unresolved.pop(eid, None)
            else:  # MLP
                url = (f"{BFF}/api/v2/results/getTeamLeaguesResultsOnDate"
                       f"?date={d}")
                cache = CACHE / "bff" / f"teamleagues_{d}.json"
                body = _cached_json(client, cache, url, last)
                for tl in body.get("data", []):
                    eid = (tl.get("matchupGroupUuid") or "").lower()
                    if eid not in unresolved:
                        continue
                    loc = tl.get("location") or {}
                    if loc.get("latitude") is None:
                        continue
                    tz = (loc.get("timezoneInfo") or {}).get(
                        "ianaTzIdentifier") or ""
                    geo[eid] = {
                        "venue": loc.get("venue") or "",
                        "city": loc.get("city") or "",
                        "state": (loc.get("stateInfo") or {}).get(
                            "abbreviation") or "",
                        "lat": loc["latitude"], "lon": loc["longitude"],
                        "iana_tz": tz,
                    }
                    unresolved.pop(eid, None)
    for eid in unresolved:
        print(f"  UNRESOLVED venue: {eid} {events[eid]['name']}")
    return geo


def guess_setting(venue: str, tour: str) -> str:
    v = venue.lower()
    if any(w in v for w in INDOOR_WORDS):
        return "indoor"
    if any(w in v for w in OUTDOOR_WORDS):
        return "outdoor"
    return "outdoor" if tour == "PPA" else "indoor"


def load_overrides() -> dict[str, str]:
    p = DATA / "venue_overrides.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["event_id"]: r["setting"] for r in csv.DictReader(f)}


def fetch_weather(client: httpx.Client, eid: str, g: dict, dates: list[str],
                  last: list[float], refresh: bool) -> dict | None:
    url = (f"{ARCHIVE}?latitude={g['lat']}&longitude={g['lon']}"
           f"&start_date={dates[0]}&end_date={dates[-1]}"
           f"&daily={','.join(DAILY_VARS)}&hourly={','.join(HOURLY_VARS)}"
           f"&timezone={g['iana_tz'] or 'auto'}"
           "&temperature_unit=fahrenheit&windspeed_unit=mph"
           "&precipitation_unit=inch")
    cache = CACHE / "openmeteo" / f"{eid}_{dates[0]}_{dates[-1]}.json"
    try:
        return _cached_json(client, cache, url, last, refresh=refresh)
    except httpx.HTTPError as e:
        print(f"  weather fetch failed for {eid}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="refetch weather even if cached")
    args = ap.parse_args()

    events = load_events()
    print(f"{len(events)} events in games.csv")
    last = [0.0]
    overrides = load_overrides()

    with httpx.Client(headers={"User-Agent": "pickleball-analytics-weather"},
                      follow_redirects=True) as client:
        geo = resolve_venues(client, events, last)
        print(f"{len(geo)} venues resolved")

        geo_rows, daily_rows, hourly_rows = [], [], []
        for eid, g in sorted(geo.items(), key=lambda kv: events[kv[0]]["dates"][0]):
            meta = events[eid]
            setting = overrides.get(eid) or guess_setting(g["venue"],
                                                          meta["tour"])
            wx = fetch_weather(client, eid, g, meta["dates"], last,
                               args.refresh)
            tz_used = (wx or {}).get("timezone", "")
            geo_rows.append({
                "event_id": eid, "event_name": meta["name"],
                "tour": meta["tour"], "first_date": meta["dates"][0],
                "last_date": meta["dates"][-1], "venue": g["venue"],
                "city": g["city"], "state": g["state"],
                "lat": g["lat"], "lon": g["lon"],
                "timezone": g["iana_tz"] or tz_used,
                "setting": setting,
                "setting_source": ("override" if eid in overrides
                                   else "heuristic"),
            })
            if not wx:
                continue
            wanted = set(meta["dates"])
            daily = wx.get("daily") or {}
            for i, d in enumerate(daily.get("time", [])):
                if d not in wanted:
                    continue
                row = {"event_id": eid, "date": d}
                for var in DAILY_VARS:
                    row[var] = (daily.get(var) or [None])[i] \
                        if i < len(daily.get(var) or []) else None
                daily_rows.append(row)
            hourly = wx.get("hourly") or {}
            for i, ts in enumerate(hourly.get("time", [])):
                if ts[:10] not in wanted:
                    continue
                row = {"event_id": eid, "local_time": ts}
                for var in HOURLY_VARS:
                    row[var] = (hourly.get(var) or [None])[i] \
                        if i < len(hourly.get(var) or []) else None
                hourly_rows.append(row)

    def write(path: Path, rows: list[dict]) -> None:
        if not rows:
            print(f"  nothing to write for {path.name}")
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")

    write(DATA / "event_geo.csv", geo_rows)
    write(DATA / "event_weather.csv", daily_rows)
    write(DATA / "event_weather_hourly.csv", hourly_rows)


if __name__ == "__main__":
    main()
