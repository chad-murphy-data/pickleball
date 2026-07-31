"""C1 shared loaders — texture-of-play tests (pace, side-outs, score shape).

Additions only: nothing in model/weather_*.py, favorites_wind.py or
end_effects.py is modified.

Two things here are METHODOLOGICAL UPGRADES over the committed tests and
are used deliberately:

1. GAME-HOUR wind, not match-start-hour wind. `data/match_times.csv`
   carries per-game end stamps `g1_end_utc..g5_end_utc` which are TRUE
   UTC (verified: `start_local`/`completed_local` are venue-local with a
   spurious trailing 'Z', the g*_end_utc stamps are ~offset hours later
   and line up with `completed_local + offset`). Converting a game's end
   stamp to the venue timezone gives the ACTUAL hour the game was played
   — no reliance on planned start times (28-35% of the committed
   match-hour joins fall back to `planned_start_local`).

2. Corrected indoor/outdoor labels from `data/venue_overrides.csv`
   (injectable, same arms as model/weather_review/b2b_lib.py).
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import team_eta  # noqa: E402


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ labels
def label_arms():
    """event_id -> 'indoor'|'outdoor'|None for three label sets."""
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    ov = read_csv(ROOT / "data/venue_overrides.csv")
    pub = dict(geo)
    c_all, c_hi = dict(geo), dict(geo)
    for r in ov:
        e, s, c = r["event_id"], r["setting"], r["confidence"]
        c_all[e] = s if s in ("indoor", "outdoor") else None
        c_hi[e] = s if (c == "high" and s in ("indoor", "outdoor")) else None
    for d in (pub, c_all, c_hi):
        for k, v in list(d.items()):
            if v not in ("indoor", "outdoor"):
                d[k] = None
    return {"published": pub, "corrected_all": c_all, "corrected_hi": c_hi}


# ----------------------------------------------------------------- weather
def load_hourly():
    """(event_id, 'YYYY-MM-DDTHH') -> dict(wind, gust, temp, precip)."""
    out = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        out[(r["event_id"], r["local_time"][:13])] = {
            "wind": fnum(r["windspeed_10m"]),
            "gust": fnum(r["windgusts_10m"]),
            "temp": fnum(r["temperature_2m"]),
            "precip": fnum(r["precipitation"]),
        }
    return out


def event_tz():
    return {r["event_id"]: r["timezone"] for r in read_csv(ROOT / "data/event_geo.csv")}


def parse_utc(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


TZ_ALIAS = {"US/Arizona": "America/Phoenix"}


def get_tz(tzname, cache={}):
    if tzname in cache:
        return cache[tzname]
    try:
        tz = ZoneInfo(TZ_ALIAS.get(tzname, tzname))
    except Exception:
        tz = None
    cache[tzname] = tz
    return tz


def local_hour_key(utc_ts, tzname):
    tz = get_tz(tzname)
    if tz is None:
        return None
    return utc_ts.astimezone(tz).strftime("%Y-%m-%dT%H")


# ------------------------------------------------------------------- games
def load_games():
    """match_id -> sorted list of game rows (DB + forfeits dropped)."""
    by = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        by[g["match_id"]].append(g)
    for mid in by:
        by[mid].sort(key=lambda r: int(r["game_number"]))
    return by


def load_v2():
    return {r["player_id"]: float(r["value_now_mean"])
            for r in read_csv(ROOT / "data/v2_players.csv")}


def game_eta(g, v2):
    vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
    if not all(v is not None for v in vals):
        return None
    return team_eta(*vals)


# ------------------------------------------------------------- regressions
def ols(X, y, ridge=1e-9):
    """Plain OLS via normal equations (numpy)."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def cluster_boot(clusters, fn, n=1000, seed=11):
    """clusters: {key: payload}. fn(list_of_payloads) -> float|None."""
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [clusters[rng.choice(keys)] for _ in keys]
        v = fn(pick)
        if v is not None and math.isfinite(v):
            vals.append(v)
    vals.sort()
    if len(vals) < 20:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def se_from_ci(lo, hi):
    return (hi - lo) / (2 * 1.959964)


def mde(lo, hi, power=0.80):
    """Minimum detectable effect at alpha=.05 two-sided, given power."""
    return 2.802 * se_from_ci(lo, hi)  # (1.96 + 0.842) * se
