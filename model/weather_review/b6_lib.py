"""Shared loaders for task B6 (variance channel + attenuation).

Read-only over committed data.  Deliberately separate from windlib.py so
the two tasks cannot contaminate each other; the game-level build here
keeps SCORES (needed for the share-variance channel) and the wind-source
provenance (needed for the measurement-error work).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ---------------------------------------------------------------- weather --

def weather_index():
    geo = {r["event_id"].lower(): r for r in read_csv(DATA / "event_geo.csv")}
    over = {r["event_id"].lower(): r for r in read_csv(DATA / "venue_overrides.csv")}
    daily = {(r["event_id"].lower(), r["date"]): r
             for r in read_csv(DATA / "event_weather.csv")}
    hourly = {}
    for r in read_csv(DATA / "event_weather_hourly.csv"):
        hourly[(r["event_id"].lower(), r["local_time"][:13])] = r
    times = {r["match_id"].lower(): r for r in read_csv(DATA / "match_times.csv")}
    return geo, over, daily, hourly, times


def hour_key(ts):
    return ts[:13] if ts else None


def add_hours(hkey, n):
    """hkey like '2026-03-25T14' -> shifted by n hours."""
    dt = datetime.strptime(hkey, "%Y-%m-%dT%H") + timedelta(hours=n)
    return dt.strftime("%Y-%m-%dT%H")


# ------------------------------------------------------------------ games --

def load_games(rating="now"):
    """Game-level frame with v2 eta, scores, wind at match hour, labels."""
    if rating == "now":
        v2 = {r["player_id"]: float(r["value_now_mean"])
              for r in read_csv(DATA / "v2_players.csv")}
    else:
        v2 = {r["player_id"]: float(r["value_now_mean"])
              for r in read_csv(DATA / "v2_players_train.csv")}
    geo, over, daily, hourly, times = weather_index()
    out = []
    for g in read_csv(DATA / "games.csv"):
        if g["is_dreambreaker"] != "False" or g["is_forfeit"] != "False":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        if s1 + s2 < T:
            continue
        ev = g["event_id"].lower()
        mt = times.get(g["match_id"].lower())
        wind = gust = temp = None
        src = "none"
        hk = None
        if mt:
            ts = mt["start_local"] or mt["planned_start_local"]
            hk = hour_key(ts)
            row = hourly.get((ev, hk)) if hk else None
            if row is not None:
                wind, gust, temp = (fnum(row["windspeed_10m"]),
                                    fnum(row["windgusts_10m"]),
                                    fnum(row["temperature_2m"]))
                src = "hour_actual" if mt["start_local"] else "hour_planned"
        if wind is None:
            row = daily.get((ev, g["date"]))
            if row is not None:
                wind, gust, temp = (fnum(row["windspeed_10m_max"]),
                                    fnum(row["windgusts_10m_max"]),
                                    fnum(row["temperature_2m_max"]))
                src = "day"
        dmax = daily.get((ev, g["date"]))
        heur = (geo.get(ev) or {}).get("setting", "unknown")
        ovr = over.get(ev)
        out.append(dict(
            game_id=g["game_id"], match=g["match_id"].lower(), event=ev,
            date=g["date"], tour=g["tour"], T=T, best_of=int(g["best_of"]),
            game_number=int(g["game_number"]), context=g["context"],
            s1=s1, s2=s2, share=s1 / (s1 + s2), won=s1 > s2,
            eta=team_eta(*vals), wind=wind, gust=gust, temp=temp,
            wx_source=src, hour_key=hk,
            wind_daymax=fnum(dmax["windspeed_10m_max"]) if dmax else None,
            setting_heur=heur,
            setting=(ovr["setting"] if ovr else heur),
            setting_conf=(ovr["confidence"] if ovr else "heuristic"),
        ))
    return out


# ------------------------------------------------- exact race-share moments --

def _race_share_moments(p, T, maxpts=60):
    """Exact E[share] and Var[share] of the winner-side point share for a
    race to T, win by 2, iid per-point prob p (team 1's share).

    Regulation scores enumerated exactly; the deuce branch is summed over
    its geometric tail (each extra pair of points is a 2-point mini-race).
    """
    q = 1.0 - p
    m1 = m2 = 0.0
    # team1 wins T-b (b = 0..T-2)
    for b in range(T - 1):
        pr = math.comb(T - 1 + b, b) * p ** T * q ** b
        s = T / (T + b)
        m1 += pr * s
        m2 += pr * s * s
    for a in range(T - 1):
        pr = math.comb(T - 1 + a, a) * q ** T * p ** a
        s = a / (T + a)
        m1 += pr * s
        m2 += pr * s * s
    deuce = math.comb(2 * T - 2, T - 1) * (p * q) ** (T - 1)
    if deuce > 0:
        both = p * p + q * q
        # after k exchanged pairs (prob (2pq)^k), someone wins the next pair
        k = 0
        rem = deuce
        while rem > 1e-14 and T - 1 + k + 2 <= maxpts:
            pr_pair = rem * both
            a = T - 1 + k
            # winner reaches a+2 vs a
            s_win = (a + 2) / (2 * a + 2)
            s_lose = a / (2 * a + 2)
            pw = pr_pair * (p * p / both)
            pl = pr_pair * (q * q / both)
            m1 += pw * s_win + pl * s_lose
            m2 += pw * s_win ** 2 + pl * s_lose ** 2
            rem *= (2 * p * q)
            k += 1
    var = max(m2 - m1 * m1, 1e-9)
    return m1, var


class ShareMoments:
    """Interpolated exact race moments + win prob on a p grid."""

    def __init__(self, n=601, lo=0.15, hi=0.85):
        self.grid = np.linspace(lo, hi, n)
        self.mean = {}
        self.sd = {}
        self.pwin = {}
        for T in (11, 15):
            m = np.empty(n)
            v = np.empty(n)
            w = np.empty(n)
            for i, p in enumerate(self.grid):
                m[i], vv = _race_share_moments(p, T)
                v[i] = vv
                w[i] = _race_win_prob(p, T)
            self.mean[T] = m
            self.sd[T] = np.sqrt(v)
            self.pwin[T] = w

    def _ip(self, tab, p, T):
        return float(np.interp(min(max(p, self.grid[0]), self.grid[-1]),
                               self.grid, tab[T]))

    def moments(self, p, T):
        return self._ip(self.mean, p, T), self._ip(self.sd, p, T)

    def win(self, p, T):
        return self._ip(self.pwin, p, T)


def _race_win_prob(p, T):
    q = 1 - p
    w = sum(math.comb(T - 1 + b, b) * p ** T * q ** b for b in range(T - 1))
    deuce = math.comb(2 * T - 2, T - 1) * (p * q) ** (T - 1)
    return w + deuce * (p * p / (p * p + q * q))


# ------------------------------------------------------------- inference --

def cluster_boot(clusters, stat, n=1000, seed=11):
    """clusters: dict key -> list of rows.  stat(list_of_rows) -> float or
    tuple.  Returns (point, lo, hi) arrays."""
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    base = stat([r for k in keys for r in clusters[k]])
    scalar = not isinstance(base, (tuple, list, np.ndarray))
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        s = []
        for i in pick:
            s.extend(clusters[keys[i]])
        v = stat(s)
        if v is None:
            continue
        draws.append([v] if scalar else list(v))
    d = np.array(draws, float)
    lo = np.nanpercentile(d, 2.5, axis=0)
    hi = np.nanpercentile(d, 97.5, axis=0)
    if scalar:
        return float(base), float(lo[0]), float(hi[0])
    return np.asarray(base, float), lo, hi


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X, X.T @ y)
