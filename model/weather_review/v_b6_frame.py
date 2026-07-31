"""VERIFIER frame for B6 — built INDEPENDENTLY of b6_lib.py.

Same declared sample definition, but coded from scratch so that a silent
sample-definition drift in b6_lib shows up as a count mismatch.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import race_dist, sigmoid  # noqa: E402

GAMMA = -0.1829


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def team_eta(a, b, c, d):
    return (a + b + GAMMA * abs(a - b)) - (c + d + GAMMA * abs(c - d))


def build():
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in rd(DATA / "v2_players.csv")}
    geo = {r["event_id"].lower(): r for r in rd(DATA / "event_geo.csv")}
    over = {r["event_id"].lower(): r for r in rd(DATA / "venue_overrides.csv")}
    daily = {(r["event_id"].lower(), r["date"]): r
             for r in rd(DATA / "event_weather.csv")}
    hourly = {}
    for r in rd(DATA / "event_weather_hourly.csv"):
        hourly[(r["event_id"].lower(), r["local_time"][:13])] = r
    times = {r["match_id"].lower(): r for r in rd(DATA / "match_times.csv")}

    out = []
    for g in rd(DATA / "games.csv"):
        if g["is_dreambreaker"] != "False" or g["is_forfeit"] != "False":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if any(v is None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        if s1 + s2 < T:
            continue
        ev = g["event_id"].lower()
        mid = g["match_id"].lower()
        mt = times.get(mid)
        wind = None
        src = "none"
        hk = None
        if mt:
            ts = mt["start_local"] or mt["planned_start_local"]
            hk = ts[:13] if ts else None
            row = hourly.get((ev, hk)) if hk else None
            if row is not None:
                wind = fnum(row["windspeed_10m"])
                src = "hour_actual" if mt["start_local"] else "hour_planned"
        if wind is None:
            row = daily.get((ev, g["date"]))
            if row is not None:
                wind = fnum(row["windspeed_10m_max"])
                src = "day"
        heur = (geo.get(ev) or {}).get("setting", "unknown")
        ovr = over.get(ev)
        eta = team_eta(*vals)
        p = sigmoid(eta)
        out.append(dict(gid=g["game_id"], match=mid, event=ev, date=g["date"],
                        tour=g["tour"], T=T, gn=int(g["game_number"]),
                        s1=s1, s2=s2, share=s1 / (s1 + s2), won=s1 > s2,
                        eta=eta, p=p, wind=wind, src=src, hk=hk,
                        heur=heur,
                        setting=(ovr["setting"] if ovr else heur)))
    return out


_WIN = {}


def win_prob(p, T):
    key = (round(p, 5), T)
    if key not in _WIN:
        _WIN[key] = race_dist(round(min(max(p, 1e-6), 1 - 1e-6), 5), T)["p_win"]
    return _WIN[key]


def cluster_boot_index(events, nboot, seed):
    """yield index arrays for a cluster bootstrap over events."""
    by = defaultdict(list)
    for i, e in enumerate(events):
        by[e].append(i)
    keys = list(by)
    idx = [np.array(by[k]) for k in keys]
    rng = np.random.default_rng(seed)
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        yield np.concatenate([idx[j] for j in pick])


if __name__ == "__main__":
    G = build()
    print("games total", len(G))
    from collections import Counter
    print("src", Counter(g["src"] for g in G))
    print("setting", Counter(g["setting"] for g in G))
    print("heur", Counter(g["heur"] for g in G))
    for lab, key in (("corrected", "setting"), ("heuristic", "heur")):
        o = [g for g in G if g[key] == "outdoor" and g["wind"] is not None]
        print(lab, "outdoor w/ wind", len(o),
              "src", Counter(g["src"] for g in o))
