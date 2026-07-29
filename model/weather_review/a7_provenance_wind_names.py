"""A7 provenance check — can the NAMED wind numbers in model/weather_thread.md
and web/insights/wind/index.html be reproduced?

model/wind_skill.py only ever reports ONE pre-specified player (Anna Bright).
The narrative additionally quotes:
  - Anna Leigh Waters  -0.020 share/10 mph, 726 outdoor games, p = 0.95
  - Tyra "Hurricane" Black +0.017, 585 games, p = 0.18
  - Jack Sock "among the most wind-positive big names", 263 games
  - a 10-name defensive-grinder cohort pooled -0.012 /10 mph, p = 0.975

None of those appear in any committed script. This file re-implements
wind_skill.py's loader + ols_slope + within-player wind permutation VERBATIM
(copied, not imported, so the committed file is untouched) and evaluates the
named cases, so the published values can be checked.

READ ONLY: writes nothing into data/ or model/*.md. Prints a table.
Deterministic: fresh seeded RNG per test (the original shares one Random(42)
stream, so exact permutation p-values differ in the 3rd decimal by design).

    python model/weather_review/a7_provenance_wind_names.py
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

MIN_GAMES = 40
N_PERM = 2000

GRINDERS = ["Tellez", "Tardio", "Patriquin", "Hewett", "Staksrud",
            "Frazier", "Smith", "Parenteau", "Johnson", "Irvine"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ols_slope(pts):
    n = len(pts)
    if n < 10:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / den


def load():
    """Byte-for-byte the loader in model/wind_skill.py:main()."""
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    names = {r["player_id"]: r["full_name"]
             for r in read_csv(ROOT / "data/v2_players.csv")}
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]

    players = defaultdict(list)
    n_games = 0
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        if geo.get(g["event_id"]) != "outdoor":
            continue
        wind = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        if wind is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        resid = s1 / (s1 + s2) - sigmoid(eta)
        n_games += 1
        for k in ("t1_p1", "t1_p2"):
            players[g[k]].append((wind, resid))
        for k in ("t2_p1", "t2_p2"):
            players[g[k]].append((wind, -resid))
    return names, dict(players), n_games


def perm_p(games, seed):
    """One-sided upper permutation p, exactly wind_skill.py's definition."""
    rng = random.Random(seed)
    b = ols_slope(games)
    perm = []
    for _ in range(N_PERM):
        winds = [x for x, _ in games]
        rng.shuffle(winds)
        s = ols_slope([(w, y) for w, (_, y) in zip(winds, games)])
        if s is not None:
            perm.append(s)
    perm.sort()
    p = sum(1 for s in perm if s >= b) / len(perm)
    lo = perm[int(0.025 * len(perm))] * 10
    hi = perm[int(0.975 * len(perm))] * 10
    return b * 10, p, lo, hi


def main():
    names, players, n_games = load()
    by_name = defaultdict(list)
    for pid, nm in names.items():
        by_name[nm].append(pid)
    print(f"loaded: {n_games} outdoor games; "
          f"{sum(1 for g in players.values() if len(g) >= MIN_GAMES)} players "
          f"with >={MIN_GAMES}\n")

    targets = ["Anna Bright", "Anna Leigh Waters", "Tyra Black", "Jack Sock"]
    print("| published name | pid found | n games | slope/10mph | one-sided p | null 95% band |")
    print("|---|---|---|---|---|---|")
    for i, nm in enumerate(targets):
        pids = [p for p in by_name.get(nm, []) if p in players]
        if not pids:
            print(f"| {nm} | NOT FOUND | - | - | - | - |")
            continue
        for pid in pids:
            g = players[pid]
            b, p, lo, hi = perm_p(g, 1000 + i)
            print(f"| {nm} | {pid[:8]} | {len(g)} | {b:+.4f} | {p:.2f} "
                  f"| [{lo:+.4f}, {hi:+.4f}] |")

    # grinder cohort: pooled slope over the union of their games
    print("\n## grinder cohort (surname match, >=MIN_GAMES, outdoor)")
    pool = []
    members = []
    for pid, g in players.items():
        if len(g) < MIN_GAMES:
            continue
        nm = names.get(pid, "")
        sur = nm.split()[-1] if nm else ""
        if sur in GRINDERS:
            members.append((nm, len(g)))
            pool.extend(g)
    members.sort()
    for nm, n in members:
        print(f"  - {nm} ({n})")
    if pool:
        b, p, lo, hi = perm_p(pool, 2222)
        print(f"\npooled: n_obs={len(pool)}  slope {b:+.4f}/10mph  "
              f"one-sided p={p:.3f}  null band [{lo:+.4f}, {hi:+.4f}]")
        print("NOTE: surname matching is a GUESS at the cohort — the published "
              "list gives surnames only for 7 of 10 and no UUIDs, so the exact "
              "membership is unrecoverable from the record.")


if __name__ == "__main__":
    main()
