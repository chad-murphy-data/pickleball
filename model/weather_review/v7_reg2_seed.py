"""V7: how fragile is favorites_wind regression 2's outdoor CI, really?

A7 flags it as SEED-FRAGILE (committed seed gives upper bound +0.0001, another
seed -0.0002).  Rather than take two seeds on faith, rebuild regression 2
independently and (a) run the cluster bootstrap at 40x the committed depth to
get the seed-free bootstrap tail probability P(c >= 0), (b) run 40 separate
2,000-draw bootstraps to see the actual spread of the 97.5% bound, and
(c) report the analytic cluster-robust interval, which has no RNG at all.

Deterministic; writes nothing.
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


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def build():
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2 = {r["player_id"]: float(r["value_now_mean"])
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
    meta = {}
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        s = geo.get(g["event_id"])
        if s not in ("outdoor", "indoor"):
            continue
        wind = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        if wind is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        meta[g["match_id"]] = (s, wind, team_eta(*vals), g["event_id"])
    gap = defaultdict(list)
    for r in read_csv(ROOT / "data/decider_serve_splits.csv"):
        m = meta.get(r["match_id"])
        if not m:
            continue
        setting, wind, eta, ev = m
        if abs(eta) < 0.1:
            continue
        ra = int(r["ra_pre"]) + int(r["ra_post"])
        wa = int(r["wa_pre"]) + int(r["wa_post"])
        rb = int(r["rb_pre"]) + int(r["rb_post"])
        wb = int(r["wb_pre"]) + int(r["wb_post"])
        if ra < 8 or rb < 8:
            continue
        g_ = (wa / ra - wb / rb) if eta > 0 else (wb / rb - wa / ra)
        gap[setting].append((ev, g_, wind / 10.0))
    return gap


def ols1(rows):
    n = len(rows)
    mx = sum(r[2] for r in rows) / n
    my = sum(r[1] for r in rows) / n
    sxx = sum((r[2] - mx) ** 2 for r in rows)
    b = sum((r[2] - mx) * (r[1] - my) for r in rows) / sxx
    return b, my - b * mx, mx, sxx


def main():
    gap = build()
    for setting in ("outdoor", "indoor"):
        rows = gap[setting]
        b, a, mx, sxx = ols1(rows)
        cl = defaultdict(list)
        for r in rows:
            cl[r[0]].append(r)
        keys = list(cl)
        G = len(keys)
        # analytic CR1
        by = defaultdict(float)
        for ev, y, x in rows:
            by[ev] += (x - mx) * (y - a - b * x)
        meat = sum(v * v for v in by.values())
        adj = (G / (G - 1)) * ((len(rows) - 1) / (len(rows) - 2))
        se = math.sqrt(adj * meat) / sxx
        print(f"\n=== {setting}: n={len(rows)} games, {G} event clusters ===")
        print(f"  point slope       {b:+.5f} per +10 mph")
        print(f"  analytic CR1 se   {se:.5f}  t={b/se:+.2f}  "
              f"95% [{b-1.96*se:+.5f}, {b+1.96*se:+.5f}]  (no RNG)")

        # deep bootstrap: seed-free tail probability
        rng = random.Random(20260731)
        NB = 40000
        draws = []
        for _ in range(NB):
            s = []
            for _ in range(G):
                s.extend(cl[rng.choice(keys)])
            draws.append(ols1(s)[0])
        draws.sort()
        p_ge0 = sum(1 for d in draws if d >= 0) / NB
        lo = draws[int(0.025 * NB)]
        hi = draws[int(0.975 * NB)]
        print(f"  {NB}-draw cluster bootstrap: [{lo:+.5f}, {hi:+.5f}], "
              f"P(slope>=0) = {p_ge0:.4f}")

        # spread of the 97.5% bound across 40 independent 2,000-draw runs
        bounds = []
        for sd in range(40):
            r2 = random.Random(1000 + sd)
            dd = []
            for _ in range(2000):
                s = []
                for _ in range(G):
                    s.extend(cl[r2.choice(keys)])
                dd.append(ols1(s)[0])
            dd.sort()
            bounds.append(dd[int(0.975 * 2000)])
        bounds.sort()
        n_pos = sum(1 for x in bounds if x > 0)
        print(f"  97.5% bound across 40 seeds (n=2000 each): "
              f"min {bounds[0]:+.5f}, median {bounds[20]:+.5f}, "
              f"max {bounds[-1]:+.5f}")
        print(f"  -> 'CI spans zero' verdict TRUE in {n_pos}/40 seeds "
              f"({'COIN FLIP' if 0 < n_pos < 40 else 'STABLE'})")


if __name__ == "__main__":
    main()
