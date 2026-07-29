"""Shared loader for the B2b end-effect re-run (corrected labels + paired).

Does NOT modify any committed model/weather_*.py file. Mirrors
model/end_effects.py's context logic exactly (same wind join, same v2 eta,
same wind bins) but makes the indoor/outdoor label map INJECTABLE so the
same statistic can be recomputed under the published heuristic labels and
under data/venue_overrides.csv.
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
from sitelib.race import team_eta  # noqa: E402

WIND_GROUPS = [("calm <8", 0, 8), ("moderate 8-14", 8, 14), ("windy 14+", 14, 99)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- labels
def label_arms():
    """Four indoor/outdoor label maps: event_id -> 'indoor'|'outdoor'|None.

    published    - data/event_geo.csv setting (what every committed test used)
    corrected_all- every data/venue_overrides.csv verdict applied at any
                   confidence; mixed/unknown events DROPPED; unaudited events
                   keep the heuristic label
    corrected_hi - only confidence=='high' verdicts applied; events whose
                   audit came back mixed/unknown at ANY confidence, or
                   indoor/outdoor at medium/low confidence, are DROPPED
                   (we trust neither the heuristic nor the weak verdict);
                   unaudited events keep the heuristic label
    audited_hi   - ONLY events with a high-confidence indoor/outdoor verdict
                   (cleanest possible label set, smallest sample)
    """
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    ov = read_csv(ROOT / "data/venue_overrides.csv")
    pub = dict(geo)
    c_all, c_hi, a_hi = dict(geo), dict(geo), {}
    for r in ov:
        e, s, c = r["event_id"], r["setting"], r["confidence"]
        if s in ("indoor", "outdoor"):
            c_all[e] = s
        else:
            c_all[e] = None
        if c == "high" and s in ("indoor", "outdoor"):
            c_hi[e] = s
            a_hi[e] = s
        else:
            c_hi[e] = None
    for d in (pub, c_all, c_hi):
        for k, v in list(d.items()):
            if v not in ("indoor", "outdoor"):
                d[k] = None
    return {"published": pub, "corrected_all": c_all,
            "corrected_hi": c_hi, "audited_hi": a_hi}


# ---------------------------------------------------------------- context
def load_matches(setting_map):
    """match_id -> dict(setting, wind, eta, tour, event, best_of, date,
    games, wind_source). Same joins as model/end_effects.py:load_context."""
    wx = {(r["event_id"], r["date"]): r
          for r in read_csv(ROOT / "data/event_weather.csv")}
    hourly, start_hour = {}, {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = float(r["windspeed_10m"])
        except (ValueError, TypeError):
            pass
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    by_match = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        by_match[g["match_id"]].append(g)
    matches = {}
    for mid, gs in by_match.items():
        g0 = gs[0]
        w = wx.get((g0["event_id"], g0["date"]))
        setting = setting_map.get(g0["event_id"])
        if not w or not setting:
            continue
        try:
            wind = float(w["windspeed_10m_max"])
        except (ValueError, TypeError):
            continue
        src = "daily"
        hw = hourly.get((g0["event_id"], start_hour.get(mid, "")))
        if hw is not None:
            wind, src = hw, "hour"
        vals = [v2.get(g0[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        eta = team_eta(*vals) if all(v is not None for v in vals) else None
        matches[mid] = {"setting": setting, "wind": wind, "eta": eta,
                        "tour": g0["tour"], "event": g0["event_id"],
                        "date": g0["date"], "wind_source": src,
                        "best_of": int(g0["best_of"] or 0),
                        "games": sorted(gs, key=lambda r: int(r["game_number"]))}
    return matches


def group_of(m):
    if m["setting"] == "indoor":
        return "INDOOR"
    for lbl, lo, hi in WIND_GROUPS:
        if lo <= m["wind"] < hi:
            return f"OUTDOOR {lbl}"
    return None


def zsq(x_pts, n1, y_pts, n2):
    x, y = x_pts / n1, y_pts / n2
    p = (x_pts + y_pts) / (n1 + n2)
    noise = p * (1 - p) * (1 / n1 + 1 / n2)
    sq = (x - y) ** 2
    return sq, noise, (sq / noise if noise > 0 else 0.0)


# ------------------------------------------------------- estimators
def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def unpaired_diff(units_t, units_c):
    """units_* : list of (event, [values]).  Grand mean of pooled values."""
    t = [v for _, vs in units_t for v in vs]
    c = [v for _, vs in units_c for v in vs]
    if not t or not c:
        return float("nan")
    return _mean(t) - _mean(c)


def paired_events(units_t, units_c):
    """{event: (t_games, c_games)} for events present in BOTH arms.
    Each *_games entry is a list of GAME payloads; a payload is the list of
    values that game contributes (1 value for Design B, 2 for Design C)."""
    T, C = defaultdict(list), defaultdict(list)
    for e, vs in units_t:
        T[e].append(list(vs))
    for e, vs in units_c:
        C[e].append(list(vs))
    return {e: (T[e], C[e]) for e in T if e in C}


def _flat(games):
    return [v for g in games for v in g]


def paired_diff(pe, weight="fe"):
    """Weighted mean of within-event differences d_e = mean_t - mean_c.

    weight:
      'unit' - w_e = 1               (average event effect)
      'att'  - w_e = n_t             (effect on the treated game; phase-1 choice)
      'fe'   - w_e = n_t*n_c/(n_t+n_c)  (inverse-variance / two-way fixed
               effects; the OLS-with-event-dummies estimator)
    """
    num = den = 0.0
    for e, (t, c) in pe.items():
        if not t or not c:
            continue
        tv, cv = _flat(t), _flat(c)
        nt, nc = len(tv), len(cv)
        w = 1.0 if weight == "unit" else (nt if weight == "att"
                                          else nt * nc / (nt + nc))
        num += w * (_mean(tv) - _mean(cv))
        den += w
    return num / den if den else float("nan")


def cluster_boot(clusters, stat, n=2000, seed=7):
    """clusters: {key: payload}. stat(list_of_payloads) -> float."""
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [clusters[rng.choice(keys)] for _ in keys]
        v = stat(s)
        if v == v:
            vals.append(v)
    vals.sort()
    if len(vals) < 20:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def t_interval(pe, weight="fe"):
    """Weighted-mean t interval on the G event-level differences, G-1 df."""
    ds, ws = [], []
    for e, (t, c) in pe.items():
        if not t or not c:
            continue
        tv, cv = _flat(t), _flat(c)
        nt, nc = len(tv), len(cv)
        w = 1.0 if weight == "unit" else (nt if weight == "att"
                                          else nt * nc / (nt + nc))
        ds.append(_mean(tv) - _mean(cv))
        ws.append(w)
    G = len(ds)
    if G < 3:
        return float("nan"), float("nan"), float("nan"), G
    W = sum(ws)
    est = sum(w * d for w, d in zip(ws, ds)) / W
    # variance of a weighted mean of independent event effects
    var = sum(w * w * (d - est) ** 2 for w, d in zip(ws, ds)) * G / (G - 1) / (W * W)
    se = math.sqrt(var)
    tcrit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
             14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
             19: 2.093, 20: 2.086}.get(G - 1, 1.96 + 2.4 / max(G - 1, 1))
    return est - tcrit * se, est + tcrit * se, se, G


def perm_test(pe, weight="fe", n=3000, seed=13):
    """Within-event permutation of the treatment label (exact under the
    sharp null of no wind effect, conditional on each event's games).
    Units are whole GAMES (payload = list of values)."""
    rng = random.Random(seed)
    # rebuild as per-event lists of game payloads
    obs = paired_diff(pe, weight)
    hits_1 = hits_2 = 0
    for _ in range(n):
        shuffled = {}
        for e, (t, c) in pe.items():
            pool = list(t) + list(c)
            rng.shuffle(pool)
            shuffled[e] = (pool[:len(t)], pool[len(t):])
        v = paired_diff(shuffled, weight)
        if v >= obs:
            hits_1 += 1
        if abs(v) >= abs(obs):
            hits_2 += 1
    return obs, (hits_1 + 1) / (n + 1), (hits_2 + 1) / (n + 1)
