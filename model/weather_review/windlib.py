"""Shared loaders for the wind-as-a-predictor holdout test (task C2).

Everything here is READ-ONLY over committed data.  The base prediction is
the FROZEN v2 _train fit (posterior means, exactly as spec_shootout.py's
`v2_plugin` reference builds it) — v2 is never refit.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

SPLIT = "2026-06-01"          # frozen holdout boundary (project protocol)
RECENT = "2026-01-01"         # window used to fit prediction-stage params
MIN_TRAIN_GAMES = 10
GAMMA_V2 = -0.16596178710460663   # posterior mean, v2 _train fit


def nelder_mead(f, x0, step=0.15, xtol=1e-5, ftol=1e-9, maxiter=2000):
    """Standard Nelder-Mead (scipy is not installed in this container).
    Deterministic; matches scipy's default reflection/expansion constants."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    if n == 0:
        return x0
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for i in range(n):
        y = x0.copy()
        y[i] = y[i] + (step * abs(y[i]) if y[i] != 0 else step)
        sim[i + 1] = y
    fs = np.array([f(s) for s in sim])
    order = np.argsort(fs)
    sim, fs = sim[order], fs[order]
    for _ in range(maxiter):
        if (np.max(np.abs(sim[1:] - sim[0])) <= xtol
                and np.max(np.abs(fs[1:] - fs[0])) <= ftol):
            break
        cen = sim[:-1].mean(axis=0)
        xr = cen + (cen - sim[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            if fr < fs[-1]:
                xc = cen + 0.5 * (cen - sim[-1])
                fc = f(xc)
                if fc <= fr:
                    sim[-1], fs[-1] = xc, fc
                else:
                    sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                    fs[1:] = [f(s) for s in sim[1:]]
            else:
                xc = sim[-1] + 0.5 * (cen - sim[-1])
                fc = f(xc)
                if fc < fs[-1]:
                    sim[-1], fs[-1] = xc, fc
                else:
                    sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                    fs[1:] = [f(s) for s in sim[1:]]
        order = np.argsort(fs)
        sim, fs = sim[order], fs[order]
    return sim[0]


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ------------------------------------------------------------- race DP --

def race_win_table(T, grid):
    """P(team1 wins race to T, win by 2) per p — exact DP (v2_holdout.py)."""
    out = np.zeros_like(grid)
    for gi, p in enumerate(grid):
        q = 1 - p
        dp = np.zeros((T + 1, T + 1))
        dp[0, 0] = 1.0
        win = 0.0
        deuce = 0.0
        for a in range(T + 1):
            for b in range(T + 1):
                if dp[a, b] == 0:
                    continue
                if a == T - 1 and b == T - 1:
                    deuce += dp[a, b]
                    continue
                if a == T:
                    win += dp[a, b]
                    continue
                if b == T:
                    continue
                if a + 1 == T and b <= T - 2:
                    win += dp[a, b] * p
                else:
                    dp[a + 1, b] += dp[a, b] * p
                dp[a, b + 1] += dp[a, b] * q
        win += deuce * (p * p / (p * p + q * q + 1e-12))
        out[gi] = win
    return out


class Racer:
    def __init__(self):
        self.grid = np.linspace(0.01, 0.99, 981)
        self.tab = {T: race_win_table(T, self.grid) for T in (11, 15)}

    def win(self, eta, T):
        eta = np.asarray(eta, float)
        p = 1.0 / (1.0 + np.exp(-eta))
        out = np.empty_like(p)
        for t in (11, 15):
            m = (T == t)
            if m.any():
                out[m] = np.interp(p[m], self.grid, self.tab[t])
        return out


# ---------------------------------------------------------------- data --

def load_games():
    rows = []
    for g in read_csv(DATA / "games.csv"):
        if g["is_forfeit"] != "False":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        rows.append(dict(
            date=g["date"], tour=g["tour"], event=g["event_id"].lower(),
            match=g["match_id"].lower(),
            T=11 if g["scoring_format"] == "sideout_11" else 15,
            us=(g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]),
            won=int(g["margin"]) > 0))
    rows.sort(key=lambda r: (r["date"], r["match"]))
    return rows


def load_v2_train():
    players = {}
    for r in read_csv(DATA / "v2_players_train.csv"):
        players[r["player_id"]] = dict(
            v=float(r["value_now_mean"]), games=int(r["games"]),
            name=r["full_name"], dynamic=r["dynamic"] == "1")
    chem = {}
    for r in read_csv(DATA / "v2_dyads_train.csv"):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = float(r["chem_logit_mean"])
    traj = {}
    for r in read_csv(DATA / "v2_trajectories_train.csv"):
        traj.setdefault(r["player_id"], {})[r["month"]] = float(r["value_mean"])
    return players, chem, traj


def value_at(u, month, players, traj):
    """Month-appropriate v2 value: monthly random-walk state for dynamic
    players (clamped at the ends of their curve), static value otherwise."""
    t = traj.get(u)
    if not t:
        return players[u]["v"]
    if month in t:
        return t[month]
    ks = sorted(t)
    return t[ks[0]] if month < ks[0] else t[ks[-1]]


def eta_for(games, players, chem, traj=None):
    """v2 eta (per-point logit): sum + gamma|gap| + chem difference.
    traj=None -> current-form values (spec_shootout's Frame); traj given ->
    month-appropriate values from the same frozen fit."""
    out = np.empty(len(games))
    for i, g in enumerate(games):
        us = g["us"]
        if traj is None:
            v = [players[u]["v"] for u in us]
        else:
            m = g["date"][:7]
            v = [value_at(u, m, players, traj) for u in us]
        c1 = chem.get(frozenset((players[us[0]]["name"], players[us[1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[us[2]]["name"], players[us[3]]["name"])), 0.0)
        out[i] = (v[0] + v[1] + GAMMA_V2 * abs(v[0] - v[1])
                  - v[2] - v[3] - GAMMA_V2 * abs(v[2] - v[3]) + c1 - c2)
    return out


# ------------------------------------------------------------- weather --

def load_weather_index():
    geo = {r["event_id"].lower(): r for r in read_csv(DATA / "event_geo.csv")}
    over = {r["event_id"].lower(): r for r in read_csv(DATA / "venue_overrides.csv")}
    daily = {(r["event_id"].lower(), r["date"]): r
             for r in read_csv(DATA / "event_weather.csv")}
    hourly = {(r["event_id"].lower(), r["local_time"][:13]): r
              for r in read_csv(DATA / "event_weather_hourly.csv")}
    times = {}
    for r in read_csv(DATA / "match_times.csv"):
        ts, planned = r["start_local"], r["planned_start_local"]
        times[r["match_id"].lower()] = (ts or planned, bool(ts))
    return geo, over, daily, hourly, times


def attach_weather(games, idx, label_source="override"):
    """Adds wind/gust/temp at MATCH HOUR (falls back to event-day max) and
    the indoor/outdoor label.  label_source: 'override' = phase-1 web-audited
    labels where available else heuristic; 'heuristic' = the labels every
    published weather test used."""
    geo, over, daily, hourly, times = idx
    for g in games:
        ev = g["event"]
        hs = times.get(g["match"])
        row, src = None, "none"
        if hs and hs[0]:
            row = hourly.get((ev, hs[0][:13]))
            if row is not None:
                src = "hour_actual" if hs[1] else "hour_planned"
        if row is None:
            row = daily.get((ev, g["date"]))
            src = "day" if row is not None else "none"
            g["wind"] = fnum(row["windspeed_10m_max"]) if row else None
            g["gust"] = fnum(row["windgusts_10m_max"]) if row else None
            g["temp"] = fnum(row["temperature_2m_max"]) if row else None
        else:
            g["wind"] = fnum(row["windspeed_10m"])
            g["gust"] = fnum(row["windgusts_10m"])
            g["temp"] = fnum(row["temperature_2m"])
        g["wx_source"] = src
        heur = (geo.get(ev, {}) or {}).get("setting", "unknown")
        if label_source == "heuristic":
            g["setting"] = heur
        else:
            o = over.get(ev)
            g["setting"] = o["setting"] if o else heur
    return games


def build(label_source="override", rating="now"):
    """Returns (games, eta, meta) for every modellable game with weather."""
    games = load_games()
    players, chem, traj = load_v2_train()
    keep = [g for g in games
            if all(u in players and players[u]["games"] >= MIN_TRAIN_GAMES
                   for u in g["us"])]
    attach_weather(keep, load_weather_index(), label_source)
    keep = [g for g in keep if g["wind"] is not None]
    eta = eta_for(keep, players, chem, traj if rating == "traj" else None)
    return keep, eta
