"""Shared loaders + sparse fixed-effect machinery for task C3.

Read-only over committed data.  Two jobs live on top of this module:
  * c3a_fixed_effects.py — within-unit identification of the favourites
    x wind interaction (kills the event-composition confound),
  * c3b_wind_skill.py    — random-slope variance component for wind skill
    plus a pooled style x wind interaction.

Nothing here modifies the committed weather scripts.
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


# ------------------------------------------------------------ label arms --

def label_arms():
    """event_id -> setting, for the four label arms of phase-2 B2a.

    a = heuristic (what every published test used)
    c = web-verified high/medium confidence, mixed+unknown DROPPED
    d = as c but mixed counted as outdoor
    """
    geo = {r["event_id"]: r["setting"] for r in read_csv(DATA / "event_geo.csv")}
    ov = {r["event_id"]: r for r in read_csv(DATA / "venue_overrides.csv")}

    def arm(name):
        out = {}
        for ev, heur in geo.items():
            o = ov.get(ev)
            if o is None:
                out[ev] = heur
                continue
            s, conf = o["setting"], o["confidence"]
            if name == "a":
                out[ev] = heur
            elif name in ("c", "d"):
                if s == "unknown":
                    out[ev] = None
                elif s == "mixed":
                    out[ev] = "outdoor" if name == "d" else None
                elif conf in ("high", "medium"):
                    out[ev] = s
                else:
                    out[ev] = heur
        return out

    audited = {ev for ev, o in ov.items()
               if o["setting"] in ("indoor", "outdoor")
               and o["confidence"] in ("high", "medium")}
    return {k: arm(k) for k in "acd"}, audited


# ---------------------------------------------------------------- games --

def load_frame(require_hour=True):
    """Game-level frame: v2 skill, point share, match-hour wind, players.

    Mirrors model/favorites_wind.py's regression-1 sample construction
    (same v2 file, same eta, same share, same wind join) but keeps the
    player UUIDs and the wind provenance so fixed effects are possible.
    """
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(DATA / "v2_players.csv")}
    hourly = {}
    for r in read_csv(DATA / "event_weather_hourly.csv"):
        hourly[(r["event_id"], r["local_time"][:13])] = r
    start_hour, actual = {}, {}
    for r in read_csv(DATA / "match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
            actual[r["match_id"]] = bool(r["start_local"])
    arms, audited = label_arms()

    rows = []
    for g in read_csv(DATA / "games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        ev, mid = g["event_id"], g["match_id"]
        hk = start_hour.get(mid)
        row = hourly.get((ev, hk)) if hk else None
        if row is None:
            if require_hour:
                continue
            wind = temp = gust = None
        else:
            wind = fnum(row["windspeed_10m"])
            gust = fnum(row["windgusts_10m"])
            temp = fnum(row["temperature_2m"])
        if wind is None:
            continue
        eta = team_eta(*vals)
        us = tuple(g[k] for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))
        rows.append(dict(
            game_id=g["game_id"], match=mid, event=ev, date=g["date"],
            tour=g["tour"], hour=hk, actual=actual.get(mid, False),
            T=11 if g["scoring_format"] == "sideout_11" else 15,
            s1=s1, s2=s2, y=s1 / (s1 + s2) - 0.5,
            eta=eta, skill=sigmoid(eta) - 0.5, w=wind / 10.0,
            wind=wind, gust=gust, temp=temp, us=us,
            pair1="|".join(sorted(us[:2])), pair2="|".join(sorted(us[2:])),
            setting_a=arms["a"].get(ev), setting_c=arms["c"].get(ev),
            setting_d=arms["d"].get(ev), audited=ev in audited,
        ))
    return rows


# ----------------------------------------------------- sparse FE absorb --

class FEBlocks:
    """A sparse fixed-effect design as a list of (index, value) blocks.

    Each block contributes  value[g] * beta[index[g]]  to row g.  Blocks
    may share a parameter space (that is how antisymmetric player /
    pair dummies are encoded: +1 for the two team-1 slots, -1 for the
    team-2 slots, all indexing the same player parameter vector).
    """

    def __init__(self, n, blocks, nparam):
        self.n = n
        self.blocks = [(np.asarray(i, np.int64), np.asarray(v, float))
                       for i, v in blocks]
        self.p = nparam

    def mv(self, b):
        """X @ b for b shaped (p,) or (p,K)."""
        if b.ndim == 1:
            out = np.zeros(self.n)
            for idx, val in self.blocks:
                out += val * b[idx]
            return out
        out = np.zeros((self.n, b.shape[1]))
        for idx, val in self.blocks:
            out += val[:, None] * b[idx]
        return out

    def rmv(self, r):
        """X.T @ r for r shaped (n,) or (n,K)."""
        if r.ndim == 1:
            out = np.zeros(self.p)
            for idx, val in self.blocks:
                out += np.bincount(idx, weights=val * r, minlength=self.p)
            return out
        out = np.zeros((self.p, r.shape[1]))
        for k in range(r.shape[1]):
            for idx, val in self.blocks:
                out[:, k] += np.bincount(idx, weights=val * r[:, k],
                                         minlength=self.p)
        return out

    def diag(self):
        d = np.zeros(self.p)
        for idx, val in self.blocks:
            d += np.bincount(idx, weights=val * val, minlength=self.p)
        return d


def absorb(fe: FEBlocks, M, ridge=1e-6, tol=1e-10, maxit=400):
    """Residualise the columns of M (n,K) on the FE span.

    Jacobi-preconditioned CG on the (ridged) normal equations, run on all
    K right-hand sides simultaneously.  Returns (resid, iters).
    """
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    d = fe.diag()
    lam = ridge * max(d.mean(), 1e-12)
    prec = 1.0 / (d + lam)
    B = fe.rmv(M)                       # (p,K)
    x = np.zeros_like(B)
    r = B.copy()
    z = prec[:, None] * r
    pdir = z.copy()
    rz = (r * z).sum(axis=0)
    nrm0 = np.maximum((B * B).sum(axis=0), 1e-300)
    it = 0
    for it in range(1, maxit + 1):
        Ap = fe.rmv(fe.mv(pdir)) + lam * pdir
        denom = (pdir * Ap).sum(axis=0)
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        alpha = rz / denom
        x += alpha * pdir
        r -= alpha * Ap
        if np.all((r * r).sum(axis=0) / nrm0 < tol):
            break
        z = prec[:, None] * r
        rz_new = (r * z).sum(axis=0)
        beta = rz_new / np.where(np.abs(rz) < 1e-300, 1e-300, rz)
        rz = rz_new
        pdir = z + beta * pdir
    return M - fe.mv(x), it


def ols_ci(X, y):
    """Plain OLS coefficients (X already includes an intercept column)."""
    XtX = X.T @ X
    return np.linalg.solve(XtX, X.T @ y)


def cluster_se(X, y, beta, groups):
    """CR1 cluster-robust covariance (clusters = groups array)."""
    n, k = X.shape
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    G = len(bounds) - 1
    Xo, ro = X[order], resid[order]
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = Xo[a:b].T @ ro[a:b]
        meat += np.outer(s, s)
    c = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    return c * XtX_inv @ meat @ XtX_inv, G
