"""Specification shootout: every reasonably defensible way to price a pro
doubles game, scored head-to-head on the frozen 2026-06-01 holdout.

The question (2026-07-18 session): did we settle on v2's specification too
quickly?  This harness tries the alternatives — different ways to combine
the four player ratings (men only, women only, weaker only, stronger only,
free per-slot loadings), different levels of analysis (match / game /
point / rally), different rating systems (Elo, platform rating), and a few
wacky-but-defensible extras — all on identical footing:

  * every free parameter is fit on TRAIN data only (games before
    2026-06-01; prediction-stage scales on the 2026 portion of train),
  * every strategy prices the SAME holdout games through the SAME race DP
    (the exact win-by-2 race used by model/v2_holdout.py),
  * differences vs the v2 reference get paired-bootstrap CIs.

Families:
  A  aggregation over the frozen v2 _train ratings (prediction stage)
  B  structural refits — ratings re-estimated under each structure
     (fast MAP, point binomial, ridge prior; static, no dynamics)
  C  level of analysis — same structure, likelihood at match / game /
     margin / point level
  D  rally level — serve/return split fit on referee logs, priced by the
     serve-aware DP (needs raw/match_logs; --rally)
  E  other systems — Elo (game + point flavors), platform rating, coin

Outputs model/spec_shootout_summary.json; the writeup lives in
model/spec_shootout.md.  Usage:

  python model/spec_shootout.py             # families A B C E
  python model/spec_shootout.py --rally     # add family D (logs required)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "model"
SPLIT = "2026-06-01"
RECENT = "2026-01-01"          # window for prediction-stage free params
MIN_TRAIN_GAMES = 10
GAMMA_V2 = -0.16596178710460663   # posterior mean, v2 _train fit
SD_V_PRIOR = 0.3765               # v2 posterior sd_v — ridge scale for refits
SEED = 20260718
N_BOOT = 4000

# ---------------------------------------------------------------- data --


def load_games():
    rows = []
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["is_forfeit"] != "False":
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        rows.append(dict(
            date=g["date"], tour=g["tour"], ctx=g["context"],
            match=g["match_id"].lower(),
            T=11 if g["scoring_format"] == "sideout_11" else 15,
            us=(g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]),
            s1=int(g["t1_score"]), s2=int(g["t2_score"]),
            won=int(g["margin"]) > 0))
    rows.sort(key=lambda r: (r["date"], r["match"]))
    return rows


def load_v2():
    players = {}
    for r in csv.DictReader((DATA / "v2_players_train.csv").open()):
        players[r["player_id"]] = dict(
            v=float(r["value_now_mean"]), sd=float(r["value_now_sd"]),
            gender=r["gender"], games=int(r["games"]), name=r["full_name"])
    chem = {}
    for r in csv.DictReader((DATA / "v2_dyads_train.csv").open()):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = float(r["chem_logit_mean"])
    return players, chem


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
        """eta, T arrays -> win prob arrays via the race tables."""
        eta = np.asarray(eta, float)
        p = 1.0 / (1.0 + np.exp(-eta))
        out = np.empty_like(p)
        for t in (11, 15):
            m = T == t
            if m.any():
                out[m] = np.interp(p[m], self.grid, self.tab[t])
        return out

    def win_mixture(self, eta, sd, T, nodes=21):
        """E[race prob] over eta ~ N(eta, sd^2) — Gauss grid, mirrors
        the draws-averaging in v2_holdout / race.game_win_prob_uncertain."""
        zs = np.linspace(-3, 3, nodes)
        ws = np.exp(-0.5 * zs ** 2)
        ws /= ws.sum()
        acc = np.zeros_like(np.asarray(eta, float))
        for z, w in zip(zs, ws):
            acc += w * self.win(eta + z * sd, T)
        return acc


# ------------------------------------------------------------- metrics --


def metrics(pw, won):
    pw = np.asarray(pw, float)
    won = np.asarray(won, bool)
    pc = np.clip(pw, 1e-6, 1 - 1e-6)
    return dict(
        n=int(len(pw)),
        accuracy=float(np.mean((pw > 0.5) == won)),
        brier=float(np.mean((pw - won) ** 2)),
        log_loss=float(np.mean(-np.where(won, np.log(pc), np.log(1 - pc)))))


def paired_boot(pw, ref, won, seed=SEED, n=N_BOOT):
    """Bootstrap the Brier difference (strategy - reference); negative
    favors the strategy."""
    pw, ref = np.asarray(pw, float), np.asarray(ref, float)
    won = np.asarray(won, bool)
    d = (pw - won) ** 2 - (ref - won) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return dict(d_brier=float(d.mean()),
                ci=[float(np.percentile(means, 2.5)),
                    float(np.percentile(means, 97.5))],
                p_better=float(np.mean(means < 0)))


# ----------------------------------------------------- family A: rules --
# Prediction-stage strategies over the frozen v2 _train ratings.  Each is
# eta(params) on a game; free params are fit on the 2026 portion of train
# by minimizing game-winner Bernoulli NLL through the race DP.


class Frame:
    """Shared arrays for a set of games under the frozen v2 ratings."""

    def __init__(self, games, players, chem):
        self.games = games
        self.won = np.array([g["won"] for g in games])
        self.T = np.array([g["T"] for g in games])
        self.tour_mlp = np.array([g["tour"] == "MLP" for g in games])
        self.ctx = np.array([g["ctx"] for g in games])
        V = np.zeros((len(games), 4))
        SD = np.zeros((len(games), 4))
        GEN = np.empty((len(games), 4), dtype="U1")
        LOGG = np.zeros((len(games), 4))
        for i, g in enumerate(games):
            for j, u in enumerate(g["us"]):
                p = players[u]
                V[i, j] = p["v"]
                SD[i, j] = p["sd"]
                GEN[i, j] = p["gender"]
                LOGG[i, j] = math.log1p(p["games"])
        self.V, self.SD, self.GEN, self.LOGG = V, SD, GEN, LOGG
        self.chem = np.array([
            chem.get(frozenset((players[g["us"][0]]["name"],
                                players[g["us"][1]]["name"])), 0.0)
            - chem.get(frozenset((players[g["us"][2]]["name"],
                                  players[g["us"][3]]["name"])), 0.0)
            for g in games])
        # gender slots for mixed games: value of the man / woman per team
        man = np.full((len(games), 2), np.nan)
        wom = np.full((len(games), 2), np.nan)
        for i in range(len(games)):
            for t in range(2):
                a, b = V[i, 2 * t], V[i, 2 * t + 1]
                ga, gb = GEN[i, 2 * t], GEN[i, 2 * t + 1]
                if {ga, gb} == {"M", "F"}:
                    man[i, t] = a if ga == "M" else b
                    wom[i, t] = b if ga == "M" else a
        self.man, self.wom = man, wom

    def sub(self, mask):
        f = object.__new__(Frame)
        f.games = [g for g, m in zip(self.games, mask) if m]
        for k in ("won", "T", "tour_mlp", "ctx", "V", "SD", "GEN", "LOGG",
                  "chem", "man", "wom"):
            setattr(f, k, getattr(self, k)[mask])
        return f

    # --- building blocks ---
    def d_sum(self):
        return (self.V[:, 0] + self.V[:, 1]) - (self.V[:, 2] + self.V[:, 3])

    def d_gap(self):
        return (np.abs(self.V[:, 0] - self.V[:, 1])
                - np.abs(self.V[:, 2] - self.V[:, 3]))

    def d_min(self):
        return 2 * (np.minimum(self.V[:, 0], self.V[:, 1])
                    - np.minimum(self.V[:, 2], self.V[:, 3]))

    def d_max(self):
        return 2 * (np.maximum(self.V[:, 0], self.V[:, 1])
                    - np.maximum(self.V[:, 2], self.V[:, 3]))

    def d_man(self):
        return self.man[:, 0] - self.man[:, 1]

    def d_wom(self):
        return self.wom[:, 0] - self.wom[:, 1]

    def d_exp(self):
        return (self.LOGG[:, 0] + self.LOGG[:, 1]
                - self.LOGG[:, 2] - self.LOGG[:, 3])

    def sd2(self):
        return (self.SD ** 2).sum(axis=1)

    def eta_v2(self):
        return self.d_sum() + GAMMA_V2 * self.d_gap() + self.chem


def fit_params(frame, racer, eta_fn, x0, bounds=None):
    """Minimize game-Bernoulli NLL through the race DP on `frame`."""
    won, T = frame.won, frame.T

    def nll(x):
        eta = eta_fn(frame, x)
        ok = np.isfinite(eta)
        pw = racer.win(eta[ok], T[ok])
        pc = np.clip(pw, 1e-9, 1 - 1e-9)
        return -np.mean(np.where(won[ok], np.log(pc), np.log(1 - pc)))

    res = minimize(nll, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-7, maxiter=2000))
    return res.x


# momentum: exponentially-decayed sum of (observed - expected point share)
# per player, walked forward in date order (holdout games see only
# strictly-earlier dates — no leakage; expectation uses the frozen etas).


def momentum_features(all_games, players, chem, hl_days=45.0):
    def eta_of(g):
        if any(u not in players for u in g["us"]):
            return None
        v = [players[u]["v"] for u in g["us"]]
        c1 = chem.get(frozenset((players[g["us"][0]]["name"],
                                 players[g["us"][1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[g["us"][2]]["name"],
                                 players[g["us"][3]]["name"])), 0.0)
        return (v[0] + v[1] + GAMMA_V2 * abs(v[0] - v[1])
                - v[2] - v[3] - GAMMA_V2 * abs(v[2] - v[3]) + c1 - c2)

    resid = np.zeros(len(all_games))
    for i, g in enumerate(all_games):
        e = eta_of(g)
        if e is None:
            continue
        resid[i] = g["s1"] / (g["s1"] + g["s2"]) - 1.0 / (1.0 + math.exp(-e))
    dates = [g["date"] for g in all_games]

    def dnum(s):
        y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
        return (y * 12 + m) * 31 + d     # monotone pseudo-day; fine for decay

    cur = defaultdict(float)
    last = {}
    feat = np.zeros(len(all_games))
    i = 0
    while i < len(all_games):
        j = i
        while j < len(all_games) and dates[j] == dates[i]:
            j += 1
        for k in range(i, j):            # price with pre-date state
            g = all_games[k]
            f = 0.0
            for slot, u in enumerate(g["us"]):
                m = cur[u] * (0.5 ** ((dnum(dates[k]) - last.get(u, dnum(dates[k]))) / hl_days))
                f += m if slot < 2 else -m
            feat[k] = f
        for k in range(i, j):            # then absorb the date's games
            g = all_games[k]
            for slot, u in enumerate(g["us"]):
                d = dnum(dates[k])
                cur[u] = cur[u] * (0.5 ** ((d - last.get(u, d)) / hl_days)) \
                    + (resid[k] if slot < 2 else -resid[k])
                last[u] = d
        i = j
    return feat


# --------------------------------------------- families B/C: MAP refit --


class MapFit:
    """Fast MAP refit of per-player values under a chosen team structure
    and observation likelihood.  Static values, ridge prior (v2's sd_v),
    no dyad/match effects — deliberately the same footing for every row so
    within-family deltas isolate the axis being tested."""

    def __init__(self, games, level="points", structure="gamma",
                 decay_hl_days=None, prior_sd=SD_V_PRIOR):
        self.level = level
        self.structure = structure
        self.prior_sd = prior_sd
        self.pidx = {}
        for g in games:
            for u in g["us"]:
                self.pidx.setdefault(u, len(self.pidx))
        if level == "match":
            by = defaultdict(list)
            for g in games:
                by[g["match"]].append(g)
            obs = []
            for m, gs in by.items():
                w1 = sum(g["won"] for g in gs)
                if 2 * w1 == len(gs):
                    continue
                obs.append(dict(us=gs[0]["us"], won=w1 * 2 > len(gs),
                               s1=0, s2=0, date=gs[0]["date"]))
            self.obs = obs
        else:
            self.obs = games
        n = len(self.obs)
        self.A = np.array([[self.pidx[u] for u in o["us"]] for o in self.obs])
        self.S1 = np.array([o["s1"] for o in self.obs], float)
        self.S2 = np.array([o["s2"] for o in self.obs], float)
        self.won = np.array([o["won"] for o in self.obs])
        self.w = np.ones(n)
        if decay_hl_days:
            def dnum(s):
                return ((int(s[:4]) * 12 + int(s[5:7])) * 31 + int(s[8:10]))
            end = max(dnum(o["date"]) for o in self.obs)
            self.w = 0.5 ** ((end - np.array([dnum(o["date"]) for o in self.obs]))
                             / decay_hl_days)

    TAU = 0.05   # softmin sharpness for min/max structures

    def team_and_grad(self, v, cols):
        """team value + d(team)/d(v_a), d/d(v_b) for one side's columns."""
        a, b = v[self.A[:, cols[0]]], v[self.A[:, cols[1]]]
        if self.structure == "sum":
            return a + b, np.ones_like(a), np.ones_like(b)
        if self.structure == "gamma":
            s = np.sign(a - b)
            return (a + b + self._gamma * np.abs(a - b),
                    1 + self._gamma * s, 1 - self._gamma * s)
        if self.structure in ("min", "max"):
            sgn = -1.0 if self.structure == "min" else 1.0
            soft = self.TAU * sgn * np.logaddexp(sgn * a / self.TAU,
                                                 sgn * b / self.TAU)
            wa = 1.0 / (1.0 + np.exp(-sgn * (a - b) / self.TAU))
            return 2 * soft, 2 * wa, 2 * (1 - wa)
        raise ValueError(self.structure)

    def fit(self, gamma_free=True, gamma0=GAMMA_V2, maxiter=400):
        n_p = len(self.pidx)
        self._gamma = gamma0
        fit_gamma = self.structure == "gamma" and gamma_free
        fit_sigma = self.level == "margin"

        def unpack(x):
            v = x[:n_p]
            i = n_p
            if fit_gamma:
                self._gamma = x[i]
                i += 1
            sig = math.exp(x[i]) if fit_sigma else None
            return v, sig

        def obj(x):
            v, sig = unpack(x)
            t1, g1a, g1b = self.team_and_grad(v, (0, 1))
            t2, g2a, g2b = self.team_and_grad(v, (2, 3))
            eta = t1 - t2
            grad_v = np.zeros(n_p)
            g_gamma = 0.0
            g_lsig = 0.0
            if self.level == "points":
                p = 1.0 / (1.0 + np.exp(-eta))
                nll = -(self.w * (self.S1 * np.log(p + 1e-12)
                                  + self.S2 * np.log(1 - p + 1e-12))).sum()
                deta = self.w * (self.S1 + self.S2) * p - self.w * self.S1
            elif self.level in ("game", "match"):
                p = 1.0 / (1.0 + np.exp(-eta))
                y = self.won.astype(float)
                nll = -(self.w * (y * np.log(p + 1e-12)
                                  + (1 - y) * np.log(1 - p + 1e-12))).sum()
                deta = self.w * (p - y)
            elif self.level == "margin":
                m = self.S1 - self.S2
                r = m - eta
                nll = (self.w * (0.5 * (r / sig) ** 2 + math.log(sig))).sum()
                deta = -self.w * r / sig ** 2
                g_lsig = (self.w * (1 - (r / sig) ** 2)).sum()
            else:
                raise ValueError(self.level)
            for cols, tg in (((0, 1), (g1a, g1b)), ((2, 3), (g2a, g2b))):
                sgn = 1.0 if cols == (0, 1) else -1.0
                np.add.at(grad_v, self.A[:, cols[0]], sgn * deta * tg[0])
                np.add.at(grad_v, self.A[:, cols[1]], sgn * deta * tg[1])
            if fit_gamma:
                a1, b1 = v[self.A[:, 0]], v[self.A[:, 1]]
                a2, b2 = v[self.A[:, 2]], v[self.A[:, 3]]
                g_gamma = (deta * (np.abs(a1 - b1) - np.abs(a2 - b2))).sum() \
                    + self._gamma / 0.3 ** 2
                nll += 0.5 * (self._gamma / 0.3) ** 2
            nll += 0.5 * (v ** 2).sum() / self.prior_sd ** 2
            grad_v += v / self.prior_sd ** 2
            grad = [grad_v]
            if fit_gamma:
                grad.append([g_gamma])
            if fit_sigma:
                grad.append([g_lsig])
            return nll, np.concatenate([np.atleast_1d(np.asarray(g, float))
                                        for g in grad])

        x0 = np.zeros(n_p + (1 if fit_gamma else 0) + (1 if fit_sigma else 0))
        if fit_gamma:
            x0[n_p] = gamma0
        if fit_sigma:
            x0[-1] = math.log(4.0)
        res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter))
        v, sig = unpack(res.x)
        self.values = {u: float(v[i]) for u, i in self.pidx.items()}
        self.gamma = float(self._gamma) if self.structure == "gamma" else None
        self.sigma = float(sig) if fit_sigma else None
        return self

    def eta(self, games):
        """Raw structure score for arbitrary games (nan if player unseen)."""
        out = np.full(len(games), np.nan)
        for i, g in enumerate(games):
            if any(u not in self.values for u in g["us"]):
                continue
            vs = [self.values[u] for u in g["us"]]

            def team(a, b):
                if self.structure == "sum":
                    return a + b
                if self.structure == "gamma":
                    return a + b + self.gamma * abs(a - b)
                if self.structure == "min":
                    return 2 * min(a, b)
                if self.structure == "max":
                    return 2 * max(a, b)
            out[i] = team(vs[0], vs[1]) - team(vs[2], vs[3])
        return out


# ------------------------------------------------------- family E: elo --


def run_elo(all_games, level="game", ks=(0.02, 0.04, 0.08, 0.12, 0.2, 0.3)):
    """Prequential Elo: predict each game from current ratings, then
    update.  K chosen by sequential log loss on TRAIN games only; holdout
    predictions keep updating walk-forward (as a live system would)."""

    def sweep(K):
        r = defaultdict(float)
        preds = np.zeros(len(all_games))
        for i, g in enumerate(all_games):
            d = (r[g["us"][0]] + r[g["us"][1]]
                 - r[g["us"][2]] - r[g["us"][3]])
            p = 1.0 / (1.0 + math.exp(-d))
            preds[i] = p
            y = (g["s1"] / (g["s1"] + g["s2"])) if level == "point" \
                else float(g["won"])
            delta = K * (y - p)
            for j, u in enumerate(g["us"]):
                r[u] += delta if j < 2 else -delta
        return preds

    train_mask = np.array([g["date"] < SPLIT for g in all_games])
    warm = np.array([g["date"] >= "2024-07-01" for g in all_games])
    best = None
    for K in ks:
        preds = sweep(K)
        pc = np.clip(preds[train_mask & warm], 1e-9, 1 - 1e-9)
        won = np.array([g["won"] for g in all_games])[train_mask & warm]
        ll = -np.mean(np.where(won, np.log(pc), np.log(1 - pc)))
        if best is None or ll < best[1]:
            best = (K, ll, preds)
    return best


# ------------------------------------------------ family D: rally logs --

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23


def load_rallies(games, until=SPLIT, since="2026-01-01"):
    """Per-rally (serving pair, returning pair, server won) from cached
    referee logs, restricted to games we model.  Uses the same
    rally→outcome attribution as harvest_logs.py."""
    raw = ROOT / "raw" / "match_logs"
    by_match = defaultdict(list)
    for g in games:
        if since <= g["date"] < until:
            by_match[g["match"]].append(g)
    rallies = []
    n_logs = 0
    for m, gs in by_match.items():
        p = raw / m[:2] / f"{m}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
        if not rows:
            continue
        n_logs += 1
        side = {}
        for g in gs:
            for j, u in enumerate(g["us"]):
                side[u.lower()] = 0 if j < 2 else 1
        team1 = frozenset(u.lower() for g in gs for u in g["us"][:2])
        team2 = frozenset(u.lower() for g in gs for u in g["us"][2:])
        rows = sorted(rows, key=lambda x: x.get("log_index", 0))
        current = None
        for r in rows:
            t = r.get("log_type")
            if t == RALLY:
                current = r
            elif t == POINT:
                if current is not None:
                    su = (current.get("server_uuid") or "").lower()
                    if su in side:
                        rallies.append((m, su, side[su], True))
                current = None
            elif t in (SIDEOUT, SECOND):
                if current is not None:
                    su = (current.get("server_uuid") or "").lower()
                    if su in side:
                        rallies.append((m, su, side[su], False))
                current = None
    return rallies, n_logs


class RallyFit:
    """Rally-level tug of war: P(serving side wins the rally) =
    sigmoid(alpha + serve_value(pair serving) - return_value(pair
    returning)).  Per-player serve/return values, ridge prior.  A game's
    (kA, kB) then feed the exact serve-aware DP — no anchoring."""

    def __init__(self, rallies, match_sides, prior_sd=0.25):
        self.prior_sd = prior_sd
        self.pidx = {}
        for m, su, s, w in rallies:
            for u in match_sides[m][0] | match_sides[m][1]:
                self.pidx.setdefault(u, len(self.pidx))
        n = len(rallies)
        self.SRV = np.zeros((n, 2), int)
        self.RET = np.zeros((n, 2), int)
        self.y = np.zeros(n)
        for i, (m, su, s, w) in enumerate(rallies):
            srv = sorted(match_sides[m][s])
            ret = sorted(match_sides[m][1 - s])
            self.SRV[i] = [self.pidx[u] for u in srv]
            self.RET[i] = [self.pidx[u] for u in ret]
            self.y[i] = w

    def fit(self, maxiter=300):
        n_p = len(self.pidx)

        def obj(x):
            alpha, s, r = x[0], x[1:n_p + 1], x[n_p + 1:]
            eta = (alpha + s[self.SRV[:, 0]] + s[self.SRV[:, 1]]
                   - r[self.RET[:, 0]] - r[self.RET[:, 1]])
            p = 1.0 / (1.0 + np.exp(-eta))
            nll = -(self.y * np.log(p + 1e-12)
                    + (1 - self.y) * np.log(1 - p + 1e-12)).sum()
            de = p - self.y
            gs = np.zeros(n_p)
            gr = np.zeros(n_p)
            np.add.at(gs, self.SRV[:, 0], de)
            np.add.at(gs, self.SRV[:, 1], de)
            np.add.at(gr, self.RET[:, 0], -de)
            np.add.at(gr, self.RET[:, 1], -de)
            nll += 0.5 * ((s ** 2).sum() + (r ** 2).sum()) / self.prior_sd ** 2
            gs += s / self.prior_sd ** 2
            gr += r / self.prior_sd ** 2
            return nll, np.concatenate([[de.sum()], gs, gr])

        res = minimize(obj, np.zeros(1 + 2 * len(self.pidx)), jac=True,
                       method="L-BFGS-B", options=dict(maxiter=maxiter))
        x = res.x
        n_p = len(self.pidx)
        self.alpha = float(x[0])
        self.serve = {u: float(x[1 + i]) for u, i in self.pidx.items()}
        self.ret = {u: float(x[1 + n_p + i]) for u, i in self.pidx.items()}
        return self


def serve_dp_win(kA, kB, T):
    """Pre-match win prob from the serve-aware DP, averaging the coin flip
    over which side opens (on its #2 server, per the standard exception)."""
    import sys
    sys.path.insert(0, str(ROOT / "web"))
    from sitelib.winprob import _table, A2, B2
    V = _table(round(kA, 6), round(kB, 6), T, T + 40)
    return 0.5 * (V[(0, 0, A2)] + V[(0, 0, B2)])


# ---------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", action="store_true",
                    help="include family D (needs raw/match_logs)")
    ap.add_argument("--quick", action="store_true",
                    help="skip families B/C/E (iterate on A)")
    args = ap.parse_args()

    games = load_games()
    players, chem = load_v2()
    racer = Racer()

    train = [g for g in games if g["date"] < SPLIT]
    recent = [g for g in train if g["date"] >= RECENT]
    hold = [g for g in games if g["date"] >= SPLIT
            and all(u in players and players[u]["games"] >= MIN_TRAIN_GAMES
                    for u in g["us"])]
    print(f"train={len(train)} (recent={len(recent)})  holdout={len(hold)}")

    F_hold = Frame(hold, players, chem)
    F_recent = Frame(recent, players, chem)
    won = F_hold.won
    results = {}
    probs = {}

    def record(name, pw, extra=None, frame=None, ref=None):
        fr = frame if frame is not None else F_hold
        m = metrics(pw, fr.won)
        row = dict(**m)
        if extra:
            row.update(extra)
        if ref is not None:
            row["vs_ref"] = paired_boot(pw, ref, fr.won)
        results[name] = row
        probs[name] = pw
        print(f"{name:24s} n={m['n']:4d} acc={m['accuracy']:.4f} "
              f"brier={m['brier']:.4f} ll={m['log_loss']:.4f}")
        return pw

    # ---- reference: v2 plugin (posterior means; draws file is not in the
    # clone).  Published draws-based numbers: 0.7738 / 0.1653 / 0.5058.
    ref = record("v2_plugin", racer.win(F_hold.eta_v2(), F_hold.T),
                 extra=dict(family="ref",
                            desc="v2 as shipped: sum + gamma|gap| + chem, "
                                 "posterior-mean plugin"))

    # v2 with uncertainty mixture (approximates draws-averaging)
    sd_h = np.sqrt(F_hold.sd2())
    record("v2_mixture", racer.win_mixture(F_hold.eta_v2(), sd_h, F_hold.T),
           extra=dict(family="ref",
                      desc="same eta, integrated over N(eta, sum sd_i^2) — "
                           "plugin->draws approximation"),
           ref=ref)

    # ---- family A ------------------------------------------------------
    def strat(name, eta_fn, x0, desc, frame_fit=F_recent, frame_ev=F_hold,
              ref_pw=None):
        x = fit_params(frame_fit, racer, eta_fn, x0)
        eta = eta_fn(frame_ev, x)
        ok = np.isfinite(eta)
        pw = racer.win(eta[ok], frame_ev.T[ok])
        fr = frame_ev if ok.all() else frame_ev.sub(ok)
        rp = ref_pw if ref_pw is not None else \
            (ref if ok.all() else racer.win(fr.eta_v2(), fr.T))
        record(name, pw, extra=dict(family="A", params=[round(float(t), 4) for t in x],
                                    desc=desc), frame=fr, ref=rp)
        return x

    strat("A_v2_rescaled", lambda f, x: x[0] * f.eta_v2(), [1.0],
          "free overall scale on the v2 eta (calibration check)")
    strat("A_sum", lambda f, x: x[0] * f.d_sum(), [1.0],
          "team = sum of the two ratings; no weakest-link, no chem")
    strat("A_weak_gamma", lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()),
          [1.0, GAMMA_V2],
          "sum + free gamma|gap| == free weights on (min, max)")
    strat("A_gamma_by_ctx",
          lambda f, x: x[0] * (f.d_sum()
                               + np.where(f.ctx == "mixed", x[1], x[2])
                               * f.d_gap()),
          [1.0, GAMMA_V2, GAMMA_V2],
          "separate weakest-link gamma for mixed vs same-gender games")
    strat("A_min_only", lambda f, x: x[0] * f.d_min(), [1.0],
          "weaker player only (gamma = -1)")
    strat("A_max_only", lambda f, x: x[0] * f.d_max(), [1.0],
          "stronger player only (gamma = +1)")
    strat("A_chem", lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap())
          + x[2] * f.chem, [1.0, GAMMA_V2, 1.0],
          "weak_gamma + free loading on the dyad chemistry term")
    strat("A_seed_order",
          lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap())
          + np.where(f.tour_mlp, x[2], x[3]),
          [1.0, GAMMA_V2, 0.0, 0.0],
          "weak_gamma + team-listed-first intercept per tour (seeding info)")
    strat("A_experience",
          lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()) + x[2] * f.d_exp(),
          [1.0, GAMMA_V2, 0.0],
          "weak_gamma + log career-games differential")

    # momentum needs features precomputed over the full timeline
    mom = momentum_features(games, players, chem)
    mom_by_id = {id(g): m for g, m in zip(games, mom)}
    F_hold.mom = np.array([mom_by_id[id(g)] for g in hold])
    F_recent.mom = np.array([mom_by_id[id(g)] for g in recent])
    Frame.d_mom = lambda self: self.mom
    strat("A_momentum",
          lambda f, x: x[0] * (f.d_sum() + x[1] * f.d_gap()) + x[2] * f.d_mom(),
          [1.0, GAMMA_V2, 0.0],
          "weak_gamma + EWMA of recent over/under-performance (45d half-life)")

    # uncertainty shrink: mixture width as a free multiple of sum sd_i^2
    def eta_shrink(f, x):
        return f.eta_v2()          # eta unchanged; width handled below

    def fit_shrink():
        def nll(a):
            pw = racer.win_mixture(F_recent.eta_v2(),
                                   abs(a[0]) * np.sqrt(F_recent.sd2()),
                                   F_recent.T)
            pc = np.clip(pw, 1e-9, 1 - 1e-9)
            return -np.mean(np.where(F_recent.won, np.log(pc), np.log(1 - pc)))
        res = minimize(nll, [1.0], method="Nelder-Mead",
                       options=dict(xatol=1e-3))
        a = abs(float(res.x[0]))
        pw = racer.win_mixture(F_hold.eta_v2(), a * sd_h, F_hold.T)
        record("A_uncert_shrink", pw,
               extra=dict(family="A", params=[round(a, 3)],
                          desc="v2 eta integrated over value uncertainty, "
                               "width fit on 2026 train"), ref=ref)
    fit_shrink()

    # ---- mixed arena ---------------------------------------------------
    mx_h = F_hold.ctx == "mixed"
    mx_r = F_recent.ctx == "mixed"
    Fh_mx, Fr_mx = F_hold.sub(mx_h), F_recent.sub(mx_r)
    ref_mx = racer.win(Fh_mx.eta_v2(), Fh_mx.T)
    record("MX_v2", ref_mx, frame=Fh_mx,
           extra=dict(family="mixed-arena",
                      desc="v2 plugin on mixed games only"))
    for nm, fn, x0, ds in [
        ("MX_men_only", lambda f, x: x[0] * f.d_man(), [2.0],
         "mixed priced from the two men alone"),
        ("MX_women_only", lambda f, x: x[0] * f.d_wom(), [2.0],
         "mixed priced from the two women alone"),
        ("MX_gender_free", lambda f, x: x[0] * f.d_man() + x[1] * f.d_wom(),
         [1.0, 1.0], "free loadings per gender slot (the mixed 'SEM')"),
        ("MX_weaker_gender", lambda f, x: x[0] * np.minimum(f.man[:, 0], f.wom[:, 0])
         - x[0] * np.minimum(f.man[:, 1], f.wom[:, 1]), [2.0],
         "mixed priced from each team's weaker member (offset-dependent!)"),
    ]:
        strat(nm, fn, x0, ds, frame_fit=Fr_mx, frame_ev=Fh_mx, ref_pw=ref_mx)

    if args.quick:
        finish(results)
        return

    # ---- family B: structural refits ----------------------------------
    def refit(name, desc, ref_pw=ref, **kw):
        mf = MapFit(train, **kw).fit()
        eta_r = mf.eta(recent)
        s = fit_params_scalar(racer, eta_r, F_recent)
        eta_h = mf.eta(hold)
        pw = racer.win(s * eta_h, F_hold.T)
        level = kw.get("level", "points")
        extra = dict(family="C" if level != "points" else "B",
                     scale=round(s, 4), desc=desc)
        if mf.gamma is not None:
            extra["gamma"] = round(mf.gamma, 4)
        if mf.sigma is not None:
            extra["sigma"] = round(mf.sigma, 3)
        record(name, pw, extra=extra, ref=ref_pw)
        return mf

    def fit_params_scalar(racer, eta, frame):
        ok = np.isfinite(eta)

        def nll(x):
            pw = racer.win(x[0] * eta[ok], frame.T[ok])
            pc = np.clip(pw, 1e-9, 1 - 1e-9)
            return -np.mean(np.where(frame.won[ok], np.log(pc), np.log(1 - pc)))
        res = minimize(nll, [1.0], method="Nelder-Mead",
                       options=dict(xatol=1e-4))
        return float(res.x[0])

    refit("B_refit_sum", "values refit under team=sum (static MAP)",
          structure="sum")
    refit("B_refit_gamma", "values refit under sum+gamma|gap| — static v2 twin",
          structure="gamma")
    refit("B_refit_min", "values refit under team=2*min (weakest link only)",
          structure="min")
    refit("B_refit_max", "values refit under team=2*max (strongest only)",
          structure="max")
    refit("B_refit_decay", "sum+gamma with 12-month-half-life game weights",
          structure="gamma", decay_hl_days=365.0)

    # ---- family C: levels of analysis ----------------------------------
    refit("C_match_level", "Bernoulli on match winners only (best-of collapsed)",
          structure="gamma", level="match")
    refit("C_game_level", "Bernoulli on game winners only (margin-blind)",
          structure="gamma", level="game")
    refit("C_margin_level", "Gaussian on point margin (v1's likelihood, "
          "static twin)", structure="gamma", level="margin", prior_sd=2.5)
    # C at point level == B_refit_gamma; cross-referenced in the writeup.

    # v1 as shipped (Gaussian margin, points scale, own holdout method)
    v1p = {r["player_id"]: float(r["value_mean"])
           for r in csv.DictReader((DATA / "results_players_train.csv").open())}
    dyid = {frozenset((d["p1_id"], d["p2_id"])): int(d["idx"])
            for d in csv.DictReader((DATA / "model_dyads_train.csv").open())}
    v1chem = [float(r["chemistry_mean"])
              for r in csv.DictReader((DATA / "results_dyads_train.csv").open())]
    from statistics import NormalDist
    nd = NormalDist()
    ok = np.array([all(u in v1p for u in g["us"]) for g in hold])
    mu = np.array([
        v1p[g["us"][0]] + v1p[g["us"][1]] - v1p[g["us"][2]] - v1p[g["us"][3]]
        + (v1chem[dyid[frozenset(g["us"][:2])]] if frozenset(g["us"][:2]) in dyid else 0)
        - (v1chem[dyid[frozenset(g["us"][2:])]] if frozenset(g["us"][2:]) in dyid else 0)
        for g, o in zip(hold, ok) if o])
    pw_v1 = np.array([nd.cdf(m / 4.70) for m in mu])
    fr1 = F_hold.sub(ok)
    record("C_v1_published", pw_v1, frame=fr1,
           extra=dict(family="C", desc="v1 Gaussian margin model as shipped "
                      "(per-season fit, sd=4.70)"),
           ref=racer.win(fr1.eta_v2(), fr1.T))

    # ---- family E ------------------------------------------------------
    hold_ids = {id(g) for g in hold}
    hmask = np.array([id(g) in hold_ids for g in games])
    K, ll, preds = run_elo(games, level="game")
    record("E_elo_game", preds[hmask],
           extra=dict(family="E", params=[K],
                      desc=f"prequential Elo on game winners (K={K})"), ref=ref)
    K2, ll2, preds2 = run_elo(games, level="point",
                              ks=(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2))
    # point-share Elo emits a point-logit; map through the race DP
    eta_e = np.log(np.clip(preds2, 1e-9, 1 - 1e-9)
                   / np.clip(1 - preds2, 1e-9, 1 - 1e-9))
    s_e = None
    tr_mask = np.array([g["date"] < SPLIT and g["date"] >= RECENT
                        for g in games])

    def nll_e(x):
        pw = racer.win(x[0] * eta_e[tr_mask],
                       np.array([g["T"] for g, m in zip(games, tr_mask) if m]))
        wtr = np.array([g["won"] for g, m in zip(games, tr_mask) if m])
        pc = np.clip(pw, 1e-9, 1 - 1e-9)
        return -np.mean(np.where(wtr, np.log(pc), np.log(1 - pc)))
    s_e = float(minimize(nll_e, [1.0], method="Nelder-Mead",
                         options=dict(xatol=1e-4)).x[0])
    record("E_elo_point", racer.win(s_e * eta_e[hmask], F_hold.T),
           extra=dict(family="E", params=[K2, round(s_e, 3)],
                      desc=f"prequential Elo on point shares (K={K2}), "
                           "race-DP priced"), ref=ref)

    record("E_coin", np.full(len(hold), 0.5),
           extra=dict(family="E", desc="coin flip floor"), ref=ref)

    # platform-rating arena (as-of-match snapshots; sparse coverage)
    platform_arena(games, hold, racer, F_hold, record, ref)

    # zero-fitted-weight ensemble of three unlike systems (a leak-free
    # blend: equal logit weights, nothing tuned)
    def logit(p):
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(pc / (1 - pc))
    members = ("v2_plugin", "C_margin_level", "E_elo_game")
    if all(m in probs and len(probs[m]) == len(hold) for m in members):
        ens = 1.0 / (1.0 + np.exp(-np.mean(
            [logit(probs[m]) for m in members], axis=0)))
        record("X_ensemble", ens,
               extra=dict(family="X",
                          desc="equal-weight logit average of v2 + margin "
                               "refit + Elo (no fitted weights)"), ref=ref)

    # ---- family D ------------------------------------------------------
    if args.rally:
        rally_family(games, train, recent, hold, players, chem, racer,
                     F_hold, F_recent, record, ref,
                     fit_params_scalar)

    finish(results)


def platform_arena(games, hold, racer, F_hold, record, ref):
    pm = json.loads((DATA / "per_match_ratings.json").read_text())
    mdate = {}
    for g in games:
        mdate.setdefault(g["match"], g["date"])
    snaps = defaultdict(list)
    for m, d in pm.items():
        dt = mdate.get(m.lower())
        if not dt:
            continue
        for u, r in d.items():
            snaps[u].append((dt, float(r)))
    for u in snaps:
        snaps[u].sort()

    def as_of(u, date):
        best = None
        for dt, r in snaps.get(u, ()):
            if dt < date:
                best = r
            else:
                break
        return best

    def game_ratings(g):
        rs = [as_of(u, g["date"]) for u in g["us"]]
        return None if any(r is None for r in rs) else rs

    rated_h = [(i, game_ratings(g)) for i, g in enumerate(hold)]
    rated_h = [(i, r) for i, r in rated_h if r]
    idx = np.array([i for i, _ in rated_h])
    if len(idx) < 30:
        print(f"platform arena skipped: only {len(idx)} rated holdout games")
        return
    R = np.array([r for _, r in rated_h])
    mask = np.zeros(len(hold), bool)
    mask[idx] = True
    fr = F_hold.sub(mask)
    ref_sub = racer.win(fr.eta_v2(), fr.T)
    record("PL_v2", ref_sub, frame=fr,
           extra=dict(family="platform-arena",
                      desc="v2 plugin on the platform-rated subset"))
    # calibrate each on pre-split rated games
    train_rated = [(g, game_ratings(g)) for g in games
                   if RECENT <= g["date"] < SPLIT]
    train_rated = [(g, r) for g, r in train_rated if r]
    Ttr = np.array([g["T"] for g, _ in train_rated])
    wtr = np.array([g["won"] for g, _ in train_rated])
    Rtr = np.array([r for _, r in train_rated])
    print(f"platform arena: {len(idx)} holdout / {len(train_rated)} train "
          "rated games")

    def run(name, feat_fn, desc):
        ftr = feat_fn(Rtr)
        fho = feat_fn(R)

        def nll(x):
            pw = racer.win(x[0] * ftr, Ttr)
            pc = np.clip(pw, 1e-9, 1 - 1e-9)
            return -np.mean(np.where(wtr, np.log(pc), np.log(1 - pc)))
        s = float(minimize(nll, [0.5], method="Nelder-Mead",
                           options=dict(xatol=1e-4)).x[0])
        record(name, racer.win(s * fho, fr.T), frame=fr,
               extra=dict(family="platform-arena", params=[round(s, 3)],
                          desc=desc), ref=ref_sub)

    run("PL_sum", lambda R: R[:, 0] + R[:, 1] - R[:, 2] - R[:, 3],
        "platform (DUPR-synced) rating sums, as-of-match")
    run("PL_min", lambda R: 2 * (np.minimum(R[:, 0], R[:, 1])
                                 - np.minimum(R[:, 2], R[:, 3])),
        "platform rating, weaker player only")
    run("PL_max", lambda R: 2 * (np.maximum(R[:, 0], R[:, 1])
                                 - np.maximum(R[:, 2], R[:, 3])),
        "platform rating, stronger player only")


def rally_family(games, train, recent, hold, players, chem, racer,
                 F_hold, F_recent, record, ref, fit_params_scalar):
    sides = {}
    for g in games:
        m = g["match"]
        if m not in sides:
            sides[m] = (frozenset(u.lower() for u in g["us"][:2]),
                        frozenset(u.lower() for u in g["us"][2:]))
    rallies, n_logs = load_rallies(games)
    print(f"rally family: {len(rallies)} rallies from {n_logs} matches")
    if len(rallies) < 50000:
        print("not enough rally logs yet — skipping family D")
        return
    rf = RallyFit(rallies, sides).fit()
    k_league = float(np.mean([r[3] for r in rallies]))

    def game_ks(g):
        us = [u.lower() for u in g["us"]]
        if any(u not in rf.serve for u in us):
            return None
        sA = rf.serve[us[0]] + rf.serve[us[1]]
        sB = rf.serve[us[2]] + rf.serve[us[3]]
        rA = rf.ret[us[0]] + rf.ret[us[1]]
        rB = rf.ret[us[2]] + rf.ret[us[3]]
        kA = 1 / (1 + math.exp(-(rf.alpha + sA - rB)))
        kB = 1 / (1 + math.exp(-(rf.alpha + sB - rA)))
        return kA, kB

    ok = np.array([game_ks(g) is not None for g in hold])
    pw = np.array([serve_dp_win(*game_ks(g), g["T"])
                   for g, o in zip(hold, ok) if o])
    fr = F_hold.sub(ok)
    record("D_rally_srvret", pw, frame=fr,
           extra=dict(family="D", k_league=round(k_league, 4),
                      alpha=round(rf.alpha, 4),
                      desc="per-player serve+return values fit on 2026H1 "
                           "rallies; serve-aware DP, no anchoring"),
           ref=racer.win(fr.eta_v2(), fr.T))

    # window-matched control: point-binomial refit on the SAME games
    ctl_games = [g for g in train if g["date"] >= "2026-01-01"]
    mf = MapFit(ctl_games, structure="gamma").fit()
    eta_r = mf.eta(recent)
    s = fit_params_scalar(racer, eta_r, F_recent)
    eta_h = mf.eta(hold)
    okc = np.isfinite(eta_h)
    frc = F_hold.sub(okc)
    record("D_points_2026only", racer.win(s * eta_h[okc], frc.T), frame=frc,
           extra=dict(family="D", desc="point-binomial refit on 2026H1 games "
                      "only — window control for the rally model"),
           ref=racer.win(frc.eta_v2(), frc.T))

    # v2 eta through the serve DP WITHOUT anchoring (does serve structure
    # alone move pre-match prices?)
    from functools import lru_cache

    @lru_cache(maxsize=4096)
    def sdw(eta_r, T):
        import sys
        sys.path.insert(0, str(ROOT / "web"))
        from sitelib.winprob import serve_probs, _table, A2, B2
        kA, kB = serve_probs(eta_r, 0.43)
        V = _table(round(kA, 6), round(kB, 6), T, T + 40)
        return 0.5 * (V[(0, 0, A2)] + V[(0, 0, B2)])

    eta_v2 = F_hold.eta_v2()
    pw = np.array([sdw(round(float(e), 2), int(t))
                   for e, t in zip(eta_v2, F_hold.T)])
    record("D_v2_serveDP", pw,
           extra=dict(family="D", desc="v2 eta through the serve-aware DP "
                      "(k=0.43) with NO anchoring"), ref=ref)


def finish(results):
    out = OUT / "spec_shootout_summary.json"
    payload = dict(split=SPLIT, generated="2026-07-18",
                   reference="v2_plugin", results=results)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
