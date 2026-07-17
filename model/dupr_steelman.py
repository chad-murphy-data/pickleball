"""Steelmanned DUPR-vs-model comparison on one identical, glitch-repaired game set.

The published baseline (`rating_comparison.py`) prices DUPR the naive way: sum the
four synced DUPR doubles ratings, difference the teams, push through a single fitted
probit slope.  That is a fair *floor* but not DUPR's best shot.  This script gives DUPR
the same courtesy we give ourselves and isolates where our edge actually comes from,
via a 2x2 of {our values, DUPR ratings} x {bare probit, our race-DP + weakest-link}:

  model        = our v2 values  -> race-to-T DP + gamma|gap| weakest link (our machinery)
  model_bare   = our v2 values  -> single probit slope (the naive form, our ratings)
  dupr_steel   = DUPR ratings    -> race-to-T DP + fitted gap term (our machinery, DUPR)
  dupr_bare    = DUPR ratings    -> single probit slope (DUPR, as published)

Rows isolate the *ratings*; columns isolate the *machinery*.  dupr_steel is the honest
"did we give DUPR a fair shot" number; the residual model - dupr_steel is the pure
ratings edge once the pipeline is held constant.

Fairness fixes baked in (levers 1 & 3):
  * Every predictor is scored on the SAME game set (both sides must rate all four),
    so no 884-vs-518 apples-to-oranges.
  * DUPR's known reset artifact (a >=5.0 player collapsing to the ~3.5 DUPR reset
    default) is screened out of its inputs before scoring -- we do not beat DUPR on
    games where its input is a platform glitch it would itself disown.  Mirrors
    web/sitelib/data.finalize_dupr.

Free of the v2 posterior draws (gitignored): uses value_now_mean point estimates as a
plug-in for our machinery, so the model cell is the plug-in v2 and may differ a hair
from the draws-integrated 77.4% headline.  Run: python model/dupr_steelman.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT = "2026-06-01"
MIN_TRAIN_GAMES = 10
GAMMA = -0.183          # v2 weakest-link, per-point logit (model/v2_results.md)
GLITCH_CEIL = 3.65      # a rating at/below this...
GLITCH_PEAK = 5.0       # ...for a player who once sat here is a DUPR reset artifact


def race_win_table(T, grid):
    """P(team1 wins race to T, win by 2) for each p in grid -- exact DP.

    Identical algebra to model/v2_holdout.py: deuce at (T-1,T-1) resolved by the
    closed form p^2 / (p^2 + q^2)."""
    out = np.zeros_like(grid)
    for gi, p in enumerate(grid):
        q = 1 - p
        dp = np.zeros((T + 1, T + 1))
        dp[0, 0] = 1.0
        win = deuce_mass = 0.0
        for a in range(T + 1):
            for b in range(T + 1):
                if dp[a, b] == 0:
                    continue
                if a == T - 1 and b == T - 1:
                    deuce_mass += dp[a, b]; continue
                if a == T:
                    win += dp[a, b]; continue
                if b == T:
                    continue
                if a + 1 == T and b <= T - 2:
                    win += dp[a, b] * p
                else:
                    dp[a + 1, b] += dp[a, b] * p
                dp[a, b + 1] += dp[a, b] * q
        win += deuce_mass * (p * p / (p * p + q * q + 1e-12))
        out[gi] = win
    return out


GRID = np.linspace(0.01, 0.99, 981)
TABLES = {11: race_win_table(11, GRID), 15: race_win_table(15, GRID)}


_ERF = np.vectorize(__import__("math").erf)


def _phi(x):
    """Standard normal CDF, vectorised (no scipy dependency)."""
    return 0.5 * (1.0 + _ERF(np.asarray(x, float) / np.sqrt(2)))


def race_prob(eta_point, T):
    """Per-point logit -> win prob through the race DP (vectorised over games)."""
    p = 1.0 / (1.0 + np.exp(-eta_point))
    return np.interp(p, GRID, TABLES[T])


def load():
    per_match = json.loads((DATA / "per_match_ratings.json").read_text())
    # peak DUPR per player across all snapshots -> reset-artifact screen
    peak = {}
    for rm in per_match.values():
        for u, r in rm.items():
            if r > peak.get(u, 0):
                peak[u] = r

    def dupr_ok(u, r):
        return not (r <= GLITCH_CEIL and peak.get(u, 0) >= GLITCH_PEAK)

    vp = {r["player_id"]: r for r in csv.DictReader((DATA / "v2_players_train.csv").open())}
    names = {r["player_id"]: r["full_name"] for r in csv.DictReader((DATA / "players.csv").open())}
    chem = {}
    for r in csv.DictReader((DATA / "v2_dyads_train.csv").open()):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = float(r["chem_logit_mean"])

    games = [g for g in csv.DictReader((DATA / "games.csv").open())
             if g["is_forfeit"] == "False"
             and g["scoring_format"] in ("sideout_11", "sideout_15")]

    rows = []
    glitch_drop = 0
    for g in games:
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        rm = per_match.get(g["match_id"])
        if not rm or any(u not in rm for u in us):
            continue
        dr = [rm[u] for u in us]
        if not all(dupr_ok(u, r) for u, r in zip(us, dr)):
            glitch_drop += 1
            continue
        mv = [vp.get(u) for u in us]
        if any(r is None or int(r["games"]) < MIN_TRAIN_GAMES for r in mv):
            continue
        v = [float(r["value_now_mean"]) for r in mv]
        c1 = chem.get(frozenset((names.get(us[0], ""), names.get(us[1], ""))), 0.0)
        c2 = chem.get(frozenset((names.get(us[2], ""), names.get(us[3], ""))), 0.0)
        rows.append({
            "T": 11 if g["scoring_format"] == "sideout_11" else 15,
            "won": int(g["margin"]) > 0,
            "pre": g["date"] < SPLIT,
            # DUPR features
            "d_sum": dr[0] + dr[1] - dr[2] - dr[3],
            "d_gap": abs(dr[0] - dr[1]) - abs(dr[2] - dr[3]),
            # model (v2 value) features
            "m_sum": v[0] + v[1] - v[2] - v[3],
            "m_eta": (v[0] + v[1] + GAMMA * abs(v[0] - v[1]))
                     - (v[2] + v[3] + GAMMA * abs(v[2] - v[3])) + c1 - c2,
        })
    return rows, glitch_drop


def scores(pw, won):
    pw = np.asarray(pw); won = np.asarray(won, float)
    acc = float(np.mean((pw > 0.5) == (won > 0.5)))
    brier = float(np.mean((pw - won) ** 2))
    pc = np.clip(pw, 1e-6, 1 - 1e-6)
    ll = float(np.mean(-(won * np.log(pc) + (1 - won) * np.log(1 - pc))))
    return {"accuracy": acc, "brier": brier, "log_loss": ll}


def fit_probit(diff, won):
    """Best single probit slope k for P = Phi(k*diff), by Brier on the given set."""
    ks = np.arange(0.05, 12.0, 0.05)
    best_k, best_b = ks[0], 1e9
    for k in ks:
        b = float(np.mean((_phi(k * diff) - won) ** 2))
        if b < best_b:
            best_b, best_k = b, k
    return float(best_k)


def fit_race_gap(diff, gap, won, T):
    """Best (k, g) for P = raceDP(sigma(k*(diff + g*gap)), T), by Brier."""
    best = (1.0, 0.0, 1e9)
    for k in np.arange(0.05, 4.0, 0.05):
        for g in np.arange(-1.2, 0.61, 0.05):
            b = 0.0
            for t in (11, 15):
                m = T == t
                if not m.any():
                    continue
                b += float(np.sum((race_prob(k * (diff[m] + g * gap[m]), t) - won[m]) ** 2))
            b /= len(won)
            if b < best[2]:
                best = (float(k), float(g), b)
    return best[0], best[1]


def main():
    rows, glitch_drop = load()
    pre = [r for r in rows if r["pre"]]
    test = [r for r in rows if not r["pre"]]

    def arr(rs, key):
        return np.array([r[key] for r in rs])

    won_pre, won_te = arr(pre, "won").astype(float), arr(test, "won").astype(float)
    T_pre, T_te = arr(pre, "T"), arr(test, "T")

    # --- fit calibration on pre-June only ---
    k_db = fit_probit(arr(pre, "d_sum"), won_pre)                       # DUPR bare
    k_mb = fit_probit(arr(pre, "m_sum"), won_pre)                       # model bare
    k_ds, g_ds = fit_race_gap(arr(pre, "d_sum"), arr(pre, "d_gap"),
                              won_pre, T_pre)                            # DUPR steelman

    # --- score on the identical holdout set ---
    d_sum_te, d_gap_te = arr(test, "d_sum"), arr(test, "d_gap")
    m_sum_te, m_eta_te = arr(test, "m_sum"), arr(test, "m_eta")

    pw_model = np.array([race_prob(np.array([e]), t)[0]
                         for e, t in zip(m_eta_te, T_te)])               # our machinery, no free param
    pw_dupr_steel = np.array([race_prob(np.array([k_ds * (d + g_ds * gp)]), t)[0]
                              for d, gp, t in zip(d_sum_te, d_gap_te, T_te)])
    pw_dupr_bare = _phi(k_db * d_sum_te)
    pw_model_bare = _phi(k_mb * m_sum_te)

    out = {
        "split": SPLIT,
        "note": "All four predictors scored on ONE identical holdout game set: "
                "every game rates all four players under BOTH systems, with DUPR "
                "reset artifacts screened out. Rows = ratings, columns = machinery.",
        "n_holdout_games": len(test),
        "n_pre_calibration_games": len(pre),
        "n_dropped_dupr_glitch": glitch_drop,
        "formats": "sideout_11 + sideout_15",
        "gamma_weakest_link": GAMMA,
        "fitted": {"dupr_bare_slope": k_db, "model_bare_slope": k_mb,
                   "dupr_steel_k": k_ds, "dupr_steel_gap": g_ds},
        "model": scores(pw_model, won_te),
        "model_bare": scores(pw_model_bare, won_te),
        "dupr_steelman": scores(pw_dupr_steel, won_te),
        "dupr_bare": scores(pw_dupr_bare, won_te),
    }
    (ROOT / "model" / "dupr_steelman.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
