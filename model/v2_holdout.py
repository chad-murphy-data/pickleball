"""v2 holdout: predict June+ 2026 games from the pre-June v2 train fit.

Win probability = E_draws[ P(win race to T | p_point) ], with the race
probability computed exactly by dynamic programming (win-by-2; deuce at
T-1,T-1 resolved by the closed form p^2 / (p^2 + q^2)), precomputed on a
p-grid for speed.

Run after fit_v2 with SRM2_SUFFIX=_train:  python model/v2_holdout.py
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


def race_win_table(T, grid):
    """P(team1 wins race to T, win by 2) for each p in grid, exact DP."""
    out = np.zeros_like(grid)
    for gi, p in enumerate(grid):
        q = 1 - p
        # dp[a][b] = prob of reaching state (a,b); a,b < T, capped at deuce
        dp = np.zeros((T + 1, T + 1))
        dp[0, 0] = 1.0
        win = 0.0
        deuce_mass = 0.0
        for a in range(T + 1):
            for b in range(T + 1):
                if dp[a, b] == 0:
                    continue
                if a == T - 1 and b == T - 1:
                    deuce_mass += dp[a, b]
                    continue
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


def main():
    z = np.load(ROOT / "model" / "v2_draws_train.npz", allow_pickle=True)
    v0, walk_last = z["v0"], z["walk_last"]
    dyn_id, is_dyn = z["dyn_id"], z["is_dyn"]
    gamma = z["gamma"]
    col = {str(pid): i for i, pid in enumerate(z["player_ids"])}
    tg = {r["player_id"]: int(r["games"])
          for r in csv.DictReader((DATA / "v2_players_train.csv").open())}
    chem = {}
    for r in csv.DictReader((DATA / "v2_dyads_train.csv").open()):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = float(r["chem_logit_mean"])
    names = {r["player_id"]: r["full_name"] for r in csv.DictReader((DATA / "players.csv").open())}

    def value_draws(u):
        i = col[u]
        v = v0[:, i].copy()
        if is_dyn[i]:
            v += walk_last[:, dyn_id[i]]
        return v

    grid = np.linspace(0.01, 0.99, 981)
    tables = {11: race_win_table(11, grid), 15: race_win_table(15, grid)}

    test = [g for g in csv.DictReader((DATA / "games.csv").open())
            if g["is_forfeit"] == "False"
            and g["scoring_format"] in ("sideout_11", "sideout_15")
            and g["date"] >= SPLIT]
    n = correct = 0
    brier = logloss = 0.0
    for g in test:
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        if any(u not in col or tg.get(u, 0) < MIN_TRAIN_GAMES for u in us):
            continue
        vs = [value_draws(u) for u in us]
        gap1 = np.abs(vs[0] - vs[1]); gap2 = np.abs(vs[2] - vs[3])
        c1 = chem.get(frozenset((names.get(us[0], ""), names.get(us[1], ""))), 0.0)
        c2 = chem.get(frozenset((names.get(us[2], ""), names.get(us[3], ""))), 0.0)
        eta = (vs[0] + vs[1] + gamma * gap1) - (vs[2] + vs[3] + gamma * gap2) + c1 - c2
        p_point = 1.0 / (1.0 + np.exp(-eta))
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        pw = float(np.mean(np.interp(p_point, grid, tables[T])))
        won = int(g["margin"]) > 0
        n += 1
        correct += ((pw > 0.5) == won)
        brier += (pw - won) ** 2
        pc = min(max(pw, 1e-6), 1 - 1e-6)
        logloss += -(np.log(pc) if won else np.log(1 - pc))
    out = {"split": SPLIT, "n": n, "accuracy": correct / n,
           "brier": brier / n, "log_loss": logloss / n,
           "v1_reference": {"accuracy": 0.752, "brier": 0.178}}
    (ROOT / "model" / "v2_holdout_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
