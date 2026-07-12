"""Temporal holdout validation: fit on games before SPLIT, predict the rest.

Usage:
  DATE_BEFORE=2026-06-01 OUT_SUFFIX=_train python scraper/build_model_data.py
  SRM_SUFFIX=_train python model/fit_srm.py
  python model/holdout_eval.py            # writes model/holdout_summary.json
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT = "2026-06-01"
SD = 4.70  # game-level noise (match + residual) from the full fit
MIN_TRAIN_GAMES = 10

nd = NormalDist()


def main():
    tp = {r["player_id"]: r for r in csv.DictReader((DATA / "results_players_train.csv").open())}
    dy_ids = {frozenset((d["p1_id"], d["p2_id"])): int(d["idx"])
              for d in csv.DictReader((DATA / "model_dyads_train.csv").open())}
    chem = [float(r["chemistry_mean"])
            for r in csv.DictReader((DATA / "results_dyads_train.csv").open())]

    def value(u):
        r = tp.get(u)
        return (float(r["value_mean"]), int(r["games"])) if r else (0.0, 0)

    test = [g for g in csv.DictReader((DATA / "games.csv").open())
            if g["is_forfeit"] == "False" and g["scoring_format"] == "sideout_11"
            and g["date"] >= SPLIT]

    n_eval = correct = 0
    brier = logloss = mae_model = mae_zero = 0.0
    buckets = {}
    for g in test:
        t1 = [g["t1_p1"], g["t1_p2"]]; t2 = [g["t2_p1"], g["t2_p2"]]
        vals = [value(u) for u in t1 + t2]
        if sum(1 for _, n in vals if n >= MIN_TRAIN_GAMES) < 4:
            continue
        c1 = chem[dy_ids[frozenset(t1)]] if frozenset(t1) in dy_ids else 0.0
        c2 = chem[dy_ids[frozenset(t2)]] if frozenset(t2) in dy_ids else 0.0
        mu = vals[0][0] + vals[1][0] - vals[2][0] - vals[3][0] + c1 - c2
        margin = int(g["margin"])
        won = margin > 0
        p1 = nd.cdf(mu / SD)
        n_eval += 1
        correct += ((mu > 0) == won)
        brier += (p1 - won) ** 2
        pc = min(max(p1, 1e-6), 1 - 1e-6)
        logloss += -(math.log(pc) if won else math.log(1 - pc))
        mae_model += abs(margin - mu); mae_zero += abs(margin)
        hi = p1 if p1 >= 0.5 else 1 - p1
        key = round(math.floor(hi * 10) / 10, 1)
        buckets.setdefault(key, [0, 0])
        buckets[key][0] += ((mu > 0) == won); buckets[key][1] += 1

    out = {
        "split": SPLIT, "test_games_evaluable": n_eval,
        "accuracy": correct / n_eval, "brier": brier / n_eval,
        "log_loss": logloss / n_eval,
        "mae_model": mae_model / n_eval, "mae_zero_baseline": mae_zero / n_eval,
        "calibration": {str(k): {"actual": v[0] / v[1], "n": v[1]}
                        for k, v in sorted(buckets.items())},
    }
    (ROOT / "model" / "holdout_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
