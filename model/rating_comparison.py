"""Platform-rating vs model head-to-head prediction test.

pickleball.com embeds its in-house per-player rating (NOT DUPR — DUPR needs an
authenticated API) as an as-of-match snapshot in every MLP match and in the
enriched PPA records. That gives a no-lookahead external benchmark.

Test: on holdout games (>= 2026-06-01) where both predictors are available,
compare winner-pick accuracy and Brier:
  A) our SRM values, trained only on pre-June games (frozen);
  B) platform rating sums, as-of-match (updated all season - an info edge).

Run after holdout_eval.py prerequisites; writes model/rating_comparison.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import NormalDist, correlation

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT = "2026-06-01"
SD = 4.70
nd = NormalDist()


def main():
    per_match = json.loads((DATA / "per_match_ratings.json").read_text())
    tp = {r["player_id"]: r for r in csv.DictReader((DATA / "results_players_train.csv").open())}
    dy_ids = {frozenset((d["p1_id"], d["p2_id"])): int(d["idx"])
              for d in csv.DictReader((DATA / "model_dyads_train.csv").open())}
    chem = [float(r["chemistry_mean"])
            for r in csv.DictReader((DATA / "results_dyads_train.csv").open())]

    games = [g for g in csv.DictReader((DATA / "games.csv").open())
             if g["is_forfeit"] == "False" and g["scoring_format"] == "sideout_11"]

    def rating_diff(g):
        rm = per_match.get(g["match_id"])
        if not rm:
            return None
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        if any(u not in rm for u in us):
            return None
        return rm[us[0]] + rm[us[1]] - rm[us[2]] - rm[us[3]]

    def model_mu(g):
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        vals = []
        for u in us:
            r = tp.get(u)
            if not r or int(r["games"]) < 10:
                return None
            vals.append(float(r["value_mean"]))
        c1 = chem[dy_ids[frozenset(us[:2])]] if frozenset(us[:2]) in dy_ids else 0.0
        c2 = chem[dy_ids[frozenset(us[2:])]] if frozenset(us[2:]) in dy_ids else 0.0
        return vals[0] + vals[1] - vals[2] - vals[3] + c1 - c2

    # probit slope for rating diffs, fitted on pre-split rated games only
    pre = [(rating_diff(g), int(g["margin"]) > 0) for g in games if g["date"] < SPLIT]
    pre = [(d, w) for d, w in pre if d is not None]
    ks = np.arange(0.2, 8.0, 0.1)
    briers = [sum((nd.cdf(k * d) - w) ** 2 for d, w in pre) / len(pre) for k in ks]
    k = float(ks[int(np.argmin(briers))])

    n = mc = rc = agree = 0
    mb = rb = 0.0
    for g in (g for g in games if g["date"] >= SPLIT):
        d = rating_diff(g)
        mu = model_mu(g)
        if d is None or mu is None:
            continue
        won = int(g["margin"]) > 0
        n += 1
        mc += ((mu > 0) == won)
        rc += ((d > 0) == won)
        agree += ((mu > 0) == (d > 0))
        mb += (nd.cdf(mu / SD) - won) ** 2
        rb += (nd.cdf(k * d) - won) ** 2

    res = {r["player_id"]: r for r in csv.DictReader((DATA / "results_players.csv").open())}
    lat = {r["player_id"]: float(r["platform_rating_latest"])
           for r in csv.DictReader((DATA / "platform_ratings.csv").open())}
    corr = {}
    for gname in ("M", "F"):
        pairs = [(float(res[u]["value_mean"]), lat[u]) for u in res
                 if u in lat and res[u]["gender"] == gname and int(res[u]["games"]) >= 40]
        corr[gname] = {"n": len(pairs), "r": correlation(*zip(*pairs))}

    out = {
        "split": SPLIT, "n_games": n, "probit_slope": k,
        "n_pre_calibration_games": len(pre),
        "model": {"accuracy": mc / n, "brier": mb / n},
        "platform_rating": {"accuracy": rc / n, "brier": rb / n},
        "agreement": agree / n,
        "value_rating_correlation": corr,
    }
    (ROOT / "model" / "rating_comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
