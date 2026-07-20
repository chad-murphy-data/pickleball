"""DreamBreaker outcome model — is a DB predictable from roster singles skill?

    python model/db_model.py              # fit -> model/db_model_summary.json
    python model/db_model.py --self-test  # verify the fitter on synthetic data

Question: does a team's mean roster SINGLES value predict who wins the MLP
DreamBreaker (the 1v1 singles tiebreaker) better than chance, and better than
the doubles-rating proxy?

Method, deliberately conservative:
  * Unit of analysis is ONE DreamBreaker, n ~= 100 — NOT per-rally. Rallies
    within a DB are correlated, so a rally-level binomial gives a CI that is
    far too narrow (pseudoreplication). This was the main flaw in the earlier
    hardcoded k=0.42 claim.
  * For each completed DB, each side's roster value = mean SINGLES value of its
    players, imputing anyone who never plays singles from the singles~doubles
    regression (fit here, not hardcoded). gap = side1 - side2.
  * Fit one-parameter logistic  P(team1 wins) = sigmoid(k * gap)  by MLE.
  * Nonparametric bootstrap OVER DBs for the CI (the honest unit).
  * Compare vs the same fit on DOUBLES values (the proxy) by mean nll.

Requires roster1/roster2 columns in data/dreambreakers.csv (emitted by
scraper/parse.py). DB files parsed before that column existed cannot be fit —
re-run scraper/parse.py against the raw cache first.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BOOTSTRAP = 2000


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_k(rows):
    """MLE of k in P(y=1) = sigmoid(k*gap). rows = [(gap, y)]. 1-D Newton."""
    k = 0.0
    for _ in range(100):
        g = h = 0.0
        for gap, y in rows:
            p = sigmoid(k * gap)
            g += gap * (y - p)
            h -= gap * gap * p * (1.0 - p)
        if abs(h) < 1e-12:
            break
        step = g / h
        k -= step
        if abs(step) < 1e-9:
            break
    return k


def mean_nll(rows, k):
    s = 0.0
    for gap, y in rows:
        p = min(max(sigmoid(k * gap), 1e-12), 1 - 1e-12)
        s -= math.log(p if y else 1 - p)
    return s / len(rows)


def bootstrap_ci(rows, b=BOOTSTRAP, seed=20260720):
    """Percentile CI for k, resampling DBs with replacement. Uses a local
    seeded RNG so the result is reproducible without touching global state."""
    n = len(rows)
    rng = random.Random(seed)
    ks = []
    for _ in range(b):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        ks.append(fit_k(sample))
    ks.sort()
    return ks[int(0.025 * b)], ks[int(0.975 * b)]


def _ols(xy):
    n = len(xy)
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    sxx = sum((x - mx) ** 2 for x, _ in xy) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in xy) / sxx
    return slope, my - slope * mx


def load_values():
    singles = {}
    for r in csv.DictReader((DATA / "singles_players.csv").open()):
        if int(r["singles_games"]) >= 20:
            singles[r["player_id"]] = float(r["singles_value"])
    doubles = {r["player_id"]: float(r["value_now_mean"])
               for r in csv.DictReader((DATA / "v2_players.csv").open())}
    # singles ~ doubles imputation, fit here (not hardcoded) for players w/ both
    both = [(doubles[p], singles[p]) for p in singles if p in doubles]
    slope, intercept = _ols(both)
    return singles, doubles, (slope, intercept)


def roster_value(uuids, singles, doubles, impute):
    slope, intercept = impute
    vals = []
    for u in uuids:
        if u in singles:
            vals.append(singles[u])
        elif u in doubles:
            vals.append(intercept + slope * doubles[u])   # impute from doubles
        # else: unknown player, skip
    return sum(vals) / len(vals) if vals else None


def build_rows(use_doubles=False):
    dbf = DATA / "dreambreakers.csv"
    reader = list(csv.DictReader(dbf.open()))
    if not reader or "roster1" not in reader[0]:
        raise SystemExit(
            "dreambreakers.csv has no roster1/roster2 columns — re-run "
            "scraper/parse.py against the raw cache before fitting the DB model.")
    singles, doubles, impute = load_values()
    src = doubles if use_doubles else singles
    other = singles if use_doubles else doubles
    rows = []
    for r in reader:
        try:
            s1, s2 = int(r["t1_score"]), int(r["t2_score"])
        except (ValueError, KeyError):
            continue
        if s1 == s2:
            continue
        r1 = [u for u in r["roster1"].split("|") if u]
        r2 = [u for u in r["roster2"].split("|") if u]
        v1 = roster_value(r1, src, other, impute)
        v2 = roster_value(r2, src, other, impute)
        if v1 is None or v2 is None:
            continue
        rows.append((v1 - v2, 1 if s1 > s2 else 0))
    return rows


def summarize(rows, label):
    k = fit_k(rows)
    lo, hi = bootstrap_ci(rows)
    picks = sum(1 for gap, y in rows if (gap > 0) == (y == 1))
    return {
        "predictor": label, "n": len(rows), "k": round(k, 4),
        "k_ci95": [round(lo, 4), round(hi, 4)],
        "mean_nll": round(mean_nll(rows, k), 4),
        "stronger_roster_win_rate": round(picks / len(rows), 4),
        "excludes_zero": lo > 0 or hi < 0,
    }


def self_test():
    """Verify the fitter + bootstrap recover a known k on synthetic DBs."""
    k_true, n = 0.45, 1500
    rng = random.Random(42)
    rows = [(g, 1 if rng.random() < sigmoid(k_true * g) else 0)
            for g in (rng.uniform(-1.0, 1.0) for _ in range(n))]
    k = fit_k(rows)
    lo, hi = bootstrap_ci(rows, b=500)
    print(f"self-test: k_true={k_true}  k_hat={k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    assert lo <= k_true <= hi, "bootstrap CI failed to cover the true k"
    assert abs(k - k_true) < 0.15, "point estimate off"
    print("self-test PASSED — fitter and bootstrap are sound.")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    singles_rows = build_rows(use_doubles=False)
    doubles_rows = build_rows(use_doubles=True)
    if len(singles_rows) < 20 or len(doubles_rows) < 20:
        print(f"only {len(singles_rows)} fittable DreamBreakers — too few to "
              f"fit; leaving db_model_summary.json unchanged.")
        return
    _, _, (slope, intercept) = load_values()
    out = {
        "n_dreambreakers": len(singles_rows),
        "unit_of_analysis": "one DreamBreaker (not per-rally)",
        "parameterization": "winner-level logit: P(team1 wins DB) = sigmoid(k * gap)",
        "singles_impute": {"intercept": round(intercept, 4), "slope": round(slope, 4)},
        "singles": summarize(singles_rows, "mean roster singles value"),
        "doubles": summarize(doubles_rows, "mean roster doubles value"),
    }
    out["singles_beats_doubles_nll"] = round(
        (out["doubles"]["mean_nll"] - out["singles"]["mean_nll"]) * len(singles_rows), 2)
    (ROOT / "model" / "db_model_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    s = out["singles"]
    verdict = ("predictive (CI excludes 0)" if s["excludes_zero"]
               else "NOT significant at the DB level (CI includes 0)")
    print(f"\nSingles-gap DB model: k={s['k']} CI{s['k_ci95']} on n={s['n']} — {verdict}")


if __name__ == "__main__":
    main()
