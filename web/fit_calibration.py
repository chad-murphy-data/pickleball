"""Fit the display-calibration layer for site win probabilities.

    python web/fit_calibration.py        # writes web/calibration.json

Two problems, one map:

1. UNDERCONFIDENCE — the published holdout check (v1 buckets; v2 Brier)
   shows favorites winning more often than the stated probability in the
   65-80% band.  We measure it out-of-sample: price every post-2026-06-01
   game with the *frozen* _train fit (data/v2_players_train.csv, exactly the
   fit behind the registered 77.4% number), then fit a logit-linear
   recalibration  l' = a + b*logit(p)  by maximum likelihood on those games.

2. NO ZEROS — a race-DP probability can be astronomically small, but there
   is always a nonzero chance a team wins (injury, meltdown, weather, a
   twin swap nobody recorded).  We estimate that tail empirically: across
   all ~36k games, how often did teams priced above 99% actually lose?
   That rate becomes a mixture floor:  p_cal = (1-eps)*sigma(l') + eps/2,
   so no probability the site ever shows is 0% or 100%.

The fitted constants are written to web/calibration.json and baked into
both the Python page builders and the simulator JS.  Frozen predictions in
model/receipts.json are never rewritten — the ledger grades what was
committed, miscalibration and all.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib import data as D
from sitelib.race import race_dist, sigmoid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "calibration.json"

SPLIT = "2026-06-01"
GAMMA_TRAIN = -0.166        # v2 _train fit posterior mean (fit_summary_train)
MIN_TRAIN_GAMES = 10        # mirror model/v2_holdout.py


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def game_win_prob_gauss(mu, sd, T):
    """Race win prob integrated over eta ~ N(mu, sd) (41-node grid)."""
    if sd <= 0:
        return race_dist(round(sigmoid(mu), 4), T)["p_win"]
    tot = ws = 0.0
    for i in range(41):
        z = -4.0 + i * 0.2
        w = math.exp(-0.5 * z * z)
        tot += w * race_dist(round(sigmoid(mu + z * sd), 4), T)["p_win"]
        ws += w
    return tot / ws


def price_holdout():
    """(p_raw, won) for every post-split game the frozen train fit can price."""
    train = {r["player_id"]: (float(r["value_now_mean"]), float(r["value_now_sd"]),
                              int(r["games"]))
             for r in csv.DictReader((DATA / "v2_players_train.csv").open())}
    names = {r["player_id"]: r["full_name"]
             for r in csv.DictReader((DATA / "players.csv").open())}
    chem = {}
    for r in csv.DictReader((DATA / "v2_dyads_train.csv").open()):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = float(r["chem_logit_mean"])

    obs = []
    for g in csv.DictReader((DATA / "games.csv").open()):
        if (g["is_forfeit"] != "False" or g["is_dreambreaker"] != "False"
                or g["scoring_format"] not in ("sideout_11", "sideout_15")
                or g["date"] < SPLIT):
            continue
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        if any(u not in train or train[u][2] < MIN_TRAIN_GAMES for u in us):
            continue
        v = [train[u][0] for u in us]
        sd = math.sqrt(sum(train[u][1] ** 2 for u in us))
        c1 = chem.get(frozenset((names.get(us[0], ""), names.get(us[1], ""))), 0.0)
        c2 = chem.get(frozenset((names.get(us[2], ""), names.get(us[3], ""))), 0.0)
        mu = ((v[0] + v[1] + GAMMA_TRAIN * abs(v[0] - v[1]))
              - (v[2] + v[3] + GAMMA_TRAIN * abs(v[2] - v[3])) + c1 - c2)
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        obs.append((game_win_prob_gauss(mu, sd, T), int(g["margin"]) > 0))
    return obs


def estimate_tail():
    """Loss rate of teams priced >=99% across ALL games (full-fit monthly
    values; descriptive, but the tail is exactly what we want to measure).
    Laplace-smoothed so a clean sheet still yields a nonzero floor."""
    players = D.load_players()
    games = D.load_games()
    mv = D.month_values(players)
    n = k = 0
    for g in games:
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        share = D.expected_share(players, mv, g)
        if share is None:
            continue
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        pw = race_dist(round(share, 4), T)["p_win"]
        won = int(g["margin"]) > 0
        p_fav, fav_won = (pw, won) if pw >= 0.5 else (1 - pw, not won)
        if p_fav >= 0.99:
            n += 1
            k += (not fav_won)
    rate = (k + 1) / (n + 2)          # Laplace
    return {"n_extreme": n, "losses": k, "loss_rate": rate, "eps": 2 * rate}


def fit_ab(obs, eps):
    """Grid MLE for p_cal = (1-eps)*sigmoid(a + b*logit(p_raw)) + eps/2."""
    lo = [(logit(p), y) for p, y in obs]

    def nll(a, b):
        s = 0.0
        for l, y in lo:
            p = (1 - eps) * sigmoid(a + b * l) + eps / 2
            s -= math.log(p if y else 1 - p)
        return s / len(lo)

    best = (0.0, 1.0, nll(0.0, 1.0))
    for bi in range(70, 226, 5):          # b in 0.70..2.25
        b = bi / 100
        for ai in range(-40, 41, 4):      # a in -0.40..0.40
            a = ai / 100
            v = nll(a, b)
            if v < best[2]:
                best = (a, b, v)
    # local refine
    a0, b0, _ = best
    for bi in range(int(b0 * 100) - 6, int(b0 * 100) + 7):
        for ai in range(int(a0 * 100) - 5, int(a0 * 100) + 6):
            v = nll(ai / 100, bi / 100)
            if v < best[2]:
                best = (ai / 100, bi / 100, v)
    return best, nll(0.0, 1.0)


def buckets(obs, a, b, eps):
    tab = {}
    for p, y in obs:
        pc = (1 - eps) * sigmoid(a + b * logit(p)) + eps / 2
        pf, yf = (pc, y) if pc >= 0.5 else (1 - pc, not y)
        k = f"{int(pf * 10) / 10:.1f}"
        d = tab.setdefault(k, [0, 0])
        d[0] += 1; d[1] += yf
    return {k: {"n": n, "actual": round(w / n, 4)}
            for k, (n, w) in sorted(tab.items())}


def main():
    obs = price_holdout()
    acc = sum((p > 0.5) == y for p, y in obs) / len(obs)
    brier_raw = sum((p - y) ** 2 for p, y in obs) / len(obs)
    print(f"holdout games priced: {len(obs)}  acc {acc:.4f}  raw brier {brier_raw:.4f}")

    tail = estimate_tail()
    print(f"tail: {tail['losses']}/{tail['n_extreme']} favorites >=99% lost "
          f"-> eps = {tail['eps']:.4f}")

    (a, b, nll_cal), nll_raw = fit_ab(obs, tail["eps"])
    brier_cal = sum(((1 - tail["eps"]) * sigmoid(a + b * logit(p)) + tail["eps"] / 2 - y) ** 2
                    for p, y in obs) / len(obs)
    print(f"fit: a={a:+.2f} b={b:.2f}  log-loss {nll_raw:.4f} -> {nll_cal:.4f}  "
          f"brier {brier_raw:.4f} -> {brier_cal:.4f}")

    out = {
        "note": "Display calibration for site win probabilities: "
                "p_cal = (1-eps)*sigmoid(a + b*logit(p_raw)) + eps/2. "
                "Fit out-of-sample on post-2026-06-01 games priced by the frozen "
                "_train fit; eps from the observed loss rate of >=99% favorites "
                "across all games (Laplace-smoothed). Regenerate with "
                "web/fit_calibration.py after each refit.",
        "a": a, "b": b, "eps": round(tail["eps"], 5),
        "fit_on": {"split": SPLIT, "n_games": len(obs),
                   "accuracy": round(acc, 4),
                   "log_loss_raw": round(nll_raw, 4),
                   "log_loss_cal": round(nll_cal, 4),
                   "brier_raw": round(brier_raw, 4),
                   "brier_cal": round(brier_cal, 4)},
        "tail": {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in tail.items()},
        "buckets_calibrated": buckets(obs, a, b, tail["eps"]),
        "buckets_raw": buckets(obs, 0.0, 1.0, 0.0),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
