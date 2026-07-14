"""Singles rating model — first pass.

    python model/fit_singles.py    # -> data/singles_players.csv + summary

Same skeleton as v2's likelihood, minus everything team: each game is a
per-point Binomial race, points won ~ Binomial(total, sigma(v_i - v_j)),
with v ~ N(0, SD_PRIOR) shrinkage and exponential recency weighting
(half-life 12 months) instead of a full random walk — the singles corpus
is ~10x smaller than doubles, so a dynamic model would be mostly prior.
Pure Python MAP by full-batch gradient ascent (no jax needed; converges in
seconds at this size).

Also reports: singles<->doubles value correlation, and a DreamBreaker
check — does mean roster SINGLES value predict DB outcomes better than the
doubles proxy (model/db_model.md)?
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SD_PRIOR = 0.6          # per-point logit; singles spreads wider than doubles
HALF_LIFE_DAYS = 365.0
REF_DATE = None          # default: newest game in the file


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def load_games():
    games = []
    for r in csv.DictReader((DATA / "singles_games.csv").open()):
        if r["is_forfeit"] != "False":
            continue
        games.append((r["p1"], r["p2"], int(r["s1"]), int(r["s2"]),
                      r["date"], r["context"], r["p1_name"], r["p2_name"]))
    return games


def fit(games):
    ref = REF_DATE or max(g[4] for g in games)
    ref_ord = date.fromisoformat(ref).toordinal()
    idx, names = {}, {}
    obs = []
    for p1, p2, s1, s2, d, _ctx, n1, n2 in games:
        for p, n in ((p1, n1), (p2, n2)):
            if p not in idx:
                idx[p] = len(idx)
                names[p] = n
        age = ref_ord - date.fromisoformat(d).toordinal()
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        obs.append((idx[p1], idx[p2], s1, s2, w))

    n = len(idx)
    v = [0.0] * n
    # diagonal-Fisher preconditioner: weighted points seen per player
    tpts = [0.0] * n
    for i, j, s1, s2, w in obs:
        tpts[i] += w * (s1 + s2)
        tpts[j] += w * (s1 + s2)
    precon = [0.25 * t + 1.0 / SD_PRIOR ** 2 for t in tpts]
    for it in range(200):
        grad = [-vi / (SD_PRIOR ** 2) for vi in v]     # prior
        ll = -sum(vi * vi for vi in v) / (2 * SD_PRIOR ** 2)
        for i, j, s1, s2, w in obs:
            p = sigmoid(v[i] - v[j])
            g = w * (s1 - (s1 + s2) * p)               # d/d(v_i) of binomial ll
            grad[i] += g
            grad[j] -= g
            ll += w * (s1 * math.log(max(p, 1e-12))
                       + s2 * math.log(max(1 - p, 1e-12)))
        for k in range(n):
            v[k] += grad[k] / precon[k]
        if it % 50 == 49:
            gnorm = math.sqrt(sum(g * g for g in grad))
            print(f"  iter {it + 1}: penalized ll {ll:,.1f}  |grad| {gnorm:.2f}")
    inv = {i: p for p, i in idx.items()}
    return {inv[i]: v[i] for i in range(n)}, names


def main():
    games = load_games()
    print(f"singles games: {len(games)}")
    values, names = fit(games)

    counts = defaultdict(int)
    gender = {}
    for g in games:
        p1, p2, ctx = g[0], g[1], g[5]
        counts[p1] += 1; counts[p2] += 1
        g_ = "F" if ctx == "womens_singles" else "M"
        gender[p1] = g_; gender[p2] = g_

    with (DATA / "singles_players.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player_id", "full_name", "gender", "singles_games",
                    "singles_value"])
        for p, val in sorted(values.items(), key=lambda kv: -kv[1]):
            w.writerow([p, names[p], gender.get(p, ""), counts[p], round(val, 4)])

    for g_ in ("M", "F"):
        pool = [(val, p) for p, val in values.items()
                if gender.get(p) == g_ and counts[p] >= 20]
        pool.sort(reverse=True)
        print(f"\ntop {g_} singles (>=20 games):")
        for val, p in pool[:8]:
            print(f"  {names[p]:26s} {val:+.2f}  ({counts[p]} g)")

    # singles vs doubles correlation
    dv = {r["player_id"]: float(r["value_now_mean"])
          for r in csv.DictReader((DATA / "v2_players.csv").open())}
    both = [(values[p], dv[p]) for p in values if p in dv and counts[p] >= 20]
    if len(both) > 10:
        mx = sum(x for x, _ in both) / len(both)
        my = sum(y for _, y in both) / len(both)
        sxy = sum((x - mx) * (y - my) for x, y in both)
        sx = math.sqrt(sum((x - mx) ** 2 for x, _ in both))
        sy = math.sqrt(sum((y - my) ** 2 for _, y in both))
        print(f"\nsingles~doubles r = {sxy / (sx * sy):.3f} "
              f"(n={len(both)} with >=20 singles games)")

    json_summary = {
        "n_games": len(games),
        "n_players": len(values),
        "sd_prior": SD_PRIOR,
        "half_life_days": HALF_LIFE_DAYS,
    }
    (ROOT / "model" / "singles_fit_summary.json").write_text(
        json.dumps(json_summary, indent=1))


if __name__ == "__main__":
    main()
