"""Holdout gate for the Bayesian singles suite (pre-June train, post-June eval).

    python model/singles_holdout.py           # cutoff 2026-06-01

Compares, on data the fits never saw:

  STATUS QUO — the pure zero-prior fit plus the production imputation rule
    (>= 10 train games: pure value; else -0.07 + 1.14 * d_train), i.e. what
    make_forecast shipped before the suite existed;
  PURE      — the zero-prior fit alone (unseen players at 0), for reference;
  SUITE     — doubles-informed prior + DB-rally evidence + fixed-point
    anchor (model/fit_singles.py), trained with the same date cutoff AND
    train-cutoff doubles values (data/v2_players_train.csv, the frozen
    SRM2_DATE_BEFORE=2026-06-01 v2 fit) — no post-cutoff information enters
    any arm.

Eval:
  (a) post-cutoff PPA singles games — winner accuracy / Brier / log-loss of
      the race-DP game win prob (T=11; 15 when the winning score is 15+).
      Primary set = both players carry TRAIN evidence (games or DB
      rallies); the cold remainder is scored separately — that is where
      the priors do the work.
  (b) post-cutoff validated DreamBreakers — team-level nll / Brier / acc at
      the production K_DB_SINGLES = 0.42 race-to-21 on the mean-participant
      gap, plus rally-level nll at the measured per-rally slope 0.49.

Gate: the suite must not lose (a) on the primary set and must win (b).
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))
import fit_singles as fs                                    # noqa: E402
from sitelib.race import (game_win_prob_uncertain, race_dist,
                          sigmoid)                          # noqa: E402

CUTOFF = os.environ.get("SINGLES_CUTOFF", "2026-06-01")
K_TEAM = 0.42          # production team-level DB constant (make_forecast)
K_RALLY = 0.49         # measured per-rally slope (db_impute 2026-08-16)
SQ_IMPUTE = (-0.07, 1.14)   # production status-quo imputation


def metrics(rows):
    """rows: (p, y).  Returns acc, brier, nll."""
    n = len(rows)
    acc = sum((p >= 0.5) == y for p, y in rows) / n
    brier = sum((p - y) ** 2 for p, y in rows) / n
    nll = -sum(math.log(max(p if y else 1 - p, 1e-12)) for p, y in rows) / n
    return acc, brier, nll


def main():
    print(f"train < {CUTOFF}; doubles prior from v2_players_train.csv")
    F = fs.fit_all(before=CUTOFF,
                   doubles_path=DATA / "v2_players_train.csv",
                   verbose=False)
    idx, counts, db_stats = F["idx"], F["counts"], F["db_stats"]
    doubles, gender = F["doubles"], F["gender"]
    pure = {p: F["v_pure"][i] for p, i in idx.items()}
    suite = {p: F["v_suite"][i] for p, i in idx.items()}
    print(f"trained: {F['n_games']} games, {F['n_rallies_used']} DB rallies, "
          f"{len(idx)} players")

    def v_pure(u):
        return pure.get(u, 0.0)

    def v_sq(u):
        if counts.get(u, 0) >= fs.RANKED_GAMES:
            return pure[u]
        if u in doubles:
            a, b = SQ_IMPUTE
            return a + b * doubles[u][0]
        return pure.get(u, 0.0)

    def v_suite(u):
        return suite.get(u, 0.0)

    arms = (("status quo", v_sq), ("pure", v_pure), ("suite", v_suite))

    # ---- (a) post-cutoff singles games -------------------------------------
    seen = {p for p in idx
            if counts.get(p, 0) > 0 or db_stats.get(p, (0, 0))[0] > 0}
    games_warm, games_cold = [], []
    for r in csv.DictReader((DATA / "singles_games.csv").open()):
        if r["is_forfeit"] != "False" or r["date"] < CUTOFF:
            continue
        s1, s2 = int(r["s1"]), int(r["s2"])
        T = 15 if max(s1, s2) >= 15 else 11
        rec = (r["p1"], r["p2"], T, s1 > s2)
        (games_warm if r["p1"] in seen and r["p2"] in seen
         else games_cold).append(rec)
    print(f"\n(a) singles games >= {CUTOFF}: {len(games_warm)} warm "
          f"(both players have train evidence) + {len(games_cold)} cold")
    sd = {p: F["sd_suite"][i] for p, i in idx.items()}
    for label, games in (("warm", games_warm), ("cold", games_cold)):
        if not games:
            continue
        print(f"  {label}:")
        for name, vf in arms:
            rows = [(race_dist(round(sigmoid(vf(p1) - vf(p2)), 4),
                               T)["p_win"], y)
                    for p1, p2, T, y in games]
            acc, brier, nll = metrics(rows)
            print(f"    {name:10s} acc {acc:.3f}  brier {brier:.4f}  "
                  f"nll {nll:.4f}")
        # the suite as consumers actually price it: integrated over value sd
        rows = [(game_win_prob_uncertain(
                    v_suite(p1) - v_suite(p2),
                    math.sqrt(sd.get(p1, 0.5) ** 2 + sd.get(p2, 0.5) ** 2),
                    T), y)
                for p1, p2, T, y in games]
        acc, brier, nll = metrics(rows)
        print(f"    suite+unc  acc {acc:.3f}  brier {brier:.4f}  "
              f"nll {nll:.4f}")

    # ---- (b) post-cutoff DreamBreakers -------------------------------------
    dates, outcome = {}, {}
    for r in csv.DictReader((DATA / "dreambreakers.csv").open()):
        if len(r.get("match_id") or "") == 36:
            try:
                t1, t2 = int(r["t1_score"]), int(r["t2_score"])
            except ValueError:
                continue
            dates[r["match_id"].lower()] = r["date"]
            outcome[r["match_id"].lower()] = t1 > t2
    team_rosters = defaultdict(lambda: (set(), set()))
    rallies = defaultdict(list)
    for r in csv.DictReader((DATA / "db_rallies.csv").open()):
        m = r["match_id"].lower()
        if dates.get(m, "") < CUTOFF:
            continue
        u1, u2 = r["player_team1"].lower(), r["player_team2"].lower()
        team_rosters[m][0].add(u1)
        team_rosters[m][1].add(u2)
        if gender.get(u1) and gender.get(u1) == gender.get(u2):
            rallies[m].append((u1, u2, r["team1_won"] == "1"))
    dbs = sorted(team_rosters)
    print(f"\n(b) DreamBreakers >= {CUTOFF}: {len(dbs)} validated "
          f"({sum(len(v) for v in rallies.values())} same-gender rallies)")
    for name, vf in arms:
        team_rows, rally_rows = [], []
        for m in dbs:
            r1, r2 = team_rosters[m]
            gap = (sum(vf(u) for u in r1) / len(r1)
                   - sum(vf(u) for u in r2) / len(r2))
            p = race_dist(round(sigmoid(K_TEAM * gap), 4), 21)["p_win"]
            team_rows.append((p, outcome[m]))
            for u1, u2, y in rallies[m]:
                rally_rows.append((sigmoid(K_RALLY * (vf(u1) - vf(u2))), y))
        acc, brier, nll = metrics(team_rows)
        racc, rbrier, rnll = metrics(rally_rows)
        print(f"    {name:10s} team acc {acc:.3f}  brier {brier:.4f}  "
              f"nll {nll:.4f}   | rally nll {rnll:.4f}")

    # sanity: how far the prior moves settled players (train fit)
    diffs = sorted(abs(suite[p] - pure[p]) for p in idx
                   if counts.get(p, 0) >= 60)
    print(f"\nsanity: median |suite-pure| among >=60g train players "
          f"{diffs[len(diffs)//2]:.3f} (p90 {diffs[int(len(diffs)*.9)]:.3f})")


if __name__ == "__main__":
    main()
