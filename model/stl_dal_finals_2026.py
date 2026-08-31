"""STL Shock vs Dallas Flash, 2026 MLP Finals (best-of-3 series): baseline vs
Dallas-hot-streak pricing.  Offline, stdlib only.

    python model/stl_dal_finals_2026.py

Uses the exact production stack (web/sitelib/race.py race DP + weakest link +
display calibration; singles-suite DreamBreaker at K_DB_SINGLES=0.42), the
same projected best lineups as data/forecasts.json, and current v2 values
(last refit 2026-08-03 — deliberately BEFORE the playoffs, so the entire
playoff run is out-of-sample for the ratings).

Hot-streak measure: for each game a team played in the window, the team-level
logit surplus = logit(observed point share) - predicted eta from current v2
values (shares clamped to [1/24, 23/24] so 11-1 stays finite).  A player's
bump = half their mean team surplus (two players share the court).  This is
FACE VALUE, no shrinkage — the point of the exercise is "what if the streak
is fully real", and the writeup (stl_dal_finals_2026.md) carries the noise
caveats.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import (calibrate, race_dist, set_calibration, sigmoid,
                          team_eta)                        # noqa: E402

CAL = json.loads((ROOT / "web" / "calibration.json").read_text())
set_calibration(CAL["a"], CAL["b"], CAL["eps"])
K_DB_SINGLES = 0.42

STL = {"WD": ("Anna Bright", "Kate Fahey"),
       "MD": ("Hayden Patriquin", "Gabriel Tardio"),
       "MXD1": ("Anna Bright", "Hayden Patriquin"),
       "MXD2": ("Kate Fahey", "Gabriel Tardio")}
DAL = {"WD": ("Danni-Elle Townsend", "Alix Truong"),
       "MD": ("JW Johnson", "Augustus Ge"),
       "MXD1": ("Danni-Elle Townsend", "JW Johnson"),
       "MXD2": ("Alix Truong", "Augustus Ge")}
# What Dallas actually ran throughout the playoffs (mixed differs from the
# best-lineup projection; Buckner played every mixed with JW).
DAL_ACTUAL = {"WD": ("Danni-Elle Townsend", "Alix Truong"),
              "MD": ("JW Johnson", "Augustus Ge"),
              "MXD1": ("Danni-Elle Townsend", "Augustus Ge"),
              "MXD2": ("Brooke Buckner", "JW Johnson")}


def load():
    vals = {}
    for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open()):
        vals[r["player_id"]] = [r["full_name"], float(r["value_now_mean"]),
                                r["gender"]]
    singles = {r["player_id"]: float(r["singles_value"])
               for r in csv.DictReader(
                   (ROOT / "data" / "singles_players.csv").open())}
    uuid = {v[0]: k for k, v in vals.items()}
    return vals, singles, uuid


def streak_surpluses(vals, uuid, names, start):
    """player -> list of team logit surpluses in MLP games since `start`."""
    core = {uuid[n] for n in names}
    per = {u: [] for u in core}
    n_games = wins = 0
    for r in csv.DictReader((ROOT / "data" / "games.csv").open()):
        if r["tour"] != "MLP" or r["date"] < start:
            continue
        if r["is_dreambreaker"] == "True" or r["is_forfeit"] == "True":
            continue
        s1 = (r["t1_p1"], r["t1_p2"])
        s2 = (r["t2_p1"], r["t2_p2"])
        d1, d2 = len(set(s1) & core), len(set(s2) & core)
        if not d1 and not d2:
            continue
        assert not (d1 and d2), "core players on both sides"
        d, o = (s1, s2) if d1 else (s2, s1)
        ds, os_ = ((int(r["t1_score"]), int(r["t2_score"])) if d1
                   else (int(r["t2_score"]), int(r["t1_score"])))
        if any(u not in vals for u in d + o):
            continue
        va, vb = [vals[u][1] for u in d], [vals[u][1] for u in o]
        eta = team_eta(va[0], va[1], vb[0], vb[1])
        obs = min(max(ds / (ds + os_), 1 / 24), 1 - 1 / 24)
        surplus = math.log(obs / (1 - obs)) - eta
        n_games += 1
        wins += ds > os_
        for u in d:
            if u in core:
                per[u].append(surplus)
    return per, n_games, wins


def bumps_from(per, vals):
    out = {}
    for u, g in per.items():
        m = sum(g) / len(g)
        se = ((sum((x - m) ** 2 for x in g) / (len(g) - 1) / len(g)) ** 0.5
              if len(g) > 1 else float("nan"))
        out[u] = (m / 2, m, se, len(g))
    return out


def price(vals, singles, uuid, dal):
    games = {}
    for slot in ("WD", "MD", "MXD1", "MXD2"):
        a = [vals[uuid[n]][1] for n in STL[slot]]
        b = [vals[uuid[n]][1] for n in dal[slot]]
        dist = race_dist(round(sigmoid(team_eta(a[0], a[1], b[0], b[1])), 4), 11)
        scores = ([(11, y, pr) for _, y, pr in dist["win_scores"]]
                  + [(x, 11, pr) for x, _, pr in dist["lose_scores"]])
        modal = max(scores, key=lambda s: s[2])
        games[slot] = {"p": calibrate(dist["p_win"]),
                       "modal": f"{modal[0]}-{modal[1]}",
                       "margin": dist["exp_margin"]}
    r1 = {uuid[n] for pair in STL.values() for n in pair}
    r2 = {uuid[n] for pair in dal.values() for n in pair}
    gap = (sum(singles[u] for u in r1) / len(r1)
           - sum(singles[u] for u in r2) / len(r2))
    p_db = race_dist(round(sigmoid(K_DB_SINGLES * gap), 4), 21)["p_win"]
    eps = CAL["eps"]
    p_db = min(max(p_db, eps / 2), 1 - eps / 2)
    dist = [1.0]
    for p in [games[s]["p"] for s in ("WD", "MD", "MXD1", "MXD2")]:
        nxt = [0.0] * (len(dist) + 1)
        for w, pr in enumerate(dist):
            nxt[w + 1] += pr * p
            nxt[w] += pr * (1 - p)
        dist = nxt
    p40, p31, p22, p13, p04 = dist[4], dist[3], dist[2], dist[1], dist[0]
    p_win = min(max(p40 + p31 + p22 * p_db, eps / 2), 1 - eps / 2)
    series = p_win * p_win * (3 - 2 * p_win)     # best-of-3
    return games, {"p_win": p_win, "p_40": p40, "p_31": p31, "p_22": p22,
                   "p_13": p13, "p_04": p04, "p_db_win": p_db,
                   "p_series": series}


def show(tag, games, tree):
    print(f"--- {tag} ---")
    for s in ("WD", "MD", "MXD1", "MXD2"):
        g = games[s]
        print(f"  {s:5s} STL {g['p'] * 100:5.1f}%  modal {g['modal']:>5s}"
              f"  margin {g['margin']:+.2f}")
    print(f"  matchup: STL {tree['p_win'] * 100:.1f}%"
          f"  (4-0 {tree['p_40'] * 100:.1f} / 3-1 {tree['p_31'] * 100:.1f}"
          f" / 2-2 {tree['p_22'] * 100:.1f} [STL DB {tree['p_db_win'] * 100:.1f}%]"
          f" / 1-3 {tree['p_13'] * 100:.1f} / 0-4 {tree['p_04'] * 100:.1f})")
    print(f"  best-of-3: STL {tree['p_series'] * 100:.2f}%"
          f"  (Dallas {(1 - tree['p_series']) * 100:.2f}%)\n")


def bumped(vals, bumps):
    v2 = {k: list(v) for k, v in vals.items()}
    for u, (b, *_) in bumps.items():
        v2[u][1] += b
    return v2


def main():
    vals, singles, uuid = load()
    dal4 = [n for pair in DAL.values() for n in pair]
    stl4 = [n for pair in STL.values() for n in pair]

    show("BASELINE — current v2 (fit 2026-08-03), projected best lineups",
         *price(vals, singles, uuid, DAL))

    for tag, start in (("playoff run, since 2026-08-07", "2026-08-07"),
                       ("wider window, since 2026-07-23", "2026-07-23")):
        for team, names in (("Dallas", dal4), ("STL", stl4)):
            per, n, w = streak_surpluses(vals, uuid, set(names), start)
            b = bumps_from(per, vals)
            print(f"{team} {tag}: {w}-{n - w} in games")
            for u in sorted(b, key=lambda u: vals[u][0]):
                bump, m, se, ng = b[u]
                print(f"  {vals[u][0]:22s} n={ng:2d}  team surplus {m:+.3f}"
                      f" logit (se {se:.3f})  -> bump {bump:+.3f}")
            if team == "Dallas":
                dal_bumps = b
            else:
                stl_bumps = b
        print()
        show(f"DALLAS HOT ({tag}) — face value, STL unchanged",
             *price(bumped(vals, dal_bumps), singles, uuid, DAL))
        if start == "2026-08-07":
            both = dict(dal_bumps)
            both.update(stl_bumps)
            show("BOTH HOT (playoff run) — same face-value logic on all 8",
                 *price(bumped(vals, both), singles, uuid, both and DAL))

    # Dallas's actual playoff lineup (Townsend/Ge + Buckner/JW mixed)
    per, _, _ = streak_surpluses(vals, uuid,
                                 set(dal4) | {"Brooke Buckner"}, "2026-08-07")
    b = bumps_from(per, vals)
    bu = uuid["Brooke Buckner"]
    print(f"Brooke Buckner playoff surplus: n={b[bu][3]}"
          f" {b[bu][1]:+.3f} logit -> bump {b[bu][0]:+.3f}"
          f" (value {vals[bu][1]:.3f}, singles {singles[bu]:.2f})\n")
    show("LINEUP VARIANT — Dallas actual playoff mixed, baseline values",
         *price(vals, singles, uuid, DAL_ACTUAL))
    show("LINEUP VARIANT — Dallas actual playoff mixed, Dallas hot",
         *price(bumped(vals, b), singles, uuid, DAL_ACTUAL))


if __name__ == "__main__":
    main()
