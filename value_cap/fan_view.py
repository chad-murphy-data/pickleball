"""fan_view.py -- what a NEW OWNER knows about the players (no model values).

(File name kept from the first draft; the owners are not fans in the stands but
people who bought a team and follow the sport without analytics.)

The records-only owner for the value-cap auction (fan_owner_spec.md). Such an owner
knows results, not ratings: per-division doubles win% with game counts,
singles win% with game counts (so who plays singles at all), how often a
player actually took the court for their 2026 MLP team, and an ORDINAL
picture of who is better than whom within a gender. Nothing here is a
dollar, a per-point value, a tie probability or a cross-gender comparison.

The ordinal picture is drawn from the v2 posterior (data/v2_players.csv:
value_now_mean / value_now_sd): one joint draw per owner, ranked within
gender, and the owner is handed ONLY the order. So the fog is the model's
own uncertainty -- stable at the top (Waters / Bright / Jorja), blurry from
about #10 down -- scaled by --sd-mult (1 = the posterior; 2 = an owner who
processes the results less efficiently than the model). Same for singles
from data/singles_players.csv. `rank_probs` reports P(rank <= k) so the fog
is visible as a table, not a vibe.

    python value_cap/fan_view.py                 # print the owner's board
    python value_cap/fan_view.py --sd-mult 2
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
YEAR = "2026"
DIVS = ("womens", "mens", "mixed")


def _rows(name):
    with (DATA / name).open() as fh:
        yield from csv.DictReader(fh)


# ------------------------------------------------------------------ records
def doubles_records(year=YEAR):
    """pid -> {div: (wins, games, pts_for, pts_against)} for pro doubles that year
    (DreamBreakers and forfeits excluded), plus 'all'."""
    rec = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for g in _rows("games.csv"):
        if not g["date"].startswith(year) or g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        try:
            s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        except ValueError:
            continue
        div = g["context"] if g["context"] in DIVS else None
        for side, sf, sa in (("t1", s1, s2), ("t2", s2, s1)):
            for k in ("p1", "p2"):
                u = g[f"{side}_{k}"].lower()
                for d in (div, "all"):
                    if d is None:
                        continue
                    r = rec[u][d]
                    r[0] += sf > sa; r[1] += 1; r[2] += sf; r[3] += sa
    return rec


def singles_records(year=None):
    """pid -> (wins, games) in PPA singles; year=None = career."""
    rec = defaultdict(lambda: [0, 0])
    for g in _rows("singles_games.csv"):
        if year and not g["date"].startswith(year):
            continue
        if g["is_forfeit"] == "True":
            continue
        try:
            s1, s2 = int(g["s1"]), int(g["s2"])
        except ValueError:
            continue
        for u, sf, sa in ((g["p1"].lower(), s1, s2), (g["p2"].lower(), s2, s1)):
            rec[u][0] += sf > sa; rec[u][1] += 1
    return rec


def mlp_usage(year=YEAR):
    """pid -> (matchups appeared in, matchups the player's modal franchise
    played, franchise). Appearance = any game of the matchup (DBs count).
    A low ratio is EITHER bench or injury -- the data cannot tell, an owner
    reading the news can (Rohrabacher's absence / Blatt subbing in)."""
    mm = {r["match_id"]: r for r in _rows("mlp_matchups_2026.csv")}
    team_matchups = defaultdict(set)
    for r in mm.values():
        team_matchups[r["team_one"]].add(r["matchup_id"])
        team_matchups[r["team_two"]].add(r["matchup_id"])
    seen = defaultdict(Counter)           # pid -> franchise -> games
    appeared = defaultdict(lambda: defaultdict(set))   # pid -> franchise -> matchup ids
    for g in _rows("games.csv"):
        if g["tour"] != "MLP" or not g["date"].startswith(year):
            continue
        m = mm.get(g["match_id"])
        if not m or m["winner_side"] not in ("1", "2"):
            continue
        try:
            t1_won = int(g["t1_score"]) > int(g["t2_score"])
        except ValueError:
            continue
        t1_is_one = (m["winner_side"] == "1") == t1_won
        for side, team in (("t1", m["team_one"] if t1_is_one else m["team_two"]),
                           ("t2", m["team_two"] if t1_is_one else m["team_one"])):
            for k in ("p1", "p2"):
                u = g[f"{side}_{k}"].lower()
                seen[u][team] += 1
                appeared[u][team].add(m["matchup_id"])
    out = {}
    for u, c in seen.items():
        fr = c.most_common(1)[0][0]   # modal franchise; appearances for OTHER teams (mid-season moves) are not counted
        out[u] = (len(appeared[u][fr]), len(team_matchups[fr]), fr)
    return out


# ------------------------------------------------------------------ ordinal beliefs
def _posterior(name, mean_col, sd_col, gender_col="gender", id_col="player_id"):
    P = {}
    for r in _rows(name):
        try:
            P[r[id_col].lower()] = (r["full_name"], r[gender_col], float(r[mean_col]), float(r[sd_col]))
        except (KeyError, ValueError):
            continue
    return P


DOUBLES_POST = _posterior("v2_players.csv", "value_now_mean", "value_now_sd")
SINGLES_POST = _posterior("singles_players.csv", "singles_value", "singles_sd")
NAME = {u: v[0] for u, v in DOUBLES_POST.items()}
GENDER = {u: v[1] for u, v in DOUBLES_POST.items()}


def draw_order(rng, post, gender, pids=None, sd_mult=1.0):
    """One owner's ordinal belief: a joint posterior draw, ranked within gender.
    Returns the list of pids best-first. Only the ORDER leaves this function."""
    pids = [u for u in (pids if pids is not None else post) if u in post and post[u][1] == gender]
    mu = np.array([post[u][2] for u in pids]); sd = np.array([post[u][3] for u in pids]) * sd_mult
    z = mu + sd * rng.standard_normal(len(pids))
    return [pids[i] for i in np.argsort(-z, kind="stable")]


def rank_probs(post, gender, pids=None, draws=4000, sd_mult=1.0, seed=0):
    """pid -> array P(rank == r) over the given pids (best = 0)."""
    rng = np.random.default_rng(seed)
    pids = [u for u in (pids if pids is not None else post) if u in post and post[u][1] == gender]
    mu = np.array([post[u][2] for u in pids]); sd = np.array([post[u][3] for u in pids]) * sd_mult
    z = mu[None, :] + sd[None, :] * rng.standard_normal((draws, len(pids)))
    ranks = np.argsort(np.argsort(-z, axis=1), axis=1)      # rank of each pid per draw
    n = len(pids)
    out = {}
    for j, u in enumerate(pids):
        out[u] = np.bincount(ranks[:, j], minlength=n) / draws
    return out


# ------------------------------------------------------------------ the owner's board
def board_pids():
    """The auction board as draft_sim defines it (mlp2026): priced pool + 2026 MLP participants."""
    import sys
    sys.path.insert(0, str(HERE))
    import draft_sim  # noqa: E402
    draft_sim.set_board("mlp2026")
    return list(draft_sim.BOARD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd-mult", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=14)
    A = ap.parse_args()
    pids = board_pids()
    dbl = doubles_records(); sgl = singles_records(); sgl_c = singles_records(None); use = mlp_usage()
    pct = lambda r: f"{r[0]/r[1]:.2f}" if r[1] else "  -- "
    for g in ("F", "M"):
        rp = rank_probs(DOUBLES_POST, g, pids, sd_mult=A.sd_mult)
        order = sorted(rp, key=lambda u: (rp[u] * np.arange(len(rp[u]))).sum())
        sp = rank_probs(SINGLES_POST, g, pids, sd_mult=A.sd_mult)
        print(f"\n=== {g}: what a new owner sees (2026 doubles records by division, singles record, MLP usage; "
              f"P(rank<=k) from the posterior x{A.sd_mult:g}) ===")
        print(f"{'player':22s} {'E[rk]':>5s} {'P<=1':>5s} {'P<=3':>5s} {'P<=5':>5s} {'P<=10':>5s} | "
              f"{'WD/MD':>9s} {'MXD':>9s} | {'sgl26':>9s} {'sglcar':>9s} {'E[sgl rk]':>9s} | {'MLP use':>8s} franchise")
        for u in order[:A.top]:
            p = rp[u]; c = np.cumsum(p)
            er = (p * np.arange(len(p))).sum() + 1
            d = dbl.get(u, {})
            own = d.get("womens" if g == "F" else "mens", [0, 0, 0, 0]); mx = d.get("mixed", [0, 0, 0, 0])
            s26 = sgl.get(u, [0, 0]); sc = sgl_c.get(u, [0, 0])
            es = (sp[u] * np.arange(len(sp[u]))).sum() + 1 if u in sp else float("nan")
            us = use.get(u)
            ustr = f"{us[0]:2d}/{us[1]:2d}" if us else "   --"
            print(f"{NAME[u][:22]:22s} {er:5.1f} {c[0]:5.2f} {c[2]:5.2f} {c[4]:5.2f} {c[9]:5.2f} | "
                  f"{pct(own)} {own[1]:3d} {pct(mx)} {mx[1]:3d} | {pct(s26)} {s26[1]:3d} {pct(sc)} {sc[1]:3d} {es:9.1f} | "
                  f"{ustr:>8s} {us[2] if us else ''}")
    # usage outliers among top-30 by expected rank: the injury/bench fog
    print("\n=== MLP 2026 usage < 75% among the owner's top 30 per gender (bench or injury -- data can't tell) ===")
    for g in ("F", "M"):
        rp = rank_probs(DOUBLES_POST, g, pids, sd_mult=A.sd_mult)
        order = sorted(rp, key=lambda u: (rp[u] * np.arange(len(rp[u]))).sum())[:30]
        for u in order:
            us = use.get(u)
            if us and us[0] < 0.75 * us[1]:
                print(f"  {NAME[u]:24s} {us[0]:2d}/{us[1]:2d} matchups  {us[2]}")


if __name__ == "__main__":
    main()
