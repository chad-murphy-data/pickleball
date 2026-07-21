"""Decompose the "surprise" of a single doubles result.

Case study: the 2026-07-12 MLP Mid-Season women's-doubles final game,
Anna Leigh Waters / Jorja Johnson LOST 6-11 to Anna Bright / Kate Fahey —
the weekend's headline upset and a graded 88%-favorite MISS on the site's
receipts.

The general question, for ANY four players + an observed result: given
the ratings we had, how surprised should the model have been, and which
source explains it?  Four candidate sources:

  1. race variance   — even at a fixed, known gap, a first-to-11 sprint is
                        noisy; a favorite loses some fixed fraction.
  2. skill error     — value estimates have posterior SDs (favorites maybe
                        overrated / underdogs underrated).
  3. structure (γ)   — the weakest-link dial (targeting / freeze-out).
  4. match shock     — v2's fitted per-match random effect sd_m ≈ 0.35.

IMPORTANT (learned the hard way, this script's own history): sources 1-2
are the honest predictive correction — integrating VALUE uncertainty is
what the holdout validates (it improves Brier) and is roughly what the
site's calibration layer already does.  Source 4 is a TRAP for
prediction: adding the per-match random effect at predict time
DOUBLE-COUNTS noise and OVER-disperses — holdout_calibration_test() below
shows Brier gets WORSE (0.1668 vs 0.1658), and the calibration buckets
show 90%-predicted favorites actually won 91.7%.  So the honest "how
surprised" number uses sources 1-3, not 4.  The match-shock row is kept
only to show what over-dispersion looks like and why we don't ship it.

The lesson that generalizes: for well-observed players a lone upset is
race variance + tight skill uncertainty — "the Tuesday the model budgets
for" — NOT evidence the ratings were wrong.  Indicting the ratings needs
many games (the convex-gap test in spec_shootout.md), never one.  For a
LOW-game player, source 2 widens and the story can differ — which is why
separating them is worth a script.

Run:  python model/iceout_waters.py            # tables + holdout validation
      python model/iceout_waters.py --json      # + iceout_waters_summary.json

Reusable: edit MATCHUP / OBSERVED / LOPSIDED_PTS to point at another game.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---- the game under the microscope (edit to reuse) --------------------
MATCHUP = dict(
    team1=("Anna Leigh Waters", "Jorja Johnson"),
    team2=("Anna Bright", "Kate Fahey"),
    chem1=0.0237, chem2=-0.0135,          # v2 dyad chem (data/v2_dyads.csv)
    T=11)
OBSERVED = (6, 11)                          # team1 - team2 final
LOPSIDED_PTS = 7                            # the "11-7 or worse" question
# v2 posterior scalars (model/v2_fit_summary.json)
GAMMA, SD_G, SD_M = -0.1829, 0.0474, 0.3523
SPLIT, MIN_TRAIN_GAMES = "2026-06-01", 10
SEED, N_MC = 20260712, 60000

# ---- race math: side-out to T, win by 2 (grid table = v2_holdout DP) ---
_GRID = np.linspace(0.01, 0.99, 981)


def _race_table(T):
    out = np.zeros_like(_GRID)
    for gi, p in enumerate(_GRID):
        q = 1 - p
        dp = np.zeros((T + 1, T + 1))
        dp[0, 0] = 1.0
        win = deuce = 0.0
        for a in range(T + 1):
            for b in range(T + 1):
                if dp[a, b] == 0:
                    continue
                if a == T - 1 and b == T - 1:
                    deuce += dp[a, b]
                    continue
                if a == T:
                    win += dp[a, b]
                    continue
                if b == T:
                    continue
                if a + 1 == T and b <= T - 2:
                    win += dp[a, b] * p
                else:
                    dp[a + 1, b] += dp[a, b] * p
                dp[a, b + 1] += dp[a, b] * q
        out[gi] = win + deuce * (p * p / (p * p + q * q + 1e-12))
    return out


_TAB = {11: _race_table(11), 15: _race_table(15)}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def race_win(eta, T=11):
    return float(np.interp(sigmoid(eta), _GRID, _TAB[T]))


def race_win_mix(eta, sd, T=11, nodes=21):
    if sd <= 0:
        return race_win(eta, T)
    zs = np.linspace(-3, 3, nodes)
    ws = np.exp(-0.5 * zs ** 2)
    ws /= ws.sum()
    return float(sum(w * race_win(eta + z * sd, T) for z, w in zip(zs, ws)))


def team_value(a, b, gamma):
    return a + b + gamma * abs(a - b)


def eta_of(V, gamma):
    t1 = team_value(V[MATCHUP["team1"][0]], V[MATCHUP["team1"][1]], gamma) + MATCHUP["chem1"]
    t2 = team_value(V[MATCHUP["team2"][0]], V[MATCHUP["team2"][1]], gamma) + MATCHUP["chem2"]
    return t1 - t2


def p_exact(a, b, p, T):
    q = 1 - p
    if a == T and b <= T - 2:
        return comb(T - 1 + b, b) * p ** T * q ** b
    if b == T and a <= T - 2:
        return comb(T - 1 + a, a) * q ** T * p ** a
    return 0.0                              # deuce region handled elsewhere


def p_loss_at_least(pts, p, T):
    """P(team1 loses with <= pts points) i.e. '11-pts or worse'."""
    q = 1 - p
    return sum(comb(T - 1 + a, a) * q ** T * p ** a for a in range(pts + 1))


# ---- data loaders -----------------------------------------------------

def load_values(path="v2_players.csv"):
    want = set(MATCHUP["team1"]) | set(MATCHUP["team2"])
    V, SD = {}, {}
    for r in csv.DictReader((DATA / path).open()):
        if r["full_name"] in want:
            V[r["full_name"]] = float(r["value_now_mean"])
            SD[r["full_name"]] = float(r["value_now_sd"])
    return V, SD


# ---- the decisive test: does the match shock help or hurt? ------------

def holdout_calibration_test():
    """On the frozen June+ holdout, compare prediction methods by Brier /
    log-loss.  This is what tells us the match shock over-disperses and the
    ~11% (value-uncertainty) number is the honest one — not 17%."""
    pl = {r["player_id"]: r for r in csv.DictReader((DATA / "v2_players_train.csv").open())}
    chem = {frozenset((r["p1_name"], r["p2_name"])): float(r["chem_logit_mean"])
            for r in csv.DictReader((DATA / "v2_dyads_train.csv").open())}
    names = {r["player_id"]: r["full_name"] for r in csv.DictReader((DATA / "players.csv").open())}
    rows = []
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["is_forfeit"] != "False" or g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        if g["date"] < SPLIT:
            continue
        us = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
        if any(u not in pl or int(pl[u]["games"]) < MIN_TRAIN_GAMES for u in us):
            continue
        v = [float(pl[u]["value_now_mean"]) for u in us]
        sd = [float(pl[u]["value_now_sd"]) for u in us]
        c1 = chem.get(frozenset((names.get(us[0], ""), names.get(us[1], ""))), 0.0)
        c2 = chem.get(frozenset((names.get(us[2], ""), names.get(us[3], ""))), 0.0)
        eta = (v[0] + v[1] + GAMMA * abs(v[0] - v[1])
               - v[2] - v[3] - GAMMA * abs(v[2] - v[3]) + c1 - c2)
        T = 11 if g["scoring_format"] == "sideout_11" else 15
        rows.append((eta, math.sqrt(sum(x * x for x in sd)), T, int(g["margin"]) > 0))

    def score(fn):
        n = len(rows)
        acc = br = ll = 0.0
        for eta, sv, T, won in rows:
            p = min(max(fn(eta, sv, T), 1e-9), 1 - 1e-9)
            acc += (p > 0.5) == won
            br += (p - won) ** 2
            ll += -(math.log(p) if won else math.log(1 - p))
        return dict(brier=br / n, log_loss=ll / n, accuracy=acc / n)

    methods = {
        "plug-in (no uncertainty)": lambda e, s, T: race_win(e, T),
        "+ value uncertainty [VALIDATED]": lambda e, s, T: race_win_mix(e, s, T),
        "+ value + match shock (sd_m)": lambda e, s, T: race_win_mix(e, math.sqrt(s * s + SD_M ** 2), T),
        "+ match shock only": lambda e, s, T: race_win_mix(e, SD_M, T)}
    res = {k: score(f) for k, f in methods.items()}
    print(f"\nHOLDOUT CALIBRATION TEST (n={len(rows)}) — which method is honest?")
    for k, m in res.items():
        flag = "  <- best Brier" if k.startswith("+ value uncertainty") else \
               "  <- OVER-disperses" if "match shock (sd_m)" in k else ""
        print(f"  {k:36s} Brier={m['brier']:.4f}  logloss={m['log_loss']:.4f}{flag}")
    return res


def conditional_loss_dist(p, T):
    """Among losing outcomes, the distribution of team1's point total.
    Includes an approximate deuce-loss mass (all 'close') in the base."""
    win = race_win(math.log(p / (1 - p)), T) if 0 < p < 1 else 0.0
    total_loss = 1 - win
    clean = {a: p_exact(a, T, p, T) for a in range(T - 1)}   # 11-a, a<=9
    clean_sum = sum(clean.values())
    deuce_mass = max(total_loss - clean_sum, 0.0)            # 10-12,11-13,... all 'close'
    return clean, clean_sum, deuce_mass, total_loss


# ---- Monte-Carlo posterior predictive ---------------------------------

def mc_bucket(V, SD, draw_vals, gdraw, match_eff, bucket_fn, rng, n=N_MC):
    acc = 0.0
    for _ in range(n):
        vv = {k: rng.normal(V[k], SD[k]) for k in V} if draw_vals else V
        g = rng.normal(GAMMA, SD_G) if gdraw else GAMMA
        e = eta_of(vv, g) + (rng.normal(0, SD_M) if match_eff else 0.0)
        acc += bucket_fn(sigmoid(e))
    return acc / n


# ---- the freeze-out MECHANISM (court coverage, asymmetric freeze) ------

def _team2_value(V, gamma=GAMMA):
    return team_value(V[MATCHUP["team2"][0]], V[MATCHUP["team2"][1]], gamma) + MATCHUP["chem2"]


def coverage_curve(V):
    """Win prob as the STRONGER team-1 player's share of the court, w,
    varies: team1 = 2*[(1-w)*weaker + w*stronger] (+ dyad chem).  w=0 fully
    iced, 0.5 equal, 1 stronger does everything.  Realized w = (1+gamma)/2
    (opponents already tilt the court toward the weaker player)."""
    a, b = (V[p] for p in MATCHUP["team1"])
    strong, weak = max(a, b), min(a, b)
    t2 = _team2_value(V)
    realized = (1 + GAMMA) / 2
    rows = []
    for w in (0.0, realized, 0.5, 0.75, 1.0):
        team1 = 2 * ((1 - w) * weak + w * strong) + MATCHUP["chem1"]
        rows.append((w, team1, race_win(team1 - t2, MATCHUP["T"])))
    return rows, realized


def asymmetric_freeze(V, SD):
    """Stronger team-1 player fully iced -> team1 = 2*weaker (the weaker
    player plays 100%), team2 kept at NORMAL weighting.  Skill dial k:
    team1 down k*SD, team2 up k*SD."""
    w1, w2 = MATCHUP["team1"]
    weaker = w1 if V[w1] <= V[w2] else w2
    o1, o2 = MATCHUP["team2"]
    rows = []
    for k in (0.0, 0.5, 1.0, 1.5, 2.0):
        team1 = 2 * (V[weaker] - k * SD[weaker]) + MATCHUP["chem1"]
        t2 = team_value(V[o1] + k * SD[o1], V[o2] + k * SD[o2], GAMMA) + MATCHUP["chem2"]
        rows.append((k, race_win(team1 - t2, MATCHUP["T"])))
    return rows, weaker


def two_of_weaker(V):
    """Sanity check: literally two copies of the weaker player (no dyad
    chem) vs the real team2 — the identity behind the freeze."""
    w1, w2 = MATCHUP["team1"]
    weaker = w1 if V[w1] <= V[w2] else w2
    return race_win(2 * V[weaker] - _team2_value(V), MATCHUP["T"]), weaker


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    V, SD = load_values()
    T = MATCHUP["T"]
    t1n = " / ".join(MATCHUP["team1"])
    obs_pts = min(OBSERVED)
    lop = LOPSIDED_PTS
    out = {"matchup": t1n + " vs " + " / ".join(MATCHUP["team2"]),
           "observed": f"{OBSERVED[0]}-{OBSERVED[1]}"}

    print("values:  " + "   ".join(f"{k} {V[k]:.2f}±{SD[k]:.2f}" for k in
                                    list(MATCHUP["team1"]) + list(MATCHUP["team2"])))
    e0 = eta_of(V, GAMMA)
    p0 = sigmoid(e0)
    win = race_win(e0, T)
    print(f"\nplug-in per-point p({t1n})={p0:.3f}  eta={e0:.3f}")
    print(f"  P(win)                = {win*100:.1f}%")
    print(f"  P(lose, any)          = {(1-win)*100:.1f}%")
    print(f"  P(lose 11-{lop} or worse)   = {p_loss_at_least(lop,p0,T)*100:.1f}%   "
          f"<-- 'or worse' question (team1 scores <= {lop})")
    print(f"  P(this exact {OBSERVED[0]}-{OBSERVED[1]})       = {p_exact(*sorted(OBSERVED),p0,T)*100:.2f}%")

    # conditional losing-score distribution (your point: mass is on close losses)
    clean, clean_sum, deuce_mass, tot = conditional_loss_dist(p0, T)
    print(f"\nGIVEN A LOSS — distribution of {t1n}'s point total "
          f"(close losses dominate):")
    for a in range(T - 2, -1, -1):
        if clean[a] / tot < 0.005 and a < obs_pts:
            continue
        mark = "  <- actual" if a == obs_pts else ""
        print(f"  11-{a:<2d}  {clean[a]/tot*100:5.1f}% of losses{mark}")
    print(f"  (10-12 / 11-13 deuce losses, all 'close'): {deuce_mass/tot*100:.0f}% of losses")
    print(f"  --> P(11-{lop} or worse | loss) = {p_loss_at_least(lop,p0,T)/tot*100:.0f}%   "
          f"P(11-{obs_pts} or worse | loss) = {p_loss_at_least(obs_pts,p0,T)/tot*100:.0f}%")

    # --- honest posterior predictive (sources 1-3; source 4 shown as trap) ---
    print(f"\nPOSTERIOR PREDICTIVE — P(loss)  [P(11-{lop} or worse)]:")
    bloss = lambda p: 1 - race_win(math.log(p / (1 - p)), T)
    blop = lambda p: p_loss_at_least(lop, p, T)
    ladder = [
        ("plug-in", False, False, False),
        ("+ skill-estimation error (VALIDATED honest number)", True, False, False),
        ("+ gamma uncertainty", True, True, False),
        ("+ match shock  [OVER-disperses OOS — do not ship]", True, True, True)]
    lad = {}
    for name, dv, gd, me in ladder:
        lv = mc_bucket(V, SD, dv, gd, me, bloss, np.random.default_rng(SEED))
        lp = mc_bucket(V, SD, dv, gd, me, blop, np.random.default_rng(SEED))
        lad[name] = dict(loss=lv, lopsided=lp)
        print(f"  {name:52s} P(loss)={lv*100:4.1f}%  [{lp*100:4.1f}%]")

    # --- BOTH DIALS AT ONCE (skill shift x weak-link gamma) ---
    print(f"\nBOTH DIALS — P(lose 11-{lop} or worse):  rows k = team1 down /"
          f" team2 up k·SD;  cols = gamma")
    gammas = [-0.183, -0.35, -0.50, -1.00]
    ks = [0.0, 0.5, 1.0, 1.5, 2.0]
    print("   k\\γ  " + "".join(f"{g:>9.2f}" for g in gammas))
    grid = {}
    for k in ks:
        vv = {MATCHUP["team1"][0]: V[MATCHUP["team1"][0]] - k * SD[MATCHUP["team1"][0]],
              MATCHUP["team1"][1]: V[MATCHUP["team1"][1]] - k * SD[MATCHUP["team1"][1]],
              MATCHUP["team2"][0]: V[MATCHUP["team2"][0]] + k * SD[MATCHUP["team2"][0]],
              MATCHUP["team2"][1]: V[MATCHUP["team2"][1]] + k * SD[MATCHUP["team2"][1]]}
        grid[k] = [p_loss_at_least(lop, sigmoid(eta_of(vv, g)), T) for g in gammas]
        print(f"  {k:4.1f}   " + "".join(f"{x*100:8.1f}%" for x in grid[k]))

    # --- the FREEZE-OUT mechanism: court coverage, not a rating error ------
    a, b = (V[p] for p in MATCHUP["team1"])
    strong = MATCHUP["team1"][0] if a >= b else MATCHUP["team1"][1]
    cov, realized = coverage_curve(V)
    print(f"\nFREEZE-OUT MECHANISM — win prob vs {strong}'s share of the court w")
    print(f"  (team1 = 2·[(1-w)·weaker + w·stronger] + chem; realized w={realized:.2f}"
          f" already tilts to the weaker side via gamma):")
    for w, team1v, wp in cov:
        tag = ("  <- fully iced (stronger touches nothing)" if w == 0.0 else
               "  <- realized court tilt (gamma)" if abs(w - realized) < 1e-9 else
               "  <- equal share" if w == 0.5 else
               "  <- stronger does everything" if w == 1.0 else "")
        print(f"    w={w:4.2f}  team1 value={team1v:+.3f}  P(win)={wp*100:5.1f}%{tag}")

    afz, weaker = asymmetric_freeze(V, SD)
    print(f"\nASYMMETRIC FREEZE — stronger fully iced (team1 = 2·{weaker}), "
          f"plus a skill dial k (team1 −k·SD, team2 +k·SD):")
    for k, wp in afz:
        print(f"    k={k:3.1f}  P(win)={wp*100:5.1f}%")

    two_wp, weaker2 = two_of_weaker(V)
    print(f"\nTWO-{weaker2.upper()} IDENTITY — literally two copies of the weaker "
          f"player (no chem) vs team2:  P(win)={two_wp*100:.1f}%")
    print(f"  (the 88%->~coin-flip collapse is the weakest-link structure, "
          f"not a claim that Waters' rating was wrong.)")

    cal = holdout_calibration_test()

    if args.json:
        out.update(lopsided_threshold_pts=lop,
                   plugin=dict(p_point=p0, p_win=win, p_loss=1 - win,
                               p_lopsided=p_loss_at_least(lop, p0, T),
                               p_exact_observed=p_exact(*sorted(OBSERVED), p0, T)),
                   cond_loss={f"11-{a}": clean[a] / tot for a in range(T - 1)},
                   ladder=lad,
                   both_dials_grid={f"k={k}": {f"g={g}": grid[k][i]
                                    for i, g in enumerate(gammas)} for k in ks},
                   freeze_out=dict(
                       realized_court_share=realized,
                       coverage_curve={f"w={w:.2f}": wp for w, _, wp in cov},
                       asymmetric_freeze={f"k={k}": wp for k, wp in afz},
                       two_of_weaker=two_wp),
                   holdout_calibration_test=cal)
        (ROOT / "model" / "iceout_waters_summary.json").write_text(json.dumps(out, indent=1))
        print("\nwrote model/iceout_waters_summary.json")


if __name__ == "__main__":
    main()
