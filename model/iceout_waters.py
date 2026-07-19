"""Decompose the "surprise" of a single doubles result.

Case study: the 2026-07-12 MLP Mid-Season women's-doubles final game,
Anna Leigh Waters / Jorja Johnson LOST 6-11 to Anna Bright / Kate Fahey —
the headline upset of the weekend, and a graded 88%-favorite MISS on the
site's receipts.

The general question this tool answers, for ANY four players + an
observed result: given the ratings we had, how surprised should the model
have been, and WHERE does the surprise come from?  It separates four
sources and lets you turn each dial:

  1. race variance   — even at a fixed skill gap, a first-to-11 sprint is
                        noisy; a favorite loses a fixed fraction outright.
  2. skill error     — our value estimates have posterior SDs; maybe the
                        favorites were overrated / underdogs underrated.
  3. structure (γ)   — the weakest-link dial: how much a team is dragged
                        toward its weaker player (targeting / freeze-out).
  4. match shock     — v2's fitted per-match random effect sd_m ≈ 0.35
                        logit: game-to-game variation beyond skill (off
                        days, tactics, adrenaline).  Already IN the model;
                        collapsed when we quote a point estimate.

The lesson that generalizes: for well-observed players, dials 2 and 3 are
stiff (tight SDs; γ can't honestly leave the data-supported range), so a
lone upset is overwhelmingly dial 1 + dial 4 — "the Tuesday the model
budgets for" — NOT evidence the ratings were wrong.  To indict the
ratings you need many games (the convex-gap test in spec_shootout), not
one.

Run:  python model/iceout_waters.py            # prints tables
      python model/iceout_waters.py --json      # + model/iceout_waters_summary.json

Reusable: edit MATCHUP / OBSERVED at the top to point at another game.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
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
SEED, N_MC = 20260712, 60000


def load_values():
    want = set(MATCHUP["team1"]) | set(MATCHUP["team2"])
    V, SD = {}, {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        if r["full_name"] in want:
            V[r["full_name"]] = float(r["value_now_mean"])
            SD[r["full_name"]] = float(r["value_now_sd"])
    return V, SD


# ---- race math: side-out to T, win by 2 -------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def team_value(a, b, gamma):
    return a + b + gamma * abs(a - b)


def eta(V, gamma):
    t1 = team_value(V[MATCHUP["team1"][0]], V[MATCHUP["team1"][1]], gamma) + MATCHUP["chem1"]
    t2 = team_value(V[MATCHUP["team2"][0]], V[MATCHUP["team2"][1]], gamma) + MATCHUP["chem2"]
    return t1 - t2


def p_exact(a, b, p, T):
    """P(final score team1=a, team2=b) given per-point p=P(team1 point)."""
    q = 1 - p
    if a == T and b <= T - 2:
        return comb(T - 1 + b, b) * p ** T * q ** b
    if b == T and a <= T - 2:
        return comb(T - 1 + a, a) * q ** T * p ** a
    if a >= T - 1 and b >= T - 1:              # deuce region
        base = comb(2 * T - 2, T - 1) * (p * q) ** (T - 1)
        m = min(a, b) - (T - 1)                 # extra exchanged pairs
        pair = 2 * p * q
        lead = a - b
        if abs(lead) != 2:
            return 0.0
        return base * pair ** m * (p * p if lead == 2 else q * q)
    return 0.0


@lru_cache(maxsize=None)
def _loss_prob(p, T):
    q = 1 - p

    @lru_cache(maxsize=None)
    def f(a, b):
        if a >= T and a - b >= 2:
            return 1.0
        if b >= T and b - a >= 2:
            return 0.0
        if a >= T + 30 or b >= T + 30:
            return 0.5
        return p * f(a + 1, b) + q * f(a, b + 1)
    r = 1 - f(0, 0)
    f.cache_clear()
    return r


def p_loss(p, T=11):
    return _loss_prob(round(p, 5), T)


def p_loss_at_least(pts, p, T=11):
    """P(team1 loses with <= `pts` points) i.e. '11-pts or worse'."""
    q = 1 - p
    return sum(comb(T - 1 + a, a) * q ** T * p ** a for a in range(pts + 1))


def score_dist(p, T=11):
    """Full outcome distribution, team1 perspective."""
    d = {}
    for a in range(T + 1):
        for b in range(T + 1):
            pr = p_exact(a, b, p, T)
            if pr > 1e-12:
                d[(a, b)] = pr
    # deuce tail beyond T,T-? folded approximately into T+k scores omitted
    return d


# ---- Monte-Carlo posterior predictive ---------------------------------

def mc_bucket(V, SD, draw_vals, gdraw, match_eff, bucket_fn, rng, n=N_MC):
    """E[bucket probability] over selected uncertainty sources."""
    acc = 0.0
    for _ in range(n):
        vv = {k: rng.normal(V[k], SD[k]) for k in V} if draw_vals else V
        g = rng.normal(GAMMA, SD_G) if gdraw else GAMMA
        e = eta(vv, g) + (rng.normal(0, SD_M) if match_eff else 0.0)
        acc += bucket_fn(sigmoid(e))
    return acc / n


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    V, SD = load_values()
    T = MATCHUP["T"]
    t1n = " / ".join(MATCHUP["team1"])
    obs_pts = min(OBSERVED)                 # team1's points in the actual loss
    lop = LOPSIDED_PTS                      # the "11-lop or worse" threshold
    rng = np.random.default_rng(SEED)
    out = {"matchup": t1n + " vs " + " / ".join(MATCHUP["team2"]),
           "observed": f"{OBSERVED[0]}-{OBSERVED[1]}"}

    print("values:  " + "   ".join(f"{k} {V[k]:.2f}±{SD[k]:.2f}" for k in
                                    list(MATCHUP["team1"]) + list(MATCHUP["team2"])))
    p0 = sigmoid(eta(V, GAMMA))
    print(f"\nplug-in per-point p({t1n})={p0:.3f}  eta={eta(V,GAMMA):.3f}")

    # outcome buckets, plug-in (P(win) from the exact DP; modal from the dist)
    d = score_dist(p0, T)
    modal = max(d, key=d.get)
    win = 1 - p_loss(p0, T)
    print(f"  P(win)                = {win*100:.1f}%   modal score {modal[0]}-{modal[1]} "
          f"({d[modal]*100:.1f}%)")
    print(f"  P(lose, any)          = {p_loss(p0,T)*100:.1f}%")
    print(f"  P(lose 11-{lop} or worse)   = {p_loss_at_least(lop,p0,T)*100:.1f}%   "
          f"<-- the question (team1 scores <= {lop})")
    print(f"  P(this exact {OBSERVED[0]}-{OBSERVED[1]})       = {p_exact(*OBSERVED,p0,T)*100:.2f}%   "
          f"(P(11-{obs_pts} or worse)={p_loss_at_least(obs_pts,p0,T)*100:.1f}%)")

    # --- posterior predictive ladder for the '11-lop or worse' bucket ---
    print(f"\nPOSTERIOR PREDICTIVE — P(lose 11-{lop} or worse):")
    bf = lambda p: p_loss_at_least(lop, p, T)
    ladder = [
        ("plug-in", False, False, False),
        ("+ skill-estimation error", True, False, False),
        ("+ gamma uncertainty", True, True, False),
        ("+ match shock (sd_m, the model's own)", True, True, True)]
    lad = {}
    for name, dv, gd, me in ladder:
        v = mc_bucket(V, SD, dv, gd, me, bf, np.random.default_rng(SEED))
        lv = mc_bucket(V, SD, dv, gd, me, lambda p: p_loss(p, T), np.random.default_rng(SEED))
        lad[name] = dict(loss=lv, lopsided=v)
        print(f"  {name:42s} P(loss)={lv*100:4.1f}%   P(11-{lop} or worse)={v*100:4.1f}%")

    # conditional: given they lost, how likely was it at least this lopsided?
    cond = p_loss_at_least(lop, p0, T) / p_loss(p0, T)
    print(f"\ngiven a loss, P(it's 11-{lop} or worse) = {cond*100:.0f}%  (plug-in)")

    # --- BOTH DIALS AT ONCE ---
    print(f"\nBOTH DIALS — P(lose 11-{lop} or worse) as a grid.")
    print("  rows: skill shift k (team1 down k·SD, team2 up k·SD)")
    print("  cols: weak-link gamma\n")
    gammas = [-0.183, -0.35, -0.50, -1.00]
    ks = [0.0, 0.5, 1.0, 1.5, 2.0]
    hdr = "   k\\γ  " + "".join(f"{g:>9.2f}" for g in gammas)
    print(hdr)
    grid = {}
    for k in ks:
        vv = {MATCHUP["team1"][0]: V[MATCHUP["team1"][0]] - k * SD[MATCHUP["team1"][0]],
              MATCHUP["team1"][1]: V[MATCHUP["team1"][1]] - k * SD[MATCHUP["team1"][1]],
              MATCHUP["team2"][0]: V[MATCHUP["team2"][0]] + k * SD[MATCHUP["team2"][0]],
              MATCHUP["team2"][1]: V[MATCHUP["team2"][1]] + k * SD[MATCHUP["team2"][1]]}
        row = []
        for g in gammas:
            p = sigmoid(eta(vv, g))
            row.append(p_loss_at_least(lop, p, T))
        grid[k] = row
        print(f"  {k:4.1f}   " + "".join(f"{x*100:8.1f}%" for x in row))

    # for context: same grid but P(loss) at the corner
    pcorner = sigmoid(eta({MATCHUP["team1"][0]: V[MATCHUP["team1"][0]] - 2 * SD[MATCHUP["team1"][0]],
                           MATCHUP["team1"][1]: V[MATCHUP["team1"][1]] - 2 * SD[MATCHUP["team1"][1]],
                           MATCHUP["team2"][0]: V[MATCHUP["team2"][0]] + 2 * SD[MATCHUP["team2"][0]],
                           MATCHUP["team2"][1]: V[MATCHUP["team2"][1]] + 2 * SD[MATCHUP["team2"][1]]}, -0.5))
    print(f"\n  corner (k=2, γ=-0.5): p({t1n})={pcorner:.3f}, "
          f"P(loss)={p_loss(pcorner,T)*100:.0f}% — even here, still <coin-flip-ish")

    if args.json:
        out.update(lopsided_threshold_pts=lop,
                   plugin=dict(p_point=p0, p_win=win, p_loss=p_loss(p0, T),
                               p_lopsided=p_loss_at_least(lop, p0, T),
                               p_exact_observed=p_exact(*OBSERVED, p0, T),
                               modal=f"{modal[0]}-{modal[1]}"),
                   ladder=lad, cond_lopsided_given_loss=cond,
                   both_dials_grid={f"k={k}": {f"g={g}": grid[k][i]
                                    for i, g in enumerate(gammas)} for k in ks})
        (ROOT / "model" / "iceout_waters_summary.json").write_text(json.dumps(out, indent=1))
        print("\nwrote model/iceout_waters_summary.json")


if __name__ == "__main__":
    main()
