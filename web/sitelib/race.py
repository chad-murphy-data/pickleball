"""Race-to-T math for the static site generator.

Mirrors model/v2_holdout.py exactly: a game is a race to T, win by 2, with
iid per-point win probability p; the deuce at (T-1, T-1) resolves with the
closed form p^2 / (p^2 + q^2).  The same DP ships in JS inside
simulator.html — keep the two in sync.
"""
from __future__ import annotations

import math
from functools import lru_cache

GAMMA = -0.1829          # weakest-link coefficient (v2 posterior mean, logit)
SD_MATCH = 0.352         # per-match random-effect sd (logit) — overdispersion


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@lru_cache(maxsize=4096)
def race_dist(p: float, T: int = 11):
    """Exact outcome distribution of one race-to-T game at point prob p.

    Returns dict with p_win, expected margin, and the final-score pmf
    (regulation scores exactly; the win-by-2 overtime branch lumped, where
    the winner's margin is exactly 2).
    """
    p = min(max(p, 1e-9), 1 - 1e-9)
    q = 1.0 - p
    win_scores = [(T, b, math.comb(T - 1 + b, b) * p**T * q**b)
                  for b in range(T - 1)]
    lose_scores = [(a, T, math.comb(T - 1 + a, a) * q**T * p**a)
                   for a in range(T - 1)]
    deuce = math.comb(2 * T - 2, T - 1) * (p * q) ** (T - 1)
    d_win = deuce * (p * p / (p * p + q * q))
    p_win = sum(pr for _, _, pr in win_scores) + d_win
    margin = (sum((T - b) * pr for _, b, pr in win_scores)
              - sum((T - a) * pr for a, _, pr in lose_scores)
              + 2 * d_win - 2 * (deuce - d_win))
    return {"p_win": p_win, "exp_margin": margin,
            "win_scores": win_scores, "lose_scores": lose_scores,
            "p_ot": deuce, "p_ot_win": d_win}


def game_win_prob(eta: float, T: int = 11) -> float:
    return race_dist(round(sigmoid(eta), 4), T)["p_win"]


def game_win_prob_uncertain(eta_mean: float, eta_sd: float, T: int = 11) -> float:
    """Win prob integrated over value uncertainty (41-node grid), matching
    the holdout methodology (posterior-averaged race probability)."""
    if eta_sd <= 0:
        return game_win_prob(eta_mean, T)
    total = wsum = 0.0
    for i in range(41):
        z = -4.0 + i * 0.2
        w = math.exp(-0.5 * z * z)
        total += w * game_win_prob(eta_mean + z * eta_sd, T)
        wsum += w
    return total / wsum


def team_eta(v1: float, v2: float, v3: float, v4: float,
             gamma: float = GAMMA) -> float:
    """Per-point logit of team (v1,v2) vs team (v3,v4), weakest link applied."""
    return (v1 + v2 + gamma * abs(v1 - v2)) - (v3 + v4 + gamma * abs(v3 - v4))


_CAL = {"a": 0.0, "b": 1.0, "eps": 0.0}


def set_calibration(a: float, b: float, eps: float):
    """Install the fitted display-calibration map (web/calibration.json)."""
    _CAL.update(a=a, b=b, eps=eps)


def calibrate(p: float) -> float:
    """p_cal = (1-eps)*sigmoid(a + b*logit(p)) + eps/2.  With eps > 0 no
    calibrated probability is ever 0 or 1 — there is always a chance."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    l = math.log(p / (1 - p))
    return (1 - _CAL["eps"]) * sigmoid(_CAL["a"] + _CAL["b"] * l) + _CAL["eps"] / 2


def value_points(v: float, gamma: float = GAMMA, T: int = 11) -> float:
    """Convert a per-point logit value to the human scale: expected margin of
    (player + average partner) vs an average pair in a race to 11.  The
    weakest-link penalty of dragging an average partner is included, so the
    map is deliberately asymmetric around zero."""
    return race_dist(round(sigmoid(v + gamma * abs(v)), 4), T)["exp_margin"]
