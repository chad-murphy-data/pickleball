"""Serve-aware live win-probability engine for side-out doubles games.

The race DP in race.py is serve-blind: fine pre-match, wrong mid-game
(down 4-7 on your own second server is worse than 4-7 about to receive).
This module prices every (score, serve-state) exactly.

Model: a game to T (win by 2) is a Markov chain on states
(a, b, s) where s ∈ {A serves #1, A serves #2, B serves #1, B serves #2}.
The serving side wins a rally with probability kA (kB) — constant within
a game — scoring a point and KEEPING the same server on a win; on a loss
the serve passes #1→#2 within the team, #2→side-out. Games open on the
starting team's #2 server (the standard first-server exception).

Calibration: two inputs pin (kA, kB) —
  * eta (per-point logit from v2, via race.team_eta): the long-run point
    share must equal p = sigmoid(eta). Points are only scored on serve,
    and a team's expected points per service turn is odds(k) = k/(1-k)
    per server slot, so point share = odds(kA) / (odds(kA) + odds(kB)).
  * k, the league serve-rally win rate between equal teams (MEASURED from
    referee logs — scraper/harvest_logs.py --summarize; ~0.43 in pro
    doubles), sets how streaky the side-out cycle is.
  Solving with the symmetric constraint odds(kA)·odds(kB) = odds(k)^2:
      odds(kA) = odds(k) · sqrt(odds(p)),   odds(kB) = odds(k) / sqrt(odds(p))
  At eta = 0 both reduce to k. The odds-split is an assumption (how serve
  advantage scales with skill gap) — validate against rally logs once the
  backfill lands, via P(serve rally win | v2 gap).

Within a score cell the four state values form one linear cycle
(A1→A2→B1→B2→A1); the closed form divides by (1 - qA²qB²) — this is the
algebraic resolution of the no-score side-out cycle that a naive
recursion loops on forever.

Stdlib only, mirrors race.py conventions. The same recursion will need a
JS twin if/when the site ships live charts — keep them in sync.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .race import sigmoid

# Measured league serve-rally win rate between equal sides (referee logs,
# 2024-26 doubles; see recon.md "getListLogs" and data/match_rally_summary).
K_DOUBLES = 0.43
# Per-match random-effect sd on eta (logit) — same overdispersion the race
# model carries; the mixture wrapper integrates over it.
SD_MATCH = 0.352
# Display floor: no probability is ever shown as 0 or 1 (house rule).
EPS_FLOOR = 0.021

# Serve states
A1, A2, B1, B2 = 0, 1, 2, 3
STATE_NAMES = {A1: "A#1", A2: "A#2", B1: "B#1", B2: "B#2"}


def _odds(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return p / (1 - p)


def serve_probs(eta: float, k: float = K_DOUBLES) -> tuple[float, float]:
    """(kA, kB): rally win prob for each side while serving."""
    r = math.sqrt(_odds(sigmoid(eta)))
    oa, ob = _odds(k) * r, _odds(k) / r
    return oa / (1 + oa), ob / (1 + ob)


@lru_cache(maxsize=256)
def _table(kA: float, kB: float, T: int, cap: int) -> dict:
    """V[(a, b, s)] = P(team A wins the game) by backward induction."""
    qA, qB = 1 - kA, 1 - kB
    denom = 1 - qA * qA * qB * qB
    V = {}

    def done(a, b):
        if a >= T and a - b >= 2:
            return 1.0
        if b >= T and b - a >= 2:
            return 0.0
        return None

    def get(a, b, s):
        d = done(a, b)
        if d is not None:
            return d
        if a >= cap or b >= cap:      # unreachable-in-practice deuce tail
            return 0.5
        return V[(a, b, s)]

    for a in range(cap - 1, -1, -1):
        for b in range(cap - 1, -1, -1):
            if done(a, b) is not None:
                continue
            wa1, wa2 = get(a + 1, b, A1), get(a + 1, b, A2)
            lb1, lb2 = get(a, b + 1, B1), get(a, b + 1, B2)
            va1 = (kA * wa1 + qA * kA * wa2
                   + qA * qA * kB * lb1 + qA * qA * qB * kB * lb2) / denom
            va2 = kA * wa2 + qA * (kB * lb1 + qB * kB * lb2 + qB * qB * va1)
            vb2 = kB * lb2 + qB * va1
            vb1 = kB * lb1 + qB * vb2
            V[(a, b, A1)], V[(a, b, A2)] = va1, va2
            V[(a, b, B1)], V[(a, b, B2)] = vb1, vb2
    return V


class ServeDP:
    """Exact win probability for one game at a fixed eta."""

    def __init__(self, eta: float, k: float = K_DOUBLES, T: int = 11):
        self.eta, self.k, self.T = eta, k, T
        kA, kB = serve_probs(eta, k)
        self._V = _table(round(kA, 6), round(kB, 6), T, T + 40)
        self._done = lambda a, b: (1.0 if (a >= T and a - b >= 2) else
                                   0.0 if (b >= T and b - a >= 2) else None)

    def p(self, a: int, b: int, state: int) -> float:
        d = self._done(a, b)
        if d is not None:
            return d
        return self._V.get((a, b, state), 0.5)


class MixedDP:
    """Posterior-mixture win probability: integrates ServeDP over a normal
    on eta (value uncertainty + per-match random effect), 21-node grid —
    the live analogue of race.game_win_prob_uncertain."""

    def __init__(self, eta_mean: float, eta_sd: float,
                 k: float = K_DOUBLES, T: int = 11, nodes: int = 21):
        if eta_sd <= 0:
            self._parts = [(1.0, ServeDP(eta_mean, k, T))]
        else:
            zs = [-3.0 + i * (6.0 / (nodes - 1)) for i in range(nodes)]
            ws = [math.exp(-0.5 * z * z) for z in zs]
            tot = sum(ws)
            self._parts = [(w / tot, ServeDP(eta_mean + z * eta_sd, k, T))
                           for z, w in zip(zs, ws)]

    def p(self, a: int, b: int, state: int) -> float:
        return sum(w * dp.p(a, b, state) for w, dp in self._parts)


def display_floor(p: float, eps: float = EPS_FLOOR) -> float:
    """Mixture floor: nothing is ever displayed as 0% or 100%."""
    return (1 - eps) * p + eps / 2


def eta_anchor(target_p: float, k: float = K_DOUBLES, T: int = 11) -> float:
    """Find eta' such that the serve DP's start-of-game win prob (average
    over who serves first) equals target_p.

    Why: v2 etas are fitted and VALIDATED through the serve-blind race
    likelihood, and the site's pre-match probabilities are graded
    receipts. The serve DP at the same eta gives favorites systematically
    lower start probs (points cluster on serve possessions — fewer
    effectively-independent trials), so feeding raw etas in would
    contradict calibrated pre-match numbers. Anchoring preserves the
    validated endpoint; the measured k then only shapes the WITHIN-game
    dynamics. Revisit once the rally-log backfill lets us fit serve-state
    dynamics directly."""
    target_p = min(max(target_p, 1e-6), 1 - 1e-6)
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        dp = ServeDP(mid, k, T)
        if 0.5 * (dp.p(0, 0, A2) + dp.p(0, 0, B2)) < target_p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def first_server_effect(k: float = K_DOUBLES, T: int = 11) -> float:
    """P(win | serve first) - P(win | receive first) between equal teams —
    a diagnostic of the opening #2-server exception."""
    dp = ServeDP(0.0, k, T)
    return dp.p(0, 0, A2) - dp.p(0, 0, B2)


def rally_race_p(a: int, b: int, p: float, T: int = 21) -> float:
    """Win prob in a RALLY-scored race (DreamBreakers): every rally scores,
    iid rally prob p, win by 2. Used for the DB panel of live charts."""
    @lru_cache(maxsize=None)
    def f(x, y):
        if x >= T and x - y >= 2:
            return 1.0
        if y >= T and y - x >= 2:
            return 0.0
        if x >= T + 40 or y >= T + 40:
            return 0.5
        return p * f(x + 1, y) + (1 - p) * f(x, y + 1)
    return f(a, b)
