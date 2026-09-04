"""value_cap/fast_tie.py -- the speed layer: every pairing's strength in one
table, every game probability from one interpolation grid, tie_win_prob
~100x faster and checked against the exact engine.

    python value_cap/fast_tie.py            # builds/loads the grid, writes pair_table.csv,
                                            # self-test vs phase1_value_model.tie_win_prob

Why this is exact and small. In the production engine a doubles game's
win probability is game_win_prob_uncertain(eta, sd) with
    eta = S(pair A) - S(pair B),   S(v1, v2) = v1 + v2 + gamma*|v1 - v2|
    sd  = sqrt(U(pair A) + U(pair B)),   U = sd1^2 + sd2^2
so "simulate every match between every pairing" reduces to (a) S and U per
pair -- pair_table.csv, 7,140 rows for a 60+60 pool (1,770 MD + 1,770 WD +
3,600 MXD) -- and (b) one function of two numbers, f(eta, sd), which the
engine evaluates by a 41-node integral over a race DP (~340 us cold).
Here f is tabulated once on a fine grid (eta step 0.005, sd step 0.01)
and read back by bilinear interpolation (~2 us). The self-test at the
bottom reports the max deviation from the exact tie probability over
random rosters; it is ~1e-5, far below anything a parity read can see.

The grid is cached under value_cap/cache/ (gitignored, ~1.5 MB, ~40 s to
rebuild). The weakest-link gamma is a SWITCH: pass a float (pooled, the
production -0.1829) or a dict {"MD":..,"WD":..,"MXD":..} for the
per-division refit (finding 1) -- the mixed-split pairing rule uses the
MXD value. Nothing about the tie format changes: top 2 per gender by
doubles value start, DreamBreaker foursome = top 2 per gender by singles
value, P(tie) = P(>=3 of 4) + P(2-2) * P(DB).
"""
from __future__ import annotations

import array
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib.race import GAMMA, game_win_prob_uncertain, race_dist, sigmoid  # noqa: E402

import phase1_value_model as p1  # noqa: E402  (installs calibration on import)

CACHE = Path(__file__).resolve().parent / "cache"
T_GAME = p1.T_GAME
K_DB = p1.K_DB_SINGLES

ETA_LO, ETA_HI, ETA_STEP = -5.0, 5.0, 0.005
SD_LO, SD_HI, SD_STEP = 0.02, 1.20, 0.01
N_ETA = int(round((ETA_HI - ETA_LO) / ETA_STEP)) + 1
N_SD = int(round((SD_HI - SD_LO) / SD_STEP)) + 1


# ------------------------------------------------------------- the grid
def _grid_path(T):
    return CACHE / f"game_grid_T{T}_{ETA_STEP}_{SD_STEP}.f64"


def load_grid(T=T_GAME, verbose=True):
    """array('d') of N_SD * N_ETA values, row = sd index, col = eta index."""
    path = _grid_path(T)
    if path.exists():
        g = array.array("d")
        g.frombytes(path.read_bytes())
        if len(g) == N_SD * N_ETA:
            return g
    t0 = time.time()
    g = array.array("d", bytes(8 * N_SD * N_ETA))
    for i in range(N_SD):
        sd = SD_LO + i * SD_STEP
        base = i * N_ETA
        for j in range(N_ETA):
            g[base + j] = game_win_prob_uncertain(ETA_LO + j * ETA_STEP, sd, T)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(g.tobytes())
    if verbose:
        print(f"built game grid T={T}: {N_SD}x{N_ETA} in {time.time()-t0:.0f}s -> {path.name}")
    return g


class GameProb:
    """f(eta, sd) by bilinear interpolation on the cached grid."""

    def __init__(self, T=T_GAME):
        self.T = T
        self.g = load_grid(T)

    def __call__(self, eta, sd):
        g = self.g
        if eta <= ETA_LO:
            x = 0.0
        elif eta >= ETA_HI:
            x = N_ETA - 1.0
        else:
            x = (eta - ETA_LO) / ETA_STEP
        if sd <= SD_LO:
            y = 0.0
        elif sd >= SD_HI:
            y = N_SD - 1.0
        else:
            y = (sd - SD_LO) / SD_STEP
        j = int(x)
        i = int(y)
        if j >= N_ETA - 1:
            j = N_ETA - 2
        if i >= N_SD - 1:
            i = N_SD - 2
        fx = x - j
        fy = y - i
        b0 = i * N_ETA + j
        b1 = b0 + N_ETA
        top = g[b0] * (1 - fx) + g[b0 + 1] * fx
        bot = g[b1] * (1 - fx) + g[b1 + 1] * fx
        return top * (1 - fy) + bot * fy


@lru_cache(maxsize=None)
def db_win(p4):
    return race_dist(p4, 21)["p_win"]


# --------------------------------------------------------------- the model
class FastTie:
    """Drop-in for phase1_value_model.tie_win_prob over a fixed player set.

    doubles/singles = the phase1 dicts; gamma = float or {"MD","WD","MXD"}.
    Players are addressed by player_id exactly as before."""

    def __init__(self, doubles, singles, gamma=GAMMA, T=T_GAME):
        self.f = GameProb(T)
        if isinstance(gamma, dict):
            self.gamma = {k: float(gamma[k]) for k in ("MD", "WD", "MXD")}
        else:
            self.gamma = {k: float(gamma) for k in ("MD", "WD", "MXD")}
        self.v = {u: d["v"] for u, d in doubles.items()}
        self.u2 = {u: d["sd"] ** 2 for u, d in doubles.items()}
        self.gender = {u: d["gender"] for u, d in doubles.items()}
        self.s = {u: p1.singles_of(u, doubles, singles) for u in doubles}
        self._lineup_cache = {}

    # -- pair primitives (the "table") --
    def S(self, a, b, division):
        va, vb = self.v[a], self.v[b]
        return va + vb + self.gamma[division] * abs(va - vb)

    def U(self, a, b):
        return self.u2[a] + self.u2[b]

    def game(self, pair_a, pair_b, division):
        eta = self.S(*pair_a, division) - self.S(*pair_b, division)
        return self.f(eta, math.sqrt(self.U(*pair_a) + self.U(*pair_b)))

    # -- lineup --
    def lineup(self, roster):
        """-> (wd_pair, md_pair, mxd1_pair, mxd2_pair, db_mean_singles); cached per roster."""
        key = roster if isinstance(roster, tuple) else tuple(roster)
        hit = self._lineup_cache.get(key)
        if hit is not None:
            return hit
        v = self.v
        women = sorted((u for u in key if self.gender[u] == "F"), key=lambda u: -v[u])
        men = sorted((u for u in key if self.gender[u] == "M"), key=lambda u: -v[u])
        w1, w2 = women[0], women[1]
        m1, m2 = men[0], men[1]
        g = self.gamma["MXD"]

        def pv(a, b):
            return v[a] + v[b] + g * abs(v[a] - v[b])
        if pv(w1, m1) + pv(w2, m2) >= pv(w1, m2) + pv(w2, m1):
            mx1, mx2 = (w1, m1), (w2, m2)
        else:
            mx1, mx2 = (w1, m2), (w2, m1)
        s = self.s
        dbw = sorted(women, key=lambda u: -s[u])[:2]
        dbm = sorted(men, key=lambda u: -s[u])[:2]
        db_mean = (s[dbw[0]] + s[dbw[1]] + s[dbm[0]] + s[dbm[1]]) / 4.0
        out = ((w1, w2), (m1, m2), mx1, mx2, db_mean)
        self._lineup_cache[key] = out
        return out

    def tie(self, roster_a, roster_b):
        wa, ma, xa1, xa2, dba = self.lineup(roster_a)
        wb, mb, xb1, xb2, dbb = self.lineup(roster_b)
        p1_ = self.game(wa, wb, "WD")
        p2_ = self.game(ma, mb, "MD")
        p3_ = self.game(xa1, xb1, "MXD")
        p4_ = self.game(xa2, xb2, "MXD")
        # distribution of wins out of 4
        d = [1.0, 0.0, 0.0, 0.0, 0.0]
        for p in (p1_, p2_, p3_, p4_):
            q = 1.0 - p
            d = [d[0] * q,
                 d[1] * q + d[0] * p,
                 d[2] * q + d[1] * p,
                 d[3] * q + d[2] * p,
                 d[4] + d[3] * p]
        p_db = db_win(round(sigmoid(K_DB * (dba - dbb)), 4))
        return d[3] + d[4] + d[2] * p_db

    __call__ = tie

    # -- the human-readable table --
    def pair_table(self, pool, names, path):
        """Write every pairing's strength S, uncertainty U, and P(beats an
        average pair of its division) -- pool = {"M": [pids], "F": [pids]}."""
        import csv
        rows = []
        divs = {"MD": [(a, b) for i, a in enumerate(pool["M"]) for b in pool["M"][i + 1:]],
                "WD": [(a, b) for i, a in enumerate(pool["F"]) for b in pool["F"][i + 1:]],
                "MXD": [(a, b) for a in pool["F"] for b in pool["M"]]}
        for div, pairs in divs.items():
            Ss = [self.S(a, b, div) for a, b in pairs]
            Us = [self.U(a, b) for a, b in pairs]
            S_avg = sum(Ss) / len(Ss)
            U_avg = sum(Us) / len(Us)
            for (a, b), S, U in zip(pairs, Ss, Us):
                rows.append({"division": div, "player_1": names[a], "player_2": names[b],
                             "pid_1": a, "pid_2": b, "S": round(S, 4), "U": round(U, 5),
                             "p_beats_avg_pair": round(self.f(S - S_avg, math.sqrt(U + U_avg)), 4)})
        rows.sort(key=lambda r: (r["division"], -r["S"]))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return len(rows)


# ------------------------------------------------------------- self-test
def selftest(n=3000, seed=0):
    from phase2_pricing import DOUBLES, NAME, POOL, SINGLES
    ft = FastTie(DOUBLES, SINGLES)
    pool = {g: [u for u, _, _ in POOL[g]] for g in ("M", "F")}
    rng = random.Random(seed)

    def rand_roster():
        return tuple(rng.sample(pool["M"], 3) + rng.sample(pool["F"], 3))
    pairs = [(rand_roster(), rand_roster()) for _ in range(n)]
    t0 = time.perf_counter()
    exact = [p1.tie_win_prob(a, b, DOUBLES, SINGLES) for a, b in pairs]
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    fast = [ft(a, b) for a, b in pairs]
    t_fast = time.perf_counter() - t0
    err = max(abs(x - y) for x, y in zip(exact, fast))
    print(f"self-test on {n} random pool rosters: max |fast - exact| = {err:.2e}; "
          f"exact {1e6*t_exact/n:.0f} us/tie (cold-ish cache), fast {1e6*t_fast/n:.1f} us/tie "
          f"({t_exact/t_fast:.0f}x)")
    # second pass: everything warm on both sides
    t0 = time.perf_counter()
    [ft(a, b) for a, b in pairs]
    t_fast2 = time.perf_counter() - t0
    print(f"fast, warm lineup cache: {1e6*t_fast2/n:.1f} us/tie")
    assert err < 5e-4, err
    n_rows = ft.pair_table(pool, NAME, Path(__file__).resolve().parent / "pair_table.csv")
    print(f"wrote pair_table.csv: {n_rows} pairings (MD/WD/MXD over the 60+60 pool)")
    return ft


if __name__ == "__main__":
    selftest()
