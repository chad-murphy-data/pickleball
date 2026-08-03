"""Closing out a game as a COMPETING-RISKS hazard problem.

From the moment a side is at 9+ and ahead, two absorbing events race:

    event L — the leader closes it out
    event T — the trailer comes back and wins

Discrete-time cause-specific hazards, one row per rally from entry onward.

Why this is worth running when everything else came back null
-------------------------------------------------------------
Duration-to-close is a deterministic function of the rally sequence, so
relative to the per-rally analyses it adds exactly ONE thing: sensitivity to
SERIAL DEPENDENCE.  The per-rally logistic assumes rallies are conditionally
independent given the score state.  If that is false — if endgames go on runs,
if a team that has been stuck on 10 keeps getting stuck — the hazard's
dependence on ELAPSED TIME will show it, and nothing else we have run would.

    h(t | state) under conditional independence depends ONLY on the state.
    Any residual dependence on t, after score-state fixed effects, IS momentum.

And the zero-clutch simulation is an exact null for this: `sim_game` draws
memoryless rallies with a constant per-side probability, so its hazard is
state-only by construction.  Real minus sim on the elapsed-time coefficients
is a clean, assumption-light test.

The mirror gap
--------------
Per player: cause-specific hazard of closing out vs of being closed out. Note
the raw gap does NOT cancel skill — a better player closes faster AND resists
longer, so the two compound. What it does cancel is player-level PACE (a
grinder plays long rallies in both directions). Skill still needs adjusting
for; that is what the state fixed effects and the skill covariate are for.

Run: python model/clutch_hazard.py            # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model"))

from sitelib import race                                    # noqa: E402
from sitelib.winprob import eta_anchor, serve_probs         # noqa: E402
import clutch_srm as cs                                      # noqa: E402
import clutch_endogeneity as ce                              # noqa: E402
from clutch_at_99_regimes import game_target                 # noqa: E402

TMAX = 14          # elapsed-rally buckets tracked from entry


def spells(seqs):
    """One spell per game: rallies from 'a side reaches 9+ and is ahead'
    to the end, with the cause. Returns rows (t, state, ended, leader_won)."""
    rows = []
    for (mid, gn, us, seq) in seqs:
        # find entry
        ent = None
        for i, (a, b, side, n, won) in enumerate(seq):
            if max(a, b) >= 9 and a != b:
                ent = i
                break
        if ent is None:
            continue
        lead0 = 0 if seq[ent][0] > seq[ent][1] else 1
        last = seq[-1]
        fa = last[0] + (1 if (last[2] == 0 and last[4]) else 0)
        fb = last[1] + (1 if (last[2] == 1 and last[4]) else 0)
        if game_target(max(fa, fb), min(fa, fb)) != 11:
            continue
        winner = 0 if fa > fb else 1
        for t, (a, b, side, n, won) in enumerate(seq[ent:]):
            ended = 1 if (ent + t) == len(seq) - 1 else 0
            # state from the LEADER's perspective, capped
            la, lb = (a, b) if lead0 == 0 else (b, a)
            srv_is_leader = 1 if side == lead0 else 0
            rows.append((min(t, TMAX), (min(la, 15), min(lb, 15), n, srv_is_leader),
                         ended, 1 if winner == lead0 else 0, mid, us, lead0))
    return rows


def hazard_by_t(rows, label):
    """P(game ends on this rally) by elapsed time, raw and state-adjusted."""
    raw = defaultdict(lambda: [0, 0])
    for (t, stt, ended, lw, mid, us, l0) in rows:
        raw[t][0] += 1
        raw[t][1] += ended
    print(f"\n  {label} — RAW hazard of the game ending, by rallies since entry")
    print(f"    {'t':>3}{'n at risk':>11}{'hazard':>9}")
    for t in sorted(raw):
        n, e = raw[t]
        if n < 200:
            continue
        print(f"    {t:>3}{n:>11}{e / n:>9.4f}")
    return raw


def state_adjusted(rows, label):
    """Logistic: ended ~ state FE + elapsed-time dummies.  The t coefficients
    after state FE are the momentum test."""
    states = {}
    for (t, stt, *_r) in rows:
        states.setdefault(stt, len(states))
    S = len(states)
    T = TMAX + 1
    X_s = np.array([states[r[1]] for r in rows])
    X_t = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows], dtype=float)
    b = np.zeros(S + T)

    def eta(b):
        return b[X_s] + b[S + X_t]

    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(eta(b), -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = np.zeros(S + T)
        h = np.zeros(S + T)
        r = y - p
        np.add.at(g, X_s, r); np.add.at(h, X_s, w)
        np.add.at(g, S + X_t, r); np.add.at(h, S + X_t, w)
        g -= b / 25.0 ** 2; h += 1 / 25.0 ** 2 + 1e-9
        step = g / h
        step = np.clip(step, -0.5, 0.5)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    # report t effects relative to t=0
    tc = b[S:] - b[S]
    print(f"\n  {label} — elapsed-time effect AFTER score-state fixed effects")
    print(f"    (logit, relative to t=0; flat = memoryless = no momentum)")
    print(f"    {'t':>3}{'coef':>9}")
    cnt = defaultdict(int)
    for r in rows:
        cnt[r[0]] += 1
    for t in range(T):
        if cnt[t] < 200:
            continue
        print(f"    {t:>3}{tc[t]:>+9.3f}")
    return tc, cnt


def main():
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]
    real, specs = [], []
    for (mid, gn), rs in by_game.items():
        s0, s1 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s0) != 2 or len(s1) != 2:
            continue
        us = s0 + s1
        if not all(u in cur for u in us):
            continue
        rs.sort(key=lambda r: r["rally_number"])
        if any(r["server_score"] is None or r["server_side"] is None for r in rs):
            continue
        real.append((mid, gn, us,
                     [(r["server_score"] if r["server_side"] == 0 else r["receiver_score"],
                       r["receiver_score"] if r["server_side"] == 0 else r["server_score"],
                       r["server_side"], r["server_number"] or 2, r["won"]) for r in rs]))
        m = rs[0]["match_date"][:7]
        specs.append((mid, gn, us, [traj[u].get(m, cur[u]["v"]) for u in us],
                      rs[0]["server_side"], rs[0]["server_number"] or 2))
    rng = np.random.default_rng(1234)
    sim = []
    for (mid, gn, us, v, s0_, n0) in specs:
        eta_ = race.team_eta(v[0], v[1], v[2], v[3])
        p0 = race.calibrate(race.game_win_prob_uncertain(eta_, race.SD_MATCH, 11))
        kA, kB = serve_probs(round(eta_anchor(p0) / 0.05) * 0.05, 0.443)
        s = ce.sim_game(rng, kA, kB, s0_, n0)
        if len(s) >= 6:
            sim.append((mid, gn, us, s))

    R, S_ = spells(real), spells(sim)
    print(f"spells: real {len({r[4] for r in R})} games, {len(R)} rally-rows;  "
          f"sim {len({r[4] for r in S_})} games, {len(S_)} rally-rows")
    lwR = np.mean([r[3] for r in R if r[0] == 0])
    lwS = np.mean([r[3] for r in S_ if r[0] == 0])
    print(f"leader eventually wins: real {lwR:.4f}   sim {lwS:.4f}")
    hazard_by_t(R, "REAL")
    hazard_by_t(S_, "SIM ")
    tR, cR = state_adjusted(R, "REAL")
    tS, cS = state_adjusted(S_, "SIM ")
    # t=0 is DEGENERATE as a reference: entry happens at 9-7 or similar, where
    # the game cannot end on that very rally, so its hazard is mechanically ~0
    # and collinear with the state effect. Comparing levels against it measures
    # the entry boundary, not momentum. The momentum question is about the
    # SHAPE across t, so re-reference to t=1 and compare shapes.
    print(f"\n{'=' * 62}\n  MOMENTUM TEST — hazard SHAPE across elapsed time\n"
          f"  (re-referenced to t=1; t=0 dropped as a degenerate boundary)\n"
          f"{'=' * 62}")
    print(f"    {'t':>3}{'real':>9}{'sim':>9}{'diff':>9}")
    mx = 0.0
    for t in range(1, TMAX + 1):
        if cR[t] < 200 or cS[t] < 200:
            continue
        rr, ss_ = tR[t] - tR[1], tS[t] - tS[1]
        d = rr - ss_
        mx = max(mx, abs(d))
        print(f"    {t:>3}{rr:>+9.3f}{ss_:>+9.3f}{d:>+9.3f}")
    print(f"\n  max |real - sim| on the SHAPE = {mx:.3f} logit")
    print("  => " + ("MOMENTUM: the real endgame is not memoryless"
                     if mx > 0.30 else
                     "NO duration dependence beyond the score state. Both arms "
                     "are flat in t:\n     once you know the score, how long "
                     "the endgame has already run tells you\n     nothing. The "
                     "real endgame is as memoryless as the simulation."))


if __name__ == "__main__":
    main()
