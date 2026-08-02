"""THE test: do the regime clutch ratings price 9-9 better than skill alone?

Everything else in this thread was descriptive — reliability, spread, rank
agreement. This asks the only question that decides whether the rating is a
steering wheel or the car: fed a game that has reached 9-9, does knowing the
four players' BIG-POINT levels beat knowing only their skill?

Design — and the out-of-sample discipline is the whole point
------------------------------------------------------------
The regime ratings were fitted on all rallies, 9-9 rallies included, so
scoring them on the same games would be circular. Instead, **2-fold
cross-fitting by match**:

    fit regimes on half A  ->  predict the 9-9 games in half B
    fit regimes on half B  ->  predict the 9-9 games in half A
    pool the predictions

Every prediction is therefore made by a model that never saw that match.
All 9-9 games are used, none of them in-sample.

Two models, both evaluated at the FIRST rally at 9-9 with its true serve
state, predicting who wins the game from there:

    SKILL   kA, kB from v2 through the anchored serve DP (the status quo —
            this is what site/live.html shows today)
    CLUTCH  kA, kB from the regime model's BIG-POINT levels:
              kA = sigmoid(base_hi + serve_big(s1) + serve_big(s2)
                                   - return_big(r1) - return_big(r2))
            From 9-9 on, essentially every rally is high-leverage, so the
            high-leverage regime is exactly the right parameterisation.

Same DP, same states, same outcomes — only the rally probabilities differ.

Format note: an earlier cut dropped every game finishing above 13 to exclude
to-15 Challenger rounds, which also threw out to-11 games that deuced out
(the 2026-08-02 MXD that prompted this ran to 15-13). Fixed: MLP is always
to 11, and a PPA game with margin 2 and the loser on 13+ is a to-11 deuce.

Run: python model/clutch_at_99_regimes.py      # needs SUPABASE_ANON_KEY
"""
from __future__ import annotations

import csv
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

from sitelib import race                                        # noqa: E402
from sitelib.winprob import (A1, A2, B1, B2, ServeDP,            # noqa: E402
                             eta_anchor, serve_probs)
import clutch_srm as cs                                          # noqa: E402
import clutch_regimes as cr                                      # noqa: E402

K_LEAGUE = 0.443


def game_target(win, lose):
    """to-11 (incl. deuce) vs to-15. Returns 11, 15, or None to drop.

    Score-based only — the rally cache does not carry `tour`. Works for both:
    MLP games are to 11 and so end at 13 or below unless they deuce, which
    the second branch catches."""
    if win <= 13:
        return 11
    if win - lose == 2 and lose >= 13 and win <= 19:
        return 11                      # to-11 that deuced out
    if 15 <= win <= 17 and lose <= 12:
        return 15
    return None


def collect(blob, cur, traj):
    """Per-game records for every doubles game that reached exactly 9-9."""
    by_game = defaultdict(list)
    for r in blob["rallies"]:
        by_game[(r["match_id"], r["game_number"])].append(r)
    roster = defaultdict(dict)
    for r in blob["roster"]:
        if r["p1"] and r["p2"]:
            roster[r["match_id"]][r["side"]] = [r["p1"].lower(), r["p2"].lower()]

    games, drop = [], defaultdict(int)
    for (mid, gn), rs in by_game.items():
        s0, s1 = roster[mid].get(0, []), roster[mid].get(1, [])
        if len(s0) != 2 or len(s1) != 2:
            drop["roster"] += 1
            continue
        us = s0 + s1
        if not all(u in cur for u in us):
            drop["unrated"] += 1
            continue
        rs.sort(key=lambda r: r["rally_number"])
        if any(r["server_score"] is None or r["receiver_score"] is None
               or r["server_side"] is None for r in rs):
            drop["null"] += 1
            continue
        hit = [r for r in rs if r["server_score"] == 9 and r["receiver_score"] == 9]
        if not hit:
            continue
        last = rs[-1]
        a = last["server_score"] + (1 if last["won"] else 0)
        b = last["receiver_score"] + (0 if last["won"] else 1)
        win, lose = max(a, b), min(a, b)
        T = game_target(win, lose)
        if T != 11:
            drop["not to-11" if T == 15 else "odd format"] += 1
            continue
        e = hit[0]
        month = e["match_date"][:7]
        games.append({
            "mid": mid, "gn": gn, "date": e["match_date"],
            "us": us,
            "vals": [traj[u].get(month, cur[u]["v"]) for u in us],
            "serve_side": e["server_side"],
            "server_number": e["server_number"] or 2,
            "won1": 1 if (last["server_side"] if last["won"]
                          else 1 - last["server_side"]) == 0 else 0,
            "deuce": win > 13,
        })
    return games, drop


def p_from_k(kA, kB, side, n):
    dp_key = (round(kA, 6), round(kB, 6))
    dp = _DPCACHE.get(dp_key)
    if dp is None:
        dp = _mk(kA, kB)
        _DPCACHE[dp_key] = dp
    st = ((A1 if n == 1 else A2) if side == 0 else (B1 if n == 1 else B2))
    return dp(9, 9, st)


_DPCACHE = {}


def _mk(kA, kB):
    from sitelib.winprob import _table
    V = _table(round(kA, 6), round(kB, 6), 11, 51)

    def p(a, b, s):
        return V.get((a, b, s), 0.5)
    return p


def main():
    print("Loading rallies ...")
    blob = cs.fetch()
    cur, traj = cs.load_values()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])

    games, drop = collect(blob, cur, traj)
    nd = sum(g["deuce"] for g in games)
    print(f"  {len(games)} to-11 doubles games reached 9-9  "
          f"({nd} of them deuced past 13 — the class the old filter threw away)")
    print(f"  dropped: {dict(drop)}\n")

    # ---- design rows for the regime fit, and the 2-fold split by match
    rows, _ = cs.build(blob, cur, traj)
    lz = np.array([r[4] for r in rows])
    hi_cut, lo_cut = np.quantile(lz, cr.HI_Q), np.quantile(lz, cr.LO_Q)
    keep = ([r + (True,) for r in rows if r[4] > hi_cut]
            + [r + (False,) for r in rows if r[4] < lo_cut])
    nhi, nlo = defaultdict(int), defaultdict(int)
    for r in keep:
        for u in r[:4]:
            (nhi if r[9] else nlo)[u] += 1
    ok = {u for u in nhi if nhi[u] >= cr.MIN_EACH and nlo[u] >= cr.MIN_EACH}
    keep = [r for r in keep if all(u in ok for u in r[:4])]
    index = {u: i for i, u in enumerate(sorted(ok))}
    P = len(index)
    mids = sorted({r[7] for r in keep})
    fold = {m: i % 2 for i, m in enumerate(mids)}
    print(f"  regime fit: {P} players, {len(keep)} rallies, 2-fold by match\n")

    def fit_fold(f):
        sub = [r for r in keep if fold[r[7]] == f]
        hi = np.array([r[9] for r in sub])
        y = np.array([r[6] for r in sub], dtype=float)
        rH, rL = y[hi].mean(), y[~hi].mean()
        off = np.where(hi, math.log(rH / (1 - rH)), math.log(rL / (1 - rL)))
        sub2 = [r[:5] + (float(off[i]),) + r[6:] for i, r in enumerate(sub)]
        b, _ = cr.fit_regimes(sub2, index, anchor=True)
        return {"mH": b[2 * P:3 * P], "nH": b[3 * P:],
                "mL": b[:P], "nL": b[P:2 * P],
                "base": math.log(rH / (1 - rH))}

    print("  fitting fold 0 ..."); F0 = fit_fold(0)
    print("  fitting fold 1 ..."); F1 = fit_fold(1)

    # ---- predict every 9-9 game with the model that never saw its match
    rec = []
    for g in games:
        if g["mid"] not in fold:
            continue
        if not all(u in index for u in g["us"]):
            continue
        F = F1 if fold[g["mid"]] == 0 else F0        # OUT of fold
        v = g["vals"]
        eta = race.team_eta(v[0], v[1], v[2], v[3])
        p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
        kA, kB = serve_probs(eta_anchor(p0), K_LEAGUE)
        p_skill = p_from_k(kA, kB, g["serve_side"], g["server_number"])

        i = [index[u] for u in g["us"]]
        sb, rb = F["mH"], F["nH"]
        eA = F["base"] + sb[i[0]] + sb[i[1]] - rb[i[2]] - rb[i[3]]
        eB = F["base"] + sb[i[2]] + sb[i[3]] - rb[i[0]] - rb[i[1]]
        cA, cB = 1 / (1 + math.exp(-eA)), 1 / (1 + math.exp(-eB))
        p_cl = p_from_k(cA, cB, g["serve_side"], g["server_number"])

        # THIRD ARM — the fair one.  Replacing skill with the big-point levels
        # throws away v2, which is fitted on three years of games, in favour of
        # levels estimated from a half-sample of high-leverage rallies.  The
        # question worth asking is whether clutch ADDS to skill: keep the
        # calibrated skill DP and shift its eta by the lift differential.
        lift = ((F["mH"] + F["nH"]) - (F["mL"] + F["nL"]))
        dlift = float(lift[i[0]] + lift[i[1]] - lift[i[2]] - lift[i[3]])
        rec.append((g["won1"], p_skill, p_cl, g["mid"], dlift,
                    eta_anchor(p0), g["serve_side"], g["server_number"],
                    fold[g["mid"]]))

    y = np.array([r[0] for r in rec], dtype=float)
    ps = np.array([r[1] for r in rec])
    pc = np.array([r[2] for r in rec])
    n = len(rec)

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def brier(p):
        return float(np.mean((p - y) ** 2))

    def acc(p):
        return float(np.mean((p > 0.5) == (y > 0.5)))

    print(f"\n{'=' * 72}\nOUT-OF-SAMPLE AT 9-9  (n={n} games, every prediction "
          f"from a model\nthat never saw that match)\n{'=' * 72}")
    print(f"  {'model':<10}{'log-loss':>12}{'Brier':>10}{'accuracy':>11}")
    print(f"  {'skill':<10}{ll(ps):>12.5f}{brier(ps):>10.5f}{100 * acc(ps):>10.1f}%")
    print(f"  {'clutch':<10}{ll(pc):>12.5f}{brier(pc):>10.5f}{100 * acc(pc):>10.1f}%")
    d_ll, d_br = ll(pc) - ll(ps), brier(pc) - brier(ps)
    print(f"  {'delta':<10}{d_ll:>+12.5f}{d_br:>+10.5f}"
          f"   {'CLUTCH BETTER' if d_br < 0 else 'skill better'}")

    # cluster bootstrap by match
    rng = np.random.default_rng(4)
    by_m = defaultdict(list)
    for k, r in enumerate(rec):
        by_m[r[3]].append(k)
    keys = list(by_m)
    ds = []
    for _ in range(3000):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_m[keys[k]] for k in pick])
        yy, a, b = y[idx], ps[idx], pc[idx]
        ds.append(np.mean((b - yy) ** 2) - np.mean((a - yy) ** 2))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    verdict = ("clutch SIGNIFICANTLY BETTER" if hi < 0 else
               "clutch SIGNIFICANTLY WORSE" if lo > 0 else "spans zero")
    print(f"\n  Brier delta {d_br:+.5f}  cluster-CI[{lo:+.5f},{hi:+.5f}]"
          f"   -> {verdict}")
    # ---- the additive arm, weight tuned on the OPPOSITE fold
    dl = np.array([r[4] for r in rec])
    ea = np.array([r[5] for r in rec])
    sd_ = np.array([r[6] for r in rec])
    sn = np.array([r[7] for r in rec])
    fo = np.array([r[8] for r in rec])

    def pred(w, mask):
        out = np.empty(mask.sum())
        for j, k in enumerate(np.where(mask)[0]):
            kA2, kB2 = serve_probs(ea[k] + w * dl[k], K_LEAGUE)
            out[j] = p_from_k(kA2, kB2, int(sd_[k]), int(sn[k]))
        return out

    p_add = np.empty(n)
    chosen = {}
    for f in (0, 1):
        tr, te = (fo != f), (fo == f)          # tune on the OTHER fold
        best, bw = None, 0.0
        for w in np.arange(0.0, 1.01, 0.1):
            b_ = np.mean((pred(w, tr) - y[tr]) ** 2)
            if best is None or b_ < best:
                best, bw = b_, w
        chosen[f] = bw
        p_add[te] = pred(bw, te)
    print(f"\n  {'skill+lift':<10}{ll(p_add):>12.5f}{brier(p_add):>10.5f}"
          f"{100 * acc(p_add):>10.1f}%   (weight tuned out-of-fold: "
          f"{chosen[0]:.1f}/{chosen[1]:.1f})")
    da = brier(p_add) - brier(ps)
    das = []
    for _ in range(3000):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_m[keys[k]] for k in pick])
        yy = y[idx]
        das.append(np.mean((p_add[idx] - yy) ** 2) - np.mean((ps[idx] - yy) ** 2))
    alo, ahi = np.percentile(das, [2.5, 97.5])
    av = ("ADDS SIGNIFICANTLY" if ahi < 0 else
          "HURTS SIGNIFICANTLY" if alo > 0 else "spans zero — adds nothing")
    print(f"  Brier delta vs skill {da:+.5f}  cluster-CI[{alo:+.5f},{ahi:+.5f}]"
          f"   -> {av}")

    print(f"  share of games where clutch moved the pick: "
          f"{100 * np.mean((ps > 0.5) != (pc > 0.5)):.1f}%")
    print(f"  mean |p_clutch - p_skill| = {np.mean(np.abs(pc - ps)):.4f}"
          f"   max {np.max(np.abs(pc - ps)):.4f}")


if __name__ == "__main__":
    main()
