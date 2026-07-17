"""Frozen analysis for ADDENDUM H3d as AMENDED (pre-unblinding) — the
window-profile confirmation on the untouched pool. One single run.

    python model/momentum_h3d.py --selftest
    python model/momentum_h3d.py

Primary: theta_3 > 0 one-sided at .01 (the sample-A-generated cell).
Family: windows {1,2,4,5,7,10} at Bonferroni two-sided .05/7 (|z|>=2.69).
Bounding: all seven 99% CIs reported; inside +/-2.5pp => family bounded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from momentum_test import RAW, match_rallies, sample_matches, team_eta  # noqa: E402
from momentum_h3b import ols_cr0                                        # noqa: E402
from momentum_h3c import rally_win, point_delta, split_games            # noqa: E402

WINDOWS = (1, 2, 3, 4, 5, 7, 10)
PRE = 10


def build_profile(gl, eta_m):
    """Moment rows: [treat, share_1..share_10 (7 cols), pre_share10,
    margin_recv, srv2, eta_recv] — H3b/H3c moment rules."""
    out = []
    for game in gl:
        n = len(game)
        for t in range(PRE, n - PRE + 1):
            r = game[t]
            if r["to_srv"]:
                continue
            P = 1 - r["side"]
            wins = [rally_win(x, P) for x in game[t:t + PRE]]
            shares = [sum(wins[:w]) / w for w in WINDOWS]
            prew = sum(rally_win(x, P) for x in game[t - PRE:t]) / PRE
            out.append([r["to_recv"], *shares, prew,
                        -r["margin"], r["srv2"],
                        eta_m if P == 0 else -eta_m])
    return out


def fit_profile(rows, cl):
    m = np.asarray(rows, float)
    cl = np.asarray(cl)
    nW = len(WINDOWS)
    X = np.column_stack([np.ones(len(m)), m[:, 0], m[:, 1 + nW],
                         m[:, 2 + nW], m[:, 3 + nW], m[:, 4 + nW]])
    res = []
    for i, w in enumerate(WINDOWS):
        beta, V = ols_cr0(X, m[:, 1 + i], cl)
        th, se = beta[1], math.sqrt(V[1, 1])
        z = th / se if se > 0 else 0.0
        res.append((w, th, se, z))
    return res


def report(res, n_treat, n_ctrl):
    lines = [f"moments: {n_treat} treated / {n_ctrl} control",
             "| W | theta_W | 99% CI | z | test |", "|---|---|---|---|---|"]
    for w, th, se, z in res:
        ci = f"[{(th-2.576*se)*100:+.2f}, {(th+2.576*se)*100:+.2f}]"
        if w == 3:
            p1 = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
            tag = f"PRIMARY one-sided p={p1:.3g} (alpha .01)"
        else:
            tag = f"family: {'DISCOVERY' if abs(z) >= 2.69 else 'ns'} (|z|>=2.69)"
        lines.append(f"| {w} | {th*100:+.2f}pp | {ci} | {z:+.2f} | {tag} |")
    bounded = all(abs(th) + 2.576 * se <= 0.025 for _, th, se, _ in res)
    lines.append("")
    lines.append("All seven 99% CIs inside ±2.5pp: "
                 + ("YES — the 1–10 rally window family is BOUNDED"
                    if bounded else "no"))
    return lines


def selftest():
    rng = random.Random(37)
    rows, cl = [], []
    k = 0.445
    for mi in range(1200):
        eta = rng.gauss(0, 0.5)
        okA = (k / (1 - k)) * math.exp(eta / 2)
        okB = (k / (1 - k)) / math.exp(eta / 2)
        kA, kB = okA / (1 + okA), okB / (1 + okB)
        a = b = 0
        side, num = rng.choice([0, 1]), 2
        game, tos = [], {0: 0, 1: 0}
        while not ((a >= 11 and a - b >= 2) or (b >= 11 and b - a >= 2)):
            recv = 1 - side
            prew5 = sum(point_delta(x, recv) for x in game[-5:])
            to_recv = 0
            if tos[recv] < 2 and rng.random() < 1 / (1 + math.exp(4 + 0.9 * prew5)):
                to_recv = 1
                tos[recv] += 1
            pwin = kA if side == 0 else kB
            y = 1 if rng.random() < pwin else 0
            ss, sr = (a, b) if side == 0 else (b, a)
            game.append(dict(y_srv=y, side=side, srv2=1 if num == 2 else 0,
                             margin=max(-7, min(7, ss - sr)),
                             to_recv=to_recv, to_srv=0, trail5c=0.0))
            if y:
                if side == 0:
                    a += 1
                else:
                    b += 1
            else:
                if num == 1:
                    num = 2
                else:
                    side, num = 1 - side, 1
        ms = build_profile([game], eta)
        rows += ms
        cl += [mi] * len(ms)
    res = fit_profile(rows, cl)
    mx = max(abs(z) for _, _, _, z in res)
    for w, th, se, z in res:
        print(f"  selftest W={w:2d}: {th*100:+.2f}pp z={z:+.2f}")
    ok = mx < 3.29
    print(f"max |z| = {mx:.2f} on true-null profile → SELFTEST "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def pool_matches():
    sample, pairs, fmts, vals = sample_matches()
    used = set(sample)
    vset = set(vals)
    pop = []
    for r in csv.DictReader(open(ROOT / "data" / "match_rally_summary.csv")):
        if r["discipline"] == "doubles" and r["score_check"] == "ok":
            m = r["match_id"]
            if m in pairs and set(pairs[m][0] + pairs[m][1]) <= vset:
                pop.append(m)
    return sorted(set(pop) - used), pairs, fmts, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    pool, pairs, fmts, vals = pool_matches()
    rows, cl, missing = [], [], 0
    for mid in pool:
        p = RAW / "match_logs" / mid[:2] / f"{mid}.json"
        try:
            logs = json.loads(p.read_text()).get("data") or []
        except FileNotFoundError:
            missing += 1
            continue
        s1, s2 = pairs[mid]
        eta = team_eta(vals[s1[0]], vals[s1[1]], vals[s2[0]], vals[s2[1]])
        gl = split_games(match_rallies(logs, s1, s2, eta, fmts.get(mid, {})))
        ms = build_profile(gl, eta)
        rows += ms
        cl += [mid] * len(ms)
    m = np.asarray(rows, float)
    n_treat, n_ctrl = int((m[:, 0] == 1).sum()), int((m[:, 0] == 0).sum())
    print(f"pool: {len(pool)} matches ({missing} missing)")
    res = fit_profile(rows, cl)
    lines = report(res, n_treat, n_ctrl)
    print("\n".join(lines))
    with (ROOT / "model" / "momentum_results.md").open("a") as fh:
        fh.write("\n## ADDENDUM H3d (amended): window-profile confirmation "
                 "on the untouched pool (single frozen run)\n\n"
                 + "\n".join(lines) + "\n")
    print("appended to model/momentum_results.md")


if __name__ == "__main__":
    main()
