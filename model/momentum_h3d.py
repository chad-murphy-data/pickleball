"""Frozen analysis for ADDENDUM H3d — fresh-sample confirmation of the
3-rally timeout hint (see model/momentum_prereg.md). One estimand, one
run, on the full untouched replication pool.

    python model/momentum_h3d.py --selftest
    python model/momentum_h3d.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from momentum_test import RAW, match_rallies, sample_matches, team_eta  # noqa: E402
from momentum_h3b import ols_cr0                                        # noqa: E402
from momentum_h3c import build, split_games                             # noqa: E402


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


def fit_share3(rows, cl, label):
    m = np.asarray(rows, float)
    cl = np.asarray(cl)
    X = np.column_stack([np.ones(len(m)), m[:, 0], m[:, 4], m[:, 6],
                         m[:, 7], m[:, 8]])
    beta, V = ols_cr0(X, m[:, 2], cl)          # col 2 = share3
    th, se = beta[1], math.sqrt(V[1, 1])
    z = th / se if se > 0 else 0.0
    p_one = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))   # H1: theta > 0
    tr = m[:, 0] == 1
    line = (f"{label}: theta3 {th*100:+.2f}pp (se {se*100:.2f}, 99% CI "
            f"[{(th-2.576*se)*100:+.2f}, {(th+2.576*se)*100:+.2f}], "
            f"z={z:+.2f}, one-sided p={p_one:.3g}) | n_treat {int(tr.sum())}, "
            f"n_ctrl {int((~tr).sum())}")
    print(line)
    return line, th, se, z, p_one


def selftest():
    # reuse H3c's endogenous zero-effect world, fit the 3-rally estimand
    import momentum_h3c as h3c
    import random
    rng = random.Random(31)
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
            prew5 = sum(h3c.point_delta(x, recv) for x in game[-5:])
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
        ms = build([game], eta)
        rows += ms
        cl += [mi] * len(ms)
    _, th, se, z, p = fit_share3(rows, cl, "selftest share3 (true 0)")
    ok = abs(z) < 3.29
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


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
        ms = build(gl, eta)
        rows += ms
        cl += [mid] * len(ms)
    print(f"pool: {len(pool)} matches ({missing} missing), "
          f"{len(rows)} moments")
    line, th, se, z, p_one = fit_share3(rows, cl, "H3d CONFIRMATORY share3")
    with (ROOT / "model" / "momentum_results.md").open("a") as fh:
        fh.write("\n## ADDENDUM H3d — fresh-sample confirmation "
                 "(single frozen run on the untouched pool)\n\n"
                 f"{line}\n")
    print("appended to model/momentum_results.md")


if __name__ == "__main__":
    main()
