"""Frozen analysis for ADDENDUM H3c — the rally-share steelman of the
timeout question (see model/momentum_prereg.md). FINAL timeout test on
this sample.

    python model/momentum_h3c.py --selftest
    python model/momentum_h3c.py
"""
from __future__ import annotations

import argparse
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

W, W3 = 10, 3


def rally_win(r, P):
    """1 if side P won rally r (serving and won, or receiving and server lost)."""
    return 1 if (r["side"] == P) == bool(r["y_srv"]) else 0


def split_games(rs):
    gl, cur = [], []
    for r in rs:
        if cur and r["trail5c"] is None and cur[-1]["trail5c"] is not None:
            gl.append(cur)
            cur = []
        cur.append(r)
    if cur:
        gl.append(cur)
    return gl


def point_delta(r, P):
    if not r["y_srv"]:
        return 0
    return 1 if r["side"] == P else -1


def build(gl, eta_m):
    """Moment rows: [treat, share10, share3, kill, prew_share, prew_pts,
    margin_recv, srv2, eta_recv] per the H3c spec (H3b moment rules)."""
    out = []
    for game in gl:
        n = len(game)
        for t in range(W, n - W + 1):
            r = game[t]
            if r["to_srv"]:
                continue
            P = 1 - r["side"]
            share10 = sum(rally_win(x, P) for x in game[t:t + W]) / W
            share3 = sum(rally_win(x, P) for x in game[t:t + W3]) / W3
            # possession-killer: does the current server possession yield
            # no further point? Follow rallies while the same side serves.
            kill = 1
            for x in game[t:]:
                if x["side"] != r["side"]:
                    break
                if x["y_srv"]:
                    kill = 0
                    break
            prew_share = sum(rally_win(x, P) for x in game[t - W:t]) / W
            prew_pts = sum(point_delta(x, P) for x in game[t - W:t])
            out.append([r["to_recv"], share10, share3, kill,
                        prew_share, prew_pts, -r["margin"], r["srv2"],
                        eta_m if P == 0 else -eta_m])
    return out


def fit(m, cl, ycol, prewcol, label, subset=None):
    m = np.asarray(m, float)
    cl = np.asarray(cl)
    if subset is not None:
        m, cl = m[subset], cl[subset]
    X = np.column_stack([np.ones(len(m)), m[:, 0], m[:, prewcol], m[:, 6],
                         m[:, 7], m[:, 8]])
    beta, V = ols_cr0(X, m[:, ycol], cl)
    th, se = beta[1], math.sqrt(V[1, 1])
    z = th / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    tr = m[:, 0] == 1
    naive = m[tr, ycol].mean() - m[tr, prewcol].mean()
    line = (f"{label}: theta {th*100:+.2f}pp (99% CI "
            f"[{(th-2.576*se)*100:+.2f}, {(th+2.576*se)*100:+.2f}], "
            f"z={z:+.2f}, p={p:.2g}) | n_treat {int(tr.sum())}, "
            f"n_ctrl {int((~tr).sum())} | naive Δshare {naive*100:+.1f}pp")
    print(line)
    return line, th, se, z, p


def selftest():
    rng = random.Random(29)
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
        ms = build([game], eta)
        rows += ms
        cl += [mi] * len(ms)
    line, th, se, z, p = fit(rows, cl, 1, 4, "selftest share10 (true 0)")
    m = np.asarray(rows, float)
    naive = m[m[:, 0] == 1, 1].mean() - m[m[:, 0] == 1, 4].mean()
    ok = abs(z) < 3.29 and naive > 0.05
    print(f"naive rally-share trap {naive*100:+.1f}pp; adjusted z={z:+.2f} "
          f"→ SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    sample, pairs, fmts, vals = sample_matches()
    rows, cl = [], []
    for mid in sample:
        p = RAW / "match_logs" / mid[:2] / f"{mid}.json"
        try:
            logs = json.loads(p.read_text()).get("data") or []
        except FileNotFoundError:
            continue
        s1, s2 = pairs[mid]
        eta = team_eta(vals[s1[0]], vals[s1[1]], vals[s2[0]], vals[s2[1]])
        gl = split_games(match_rallies(logs, s1, s2, eta, fmts.get(mid, {})))
        ms = build(gl, eta)
        rows += ms
        cl += [mid] * len(ms)

    m = np.asarray(rows, float)
    lines = ["", "## ADDENDUM H3c — rally-share steelman (single frozen run; "
             "FINAL timeout test on this sample)", ""]
    l, *_ = fit(rows, cl, 1, 4, "H3c PRIMARY share10")
    lines.append(l)
    l, *_ = fit(rows, cl, 2, 4, "S1 share3")
    lines.append(l)
    l, *_ = fit(rows, cl, 3, 4, "S2 possession-killer")
    lines.append(l)
    severe = m[:, 5] <= -4
    l, *_ = fit(rows, cl, 1, 4, "S3 share10 | severe trigger (prew_pts <= -4)",
                subset=severe)
    lines.append(l)
    with (ROOT / "model" / "momentum_results.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("appended to model/momentum_results.md")


if __name__ == "__main__":
    main()
