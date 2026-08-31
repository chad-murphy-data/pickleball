"""Frozen analysis for pre-registration ADDENDUM H3b (windowed timeout
effect) — see model/momentum_prereg.md. Run --selftest first, then once.

    python model/momentum_h3b.py --selftest   # endogenous-timeout null sim
    python model/momentum_h3b.py              # THE run (appends results md)
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
from momentum_test import (RAW, match_rallies, sample_matches,   # noqa: E402
                           team_eta)

W = 10          # primary window (rallies), pre and post
W2 = 5          # secondary window


def rally_delta(r, P):
    """Point contribution of one resolved rally to side P's differential."""
    if not r["y_srv"]:
        return 0
    return 1 if r["side"] == P else -1


def build_moments(rallies_by_game, eta_m, window):
    """Moment rows: (treat, Y, prew, margin, srv2, eta_recv) per the spec."""
    out = []
    for game in rallies_by_game:
        n = len(game)
        for t in range(window, n - window + 1):
            r = game[t]
            if r["to_srv"]:
                continue
            P = 1 - r["side"]                       # receiving side
            Y = sum(rally_delta(x, P) for x in game[t:t + window])
            prew = sum(rally_delta(x, P) for x in game[t - window:t])
            out.append((r["to_recv"], Y, prew,
                        -r["margin"], r["srv2"],     # margin → receiver persp
                        eta_m if P == 0 else -eta_m))
    return out


def ols_cr0(X, y, clusters):
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtX = np.linalg.inv(X.T @ X)
    beta = XtX @ (X.T @ y)
    resid = (y - X @ beta)[:, None] * X
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        s = resid[clusters == c].sum(axis=0)
        meat += np.outer(s, s)
    V = XtX @ meat @ XtX
    return beta, V


def run(moments, clusters, label):
    m = np.asarray(moments, float)
    cl = np.asarray(clusters)
    X = np.column_stack([np.ones(len(m)), m[:, 0], m[:, 2], m[:, 3],
                         m[:, 4], m[:, 5]])
    beta, V = ols_cr0(X, m[:, 1], cl)
    th, se = beta[1], math.sqrt(V[1, 1])
    z = th / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    treat = m[:, 0] == 1
    naive = m[treat, 1].mean() - m[treat, 2].mean()   # after minus before
    line = (f"{label}: theta {th:+.3f} pts (se {se:.3f}, 99% CI "
            f"[{th - 2.576 * se:+.3f}, {th + 2.576 * se:+.3f}], z={z:+.2f}, "
            f"p={p:.2g}) | n_treat {int(treat.sum())}, n_ctrl "
            f"{int((~treat).sum())} | trigger profile: mean prew "
            f"{m[treat, 2].mean():+.2f} (treat) vs {m[~treat, 2].mean():+.2f} "
            f"(ctrl) | NAIVE after-minus-before {naive:+.2f}")
    print(line)
    return line, th, se, z, p


# ---------- selftest: endogenous timeouts, zero true effect ----------

def selftest():
    rng = random.Random(13)
    moments, cl = [], []
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
            # endogenous hazard: receiver calls TO more after a bad stretch
            prew5 = sum(rally_delta(x, recv) for x in game[-5:])
            to_recv = 0
            if tos[recv] < 2 and rng.random() < 1 / (1 + math.exp(4 + 0.9 * prew5)):
                to_recv = 1
                tos[recv] += 1
            pwin = kA if side == 0 else kB
            y = 1 if rng.random() < pwin else 0
            ss, sr = (a, b) if side == 0 else (b, a)
            game.append(dict(y_srv=y, side=side, srv2=1 if num == 2 else 0,
                             margin=max(-7, min(7, ss - sr)),
                             to_recv=to_recv, to_srv=0))
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
        ms = build_moments([game], eta, W)
        moments += ms
        cl += [mi] * len(ms)
    line, th, se, z, p = run(moments, cl, "selftest W=10 (true effect 0)")
    m = np.asarray(moments, float)
    naive = m[m[:, 0] == 1, 1].mean() - m[m[:, 0] == 1, 2].mean()
    ok = abs(z) < 3.29 and naive > 1.0
    print(f"naive trap shows {naive:+.2f} pts of fake 'improvement'; "
          f"adjusted z={z:+.2f} → SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


# ---------- the run ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    sample, pairs, fmts, vals = sample_matches()
    mom10, cl10, mom5, cl5 = [], [], [], []
    for mid in sample:
        p = RAW / "match_logs" / mid[:2] / f"{mid}.json"
        try:
            rows = json.loads(p.read_text()).get("data") or []
        except FileNotFoundError:
            continue
        s1, s2 = pairs[mid]
        eta = team_eta(vals[s1[0]], vals[s1[1]], vals[s2[0]], vals[s2[1]])
        rs = match_rallies(rows, s1, s2, eta, fmts.get(mid, {}))
        # split back into games: match_rallies resets the trailing window
        # each game, so a not-None → None transition in trail5c marks a
        # game boundary (games always exceed 5 rallies, so mid-game
        # trail5c is not-None from rally 6 on).
        gl, cur = [], []
        for r in rs:
            if cur and r["trail5c"] is None and cur[-1]["trail5c"] is not None:
                gl.append(cur)
                cur = []
            cur.append(r)
        if cur:
            gl.append(cur)
        m10 = build_moments(gl, eta, W)
        m5 = build_moments(gl, eta, W2)
        mom10 += m10
        cl10 += [mid] * len(m10)
        mom5 += m5
        cl5 += [mid] * len(m5)

    lines = ["", "## ADDENDUM H3b — windowed timeout effect (single frozen run)", ""]
    l1, *_ = run(mom10, cl10, "H3b W=10 (primary)")
    l2, *_ = run(mom5, cl5, "H3b W=5 (secondary)")
    lines += [l1, "", l2, ""]
    with (ROOT / "model" / "momentum_results.md").open("a") as fh:
        fh.write("\n".join(lines))
    print("appended to model/momentum_results.md")


if __name__ == "__main__":
    main()
