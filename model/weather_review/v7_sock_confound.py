"""V7 follow-up: is Jack Sock's nominally-significant wind slope a TIME
artifact?  v2 ratings are CURRENT FORM applied retroactively, so any player
whose true form trended while the tour's venue mix trended in wind gets a
spurious wind slope.  Control for date (and for tour) and see what survives.

Also: how much does the whole 552-player z-scan move under the date control
(is Sock special, or does everyone shift)?

Deterministic; read-only.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model/weather_review"))
from sitelib.race import sigmoid, team_eta  # noqa: E402
from v7_sock_verify import load, norm_cdf  # noqa: E402


def ols_multi(X, y):
    """Plain OLS via normal equations (small k)."""
    k = len(X[0])
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    # gaussian elimination with inverse
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
         for i, row in enumerate(XtX)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(k):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    inv = [row[k:] for row in A]
    beta = [sum(inv[i][j] * Xty[j] for j in range(k)) for i in range(k)]
    n = len(y)
    resid = [yy - sum(b * r[i] for i, b in enumerate(beta)) for r, yy in zip(X, y)]
    s2 = sum(e * e for e in resid) / (n - k)
    se = [math.sqrt(s2 * inv[i][i]) for i in range(k)]
    return beta, se, resid, inv


def main():
    players, names, n_games = load()
    pid = next(p for p, x in names.items() if x == "Jack Sock")
    rows = players[pid]
    d0 = date(2024, 1, 1)
    days = [(date(*map(int, r[4][:10].split("-"))) - d0).days / 365.25 for r in rows]
    w = [r[0] for r in rows]
    y = [r[1] for r in rows]
    n = len(y)
    print(f"Jack Sock n={n}, date span {min(r[4] for r in rows)} .. "
          f"{max(r[4] for r in rows)}")

    b1, se1, _, _ = ols_multi([[1.0, wi] for wi in w], y)
    print(f"  wind only        : {b1[1]*10:+.4f} /10mph  se {se1[1]*10:.4f}  "
          f"t {b1[1]/se1[1]:+.2f}  p1 {1-norm_cdf(b1[1]/se1[1]):.4f}")

    b2, se2, _, _ = ols_multi([[1.0, wi, d] for wi, d in zip(w, days)], y)
    print(f"  + linear date    : {b2[1]*10:+.4f} /10mph  se {se2[1]*10:.4f}  "
          f"t {b2[1]/se2[1]:+.2f}  p1 {1-norm_cdf(b2[1]/se2[1]):.4f}   "
          f"(date {b2[2]:+.4f}/yr, t {b2[2]/se2[2]:+.2f})")

    b3, se3, _, _ = ols_multi(
        [[1.0, wi, d, d * d] for wi, d in zip(w, days)], y)
    print(f"  + quadratic date : {b3[1]*10:+.4f} /10mph  se {se3[1]*10:.4f}  "
          f"t {b3[1]/se3[1]:+.2f}  p1 {1-norm_cdf(b3[1]/se3[1]):.4f}")

    # year fixed effects
    yrs = sorted({r[4][:4] for r in rows})
    print(f"  years: {yrs}")
    X = []
    for wi, r in zip(w, rows):
        row = [1.0, wi] + [1.0 if r[4][:4] == yy else 0.0 for yy in yrs[1:]]
        X.append(row)
    b4, se4, _, _ = ols_multi(X, y)
    print(f"  + year FE        : {b4[1]*10:+.4f} /10mph  se {se4[1]*10:.4f}  "
          f"t {b4[1]/se4[1]:+.2f}  p1 {1-norm_cdf(b4[1]/se4[1]):.4f}")

    # within-event demeaning: only within-event wind variation (match hour)
    ev = defaultdict(list)
    for i, r in enumerate(rows):
        ev[r[2]].append(i)
    Xd, yd, kept = [], [], 0
    for k, idx in ev.items():
        if len(idx) < 2:
            continue
        mw = sum(w[i] for i in idx) / len(idx)
        my = sum(y[i] for i in idx) / len(idx)
        if max(w[i] for i in idx) - min(w[i] for i in idx) < 1e-9:
            continue
        kept += len(idx)
        for i in idx:
            Xd.append([w[i] - mw])
            yd.append(y[i] - my)
    if Xd:
        num = sum(r[0] * yy for r, yy in zip(Xd, yd))
        den = sum(r[0] ** 2 for r in Xd)
        b = num / den
        res = [yy - b * r[0] for r, yy in zip(Xd, yd)]
        s2 = sum(e * e for e in res) / (len(yd) - 1 - len(ev))
        se = math.sqrt(s2 / den)
        print(f"  WITHIN-EVENT only: {b*10:+.4f} /10mph  se {se*10:.4f}  "
              f"t {b/se:+.2f}   (n={kept} games, {len(ev)} events)")

    # how much between-event wind variation is there for him?
    print("\n  variance decomposition of his wind exposure:")
    mw = sum(w) / n
    tot = sum((x - mw) ** 2 for x in w)
    within = 0.0
    for k, idx in ev.items():
        m = sum(w[i] for i in idx) / len(idx)
        within += sum((w[i] - m) ** 2 for i in idx)
    print(f"    total {tot:.0f}, within-event {within:.0f} "
          f"({100*within/tot:.0f}%), between-event {100*(1-within/tot):.0f}%")

    # ---- population: how many players flip nominal significance with date ctrl
    print("\n=== population scan: wind-only t vs date-controlled t ===")
    flips = 0
    n_sig_a = n_sig_b = 0
    ts = []
    for p, rws in players.items():
        wi = [r[0] for r in rws]
        yy = [r[1] for r in rws]
        dd = [(date(*map(int, r[4][:10].split("-"))) - d0).days / 365.25
              for r in rws]
        try:
            ba, sa, _, _ = ols_multi([[1.0, x] for x in wi], yy)
            bb, sb, _, _ = ols_multi([[1.0, x, d] for x, d in zip(wi, dd)], yy)
        except ZeroDivisionError:
            continue
        ta, tb = ba[1] / sa[1], bb[1] / sb[1]
        ts.append((p, ta, tb))
        n_sig_a += abs(ta) > 1.96
        n_sig_b += abs(tb) > 1.96
        flips += (abs(ta) > 1.96) != (abs(tb) > 1.96)
    print(f"  players: {len(ts)}; |t|>1.96 wind-only {n_sig_a}, "
          f"date-controlled {n_sig_b}; flips {flips}")
    sock = next(x for x in ts if x[0] == pid)
    print(f"  Sock: t {sock[1]:+.2f} -> {sock[2]:+.2f}")
    rank_a = 1 + sum(1 for _, ta, _ in ts if abs(ta) > abs(sock[1]))
    rank_b = 1 + sum(1 for _, _, tb in ts if abs(tb) > abs(sock[2]))
    print(f"  Sock |t| rank: {rank_a} -> {rank_b} of {len(ts)}")


if __name__ == "__main__":
    main()
