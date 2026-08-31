"""Hitter-first chain — the missile's other half (exploration, not gate).

From the pose npz ALONE (no labels, no track assigns): per-track
box-normalized wrist speed and arm reach, baseline-normalized per
track, combined into a per-frame "someone is hitting" excitement
signal; peak-picking yields PREDICTED CONTACTS — (time, wrist pixel
position of the most excited track). These are the hitter-side
anchors the recursive ball<->hitter loop feeds to the ball decoder
(and the ball path's turns feed back).

Verified feature basis: wrist speed held 74%/85% and reach 63%/74%
on 84 untouched contacts (2026-08-30 verification); fakes are faster
than hits on speed alone, so predicted contacts are ANCHOR HINTS
with a confidence, never hard constraints — the decoder prices them,
the ball evidence can veto them. That asymmetry is the EM lesson
(soft beliefs, not verdicts).

Scoring (train 6/7 tuning; rally 1 = tracker-unseen demo check):
recall/precision of predicted contacts vs the manual taps at
±0.15 s, and anchor position error vs the user's ball click at the
matched contact (ball-at-contact is near the striking wrist).

Usage:
    python3 vision/hitter_chain.py --npz pose/r0006.npz --rally 6
    python3 vision/hitter_chain.py --npz pose/r0001.npz --rally 1 --dump anchors.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ball_audit import load_impacts  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"

LSHO, RSHO, LWRI, RWRI = 5, 6, 9, 10
CONF = 0.3
SMOOTH_S = 0.10        # excitement smoothing
MIN_SEP_S = 0.35       # min separation between predicted contacts
Z_MIN = 1.2            # excitement threshold (per-track z units)
JUNK_AREA = 1280 * 720 * 0.20


def track_signals(z, tid):
    m = np.where(z["track"] == tid)[0]
    t, k, c = z["t"][m], z["kpt"][m], z["kpc"][m]
    box = z["box"][m]
    h = np.maximum(box[:, 3] - box[:, 1], 20.0)
    big = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) > JUNK_AREA
    n = len(m)
    speed = np.full(n, np.nan)
    wx = np.full(n, np.nan)
    wy = np.full(n, np.nan)
    for i in range(n):
        if big[i]:
            continue
        best = None
        for w in (LWRI, RWRI):
            if c[i, w] > CONF:
                if best is None or c[i, w] > best[0]:
                    best = (c[i, w], k[i, w, 0], k[i, w, 1])
        if best:
            wx[i], wy[i] = best[1], best[2]
        if i and not big[i - 1]:
            dt = t[i] - t[i - 1]
            if 0 < dt < 0.1:
                vals = []
                for w in (LWRI, RWRI):
                    if c[i, w] > CONF and c[i - 1, w] > CONF:
                        vals.append(np.linalg.norm(k[i, w] - k[i - 1, w])
                                    / h[i] / dt)
                if vals:
                    speed[i] = max(vals)
    reach = np.full(n, np.nan)
    for i in range(n):
        if big[i]:
            continue
        vals = []
        for w, s in ((LWRI, LSHO), (RWRI, RSHO)):
            if c[i, w] > CONF and c[i, s] > CONF:
                vals.append(np.linalg.norm(k[i, w] - k[i, s]) / h[i])
        if vals:
            reach[i] = max(vals)
    return t, speed, reach, wx, wy


def zn(x):
    v = x[~np.isnan(x)]
    if len(v) < 10:
        return np.full_like(x, np.nan)
    return (x - np.median(v)) / (np.std(v) + 1e-9)


def predict_contacts(npz_path, t_lo=None, t_hi=None):
    z = np.load(npz_path)
    # main tracks = 4 longest (automated; no identities needed)
    tids = sorted(set(z["track"].tolist()),
                  key=lambda k: -(z["track"] == k).sum())[:4]
    per = []
    for tid in tids:
        t, sp, re, wx, wy = track_signals(z, tid)
        exc = np.nanmax(np.vstack([zn(sp), zn(re)]), axis=0)
        # smooth in time
        order = np.argsort(t)
        t, exc, wx, wy = t[order], exc[order], wx[order], wy[order]
        sm = np.copy(exc)
        for i in range(len(t)):
            m = np.abs(t - t[i]) <= SMOOTH_S
            v = exc[m]
            v = v[~np.isnan(v)]
            sm[i] = v.mean() if len(v) else np.nan
        per.append((tid, t, sm, wx, wy))
    # global excitement = max over tracks; peak-pick
    events = []
    allt = np.unique(np.concatenate([p[1] for p in per]))
    if t_lo is not None:
        allt = allt[(allt >= t_lo) & (allt <= t_hi)]
    for tq in allt:
        best = None
        for tid, t, sm, wx, wy in per:
            i = np.argmin(np.abs(t - tq))
            if abs(t[i] - tq) > 0.03 or np.isnan(sm[i]):
                continue
            if best is None or sm[i] > best[0]:
                best = (sm[i], tid, wx[i], wy[i])
        if best and best[0] >= Z_MIN and not np.isnan(best[2]):
            events.append((float(tq), float(best[0]), int(best[1]),
                           float(best[2]), float(best[3])))
    # greedy peak-pick with min separation
    picked = []
    for ev in sorted(events, key=lambda e: -e[1]):
        if all(abs(ev[0] - p[0]) >= MIN_SEP_S for p in picked):
            picked.append(ev)
    return sorted(picked)


def score(picked, rally):
    imps, dead = load_impacts(rally=rally)
    hit, used = 0, set()
    errs = []
    balls = {}
    for r in csv.DictReader(open(DATA / f"ball_path_r{rally}.csv")):
        if r["x"]:
            balls[round(float(r["t_s"]) * 30)] = (float(r["x"]), float(r["y"]))
    for t0 in imps:
        m = [(abs(t0 - e[0]), i) for i, e in enumerate(picked)
             if i not in used and abs(t0 - e[0]) <= 0.15]
        if m:
            d, i = min(m)
            used.add(i)
            hit += 1
            b = balls.get(round(picked[i][0] * 30)) or balls.get(round(t0 * 30))
            if b:
                errs.append(math.hypot(picked[i][3] - b[0],
                                       picked[i][4] - b[1]))
    prec_n = sum(1 for e in picked if imps[0] - 0.2 <= e[0] <= dead)
    print(f"  rally {rally}: predicted {len(picked)} contacts "
          f"({prec_n} in-rally); recall {hit}/{len(imps)} "
          f"({100*hit/len(imps):.0f}%) at ±0.15s; "
          f"precision {100*hit/max(prec_n,1):.0f}%; "
          f"anchor→ball-click median {np.median(errs):.0f}px"
          if errs else "  no matched anchors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--dump")
    a = ap.parse_args()
    z = np.load(a.npz)
    picked = predict_contacts(a.npz, float(z["t"].min()), float(z["t"].max()))
    if (DATA / f"ball_path_r{a.rally}.csv").exists():
        score(picked, a.rally)
    else:
        print(f"  rally {a.rally}: predicted {len(picked)} contacts "
              "(no ground truth available)")
    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "excitement_z", "track", "wrist_x", "wrist_y"])
            w.writerows([(round(e[0], 3), round(e[1], 2), e[2],
                          round(e[3], 1), round(e[4], 1)) for e in picked])
        print(f"  dumped {a.dump}")


if __name__ == "__main__":
    main()
