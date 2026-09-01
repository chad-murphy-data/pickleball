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
LELB, RELB = 7, 8
CONF = 0.3
EXT_LAM = 0.5          # paddle point = wrist + EXT_LAM*(wrist-elbow).
                       # Measured 2026-08-31 vs contact ball clicks:
                       # wrist 20px median (18 on occluded contacts),
                       # extension 15.0/13.7px, plateau lam 0.5-0.6.
                       # Dumped as paddle_x/y NEXT TO wrist_x/y: the
                       # decoder's anchor flags stay on the WRIST
                       # (60px radius covers the paddle anyway; feeding
                       # it the paddle point instead LOST V — r6
                       # 71.3->69.7, r1 77.7->75.2 — the wrist region
                       # also covers the incoming flight). The paddle
                       # point is for contact-POSITION consumers (3D
                       # priors, replay), not decoder flags.
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
    pxa = np.full(n, np.nan)
    pya = np.full(n, np.nan)
    for i in range(n):
        if big[i]:
            continue
        best = None
        for w, e in ((LWRI, LELB), (RWRI, RELB)):
            if c[i, w] > CONF:
                if best is None or c[i, w] > best[0]:
                    px, py = k[i, w, 0], k[i, w, 1]
                    ex, ey = px, py
                    if c[i, e] > CONF:
                        vx, vy = px - k[i, e, 0], py - k[i, e, 1]
                        if math.hypot(vx, vy) > 5:
                            ex, ey = px + EXT_LAM * vx, py + EXT_LAM * vy
                    best = (c[i, w], px, py, ex, ey)
        if best:
            wx[i], wy[i] = best[1], best[2]
            pxa[i], pya[i] = best[3], best[4]
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
    return t, speed, reach, wx, wy, pxa, pya


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
        t, sp, re, wx, wy, pxa, pya = track_signals(z, tid)
        exc = np.nanmax(np.vstack([zn(sp), zn(re)]), axis=0)
        # smooth in time
        order = np.argsort(t)
        t, exc, wx, wy = t[order], exc[order], wx[order], wy[order]
        pxa, pya = pxa[order], pya[order]
        sm = np.copy(exc)
        for i in range(len(t)):
            m = np.abs(t - t[i]) <= SMOOTH_S
            v = exc[m]
            v = v[~np.isnan(v)]
            sm[i] = v.mean() if len(v) else np.nan
        per.append((tid, t, sm, wx, wy, pxa, pya))
    # global excitement = max over tracks; peak-pick
    events = []
    allt = np.unique(np.concatenate([p[1] for p in per]))
    if t_lo is not None:
        allt = allt[(allt >= t_lo) & (allt <= t_hi)]
    for tq in allt:
        best = None
        for tid, t, sm, wx, wy, pxa, pya in per:
            i = np.argmin(np.abs(t - tq))
            if abs(t[i] - tq) > 0.03 or np.isnan(sm[i]):
                continue
            if best is None or sm[i] > best[0]:
                best = (sm[i], tid, wx[i], wy[i], pxa[i], pya[i])
        if best and best[0] >= Z_MIN and not np.isnan(best[2]):
            events.append((float(tq), float(best[0]), int(best[1]),
                           float(best[2]), float(best[3]),
                           float(best[4]), float(best[5])))
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


def blur_gap_fill(npz_path, clip, offset, picked,
                  radius=70, lag=6, thresh=18, z_min=1.2, min_sep=0.35):
    """User idea 2026-08-31: a swinging paddle SMEARS — per-track
    motion-diff mass near the wrist is a second contact channel.
    Standalone it is weak (fires only on hard swings: 24%/57%/22%
    recall on r1/r6/r7 vs the pose channel's 72/57/67) but it is
    COMPLEMENTARY — on r6 it caught the 148.57 smash the pose channel
    missed outright, the contact whose absence broke check 3's
    segmentation. So it runs as GAP-FILL only: blur peaks further
    than min_sep from every pose anchor are appended. Measured safe:
    decoder V/turns identical on r1/r6/r7, r7's check-3 PASS
    unchanged, with the r6 bound recovered."""
    import cv2
    z = np.load(npz_path)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    diffs = []
    for i in range(len(frames)):
        a = frames[max(0, i - lag)]
        b = frames[min(len(frames) - 1, i + lag)]
        diffs.append((cv2.min(cv2.absdiff(frames[i], a),
                              cv2.absdiff(frames[i], b)) > thresh)
                     .astype(np.uint8))
    tids = sorted(set(z["track"].tolist()),
                  key=lambda k: -(z["track"] == k).sum())[:4]
    sig = {}
    for tid in tids:
        m = np.where(z["track"] == tid)[0]
        t, kpt, kpc = z["t"][m], z["kpt"][m], z["kpc"][m]
        rows = []
        for i in range(len(m)):
            ws = [(kpt[i, w], kpt[i, e] if kpc[i, e] > CONF else None)
                  for w, e in ((LWRI, LELB), (RWRI, RELB))
                  if kpc[i, w] > CONF]
            if not ws:
                continue
            fi = round((float(t[i]) - offset) * fps)
            if not (0 <= fi < len(frames)):
                continue
            best = (0, None, None)
            for w, el in ws:
                x0 = max(0, int(w[0] - radius))
                x1 = min(diffs[fi].shape[1], int(w[0] + radius))
                y0 = max(0, int(w[1] - radius))
                y1 = min(diffs[fi].shape[0], int(w[1] + radius))
                e = int(diffs[fi][y0:y1, x0:x1].sum())
                if e > best[0]:
                    best = (e, w, el)
            rows.append((float(t[i]), best[0], best[1], best[2]))
        if len(rows) < 20:
            continue
        ts = np.array([r[0] for r in rows])
        es = np.array([r[1] for r in rows], float)
        zz = (es - np.median(es)) / (es.std() + 1e-9)
        sm = np.copy(zz)
        for i in range(len(ts)):
            mm = np.abs(ts - ts[i]) <= SMOOTH_S
            sm[i] = zz[mm].mean()
        sig[tid] = (ts, sm, rows)
    if not sig:
        return []
    events = []
    for tq in np.unique(np.concatenate([s[0] for s in sig.values()])):
        best = None
        for tid, (ts, sm, rows) in sig.items():
            i = int(np.argmin(np.abs(ts - tq)))
            if abs(ts[i] - tq) > 0.03:
                continue
            if best is None or sm[i] > best[0]:
                w, el = rows[i][2], rows[i][3]
                if w is None:
                    continue    # no measurable wrist on this row (r10
                                # crash 2026-09-01): it cannot nominate
                                # a blur event
                px, py = w
                if el is not None:
                    vx, vy = w[0] - el[0], w[1] - el[1]
                    if math.hypot(vx, vy) > 5:
                        px, py = w[0] + EXT_LAM * vx, w[1] + EXT_LAM * vy
                best = (float(sm[i]), tid, float(w[0]), float(w[1]),
                        float(px), float(py))
        if best is not None and best[0] >= z_min:
            events.append((float(tq),) + best)
    peaks = []
    for ev in sorted(events, key=lambda e: -e[1]):
        if all(abs(ev[0] - p[0]) >= min_sep for p in peaks):
            peaks.append(ev)
    pose_t = [e[0] for e in picked]
    out = []
    for (tv, zv, tid, wx, wy, px, py) in sorted(peaks):
        if all(abs(tv - pt) >= min_sep for pt in pose_t):
            out.append((tv, zv, int(tid), wx, wy, px, py))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--dump")
    ap.add_argument("--clip", help="rally clip: adds the BLUR gap-fill "
                                   "channel (see blur_gap_fill)")
    ap.add_argument("--offset", type=float,
                    help="VOD time of clip frame 0 (required with --clip)")
    a = ap.parse_args()
    z = np.load(a.npz)
    picked = predict_contacts(a.npz, float(z["t"].min()), float(z["t"].max()))
    if a.clip:
        if a.offset is None:
            raise SystemExit("--clip needs --offset")
        extra = blur_gap_fill(a.npz, a.clip, a.offset, picked)
        print(f"  blur gap-fill: +{len(extra)} anchors")
        picked = sorted(picked + extra)
    if (DATA / f"ball_path_r{a.rally}.csv").exists():
        score(picked, a.rally)
    else:
        print(f"  rally {a.rally}: predicted {len(picked)} contacts "
              "(no ground truth available)")
    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "excitement_z", "track", "wrist_x",
                        "wrist_y", "paddle_x", "paddle_y"])
            w.writerows([(round(e[0], 3), round(e[1], 2), e[2],
                          round(e[3], 1), round(e[4], 1),
                          round(e[5], 1), round(e[6], 1)) for e in picked])
        print(f"  dumped {a.dump}")


if __name__ == "__main__":
    main()
