"""Corridor v2 — the owner's heuristic, mechanized (2026-09-01):
"start at the paddle that just hit the ball and scan a path from
there at frame n+1, n+2, n+3. If I lost it, figure out the next
contact and do t-1, t-2, t-3."

Pass 1: full-frame motion candidates once per frame (thr identical to
v1; area 3-600; top 400/frame by peak).
Pass 2: per corridor (contact A -> contact B), a FORWARD chain from
paddle A and a BACKWARD chain from paddle B, each propagating
frame-by-frame with velocity prediction, coasting through <=4 misses
with a growing window; merged meet-in-the-middle (near your own
anchor you are trusted).

Seeding at a contact: candidates within R1 px of the paddle at the
first step; each seed probed 5 frames; best-supported seed wins.

Scored vs the owner's V/S clicks; null = same chains from anchors
displaced ~200 px; decode baseline = current position stream.

Usage: python3 corridor_chain.py <rally> [--thr 14] [--rth 0.5]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, paddle_series          # noqa: E402
from corridor_lab import (CLIPS, SP, load_truth, prod_contacts,  # noqa
                          corridors, decode_recall, R_MAIN)

AREA_MIN, AREA_MAX = 3, 600
R1 = 90.0            # seed search radius around the anchor paddle
PROBE = 5            # seed probe depth (frames)
MAX_MISS = 4
CAND_CAP = 400


PEAK_CAP = 600
K3 = np.ones((3, 3), np.uint8)
K7 = np.ones((7, 7), np.uint8)
K9 = np.ones((9, 9), np.uint8)


def frame_candidates(rally, f_lo, f_hi, thr, mode="cc"):
    """mode 'cc' = production connected components (area 3-600,
    centroid per component). mode 'peak' = the blob-decomposition +
    appearance emitter (owner-specced 2026-09-01): every ball-scale
    local max of the motion image is its own candidate — no area cap,
    no centroid summarization — ranked by hand-built ball appearance:
    small-scale brightness (white tophat: bright blob OR thin streak
    both survive) + yellowness ((R+G)/2 - B) + motion. No training."""
    cap = cv2.VideoCapture(str(SP / CLIPS[rally]))
    from collections import deque
    buf = deque(maxlen=5)
    cbuf = deque(maxlen=5)
    cands = {}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cbuf.append(frame)
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += 1
        mid = fi - 3
        if len(buf) == 5 and f_lo <= mid <= f_hi:
            a = cv2.absdiff(buf[2], buf[0])
            b = cv2.absdiff(buf[2], buf[4])
            motion = cv2.min(a, b)
            if mode == "peak":
                mx = cv2.dilate(motion, K7)
                pk = (motion >= mx) & (motion >= thr)
                ys, xs = np.nonzero(pk)
                cs = []
                if len(xs):
                    th = cv2.morphologyEx(buf[2], cv2.MORPH_TOPHAT, K9)
                    col = cbuf[2].astype(np.int16)
                    yel = np.clip((col[..., 2] + col[..., 1]) // 2
                                  - col[..., 0], 0, 255).astype(np.uint8)
                    th3 = cv2.dilate(th, K3)
                    yel3 = cv2.dilate(yel, K3)
                    sc = (th3[ys, xs].astype(float)
                          + 0.5 * yel3[ys, xs].astype(float)
                          + 0.3 * motion[ys, xs].astype(float))
                    keep = np.argsort(-sc)[:PEAK_CAP]
                    cs = [(float(xs[i]), float(ys[i]), 1,
                           float(motion[ys[i], xs[i]])) for i in keep]
                cands[mid] = cs
            else:
                _, mask = cv2.threshold(motion, thr, 255,
                                        cv2.THRESH_BINARY)
                mask = cv2.dilate(mask, K3)
                n, lab, stats, cent = \
                    cv2.connectedComponentsWithStats(mask)
                cs = []
                for i in range(1, n):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if not (AREA_MIN <= area <= AREA_MAX):
                        continue
                    x, y = cent[i]
                    peak = float(motion[lab == i].max())
                    cs.append((float(x), float(y), int(area), peak))
                cs.sort(key=lambda c: -c[3])
                cands[mid] = cs[:CAND_CAP]
        if fi > f_hi + 3:
            break
    cap.release()
    return cands


def propagate(cands, f0, p0, v0, f_stop, step, max_steps=None):
    pos = np.array(p0, float)
    vel = np.array(v0, float)
    miss, out, f, n = 0, {}, f0, 0
    out[f0] = (float(p0[0]), float(p0[1]))
    while True:
        f += step
        if (step > 0 and f > f_stop) or (step < 0 and f < f_stop):
            break
        n += 1
        if max_steps and n > max_steps:
            break
        pred = pos + vel
        speed = float(np.hypot(*vel))
        r = min(70.0, 26.0 + 13.0 * miss + 0.35 * speed)
        best = None
        for (x, y, area, peak) in cands.get(f, ()):
            d = float(np.hypot(x - pred[0], y - pred[1]))
            if d <= r and (best is None or d < best[0]):
                best = (d, x, y)
        if best is None:
            miss += 1
            pos = pred
            if miss > MAX_MISS:
                break
        else:
            _, x, y = best
            newv = np.array([x, y]) - pos
            vel = 0.65 * newv + 0.35 * vel
            pos = np.array([x, y])
            miss = 0
            out[f] = (float(x), float(y))
    return out


def chain_from(cands, f_anchor, p_anchor, f_stop, step):
    """seed by beam probe at the first step, then run to f_stop."""
    seeds = []
    for df in (1, 2):
        fs = f_anchor + df * step
        for (x, y, area, peak) in cands.get(fs, ()):
            if np.hypot(x - p_anchor[0], y - p_anchor[1]) <= R1:
                seeds.append((fs, x, y, peak))
        if seeds:
            break
    best, best_score = None, -1.0
    for fs, x, y, peak in seeds[:12]:
        v0 = (np.array([x, y]) - p_anchor) / max(1, abs(fs - f_anchor))
        probe = propagate(cands, fs, (x, y), v0, f_stop, step,
                          max_steps=PROBE)
        fl = max(probe) if step > 0 else min(probe)
        dep = float(np.hypot(probe[fl][0] - p_anchor[0],
                             probe[fl][1] - p_anchor[1]))
        if dep < 14.0:          # arm hover — the ball departs
            continue
        score = len(probe) * 10.0 + 0.6 * dep + 0.01 * peak
        if score > best_score:
            best_score, best = score, (fs, (x, y), v0)
    track = {f_anchor: (float(p_anchor[0]), float(p_anchor[1]))}
    if best is None:
        return track
    fs, ps, v0 = best
    track.update(propagate(cands, fs, ps, v0, f_stop, step))
    return track


def merged_tracks(cands, cors, t0, disp=None):
    """dict frame -> (x, y); meet-in-the-middle merge per corridor."""
    track = {}
    for (ta, tb, A, B, wx, wy) in cors:
        fa, fb = int(round((ta - t0) * 60)), int(round((tb - t0) * 60))
        if fb - fa < 3:
            continue
        A2 = (A[0] + disp[0], A[1] + disp[1]) if disp else A
        B2 = (B[0] + disp[0], B[1] + disp[1]) if disp else B
        fw = chain_from(cands, fa, A2, fb, +1)
        bw = chain_from(cands, fb, B2, fa, -1)
        for f in range(fa, fb + 1):
            has_f, has_b = f in fw, f in bw
            if not (has_f or has_b):
                continue
            if has_f and has_b:
                pf, pb = fw[f], bw[f]
                if np.hypot(pf[0] - pb[0], pf[1] - pb[1]) <= 25:
                    track[f] = ((pf[0] + pb[0]) / 2,
                                (pf[1] + pb[1]) / 2, True)
                else:
                    p = pf if (f - fa) <= (fb - f) else pb
                    track[f] = (p[0], p[1], False)
            else:
                p = fw[f] if has_f else bw[f]
                track[f] = (p[0], p[1], False)
    return track


def score(track, truth, t0, dec, tag):
    n = len(truth)
    hits = {R: 0 for R in (8, 12, 20)}
    added = have = 0
    ag_n = ag_hit = ag_add = 0
    for (t, tx, ty, vis), d in zip(truth, dec):
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        if p is None:
            continue
        have += 1
        dd = float(np.hypot(p[0] - tx, p[1] - ty))
        for R in hits:
            hits[R] += dd <= R
        if dd <= R_MAIN and not d:
            added += 1
        if p[2]:                     # fw/bw agreement point
            ag_n += 1
            ag_hit += dd <= R_MAIN
            ag_add += dd <= R_MAIN and not d
    nag = sum(1 for p in track.values() if p[2])
    print(f"  {tag:12s} trackpts {len(track):4d}  at-click {have}/{n}"
          f"  r@8 {hits[8]}  r@12 {hits[12]}  r@20 {hits[20]}"
          f"  ADDED@12 {added}  | agree {nag}pts"
          f" prec {ag_hit}/{ag_n} add {ag_add}")
    return hits[12], added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=14)
    ap.add_argument("--rth", type=float, default=0.5)
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    truth = load_truth(a.rally)
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cands = frame_candidates(a.rally, f_lo, f_hi, a.thr)
    mc = float(np.mean([len(v) for v in cands.values()]))
    dec = decode_recall(c, truth)
    print(f"rally {a.rally}: {len(truth)} V/S clicks, "
          f"{len(cands)} frames, {mc:.0f} cands/frame, "
          f"decode@12 {sum(dec)}/{len(dec)}")
    rng = np.random.default_rng(20260901)
    for name, times in (("prod", prod_contacts(c, series, a.rth)),
                        ("oracle", list(c["imps"]))):
        cors = corridors(c, series, times)
        tr = merged_tracks(cands, cors, t0)
        score(tr, truth, t0, dec, name)
        for vis in ("V", "S"):
            tt = [x for x in truth if x[3] == vis]
            dd = [d for x, d in zip(truth, dec) if x[3] == vis]
            score(tr, tt, t0, dd, f"  [{vis}]")
        for k in range(2):
            d = (float(rng.uniform(160, 240)) * rng.choice([-1, 1]),
                 float(rng.uniform(80, 140)) * rng.choice([-1, 1]))
            trn = merged_tracks(cands, cors, t0, disp=d)
            score(trn, truth, t0, dec, f"{name}-null{k}")


if __name__ == "__main__":
    main()
