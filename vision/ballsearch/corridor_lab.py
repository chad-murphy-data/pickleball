"""Pose-corridor re-search experiment (owner-approved 2026-09-01).

Between consecutive contacts the ball flies one arc from paddle A to
paddle B. Build a per-frame search window around the straight-line
time interpolation of the two paddle endpoints, run an AGGRESSIVE
motion detector inside it only, and score against the owner's
ball-path clicks (V/S rows; I/N excluded).

Arms:
  prod   = contact times the pipeline has today (app-union claim_bounds)
  oracle = hand-timestamped taps (segmentation ceiling; diagnostic)

Metrics per rally x arm (R = 12 px unless noted):
  cover    truth click falls inside its corridor window
  any      some detector candidate within R of truth
  top1     candidate nearest window center within R of truth
  interp   window center alone within R (the prior-answers-it null)
  decode   current position stream has a point within R (what we add to)
  added    top1 hit where decode missed
  displ    displaced-window arm: candidate-found rate (hallucination)

No training anywhere; detector = frame differencing + connected
components. Endpoints chosen by decode proximity, never by truth.

Usage: python3 corridor_lab.py <rally> [--rth 0.5] [--thr 14]
"""
import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br                       # noqa: E402
from claim_lab import (load, paddle_series, paddle_at,  # noqa: E402
                       paddle_series as _ps)
from approach_lab import approach_events          # noqa: E402

CLIPS = {9: "r9_clip.mp4", 10: "r10_clip.mp4", 7: "r7_clip.mp4",
         6: "r6_clip.mp4"}
SP = Path(__file__).parent
AREA_MIN, AREA_MAX = 3, 600
RADII = (8, 12, 20)
R_MAIN = 12


def load_truth(rally):
    """(t, x, y, vis) for V/S rows only."""
    out = []
    with open(f"/home/user/pickleball/data/vision/ball_path_r{rally}"
              ".csv") as f:
        for row in csv.DictReader(f):
            if row["vis"] not in ("V", "S"):
                continue
            out.append((float(row["t_s"]), float(row["x"]),
                        float(row["y"]), row["vis"]))
    return out


def ref_at(c, t):
    ts = np.array([p[0] for p in c["timing_ref"]])
    xs = np.array([p[1] for p in c["timing_ref"]])
    ys = np.array([p[2] for p in c["timing_ref"]])
    return (float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)))


def prod_contacts(c, series, rth):
    evs = [e for e in approach_events(c, series, rth)
           if c["serve"] - 0.3 <= e[0] < c["end"]]
    anc, zs = list(c["anchors"]), list(c["zs"])
    for t, rel, tid in evs:
        p = paddle_at(series, tid, t, tol=0.12)
        if p is None:
            continue
        anc.append((t, tid, p[0], p[1], p[0], p[1]))
        zs.append(3.0 - rel)
    dd = br.dedupe_anchors(anc, zs, br.track_sides(c["floors"]),
                           c["turns"])
    return sorted(br.claim_bounds(c["turns"], c["angs"],
                                  c["timing_ref"], dd))


def endpoint(c, series, t):
    """paddle nearest the decode at t; decode point if no paddle."""
    ref = ref_at(c, t)
    best = None
    for tid in series:
        p = paddle_at(series, tid, t, tol=0.12)
        if p is None:
            continue
        d = float(np.hypot(p[0] - ref[0], p[1] - ref[1]))
        if best is None or d < best[0]:
            best = (d, (float(p[0]), float(p[1])))
    return best[1] if best else ref


def corridors(c, series, times):
    """[(ta, tb, A(x,y), B(x,y), wx, wy)] between consecutive
    contacts, plus the final flight to rally end."""
    ts = sorted(t for t in times if c["serve"] - 0.2 <= t <= c["end"])
    ts = ts + [c["end"]]
    out = []
    for ta, tb in zip(ts[:-1], ts[1:]):
        if tb - ta < 0.08:
            continue
        A, B = endpoint(c, series, ta), endpoint(c, series, tb)
        L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
        wx = min(140.0, 40.0 + 0.20 * L)
        wy = min(170.0, 55.0 + 0.30 * L)
        out.append((ta, tb, A, B, wx, wy))
    return out


def window_at(cor, t):
    ta, tb, A, B, wx, wy = cor
    a = (t - ta) / (tb - ta)
    cx = A[0] + a * (B[0] - A[0])
    cy = A[1] + a * (B[1] - A[1])
    return cx, cy, wx, wy


def detect(gray_m2, gray_0, gray_p2, cx, cy, wx, wy, thr):
    """candidates [(x, y, area, peak)] inside the window."""
    H, W = gray_0.shape
    x0, x1 = int(max(0, cx - wx)), int(min(W, cx + wx))
    y0, y1 = int(max(0, cy - wy)), int(min(H, cy + wy))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    a = cv2.absdiff(gray_0[y0:y1, x0:x1], gray_m2[y0:y1, x0:x1])
    b = cv2.absdiff(gray_0[y0:y1, x0:x1], gray_p2[y0:y1, x0:x1])
    motion = cv2.min(a, b)
    _, mask = cv2.threshold(motion, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (AREA_MIN <= area <= AREA_MAX):
            continue
        x, y = cent[i]
        peak = float(motion[lab == i].max())
        out.append((x0 + x, y0 + y, int(area), peak))
    return out


def sweep(rally, arms, thr):
    """one pass over the clip, evaluating every scheduled click."""
    c = arms["_c"]
    truth = arms["_truth"]
    rng = np.random.default_rng(20260901)
    cap = cv2.VideoCapture(str(SP / CLIPS[rally]))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # schedule: clip frame -> [(arm, truth_i, window, displaced_window)]
    sched = {}
    stats = {name: [] for name in arms if not name.startswith("_")}
    for i, (t, tx, ty, vis) in enumerate(truth):
        f = int(round((t - c["t0"]) * 60))
        for name, cors in arms.items():
            if name.startswith("_"):
                continue
            cor = next((co for co in cors if co[0] <= t <= co[1]), None)
            if cor is None:
                stats[name].append(dict(i=i, vis=vis, covered=False))
                continue
            cx, cy, wx, wy = window_at(cor, t)
            dx = float(rng.uniform(180, 260)) * (1 if cx < W / 2 else -1)
            dy = float(rng.uniform(90, 150)) * (1 if cy < H / 2 else -1)
            rec = dict(i=i, vis=vis, covered=abs(tx - cx) <= wx
                       and abs(ty - cy) <= wy, cx=cx, cy=cy)
            sched.setdefault(f, []).append(
                (name, rec, (cx, cy, wx, wy),
                 (cx + dx, cy + dy, wx, wy), (tx, ty)))
            stats[name].append(rec)
    buf = deque(maxlen=5)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += 1
        mid = fi - 3            # center of the 5-frame buffer
        if len(buf) == 5 and mid in sched:
            g_m2, g_0, g_p2 = buf[0], buf[2], buf[4]
            for name, rec, w, wd, (tx, ty) in sched[mid]:
                cands = detect(g_m2, g_0, g_p2, *w, thr)
                rec["ncand"] = len(cands)
                for R in RADII:
                    rec[f"any{R}"] = any(
                        np.hypot(x - tx, y - ty) <= R
                        for x, y, *_ in cands)
                if cands:
                    x, y, *_ = min(cands, key=lambda cd: np.hypot(
                        cd[0] - w[0], cd[1] - w[1]))
                    rec["top1"] = float(np.hypot(x - tx, y - ty))
                dc = detect(g_m2, g_0, g_p2, *wd, thr)
                rec["disp_found"] = bool(dc)
    cap.release()
    return stats


def decode_recall(c, truth, R=R_MAIN):
    vs = [(c["t0"] + f / 60.0, x, y) for f, x, y in c["visited"]]
    hit = []
    for t, tx, ty, vis in truth:
        near = [np.hypot(x - tx, y - ty) for vt, x, y in vs
                if abs(vt - t) <= 0.025]
        hit.append(bool(near) and min(near) <= R)
    return hit


def interp_hits(stats_arm, truth, R=R_MAIN):
    out = []
    for rec, (t, tx, ty, vis) in zip(stats_arm, truth):
        out.append("cx" in rec and
                   float(np.hypot(rec["cx"] - tx, rec["cy"] - ty)) <= R)
    return out


def report(name, recs, truth, dec, R=R_MAIN):
    itp = interp_hits(recs, truth, R)
    n = len(recs)
    cov = sum(1 for r in recs if r.get("covered"))
    ev = [r for r in recs if "ncand" in r]
    anyR = sum(1 for r in ev if r.get(f"any{R}"))
    top1 = sum(1 for r in ev if r.get("top1", 1e9) <= R)
    disp = sum(1 for r in ev if r.get("disp_found"))
    add = sum(1 for r, d in zip(recs, dec)
              if r.get("top1", 1e9) <= R and not d)
    nmiss = sum(1 for d in dec if not d)
    addany = sum(1 for r, d in zip(recs, dec)
                 if r.get(f"any{R}") and not d)
    print(f"  {name:7s} cover {cov}/{n}  any@{R} {anyR}/{len(ev)}"
          f"  top1@{R} {top1}/{len(ev)}  interp@{R} {sum(itp)}/{n}"
          f"  decode@{R} {sum(dec)}/{n}  ADDED {add}"
          f"  any-in-decode-holes {addany}/{nmiss}"
          f"  displ-found {disp}/{len(ev)}"
          f"  cands/eval {np.mean([r['ncand'] for r in ev]):.1f}")
    for vis in ("V", "S"):
        sub = [(r, d) for r, d, tr in zip(recs, dec, truth)
               if tr[3] == vis]
        e2 = [r for r, _ in sub if "ncand" in r]
        t2 = sum(1 for r in e2 if r.get("top1", 1e9) <= R)
        a2 = sum(1 for r, d in sub
                 if r.get("top1", 1e9) <= R and not d)
        print(f"    [{vis}] top1 {t2}/{len(e2)}  "
              f"decode {sum(d for _, d in sub)}/{len(sub)}  added {a2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--rth", type=float, default=0.5)
    ap.add_argument("--thr", type=int, default=14)
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    truth = load_truth(a.rally)
    print(f"rally {a.rally}: {len(truth)} V/S truth clicks, "
          f"thr={a.thr}")
    arms = {"_c": c, "_truth": truth}
    arms["prod"] = corridors(c, series, prod_contacts(c, series, a.rth))
    arms["oracle"] = corridors(c, series, list(c["imps"]))
    print(f"  corridors: prod {len(arms['prod'])}, "
          f"oracle {len(arms['oracle'])}")
    stats = sweep(a.rally, arms, a.thr)
    dec = decode_recall(c, truth)
    print(f"  decode baseline @12: {sum(dec)}/{len(dec)}")
    for name in ("prod", "oracle"):
        report(name, stats[name], truth, dec)


if __name__ == "__main__":
    main()
