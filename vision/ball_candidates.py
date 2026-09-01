"""Ball candidate stage — the first gated component of ball_gate.md.

BUILD LICENSE: ball_gate.md FROZEN 2026-08-31, sealed rally-8 ball
pass committed first (PR #101). This stage runs on TRAIN rallies
only during development; per the gate, NO tracker component touches
rally 8's clip until the graded run.

Recall over precision, per the gate: multiple or zero candidates per
frame, the decoder does the choosing. Method — pure classical motion:

  for frame i, d = min(|f_i - f_{i-L}|, |f_i - f_{i+L}|) over gray
  frames (L = 0.1 s at the clip fps; the min kills both-direction
  ghosting), threshold, dilate, connected components, size-filtered
  (ball ~5-12 px at 720p, smears elongated), top-K by peak diff.

Inputs: a rally clip (user-cut, known VOD offset) — nothing labeled.
Output: data/vision/ball_candidates_r{N}.csv.gz
  (frame, t_s [VOD clock], x, y, area, score) — multiple rows/frame.

Diagnostic (train only): candidate RECALL vs the user's ball pass —
fraction of V frames with a candidate within 25 px (S: 40 px),
matched within +/-1 video frame (browser-vs-decoder seek jitter,
measured on the alignment check 2026-08-31). This number decides
whether the decoder has anything to decode; it is a development
readout, not a gate check.

Usage:
    python3 vision/ball_candidates.py --clip r6_clip.mp4 --offset 144.80 --rally 6
    python3 vision/ball_candidates.py ... --score   # + recall vs train labels
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"

LAG_S = 0.10          # motion-diff lag
THRESH = 18           # gray abs-diff threshold
MIN_AREA, MAX_AREA = 4, 600
TOP_K = 40
SEALED = {10}        # gate holdout (--score refuses; extraction allowed);
                     # r10 sealed 2026-08-31 (the re-grade rally); r8/r9 train


def candidates_for_clip(clip, offset):
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    lag = max(1, round(LAG_S * fps))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    n = len(frames)
    out = []
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(n):
        a = frames[max(0, i - lag)]
        b = frames[min(n - 1, i + lag)]
        d = cv2.min(cv2.absdiff(frames[i], a), cv2.absdiff(frames[i], b))
        _, m = cv2.threshold(d, THRESH, 255, cv2.THRESH_BINARY)
        m = cv2.dilate(m, kern)
        ncc, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        cands = []
        for c in range(1, ncc):
            area = stats[c, cv2.CC_STAT_AREA]
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            x, y = cent[c]
            peak = float(d[lab == c].max())
            cands.append((peak, float(x), float(y), int(area)))
        cands.sort(reverse=True)
        t = offset + i / fps
        for peak, x, y, area in cands[:TOP_K]:
            out.append((i, round(t, 3), round(x, 1), round(y, 1),
                        area, round(peak, 1)))
    return out, fps, n


def write_out(rows, path):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x", "y", "area", "score"])
        w.writerows(rows)


def score_recall(rows, rally, fps):
    """Train-only diagnostic: candidate within tolerance of the user's
    click, matched within +/-1 video frame."""
    byf = {}
    for fr, t, x, y, area, sc in rows:
        byf.setdefault(fr, []).append((x, y))
    labels = [r for r in csv.DictReader(
        open(DATA / f"ball_path_r{rally}.csv")) if r["x"]]
    stats = {}
    for cls, tol in (("V", 25.0), ("S", 40.0)):
        hit = tot = 0
        offset_t0 = rows[0][1] - rows[0][0] / fps
        for r in labels:
            if r["vis"] != cls:
                continue
            tot += 1
            f0 = round((float(r["t_s"]) - offset_t0) * fps)
            cx, cy = float(r["x"]), float(r["y"])
            found = any(
                np.hypot(x - cx, y - cy) <= tol
                for df in (-1, 0, 1)
                for x, y in byf.get(f0 + df, ()))
            hit += int(found)
        stats[cls] = (hit, tot)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--offset", type=float, required=True,
                    help="VOD time of clip frame 0 (the ffmpeg -ss value)")
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--score", action="store_true",
                    help="train-only: recall vs the user's ball pass")
    ap.add_argument("--graded-run", action="store_true",
                    help="unlock a SEALED rally for the one graded run")
    a = ap.parse_args()
    if a.rally in SEALED and not a.graded_run:
        raise SystemExit(f"rally {a.rally} is the gate's SEALED holdout — "
                         "refusing (pass --graded-run only for the one "
                         "graded run, after readiness passes)")
    if a.rally in SEALED and a.score:
        raise SystemExit("scoring against the sealed ball pass happens in "
                         "the grading harness, not here")
    rows, fps, n = candidates_for_clip(a.clip, a.offset)
    out = DATA / f"ball_candidates_r{a.rally}.csv.gz"
    write_out(rows, out)
    per = len(rows) / max(n, 1)
    print(f"rally {a.rally}: {n} frames @ {fps:.0f} fps -> "
          f"{len(rows)} candidates ({per:.1f}/frame) -> {out}")
    if a.score:
        st = score_recall(rows, a.rally, fps)
        for cls, (hit, tot) in st.items():
            pc = 100 * hit / tot if tot else 0
            print(f"  {cls}-frame candidate recall: {pc:.1f}% ({hit}/{tot})")


if __name__ == "__main__":
    main()
