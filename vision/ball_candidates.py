"""Can a DUMB classical detector propose the ball? (probe, 2026-08-20)

User question: could a local model — or just a regular classifier —
replicate what the VLM did? Decomposing the 93%-recall localization
result honestly, the ball was the primary cue in nearly every window
and posture was corroboration. So the pipeline that matters is
classical:

    motion-differenced candidates -> trajectory linking -> direction
    change = contact -> nearest player = hitter -> gap = pace

Every step after the first is arithmetic we can already write. This
script probes ONLY the first step, because it is the one that could
fail outright: at each frame where a ball position was marked and
USER-VERIFIED, does a plain 3-frame motion difference put the ball in
the top-K candidates? No training, no weights, no labels at inference.

Decisive in the negative: if the ball is not in the candidate set, no
tracker built on this can find it, and the classical route is dead
without a learned detector. If it IS there, the remaining work is
geometry.

Ground truth = the 25 circles the user confirmed correct
(vision/vlm_ball_calls.py), minus the two they rejected. Positions are
recovered from cell fractions through the crop the grids were cut with,
so this measures against hand-checked pixels, not against my memory.

    python3 vision/ball_candidates.py --video full_match.mp4.webm
    python3 vision/ball_candidates.py --selftest
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from swing_probe import ffmpeg_bin

# the crop vlm_localize_sample.py used: w,h,x,y as fractions
CROP_W, CROP_H, CROP_X, CROP_Y = 0.70, 0.85, 0.15, 0.10
STEP_S = 0.15
DIFF_DT = 1.0 / 30.0     # frame spacing for the 3-frame difference
TOL_PX = 22              # a candidate counts as the ball within this
KS = (1, 3, 5, 10, 25)

# user-VERIFIED positions only (w01 cells 7 and 9 rejected: occlusion
# and a shoe). (window, cell): (x_frac, y_frac) within the cell.
VERIFIED = {
 (1, 1): (0.573, 0.244), (1, 2): (0.508, 0.264), (1, 3): (0.433, 0.286),
 (1, 4): (0.363, 0.361), (1, 5): (0.298, 0.422), (1, 6): (0.271, 0.486),
 (13, 1): (0.443, 0.361), (13, 4): (0.401, 0.181), (13, 5): (0.422, 0.181),
 (13, 8): (0.370, 0.208),
 (24, 1): (0.401, 0.089), (24, 4): (0.258, 0.486), (24, 5): (0.206, 0.583),
 (24, 6): (0.223, 0.444), (24, 7): (0.290, 0.278), (24, 9): (0.385, 0.125),
 (27, 1): (0.548, 0.436), (27, 2): (0.599, 0.561), (27, 3): (0.718, 0.694),
 (27, 4): (0.706, 0.583), (27, 5): (0.687, 0.411), (27, 6): (0.679, 0.292),
 (27, 7): (0.658, 0.208), (27, 8): (0.613, 0.236), (27, 9): (0.672, 0.194),
}


def cell_to_frame(xf, yf, W, H):
    """Cell fraction -> full-frame pixel, through the grid's crop."""
    return ((CROP_X + xf * CROP_W) * W, (CROP_Y + yf * CROP_H) * H)


def cell_time(t0, cell):
    return t0 + (cell - 1) * STEP_S


def candidates(prev, cur, nxt, min_a=2, max_a=120):
    """Small things that moved between BOTH neighbours and are bright.
    Deliberately dumb: no colour model, no shape prior, no learning.
    Motion-differencing is what kills the shoe false-positive class —
    a shoe travels with its player, the ball does not."""
    d = np.minimum(np.abs(cur.astype(np.int16) - prev),
                   np.abs(cur.astype(np.int16) - nxt)).max(axis=2)
    m = (d > 28).astype(np.uint8)
    import cv2
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (min_a <= a <= max_a) or max(w, h) > 26:
            continue
        if min(w, h) / max(max(w, h), 1) < 0.25:      # reject slivers
            continue
        x, y = cent[i]
        # score: bright, compact, strongly-moving
        patch = d[max(0, int(y) - 3):int(y) + 4, max(0, int(x) - 3):int(x) + 4]
        out.append((float(patch.mean()), x, y, a))
    out.sort(reverse=True)
    return out


def grab(video, t, W):
    import cv2
    p = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{max(t, 0):.3f}",
         "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "-"], capture_output=True)
    if not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def selftest():
    W, H = 1920, 1080
    x, y = cell_to_frame(0.5, 0.5, W, H)
    assert abs(x - (0.15 + 0.35) * W) < 1e-6, x
    assert abs(y - (0.10 + 0.425) * H) < 1e-6, y
    assert cell_to_frame(0, 0, W, H) == (0.15 * W, 0.10 * H)
    assert abs(cell_time(10.0, 1) - 10.0) < 1e-9
    assert abs(cell_time(10.0, 9) - 11.2) < 1e-9
    # candidate finder: a synthetic 6px "ball" that moves, on a static
    # background with a static bright blob (the "shoe") that must NOT fire
    import cv2
    def frame(bx):
        im = np.full((300, 400, 3), 40, np.uint8)
        cv2.circle(im, (120, 200), 7, (230, 230, 230), -1)   # static shoe
        cv2.circle(im, (bx, 100), 3, (60, 240, 240), -1)     # moving ball
        return im
    cs = candidates(frame(150), frame(180), frame(210))
    assert cs, "no candidates on a clean synthetic"
    top = cs[:5]
    assert any(abs(c[1] - 180) < 8 and abs(c[2] - 100) < 8 for c in top), \
        f"ball not in top-5: {top}"
    assert not any(abs(c[1] - 120) < 10 and abs(c[2] - 200) < 10
                   for c in cs), "static shoe fired — motion gate broken"
    print("selftest: crop maths, cell timing, ball found, shoe rejected OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--key", default="vlm_loc_key.csv",
                    help="the localization ANSWER_KEY_LOC.csv (for t0)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video:
        raise SystemExit("--video required (or --selftest)")
    import cv2

    t0 = {}
    for r in csv.DictReader(open(a.key)):
        t0[int(r["window"].split(".")[0][1:])] = float(r["t0_s"])

    probe = grab(a.video, 60.0, None)
    if probe is None:
        raise SystemExit("could not decode a frame")
    H, W = probe.shape[:2]
    print(f"video {W}x{H}; probing {len(VERIFIED)} user-verified ball "
          f"positions\n")

    hits = {k: 0 for k in KS}
    ncand, n = [], 0
    for (win, cell), (xf, yf) in sorted(VERIFIED.items()):
        if win not in t0:
            continue
        t = cell_time(t0[win], cell)
        cur = grab(a.video, t, W)
        prv = grab(a.video, t - DIFF_DT, W)
        nxt = grab(a.video, t + DIFF_DT, W)
        if any(f is None for f in (cur, prv, nxt)):
            print(f"  w{win:02d}c{cell}: decode failed")
            continue
        gx, gy = cell_to_frame(xf, yf, W, H)
        cs = candidates(prv, cur, nxt)
        ncand.append(len(cs))
        rank = next((i for i, c in enumerate(cs)
                     if abs(c[1] - gx) <= TOL_PX and abs(c[2] - gy) <= TOL_PX),
                    None)
        n += 1
        for k in KS:
            hits[k] += rank is not None and rank < k
        print(f"  w{win:02d}c{cell}: {len(cs):>3} candidates, ball at rank "
              + ("MISS" if rank is None else f"{rank + 1}"))

    print(f"\nballs in top-K of a dumb motion-difference detector "
          f"(n={n}):")
    for k in KS:
        print(f"   top-{k:<3} {hits[k]}/{n} = {hits[k]/max(n,1):.0%}")
    if ncand:
        ncand.sort()
        print(f"median candidates per frame: {ncand[len(ncand)//2]}")
    print("\nREADING: high top-25 recall = the ball IS in the candidate "
          "set and\nthe rest is tracking geometry. Low = the classical "
          "route needs a\nlearned detector, and this probe says so "
          "cheaply.")


if __name__ == "__main__":
    main()
