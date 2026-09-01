"""check_clip — verify an owner-cut rally clip against the COMMITTED
candidate CSV before any cache is built from it (2026-09-01, written
for the r10 re-cut: r10_clip.mp4 was never staged in Drive, so the
owner re-cuts it from full_match.mp4 and this proves the re-cut is the
same frames the sealed candidate CSV was extracted from).

Three checks, all against data/vision/ball_candidates_r{N}.csv.gz
(frame, t_s, x, y, area, score; frame = 0-based clip index; the
extractor emits nothing at frame 0 and the last frame, so the clip
must hold max(frame) + 2 frames):
  1. container: frame count == max(frame)+2, 1280x720, ~60 fps
  2. offset:    t_s(frame k) - k/fps == the --offset you will pass to
                ball_candidates.py / that ball_decoder derives t0 from
  3. alignment: re-run the extractor (vision/ball_candidates.py, pure
                classical motion, deterministic) on the new clip and
                match the committed strong candidates (score >= 60)
                against the recomputed ones at frame shift d in
                -3..+3, within 3 px.  A correct cut peaks at d = 0 with
                a high match rate; an off-by-n cut peaks at d = n; a
                different source (other VOD, other scaling) matches
                nothing at any d.
Re-encoding changes pixel noise, so the d = 0 rate need not be 100%;
the discriminating fact is that d = 0 dominates every other shift.
Validated on the staged r9 clip (the clip its CSV came from): d = 0
matches 1.000, d = +-1 0.56, +-2 0.40, +-3 0.31 -> PASS.

Usage: python3 check_clip.py <rally> <clip.mp4>
Exit 0 = PASS (all three), 1 = FAIL, with the numbers printed.
"""
import csv
import gzip
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
from ball_candidates import candidates_for_clip  # noqa: E402

DATA = Path("/home/user/pickleball/data/vision")
STRONG = 60.0
R_MATCH = 3.0
SHIFTS = range(-3, 4)


def load_committed(rally):
    with gzip.open(DATA / f"ball_candidates_r{rally}.csv.gz", "rt") as f:
        rd = csv.DictReader(f)
        rows = [(int(r["frame"]), float(r["t_s"]), float(r["x"]),
                 float(r["y"]), float(r["score"])) for r in rd]
    return rows


def probe(clip):
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, w, h, fps


def main():
    rally, clip = int(sys.argv[1]), sys.argv[2]
    rows = load_committed(rally)
    fmax = max(r[0] for r in rows)
    want_n = fmax + 2
    ok = True

    n, w, h, fps = probe(clip)
    c1 = (n == want_n and (w, h) == (1280, 720) and abs(fps - 60) < 0.5)
    print(f"[1] container: {n} frames (want {want_n}), {w}x{h} "
          f"(want 1280x720), {fps:.3f} fps (want 60)  ->  "
          f"{'ok' if c1 else 'FAIL'}")
    ok &= c1

    offs = sorted({round(t - f / 60.0, 2) for f, t, *_ in rows})
    off = np.median([t - f / 60.0 for f, t, *_ in rows])
    print(f"[2] offset: committed CSV implies clip frame 0 at VOD "
          f"{off:.3f} s (span of per-row estimates {offs[0]}..{offs[-1]});"
          f" cut with  ffmpeg -ss {off:.2f}  and pass --offset {off:.2f}")

    print("[3] alignment: re-running the extractor on the clip ...")
    out, fps2, n2 = candidates_for_clip(clip, off)
    new = {}
    for f, t, x, y, area, sc in out:
        new.setdefault(f, []).append((x, y))
    new = {f: np.array(v) for f, v in new.items()}
    strong = [(f, x, y) for f, t, x, y, sc in rows if sc >= STRONG]
    print(f"    committed strong candidates (score>={STRONG:.0f}): "
          f"{len(strong)} over {len({f for f, *_ in strong})} frames")
    rates = {}
    for d in SHIFTS:
        hit = 0
        for f, x, y in strong:
            arr = new.get(f + d)
            if arr is not None and np.hypot(arr[:, 0] - x,
                                            arr[:, 1] - y).min() <= R_MATCH:
                hit += 1
        rates[d] = hit / max(1, len(strong))
    best = max(rates, key=rates.get)
    line = "  ".join(f"d={d:+d}: {rates[d]:.3f}" for d in SHIFTS)
    print(f"    match rate within {R_MATCH:.0f} px by frame shift:  {line}")
    others = max(v for d, v in rates.items() if d != 0)
    # slow candidates persist across adjacent frames, so d=+-1 sits
    # near 0.56 even on the IDENTICAL clip (r9: 1.000 / 0.559 / 0.402
    # / 0.314 at d = 0 / 1 / 2 / 3); the test is a clear margin at 0.
    c3 = best == 0 and rates[0] >= 0.5 and rates[0] - others >= 0.15
    print(f"    best shift d={best:+d} at {rates[best]:.3f}; next-best "
          f"non-zero {others:.3f}  ->  {'ok' if c3 else 'FAIL'}")
    ok &= c3

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
