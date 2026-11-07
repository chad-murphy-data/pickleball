"""Frozen gate scorers, shared by every stack that wants a gate verdict.

Extracted verbatim from ball_grade.py (2026-09-03) so that the graded
battery and any challenger stack score through ONE implementation.
The logic is frozen by ball_gate.md Scoring; do not change it here.
Extraction was verified by re-running ball_grade.py on a train rally
and confirming identical CHECK 1 output.

CHECK 1 (ball_gate.md Scoring 1) — PATH, V-frame hit rate:
  fraction of the owner's visible-ball (V) frames on the gate panel
  where the predicted position is within 25 px of the click. S (smear)
  frames scored at 40 px, reported separately. I and N frames are
  NEVER scored (the auto-label poison lesson: a claim where no human
  can see the ball is unfalsifiable).
  Bars: PASS >= 70% V, FAIL < 40% V.
"""
from __future__ import annotations

import math

TOL = (("V", 25.0), ("S", 40.0))
BAR_PASS = 70.0
BAR_FAIL = 40.0


def check1(per_frame, labels, imps, t0, fps=60.0):
    """Score a position stream against the owner's click path.

    per_frame  {frame_index: (t, x, y)} — the stack's position stream.
    labels     rows of data/vision/ball_path_r{N}.csv (x, y, t_s, vis).
    imps       contact times; imps[0]..imps[-1]+0.5 is the gate panel.
    t0         clip time origin, so frame = round((t - t0) * fps).

    Returns a dict per class with hit rate, hits, total, and the CLAIM
    count (scored frames where the stream predicted anything at all).
    Claims are diagnostic only and change no frozen quantity: they
    separate a coverage failure from a placement failure, since an
    unclaimed frame is charged as a miss by the rule above.
    """
    p_lo, p_hi = imps[0], imps[-1] + 0.5
    out = {}
    for cls, tol in TOL:
        hit = tot = claimed = 0
        for r in labels:
            if not r["x"] or r["vis"] != cls:
                continue
            tt = float(r["t_s"])
            if not (p_lo <= tt <= p_hi):
                continue
            tot += 1
            f0 = round((tt - t0) * fps)
            best = min((math.hypot(per_frame[g][1] - float(r["x"]),
                                   per_frame[g][2] - float(r["y"]))
                        for g in (f0 - 1, f0, f0 + 1) if g in per_frame),
                       default=1e9)
            claimed += int(best < 1e9)
            hit += int(best <= tol)
        out[cls] = dict(rate=100 * hit / max(tot, 1), hit=hit, tot=tot,
                        claimed=claimed)
    v = out["V"]["rate"]
    out["pass"] = v >= BAR_PASS
    out["fail"] = v < BAR_FAIL
    return out
