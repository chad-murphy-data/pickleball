"""The ball as a voter on WHO — not as a contact detector.

THE USER'S REFRAMING (2026-08-21), and it is the better question.

The ball was closed as a channel because it is findable in only ~64% of
in-play frames, and ball_track.py was measured as a CONTACT-TIMING
instrument and set aside. Neither verdict applies here, for two
reasons:

  1 WE DO NOT NEED THE BALL AT CONTACT. Asking "who is the ball nearest
    when it is struck" targets the single worst instant — fastest,
    most motion-blurred, and occluded by the very player swinging at
    it. Asking instead where the ball FLIES BETWEEN contacts targets
    the easy interval: free flight, unoccluded, against open court.
    One flight yields TWO attributions (who it left, who it reaches),
    and needs the ball anywhere in the interval rather than at a
    specific frame.

  2 WE DO NOT NEED TIMING. The contacts are already placed (the pose
    decoder finds a detection within 0.35s of every true contact) and
    the sides are already exact (alternation). What is still in doubt
    is one binding per side: which of two players is the log's server,
    and which is its receiver. The ball's trajectory answers exactly
    that and nothing else.

ball_track.py's own design already says this is the right cut:
track_all returns non-overlapping flight SEGMENTS precisely because
"a ball is most likely to be occluded exactly AT a contact... the
contact becomes the JOIN between consecutive segments, which is what
the physics actually shows (ball vanishes into the player, reappears
going the other way)". Segments ARE inter-contact flights. This module
reuses that survivor for a job it was never measured at, and never
asks it for a timestamp.

HONEST ABSTENTION IS THE POINT. On the ~36% of frames where the ball
is unfindable this returns None and the voter stays silent, which the
cascade already handles — and this thread measured that silence beats
guessing for a trusted voter (widening the contact voter to fire more
cost 5 points of geometry).

    python3 ball_voter.py --selftest
    python3 ball_voter.py --video full_match.mp4.webm --t0 8.8 --t1 20.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_FPS = 30.0
DEFAULT_WIDTH = 1280
MIN_SEEN = 4          # a flight needs this many OBSERVED points to vote


def flight_segments(video, t0, t1, fps=DEFAULT_FPS, width=DEFAULT_WIDTH):
    """[(t_start, (x,y), t_end, (x,y), n_seen)] for each ball flight.

    Only OBSERVED points bound a segment. ball_track.py flags coasted
    stretches because "a coasted stretch is linear BY CONSTRUCTION";
    a flight whose ends are both coasted is an extrapolation, not an
    observation, and must not be allowed to name anybody."""
    from ball_candidates import candidates
    from ball_track import track_all
    from swing_probe import decode_window

    frames, cand = [], []
    for fr in decode_window(video, t0, t1 - t0, fps, width):
        frames.append(fr)
        if len(frames) == 3:
            cand.append(candidates(frames[0], frames[1], frames[2]))
            frames.pop(0)
    if not cand:
        return []
    out = []
    for tr in track_all(cand, fps):
        seen = [(f, x, y) for f, x, y, ok in tr if ok]
        if len(seen) < MIN_SEEN:
            continue
        f0, x0, y0 = seen[0]
        f1, x1, y1 = seen[-1]
        # candidate index i was built from the middle of frames i..i+2
        out.append((t0 + (f0 + 1.0) / fps, (x0, y0),
                    t0 + (f1 + 1.0) / fps, (x1, y1), len(seen)))
    return sorted(out)


def nearest_track(rd, tids, t, pt, scale=1.0, box_at=None,
                  max_dist=None):
    """Which of `tids` is closest to image point `pt` at time t.

    Returns (tid, dist, margin) where margin is how much closer the
    winner is than the runner-up — the caller uses it to abstain on a
    near-tie rather than break it with noise."""
    best = []
    for tid in tids:
        c = box_at(rd["tracks"][tid], t)
        if c is None:
            continue
        d = ((c[0] - pt[0] * scale) ** 2 +
             (c[1] - pt[1] * scale) ** 2) ** 0.5
        best.append((d, tid))
    if len(best) < 2:
        return None, None, None
    best.sort()
    (d0, t0_), (d1, _t1) = best[0], best[1]
    if max_dist is not None and d0 > max_dist:
        return None, d0, None
    return t0_, d0, d1 - d0


def serve_flight_vote(rd, tids_serving, tids_receiving, video, t0, t1,
                      box_at, scale=1.0, min_margin=20.0):
    """(server_tid, receiver_tid) from the first ball flight, or Nones.

    The serve flight leaves the SERVER and arrives at the RECEIVER, and
    the referee log names both people — so one trajectory votes on both
    disputed bindings at once. Abstains (None) whenever the flight is
    missing, too short, or the two candidates are within min_margin of
    the endpoint, because a coin-flip vote from a trusted channel is
    worse than no vote."""
    segs = flight_segments(video, t0, t1)
    if not segs:
        return None, None
    ta, pa, tb, pb, _n = segs[0]
    srv, _d, m_s = nearest_track(rd, tids_serving, ta, pa, scale,
                                 box_at=box_at)
    rcv, _d2, m_r = nearest_track(rd, tids_receiving, tb, pb, scale,
                                  box_at=box_at)
    if m_s is not None and m_s < min_margin:
        srv = None
    if m_r is not None and m_r < min_margin:
        rcv = None
    return srv, rcv


def selftest():
    class _S(dict):
        pass

    def mk(cx, y):
        ts = [i * 0.1 for i in range(20)]
        z = _S(t=ts, cx=[cx] * 20, ynorm=[y] * 20)
        z["side"] = 0
        return z

    def box_at(ser, t):
        i = min(range(len(ser["t"])),
                key=lambda k: abs(ser["t"][k] - t))
        return float(ser["cx"][i]), float(ser["ynorm"][i])

    rd = {"tracks": {1: mk(300.0, 700.0), 2: mk(900.0, 700.0),
                     3: mk(300.0, 200.0), 4: mk(900.0, 200.0)}}
    # a point right on top of track 2 must name track 2, with a big
    # margin over track 1
    tid, d, margin = nearest_track(rd, [1, 2], 0.5, (900.0, 700.0),
                                   box_at=box_at)
    assert tid == 2 and d < 1.0 and margin > 500, (tid, d, margin)
    # dead centre between the two must produce ~zero margin, which is
    # what the caller abstains on
    tid_m, _d, margin_m = nearest_track(rd, [1, 2], 0.5, (600.0, 700.0),
                                        box_at=box_at)
    assert margin_m is not None and margin_m < 1.0, margin_m
    # fewer than two candidates cannot vote
    assert nearest_track(rd, [1], 0.5, (300.0, 700.0),
                         box_at=box_at)[0] is None
    # max_dist rejects a point nowhere near anybody
    assert nearest_track(rd, [1, 2], 0.5, (600.0, 5000.0),
                         box_at=box_at, max_dist=100.0)[0] is None
    print("selftest OK: nearest_track picks, margins, abstention on a "
          "tie, and rejection of a far point")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.video:
        return selftest()
    segs = flight_segments(a.video, a.t0, a.t1)
    print(f"{len(segs)} ball flights in [{a.t0}, {a.t1}]")
    for ta, pa, tb, pb, n in segs:
        print(f"  {ta:7.2f}s ({pa[0]:6.0f},{pa[1]:6.0f}) -> "
              f"{tb:7.2f}s ({pb[0]:6.0f},{pb[1]:6.0f})   "
              f"{n} observed points, {tb - ta:.2f}s")


if __name__ == "__main__":
    main()
