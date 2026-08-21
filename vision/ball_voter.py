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


def serve_flight_vote(*_a, **_k):
    """RETIRED 2026-08-21 — its premise was falsified by real video.

    It assumed the FIRST tracked flight was the serve. On rally 1 the
    serve is at 10.24s and the first flight begins at 11.60s: the serve
    flight is simply not acquired, so this would have named whoever
    happened to be near an unrelated later flight. Use ball_at_contact,
    which matches flights to the contact times we already know."""
    raise NotImplementedError(
        "serve_flight_vote assumed flight[0] is the serve; real video "
        "disproved it (rally 1 serve 10.24s, first flight 11.60s). "
        "Use ball_at_contact(segs, t_contact).")


def dedupe(segs, t_tol=0.15, xy_tol=120.0):
    """Collapse flights that are the same ball tracked twice.

    MEASURED ON REAL VIDEO (2026-08-21, rally 1): track_all returned
    11.60->12.37 and 11.60->12.43, and again at 12.47, 24.60 and 32.90 —
    near-identical start times and endpoints. Left in, each duplicate
    votes again and turns one observation into a false majority."""
    out = []
    for seg in sorted(segs):
        ta, pa, tb, pb, n = seg
        dup = False
        for i, (ta2, pa2, tb2, pb2, n2) in enumerate(out):
            if abs(ta - ta2) <= t_tol and \
                    abs(pa[0] - pa2[0]) + abs(pa[1] - pa2[1]) <= xy_tol:
                dup = True
                if n > n2:          # keep the better-observed copy
                    out[i] = seg
                break
        if not dup:
            out.append(seg)
    return out


def ball_at_contact(segs, t_contact, tol=0.30):
    """Where the ball was when the contact at t_contact happened.

    THE FIX FOR THE SERVE-FLIGHT ASSUMPTION. This module first assumed
    the FIRST flight was the serve; real video says otherwise — rally
    1's serve is at 10.24s and the first tracked flight starts at
    11.60s, so the serve flight was simply not acquired. But every
    contact time is already known, so flights can be matched to
    contacts instead of counted from the start:

      a flight STARTING at t_contact  -> the ball is LEAVING that
                                         hitter, so its origin is at
                                         the hitter
      a flight ENDING at t_contact    -> the ball is ARRIVING at that
                                         hitter, so its endpoint is

    Departure is preferred over arrival: the outgoing point is struck
    at the hitter, whereas an incoming track can be lost early and end
    short of them. Returns (point, kind) or (None, None) — abstention
    when no flight is near, which is most of the value here."""
    best = None
    for ta, pa, tb, pb, _n in segs:
        for t_e, pt, kind, rank in ((ta, pa, "leaves", 0),
                                    (tb, pb, "arrives", 1)):
            d = abs(t_e - t_contact)
            if d > tol:
                continue
            key = (rank, d)
            if best is None or key < best[0]:
                best = (key, pt, kind)
    return (best[1], best[2]) if best else (None, None)


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
    # DEDUPE: the real duplicate shape from rally 1
    segs_d = [(11.60, (1148.0, 598.0), 12.37, (1019.0, 444.0), 24),
              (11.60, (1155.0, 607.0), 12.43, (968.0, 409.0), 26),
              (13.10, (596.0, 123.0), 13.93, (838.0, 330.0), 26)]
    dd = dedupe(segs_d)
    assert len(dd) == 2, dd
    assert dd[0][4] == 26, "must keep the better-observed copy"

    # BALL AT CONTACT — run on the DEDUPED list, which is the intended
    # order: on raw input the duplicate pair ties exactly and the winner
    # is arbitrary, which is precisely why dedupe comes first.
    pt, kind = ball_at_contact(dd, 11.62)
    assert kind == "leaves" and pt == (1155.0, 607.0), (pt, kind)
    pt2, kind2 = ball_at_contact(dd, 12.41)
    assert kind2 == "arrives", (pt2, kind2)
    # rally 1's real serve at 10.24s has no flight — the case that broke
    # the original "first flight is the serve" design. Must abstain.
    pt3, kind3 = ball_at_contact(dd, 10.24)
    assert pt3 is None and kind3 is None, (pt3, kind3)

    print("selftest OK: nearest_track picks, margins, abstention on a "
          "tie, rejection of a far point, duplicate collapse, and "
          "contact-matched ball points")


def coverage(video, t0, t1, contacts, tol=0.30):
    """How many known contacts have a ball flight at them.

    THE GATE THIS CHANNEL HAS TO PASS BEFORE IT VOTES. A dense-looking
    flight list means nothing if the flights sit between the contacts
    we care about — rally 1 produced 27 flights and still missed the
    serve. Prints per-contact hit/miss so the coverage is read rather
    than assumed."""
    segs = dedupe(flight_segments(video, t0, t1))
    hits = []
    for t in contacts:
        pt, kind = ball_at_contact(segs, t, tol)
        hits.append((t, pt, kind))
    n_hit = sum(1 for _t, pt, _k in hits if pt is not None)
    return segs, hits, n_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--rally", type=int,
                    help="rally_cum: score ball coverage against that "
                         "rally's hand-labelled contact times")
    ap.add_argument("--labels", default="contact_labels_chicago0725.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.video:
        return selftest()
    if a.rally:
        import csv
        cs = sorted(float(r["t_refined_s"] or r["t_tap_s"])
                    for r in csv.DictReader(open(a.labels))
                    if int(r["rally_cum"]) == a.rally
                    and r.get("contact", "1") == "1")
        if not cs:
            raise SystemExit(f"no labelled contacts for rally {a.rally}")
        t0 = a.t0 if a.t0 is not None else cs[0] - 2.0
        t1 = a.t1 if a.t1 is not None else cs[-1] + 2.0
        segs, hits, n_hit = coverage(a.video, t0, t1, cs)
        print(f"rally {a.rally}: {len(cs)} labelled contacts, "
              f"{len(segs)} flights after dedupe\n")
        for t, pt, kind in hits:
            where = f"{kind:<8} ({pt[0]:5.0f},{pt[1]:5.0f})" if pt \
                else "-- no flight within 0.30s --"
            print(f"  contact {t:7.2f}s   {where}")
        print(f"\nBALL COVERAGE {n_hit}/{len(cs)} = {n_hit / len(cs):.0%}"
              f" of contacts have a flight endpoint.\n  This is the "
              f"channel's ceiling: it can only vote where it fires, and "
              f"abstains elsewhere.")
        return
    segs = dedupe(flight_segments(a.video, a.t0, a.t1))
    print(f"{len(segs)} ball flights (deduped) in [{a.t0}, {a.t1}]")
    for ta, pa, tb, pb, n in segs:
        print(f"  {ta:7.2f}s ({pa[0]:6.0f},{pa[1]:6.0f}) -> "
              f"{tb:7.2f}s ({pb[0]:6.0f},{pb[1]:6.0f})   "
              f"{n} observed points, {tb - ta:.2f}s")


if __name__ == "__main__":
    main()
