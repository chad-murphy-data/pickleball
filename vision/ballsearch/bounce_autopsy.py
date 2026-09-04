"""Why are 21 of 34 human bounces missing?  One bucket per miss.

The owner's rule (2026-09-04) is the frame: *"every pair of contacts has
to make a call -- bounce or not, AND ONLY ONE call."*  A second bounce
ends the rally, so between two consecutive contacts there is exactly 0 or
1 bounce.  That is a hard constraint, not a prior, and the fitter already
honours it: `court3d.fit_segment` returns one `kind` and one `bounce_xy`
per segment, and segments run bound-to-bound.

So the rule does not change the fitter -- it turns the recall number into
a diagnostic.  A tracked segment whose span swallows a MISSED contact
covers two real flights while still being allowed only one bounce: any
second bounce in there is structurally unreachable, no matter how good
the ball detector gets.  This script counts how many of the 21 are that,
versus how many are the fitter or the ball.

Buckets, in the order they are tested (first one that applies wins):

  NO WINDOW   the human bounce falls outside the tracked bounds entirely
  CAPPED      its segment also holds another human bounce that WAS called
              -- one flight's worth of answer for two flights of ball
  NO SEG      segment is None: fewer than 5 tracked points in the span
  NOT OK      segment fit, failed the plausibility test
  CALLED ARC  segment ok, fitter looked and said no bounce
  WRONG TIME  called a bounce, but > BOUNCE_MATCH_S from this one

Read-only autopsy on the c3 cache; no knob is tuned and no gate moves.

    python3 vision/ballsearch/bounce_autopsy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ball_replicate as br                                    # noqa: E402
from claim_lab import load as c3load                           # noqa: E402

RAL = [7, 9, 10, 17]
NEAR_S = 0.30


def tracked(c):
    """The shipped tracked side, exactly as check 3 builds it."""
    rally, anchors, floors = c["rally"], c["anchors"], c["floors"]
    serve, end = c["imps"][0], c["dead"]
    t_obs, t_bounds, t_evs = br.tracked_side(
        rally, anchors, floors, serve, end)
    segs, cons, bounds, evs = br.crossing_demotion(
        c["P"], t_obs, t_bounds, t_evs, floors, anchors)
    return t_obs, segs, bounds, evs


def main():
    tally, rows = {}, []
    n_hum = n_match = 0
    capped_segs = 0
    for r in RAL:
        c = c3load(r)
        h_con = list(c["imps"])
        h_bnc = [(float(s["ts"]), np.asarray(s["bounce_xy"], float))
                 for s in c["h_segs"]
                 if s and s.get("ok") and s["kind"] == "bounce"]
        t_obs, segs, bounds, evs = tracked(c)

        # which tracked segment holds each human bounce, and what it said
        def seg_of(ts):
            for k in range(len(bounds) - 1):
                if bounds[k] <= ts <= bounds[k + 1]:
                    return k
            return None

        called = {}                       # seg index -> called bounce ts
        for k, s in enumerate(segs):
            if s and s.get("ok") and s["kind"] == "bounce":
                called[k] = float(s["ts"])

        # a segment is CAPPED when it holds >1 human bounce
        holds = {}
        for ts, _ in h_bnc:
            k = seg_of(ts)
            if k is not None:
                holds.setdefault(k, []).append(ts)
        capped_segs += sum(1 for k, v in holds.items() if len(v) > 1)

        for ts, xy in h_bnc:
            n_hum += 1
            k = seg_of(ts)
            near = sum(1 for o in t_obs if abs(o[0] - ts) <= 0.15)
            ev = any(abs(e - ts) <= NEAR_S for e in evs)
            if k is None:
                b = "NO WINDOW"
            elif k in called and abs(called[k] - ts) <= br.BOUNCE_MATCH_S:
                b = "matched"
                n_match += 1
            elif len(holds.get(k, [])) > 1 and k in called:
                b = "CAPPED"
            elif segs[k] is None:
                b = "NO SEG"
            elif not segs[k].get("ok"):
                b = "NOT OK"
            elif k in called:
                b = "WRONG TIME"
            else:
                b = "CALLED ARC"
            tally[b] = tally.get(b, 0) + 1
            rows.append((r, ts, b, near, ev,
                         len(holds.get(k, [])) if k is not None else 0))

        # missed contacts: how many human contacts have no tracked bound
        cmiss = sum(1 for hc in h_con
                    if not any(abs(hc - b) <= br.MATCH_S for b in bounds))
        print(f"r{r}: {len(h_bnc)} human bounces, {len(h_con)} human "
              f"contacts, {cmiss} contacts missing from the tracked "
              f"bounds, {len(bounds)-1} tracked segments")

    print(f"\n{n_match}/{n_hum} human bounces matched\n")
    print(f"{'bucket':12s} {'n':>3}")
    for b, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{b:12s} {n:>3}")
    print(f"\nsegments holding >1 human bounce (rule violations "
          f"by construction): {capped_segs}")

    print(f"\n{'rally':>5} {'t':>8} {'bucket':12s} {'obs+-.15':>9} "
          f"{'marker':>7} {'hb in seg':>10}")
    for r, ts, b, near, ev, nh in rows:
        if b == "matched":
            continue
        print(f"{r:>5} {ts:>8.2f} {b:12s} {near:>9} "
              f"{'yes' if ev else 'no':>7} {nh:>10}")


if __name__ == "__main__":
    main()
