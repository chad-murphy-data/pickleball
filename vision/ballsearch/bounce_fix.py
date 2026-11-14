"""Ship the four audit fixes, measured one at a time.

The owner's ordered list from the video audit:

  1  RALLY BOUNDS + junk filter -- no track before the serve anchor, none
     after the ball dies, and nothing physically impossible in between
     (path_physics).  Turns are re-detected on the CLEANED path.
  2  ASSERT the two guaranteed bounces (serve, return) instead of hunting
     for them.  The double-bounce rule makes them certain by position in
     the sequence.
  3  FIX THE CLAIM -- a spatial gate on the anchor, so an anchor cannot
     take a turn that happened somewhere else.  The 16 s swap the owner
     photographed: a bounce at 54 deg beat the real contact at 48 because
     claiming is greedy on angle alone.
  4  LOWER THE TURN THRESHOLD once 1 and 3 are in, to recover the shallow
     cross-court serve bounces the 25 deg minimum currently deletes.

Plus the discriminator measured in `a16e2ba` and never wired in: a real
bounce FALLS THEN RISES in image y.  A sign test, nothing to tune.

Scored on the c3 cache (0.2 s a rally) against the owner's own
reconstruction, so every arm is one command:

    python3 vision/ballsearch/bounce_fix.py            # all arms, all rallies

RESULT (2026-09-04, r7/r9/r10/r17, 128 turns, 35 real bounces, 79 contacts):

  arm                          emitted  real  junk  miss | b.recall  prec | contacts
  baseline (today)                 30     6    12    12 |    8/35    20% |   63/79
  sign test ALONE                  14     6     2     6 |    7/35    43% |   63/79
  1 rally bounds + filter          28     6    10    12 |    8/35    21% |   56/79
  1+sign                           16     6     3     7 |    7/35    38% |   56/79
  1+sign+3 claim gate              34    11     7    16 |   14/35    32% |   41/79
  ...+4 turn_deg 18                34    11     7    16 |   14/35    32% |   40/79
  ...+2 assert 2 bounces           37    12     7    18 |   16/35    32% |   38/79

ONE of the four ships, and it is none of the four: the SIGN TEST, which
kills 10 of 12 junk markers, keeps all 6 real bounces and costs zero
contacts.  It is now wired into ball_replicate.tracked_side.

Why the owner's four did not:

  1  RE-DETECTING turns on the physics-cleaned path costs 7 contacts
     (63 -> 56) and removes only junk the sign test removes for free.
     The filter itself stays shipped as a PATH cleaner; this is about
     the turn stream, and the turn stream does not want it.
  3  The claim radius gate buys bounce recall (8 -> 14) with a quarter
     of the contacts (63 -> 41).  Same failure mode as the 2026-09-01
     rejection recorded in claim_bounds' docstring, re-measured here on
     the cleaned input rather than assumed.  The 16 s swap is real; a
     radius is not its fix.
  4  turn_deg 18 (and 20, and min_sep 0.12/0.18) is EXACTLY null -- same
     34/11/7/16, one contact worse.  The shallow serve bounces are not
     being deleted by the threshold, they are not in the path.
  2  Asserting the two guaranteed bounces reads real 9 / recall 11 only
     if the window boundaries come from c["imps"], i.e. from the owner's
     own labelled contact times -- a leak.  Rebuilt on `serve` plus
     machine-derived anchors it is marginal standalone and negative on
     top of the claim gate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ball_replicate as br                                    # noqa: E402
from make_ball_audit import detect_events                      # noqa: E402
from claim_lab import load as c3load                           # noqa: E402
import path_physics as pp                                      # noqa: E402
import numpy as np                                             # noqa: E402

NEAR_S = 0.30       # a turn is "the same event" as a truth time within this
WIN = 0.12          # s either side for the local velocity read
CLAIM_R = 120.0     # px: how far an anchor's paddle may be from the turn
SERVE_WIN = (0.15, 0.95)   # s after the serve contact to look for its bounce


def local_vel(pts, e, win=WIN):
    w = [p for p in pts if abs(p[0] - e) <= win]
    out = []
    for seg in ([p for p in w if p[0] < e], [p for p in w if p[0] > e]):
        if len(seg) < 2 or seg[-1][0] - seg[0][0] < 1e-3:
            out.append((float("nan"),) * 3)
            continue
        dt = seg[-1][0] - seg[0][0]
        out.append(((seg[-1][1] - seg[0][1]) / dt,
                    (seg[-1][2] - seg[0][2]) / dt, 0.0))
    return out[0], out[1]


def falls_then_rises(pts, e):
    """Image y grows DOWN, so a bounce is dy_pre > 0 > dy_post."""
    (_, dy0, _), (_, dy1, _) = local_vel(pts, e)
    if dy0 != dy0 or dy1 != dy1:
        return True                      # unmeasurable -> do not delete
    return dy0 > 0 > dy1


def assert_bounces(imps, turns, emitted):
    """The two bounces the RULES guarantee: the serve must bounce before
    the return, and the return must bounce before the third shot.  They
    are not detected, they are asserted -- so the only question is which
    already-emitted turn is each one, and a window between the two known
    contacts answers it without any angle test."""
    out = []
    for k in (0, 1):
        if len(imps) < k + 2:
            break
        lo, hi = imps[k], imps[k + 1]
        cand = [e for e in turns if lo + 0.05 < e < hi - 0.02]
        if cand:                          # the turn nearest mid-window
            mid = (lo + hi) / 2
            out.append(min(cand, key=lambda e: abs(e - mid)))
    return out


def run(rally, use_filter, sign_test, claim_r, turn_deg, assert_db):
    c = c3load(rally)
    serve, end = c["imps"][0], c["dead"]
    pts = np.asarray(c["timing_ref"], float)

    if use_filter:
        z = np.load(c["npz"])
        kept, _, _ = pp.clean(pts, serve=serve, dead=end, P=c["P"],
                              occ_mask=pp.in_player_box(pts, z))
        ref = [tuple(p[:3]) for p in kept]
    else:
        ref = [tuple(p[:3]) for p in pts]

    if use_filter or turn_deg != 25.0:
        turns = [e for e in detect_events(ref, turn_deg=turn_deg)
                 if serve - 0.3 <= e < end - 0.05]
        angs = br.turn_angles(ref, turns)
    else:
        turns = [e for e in c["turns"] if serve - 0.3 <= e < end - 0.05]
        angs = c["angs"]

    claimed = set(br.claim_bounds(turns, angs, ref, c["anchors"],
                                  claim_r=claim_r))
    emitted = [e for e in turns if e not in claimed]
    if sign_test:
        emitted = [e for e in emitted if falls_then_rises(ref, e)]
    if assert_db:
        for e in assert_bounces(list(c["imps"]), turns, emitted):
            if e not in emitted:
                emitted.append(e)
            claimed.discard(e)
        emitted = sorted(set(emitted))

    h_con = list(c["imps"])
    h_bnc = [float(s["ts"]) for s in c["h_segs"]
             if s and s.get("ok") and s["kind"] == "bounce"]

    def truth_of(e):
        dc = min((abs(e - x) for x in h_con), default=9)
        db = min((abs(e - x) for x in h_bnc), default=9)
        if dc <= NEAR_S and dc <= db:
            return "CONTACT"
        return "BOUNCE" if db <= NEAR_S else "WOBBLE"

    real = sum(1 for e in emitted if truth_of(e) == "BOUNCE")
    junk = sum(1 for e in emitted if truth_of(e) == "WOBBLE")
    miss = sum(1 for e in emitted if truth_of(e) == "CONTACT")
    # bounce RECALL: how many human bounces have an emitted turn near them
    got = sum(1 for hb in h_bnc if any(abs(hb - e) <= NEAR_S for e in emitted))
    # contact recall via the claim
    cgot = sum(1 for hc in h_con
               if any(abs(hc - e) <= NEAR_S for e in claimed))
    return dict(emitted=len(emitted), real=real, junk=junk, miss=miss,
                got=got, nb=len(h_bnc), cgot=cgot, nc=len(h_con))


ARMS = [
    ("baseline (today)",       dict(use_filter=0, sign_test=0, claim_r=None, turn_deg=25.0, assert_db=0)),
    ("sign test ALONE  <- SHIPPED",
                               dict(use_filter=0, sign_test=1, claim_r=None, turn_deg=25.0, assert_db=0)),
    ("1 filter",               dict(use_filter=1, sign_test=0, claim_r=None, turn_deg=25.0, assert_db=0)),
    ("1+sign",                 dict(use_filter=1, sign_test=1, claim_r=None, turn_deg=25.0, assert_db=0)),
    ("1+sign+3 claim gate",    dict(use_filter=1, sign_test=1, claim_r=CLAIM_R, turn_deg=25.0, assert_db=0)),
    ("...+4 turn_deg 18",      dict(use_filter=1, sign_test=1, claim_r=CLAIM_R, turn_deg=18.0, assert_db=0)),
    ("...+2 assert 2 bounces", dict(use_filter=1, sign_test=1, claim_r=CLAIM_R, turn_deg=18.0, assert_db=1)),
]

RAL = [7, 9, 10, 17]


def main():
    print(f"{'arm':26s} {'emitted':>8} {'real':>5} {'junk':>5} {'miss':>5} "
          f"| {'bounce recall':>14} {'precision':>10} | {'contact recall':>15}")
    for name, kw in ARMS:
        T = dict(emitted=0, real=0, junk=0, miss=0, got=0, nb=0, cgot=0, nc=0)
        for r in RAL:
            for k, v in run(r, **kw).items():
                T[k] += v
        prec = T["real"] / max(T["emitted"], 1)
        print(f"{name:26s} {T['emitted']:>8} {T['real']:>5} {T['junk']:>5} "
              f"{T['miss']:>5} | {T['got']:>6}/{T['nb']:<7} "
              f"{100*prec:>9.0f}% | {T['cgot']:>7}/{T['nc']:<7}")


if __name__ == "__main__":
    main()
