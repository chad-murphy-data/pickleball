"""Where did the ball bounce, WITHOUT tracking the ball (owner ask 2026-09-03:
"where we can start using non-tracking methods ... for 'good enough' answers").

A ball that bounces is struck by someone right after it bounces.  So the
receiver's own court position at their contact is a proxy for the bounce
point -- offset by however far in front of themselves they take the ball.
If that offset is small and consistent, a bounce MAP costs nothing: player
court positions (homography exact at 0.06 ft) and contact times (the timing
stream already beats human labels) are both solved channels.  No ball.

Truth = the bounce points solved from the OWNER's clicked ball path
(court3d.fit_segment, `h_segs` kind="bounce"), which is the arm verified
exact on r10 (13/13 bounces across 26/26 segments, all 5 disputed calls
confirmed real).  So this grades the PROXY, not the tracker.

Three estimators, cheapest first:
  feet       the receiver's feet at their contact
  feet+lead  feet, moved LEAD_FT toward the previous hitter (a receiver
             takes the ball in front of themselves); LEAD_FT is fixed at
             the median observed offset on TRAIN and re-read on EVAL
  midpoint   halfway between the two hitters (the no-information control)

Reports total error and, separately, the DEPTH error (y, distance from the
net) -- depth is the axis a bounce map is actually read on (deep vs short),
and it is also the axis one camera cannot see for a ball in flight.

    python3 bounce_proxy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import pathfirst as pf                                       # noqa: E402
from rally_stats import players                              # noqa: E402
from geom_speed import contacts, clicks, hitter_track, foot_xy  # noqa: E402

TRAIN = [2, 3, 4, 5, 6, 7, 17]
EVAL = [9, 10]


def rally_rows(rally, lead):
    ctx = pf.context(rally)
    P, z = ctx["P"], np.load(ctx["c"]["npz"])
    pls = players(ctx)
    cl = clicks(rally)
    cs = [s for s in contacts(rally) if s["type"] != "whiff"]
    for s in cs:
        s["tid"] = hitter_track(ctx, pls, s["t"], cl)
        s["xy"] = foot_xy(z, P, s["tid"], s["t"]) if s["tid"] is not None else None
    rows = []
    for seg in ctx["c"]["h_segs"]:
        if seg["kind"] != "bounce" or not seg.get("ok"):
            continue
        tb = float(seg["ts"])
        truth = np.asarray(seg["bounce_xy"], float)
        nxt = [s for s in cs if s["t"] > tb and s["xy"] is not None]
        prv = [s for s in cs if s["t"] <= tb and s["xy"] is not None]
        if not nxt or not prv:
            continue
        rec, hit = nxt[0], prv[-1]
        if rec["t"] - tb > 1.2:                    # the bounce must belong to this shot
            continue
        d = rec["xy"] - hit["xy"]
        n = np.linalg.norm(d)
        lead_xy = rec["xy"] - lead * d / n if n > 1e-6 else rec["xy"]
        rows.append(dict(rally=rally, t=tb, truth=truth, feet=rec["xy"],
                         lead=lead_xy, mid=(rec["xy"] + hit["xy"]) / 2,
                         gap=rec["t"] - tb))
    return rows


def summarise(name, rows, lead):
    print(f"\n=== {name}: {len(rows)} human-fit bounces matched to a following contact")
    if not rows:
        return
    print(f"  {'estimator':12s} {'median err':>11} {'p90':>7} {'median DEPTH err':>17} "
          f"{'depth bias':>11}")
    for key, lab in (("feet", "feet"), ("lead", f"feet+lead {lead:.1f}ft"), ("mid", "midpoint")):
        e = np.array([np.linalg.norm(r[key] - r["truth"]) for r in rows])
        dy = np.array([r[key][1] - r["truth"][1] for r in rows])
        print(f"  {lab:12s} {np.median(e):8.1f} ft {np.percentile(e, 90):6.1f} "
              f"{np.median(np.abs(dy)):14.1f} ft {np.median(dy):+10.1f} ft")
    # how far in front of themselves does a receiver take the ball?
    off = np.array([np.linalg.norm(r["feet"] - r["truth"]) for r in rows])
    print(f"  receiver-to-bounce offset: median {np.median(off):.1f} ft, "
          f"IQR {np.percentile(off, 25):.1f}-{np.percentile(off, 75):.1f}, "
          f"time from bounce to contact {np.median([r['gap'] for r in rows]):.2f} s")
    # is the bounce SIDE (near / far court) and half (left / right) right?
    same_side = np.mean([(r["feet"][1] > 22) == (r["truth"][1] > 22) for r in rows])
    same_half = np.mean([(r["feet"][0] > 10) == (r["truth"][0] > 10) for r in rows])
    kitchen = np.mean([(abs(r["feet"][1] - 22) < 7) == (abs(r["truth"][1] - 22) < 7)
                       for r in rows])
    print(f"  categorical agreement -- court side {100 * same_side:.0f}%, "
          f"left/right half {100 * same_half:.0f}%, kitchen-vs-deep {100 * kitchen:.0f}%")


def main():
    tr0 = []
    for r in TRAIN:
        tr0 += rally_rows(r, 0.0)
    lead = float(np.median([np.linalg.norm(r["feet"] - r["truth"]) for r in tr0])) if tr0 else 0.0
    print(f"LEAD_FT fixed on TRAIN at the median receiver-to-bounce offset: {lead:.1f} ft")
    tr = []
    for r in TRAIN:
        tr += rally_rows(r, lead)
    summarise("TRAIN " + "+".join(f"r{r}" for r in TRAIN), tr, lead)
    ev = []
    for r in EVAL:
        ev += rally_rows(r, lead)
    summarise("EVAL " + "+".join(f"r{r}" for r in EVAL)
              + " (spent evaluation; lead comes from TRAIN, nothing refit)", ev, lead)


if __name__ == "__main__":
    main()
