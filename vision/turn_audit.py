"""Why the tracker over-calls bounces (owner-approved lever, 2026-09-04).

bounce_replication.md measured the automated bounce POSITION at 1.9 ft
median but matched only 13 of 34 real bounces, and the tracker emits
MORE bounces than exist (r9: 19 vs 13).  The cause is one line in
ball_replicate.tracked_side:

    bounce_evs = [e for e in turns if e not in claimed]

A bounce is the RESIDUAL category.  Every direction change in the path
that no hitter-chain anchor claims is declared a bounce -- so every
tracking wobble becomes a bounce marker, and the real bounces get
diluted.  Nothing ever asks whether the turn LOOKS like a bounce.

This dumps every turn with the features that could answer that, graded
against the owner's own reconstruction, so the discriminator is chosen
from evidence rather than guessed:

  ang          the 2D turn angle the claiming gate already uses
  dy_pre/post  vertical pixel velocity before/after (image y grows DOWN,
               so a real bounce is falling -> rising: dy_pre > 0 > dy_post)
  sp_pre/post  speed either side; a bounce loses energy, a paddle adds it
  claimed      an anchor took this turn as a contact
  truth        CONTACT / BOUNCE / WOBBLE from the human reconstruction

    python3 vision/turn_audit.py --rally 9
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_replicate as br                                   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "vision"
WIN = 0.12          # s either side for the local velocity read
NEAR_S = 0.30       # a turn is "the same event" as a truth time within this


def local_vel(pts, e, win=WIN):
    """(vx, vy, speed) just before and just after e, px/s."""
    w = [p for p in pts if abs(p[0] - e) <= win]
    out = []
    for seg in ([p for p in w if p[0] < e], [p for p in w if p[0] > e]):
        if len(seg) < 2 or seg[-1][0] - seg[0][0] < 1e-3:
            out.append((float("nan"),) * 3)
            continue
        dt = seg[-1][0] - seg[0][0]
        vx = (seg[-1][1] - seg[0][1]) / dt
        vy = (seg[-1][2] - seg[0][2]) / dt
        out.append((vx, vy, math.hypot(vx, vy)))
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--npz", help="unused; kept so old command lines work")
    ap.add_argument("--anchors", help="unused; the cache carries the anchors")
    a = ap.parse_args()

    # The c3 cache already holds every stream ball_replicate recomputes
    # (timing_ref, turns, angs, anchors, the human reconstruction) and
    # loads in 0.2 s -- re-running the decode here cost ~10 min a rally
    # for nothing.  claim_bounds below is ball_replicate's own function,
    # so the CONTACT/BOUNCE split is identical to the pipeline's.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ballsearch"))
    from claim_lab import load as c3load                       # noqa: E402
    c = c3load(a.rally)
    serve, end = c["imps"][0], c["dead"]
    timing_ref = c["timing_ref"]
    turns = [e for e in c["turns"] if serve - 0.3 <= e < end - 0.05]
    angs = c["angs"]
    anchors = c["anchors"]
    claimed = set(br.claim_bounds(turns, angs, timing_ref, anchors))
    h_contacts = list(c["imps"])
    h_bounces = [float(sg["ts"]) for sg in c["h_segs"]
                 if sg and sg.get("ok") and sg["kind"] == "bounce"]

    def near(t, xs):
        if not xs:
            return None
        d = min(abs(t - x) for x in xs)
        return d if d <= NEAR_S else None

    rows = []
    for e in sorted(turns):
        (vx0, vy0, s0), (vx1, vy1, s1) = local_vel(timing_ref, e)
        dc, db = near(e, h_contacts), near(e, h_bounces)
        truth = ("CONTACT" if dc is not None and (db is None or dc <= db)
                 else ("BOUNCE" if db is not None else "WOBBLE"))
        rows.append(dict(rally=a.rally, t=round(e, 3),
                         ang=round(angs.get(e, 0.0), 1),
                         claimed=int(e in claimed), truth=truth,
                         dy_pre=round(vy0, 1), dy_post=round(vy1, 1),
                         sp_pre=round(s0, 1), sp_post=round(s1, 1),
                         sp_ratio=(round(s1 / s0, 2)
                                   if s0 == s0 and s0 > 1 else "")))

    p = OUT / f"turn_audit_r{a.rally}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"rally {a.rally}: {len(turns)} turns, {len(claimed)} claimed as "
          f"contacts, {len(turns) - len(claimed)} emitted as BOUNCES")
    print(f"truth: {len(h_contacts)} human contacts, {len(h_bounces)} human bounces")
    print(f"-> {p}")
    print(f"\n{'t':>9} {'ang':>6} {'clm':>4} {'truth':8s} {'dy_pre':>8} "
          f"{'dy_post':>8} {'sp_pre':>8} {'sp_post':>8} {'ratio':>6}")
    for r in rows:
        flag = ""
        if not r["claimed"]:
            flag = {"WOBBLE": "  <-- FALSE BOUNCE",
                    "BOUNCE": "  ok bounce",
                    "CONTACT": "  <-- MISSED CONTACT"}[r["truth"]]
        print(f"{r['t']:9.2f} {r['ang']:6.1f} {r['claimed']:4d} {r['truth']:8s} "
              f"{r['dy_pre']:8.1f} {r['dy_post']:8.1f} {r['sp_pre']:8.1f} "
              f"{r['sp_post']:8.1f} {str(r['sp_ratio']):>6}{flag}")

    # --- does any single feature separate real bounces from wobbles? ---
    un = [r for r in rows if not r["claimed"]]
    real = [r for r in un if r["truth"] == "BOUNCE"]
    junk = [r for r in un if r["truth"] == "WOBBLE"]
    miss = [r for r in un if r["truth"] == "CONTACT"]
    print(f"\nunclaimed turns (ALL emitted as bounces today): {len(real)} real "
          f"bounce / {len(junk)} junk / {len(miss)} missed contact")
    for k in ("ang", "dy_pre", "dy_post", "sp_pre", "sp_post"):
        rv = np.array([r[k] for r in real], float)
        jv = np.array([r[k] for r in junk], float)
        rv, jv = rv[~np.isnan(rv)], jv[~np.isnan(jv)]
        if len(rv) and len(jv):
            print(f"  {k:8s} real median {np.median(rv):8.1f}   "
                  f"junk median {np.median(jv):8.1f}")
    for name, grp in (("real", real), ("junk", junk)):
        n = sum(1 for r in grp if r["dy_pre"] > 0 > r["dy_post"])
        if grp:
            print(f"  falling->rising (dy_pre>0>dy_post): {name} "
                  f"{n}/{len(grp)} = {100 * n / len(grp):.0f}%")


if __name__ == "__main__":
    main()
