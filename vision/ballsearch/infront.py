"""Owner's rule, 2026-09-04: "the ball always bounces in front of the person
making contact, and they rarely hit it behind them."

Tested as a SIGN, not an offset -- bounce_proxy.py already uses "in front"
as an average shift (LEAD_FT = 8.5); the question here is whether it is a
CONSTRAINT that never breaks.  Front = toward the previous hitter, since
that is where the ball came from.

The two halves of the rule must be tested separately, because only one of
them is confounded.  A BOUNCE is on the floor, z = 0, exactly where the
homography is exact.  A CONTACT is at paddle height, and this project maps
everything to z = 0, where a ball at height projects DEEPER than it really
is.  So "behind the feet" means something for bounces and means nothing
for contacts.

RESULT (57 human-solved bounces, 123 contacts, r2-r7 + r9 + r10 + r17):

  BOUNCES     in front of the receiver   98%  (56/57)
              distance in front          median 7.8 ft, IQR 4.1-10.5
              lateral offset from the
              receiver->hitter line      median 1.4 ft, p90 2.8 ft

  CONTACTS    in front of the hitter     46%  -- a coin flip
              signed distance            median -1.0 ft, IQR -6.3 to +7.5

So the owner's bounce rule holds as close to a law as this project has:
the bounce is not merely in front, it sits in a NARROW CORRIDOR along the
line between the two players, 2-10 ft ahead of the receiver.  That is a
search prior, not just a filter -- it says where to look, which is the
recall half of the bounce problem (38%), not only the precision half.

The owner's contact observation is REAL in the data and is NOT an error:
contacts read behind the feet about half the time because of the z = 0
projection above.  Do not use "behind the feet" as a defect signal on
contacts.  (This is the same confound that made the earlier behind-the-feet
junk filter null -- see path_physics.md.)

The single counterexample is in r17 (5 bounces, 80%, min -6.3 ft), which is
already the known-bad rally for hitter attribution (4 side-alternation
violations, 14/17 track purity, PR #116).  Likely a mislabelled hitter
rather than a real backward bounce; not chased.

    python3 vision/ballsearch/infront.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, "/home/user/pickleball/vision/ballsearch")
import pathfirst as pf
from rally_stats import players
from geom_speed import contacts, clicks, hitter_track, foot_xy

RAL = [2, 3, 4, 5, 6, 7, 9, 10, 17]

rows, crows = [], []
for rally in RAL:
    try:
        ctx = pf.context(rally)
        P, z = ctx["P"], np.load(ctx["c"]["npz"])
        pls = players(ctx)
        cl = clicks(rally)
        cs = [s for s in contacts(rally) if s["type"] != "whiff"]
        for s in cs:
            s["tid"] = hitter_track(ctx, pls, s["t"], cl)
            s["xy"] = foot_xy(z, P, s["tid"], s["t"]) if s["tid"] is not None else None
    except Exception as e:
        print(f"r{rally}: skipped ({type(e).__name__})")
        continue

    # --- BOUNCES (z=0, no projection confound) ---
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
        if rec["t"] - tb > 1.2:
            continue
        d = rec["xy"] - hit["xy"]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        u = d / n                                  # receiver -> away from hitter
        front = float(np.dot(truth - rec["xy"], -u))   # + = toward hitter = in front
        w = truth - rec["xy"]
        lat = float(abs(-u[0] * w[1] + u[1] * w[0]))
        rows.append((rally, front, lat))

    # --- CONTACTS (at paddle height; projected to z=0) ---
    for i, s in enumerate(cs):
        if s["xy"] is None or i == 0:
            continue
        prev = cs[i - 1]
        if prev["xy"] is None:
            continue
        d = s["xy"] - prev["xy"]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        # where the ball was when struck, as the human path saw it, at z=0
        cp = [c for c in ctx["c"]["hum"][0] if abs(c[0] - s["t"]) < 0.03]
        if not len(cp):
            continue
        px = np.asarray(cp[0], float)
        Hi = np.linalg.inv(np.asarray(P)[:, [0, 1, 3]])
        v = Hi @ np.array([px[1], px[2], 1.0])
        gxy = v[:2] / v[2]
        u = d / n
        crows.append((rally, float(np.dot(gxy - s["xy"], -u))))

f = np.array([r[1] for r in rows])
lat = np.array([r[2] for r in rows])
print(f"\nBOUNCES (z=0, no confound): n={len(f)}")
print(f"  in front of the receiver : {100*np.mean(f > 0):.0f}%   "
      f"({int((f > 0).sum())}/{len(f)})")
print(f"  distance in front        : median {np.median(f):.1f} ft, "
      f"IQR {np.percentile(f,25):.1f} to {np.percentile(f,75):.1f}, min {f.min():.1f}")
print(f"  lateral offset           : median {np.median(lat):.1f} ft, "
      f"p90 {np.percentile(lat,90):.1f}")
for rr in RAL:
    v = np.array([r[1] for r in rows if r[0] == rr])
    if len(v):
        print(f"    r{rr:<3d} n={len(v):3d}  in front {100*np.mean(v>0):3.0f}%  "
              f"median {np.median(v):5.1f} ft  min {v.min():6.1f}")

c = np.array([r[1] for r in crows])
if len(c):
    print(f"\nCONTACTS (paddle height, projected to z=0): n={len(c)}")
    print(f"  in front of the hitter   : {100*np.mean(c > 0):.0f}%   "
          f"({int((c > 0).sum())}/{len(c)})")
    print(f"  signed distance          : median {np.median(c):+.1f} ft, "
          f"IQR {np.percentile(c,25):+.1f} to {np.percentile(c,75):+.1f}")
    print("  (negative = reads BEHIND the feet; expected from height->depth)")
