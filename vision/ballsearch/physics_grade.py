"""Grade the physics filter with no new labels.

The owner-clicked human path says where the ball actually was.  So every
tracked point can be called GOOD (near the human path at the same time)
or JUNK (far from it), and the filter is graded on which it removes:

  junk removed   = the win
  good removed   = the cost

No thresholds are tuned on this -- V_MAX comes from the human paths'
own speed distribution, not from this score.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load                       # noqa: E402
import path_physics as pp                        # noqa: E402

NEAR_PX = 25.0      # a tracked point this close to the human path is the ball
TOL_S = 0.05        # human sample must exist this close in time


def label(pts, hum):
    """GOOD / JUNK / unknown for each tracked point vs the human path."""
    hum = np.asarray(hum, float)
    lab = np.full(len(pts), "?", dtype=object)
    for i, (t, x, y) in enumerate(pts[:, :3]):
        j = int(np.argmin(np.abs(hum[:, 0] - t)))
        if abs(hum[j, 0] - t) > TOL_S:
            continue
        d = float(np.hypot(x - hum[j, 1], y - hum[j, 2]))
        lab[i] = "GOOD" if d <= NEAR_PX else "JUNK"
    return lab


def main():
    rallies = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6, 7, 9, 10, 17]
    tot = dict(gk=0, gd=0, jk=0, jd=0, uk=0, ud=0)
    print(f"{'rally':>6} {'pts':>5} {'good':>5} {'junk':>5} "
          f"| {'junk cut':>9} {'good cut':>9} | reasons")
    for r in rallies:
        c = load(r)
        pts = np.asarray(c["timing_ref"], float)
        lab = label(pts, c["hum"][0])
        z = np.load(c["npz"])
        _, keep, reason = pp.clean(pts, serve=c["imps"][0], dead=c["dead"],
                                   P=c["P"], occ_mask=pp.in_player_box(pts, z))
        g, j = lab == "GOOD", lab == "JUNK"
        gd, jd = int((g & ~keep).sum()), int((j & ~keep).sum())
        gk, jk = int((g & keep).sum()), int((j & keep).sum())
        ud = int(((lab == "?") & ~keep).sum())
        uk = int(((lab == "?") & keep).sum())
        tot["gd"] += gd; tot["jd"] += jd; tot["gk"] += gk
        tot["jk"] += jk; tot["ud"] += ud; tot["uk"] += uk
        rs = {k: v for k, v in pp.summarize(reason).items() if k != "kept"}
        print(f"{r:>6} {len(pts):>5} {g.sum():>5} {j.sum():>5} "
              f"| {jd:>4}/{int(j.sum()):<4} {gd:>4}/{int(g.sum()):<4} | {rs}")
    print("\nPOOLED")
    print(f"  junk points removed : {tot['jd']:5d} / {tot['jd']+tot['jk']:<5d} "
          f"({100*tot['jd']/max(1,tot['jd']+tot['jk']):.1f}%)")
    print(f"  good points removed : {tot['gd']:5d} / {tot['gd']+tot['gk']:<5d} "
          f"({100*tot['gd']/max(1,tot['gd']+tot['gk']):.1f}%)")
    print(f"  unjudgeable removed : {tot['ud']:5d} / {tot['ud']+tot['uk']:<5d} "
          f"(no human sample within {TOL_S}s -- mostly pre-serve)")


if __name__ == "__main__":
    main()
