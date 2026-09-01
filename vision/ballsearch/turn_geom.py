"""Turn geometry vs truth: do bounce-turns look like a V (vy flips
down->up, vx keeps sign) while contact-turns flip horizontally?

Labels: tap = within 0.15 s of a hand-timestamped contact;
bounce = within 0.15 s of a HUMAN-FIT bounce time (truth-grade r10).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, human_bounce_times     # noqa: E402


def turn_geoms(pts, min_ang=20.0):
    """(t, ang, vx0, vy0, vx1, vy1) direction events, non-max suppressed."""
    evs = []
    for i in range(2, len(pts) - 2):
        t, x, y = pts[i]
        vx0 = x - pts[i - 2][1]
        vy0 = y - pts[i - 2][2]
        vx1 = pts[i + 2][1] - x
        vy1 = pts[i + 2][2] - y
        n0 = np.hypot(vx0, vy0)
        n1 = np.hypot(vx1, vy1)
        if n0 < 2 or n1 < 2:
            continue
        cos = (vx0 * vx1 + vy0 * vy1) / (n0 * n1)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if ang >= min_ang:
            evs.append((t, ang, vx0, vy0, vx1, vy1))
    out = []
    for e in sorted(evs, key=lambda e: -e[1]):
        if all(abs(e[0] - u[0]) >= 0.15 for u in out):
            out.append(e)
    return sorted(out)


def vfeat(vx0, vy0, vx1, vy1):
    v_shape = vy0 > 0 and vy1 < 0            # image y down: fall then rise
    vx_keep = np.sign(vx0) == np.sign(vx1) and abs(vx0) > 1 and abs(vx1) > 1
    vx_flip = np.sign(vx0) != np.sign(vx1) and abs(vx0) > 1 and abs(vx1) > 1
    vert0 = abs(vy0) > abs(vx0)              # falling steeper than moving
    return v_shape, vx_keep, vx_flip, vert0


def main():
    rows = {"tap": [], "bounce": [], "both": [], "none": []}
    for rally in [int(x) for x in sys.argv[1:]] or [7, 9, 10, 6]:
        c = load(rally)
        hb = human_bounce_times(c)
        imps = c["imps"]
        evs = turn_geoms(c["timing_ref"], 20.0)
        print(f"== rally {rally}: {len(evs)} turns>=20deg, "
              f"{len(imps)} taps, {len(hb)} human bounces")
        for t, ang, vx0, vy0, vx1, vy1 in evs:
            near_tap = any(abs(t - u) <= 0.15 for u in imps)
            near_b = any(abs(t - u) <= 0.15 for u in hb)
            lab = ("both" if near_tap and near_b else
                   "tap" if near_tap else
                   "bounce" if near_b else "none")
            f = vfeat(vx0, vy0, vx1, vy1)
            rows[lab].append((ang,) + f)
            if lab in ("bounce", "both"):
                print(f"   {lab:6s} {t:7.2f} ang {ang:5.1f} "
                      f"v0=({vx0:6.1f},{vy0:6.1f}) v1=({vx1:6.1f},{vy1:6.1f})"
                      f" V={f[0]} keepx={f[1]} flipx={f[2]}")
    print("\n== pooled ==")
    for lab, r in rows.items():
        if not r:
            continue
        a = np.array(r, dtype=float)
        print(f"  {lab:6s} n={len(r):3d}  ang med {np.median(a[:,0]):5.1f}"
              f"  V {a[:,1].mean():.2f}  keepx {a[:,2].mean():.2f}"
              f"  flipx {a[:,3].mean():.2f}  vert0 {a[:,4].mean():.2f}")


if __name__ == "__main__":
    main()
