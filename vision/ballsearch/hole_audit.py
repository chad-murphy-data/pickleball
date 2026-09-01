"""Hole audit: for every tap the production bounds MISS, what
evidence exists within +/-0.15 s?

  turn40  — timing-stream turn at the frozen 40-deg battery angle
  turn20  — timing-stream direction event at >=20 deg (claiming-only
            candidate set; check 2 stays frozen)
  peak    — pose excitement peak (any z, min-sep applied per side)
  prox    — paddle-proximity local minimum of the decoded path
And the parity signal: would the claimed-bound chain flag the hole?
(consecutive same-side bounds, or gap > 1.6x local cadence)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br           # noqa: E402
import hitter_chain as hc             # noqa: E402
from claim_lab import (load, paddle_series, paddle_at,   # noqa: E402
                       claim_production, path_at, prox_min)
from make_ball_audit import load_impacts  # noqa: E402


def turn_events(pts, min_ang):
    """direction events on the timing stream at a given angle floor —
    mirrors detect_events geometry via turn_angles on a dense grid."""
    evs = []
    ts = [p[0] for p in pts]
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
            evs.append((t, ang))
    # non-max suppress within 0.15 s
    out = []
    for t, ang in sorted(evs, key=lambda e: -e[1]):
        if all(abs(t - u) >= 0.15 for u, _ in out):
            out.append((t, ang))
    return sorted(t for t, _ in out)


def pose_peaks_all(npz_path):
    """excitement peaks with NO z floor (min-sep still applied)."""
    z = np.load(npz_path)
    old = hc.Z_MIN
    hc.Z_MIN = -99.0
    try:
        picked = hc.predict_contacts(npz_path,
                                     float(z["t"].min()),
                                     float(z["t"].max()))
    finally:
        hc.Z_MIN = old
    return picked                       # (t, z, track, ...)


def main():
    for rally in [int(x) for x in sys.argv[1:]] or [7]:
        c = load(rally)
        imps, _ = load_impacts(rally=rally)
        series = paddle_series(c["npz"])
        _, bounds = claim_production(c)[0], claim_production(c)[1]
        t20 = turn_events(c["timing_ref"], 20.0)
        peaks = pose_peaks_all(c["npz"])
        missed = [t for t in imps
                  if not any(abs(t - b) <= 0.15 for b in bounds)]
        print(f"== rally {rally}: bounds {len(bounds)}, "
              f"missed taps {len(missed)}/{len(imps)}")
        for t in missed:
            has40 = any(abs(t - e) <= 0.15 for e in c["turns"])
            has20 = any(abs(t - e) <= 0.15 for e in t20)
            pk = [p for p in peaks if abs(t - p[0]) <= 0.15]
            pkz = max((p[1] for p in pk), default=None)
            rel, tid = prox_min(series, path_at(c["timing_ref"], t), t)
            print(f"   tap {t:7.2f}: turn40 {'Y' if has40 else '-'}  "
                  f"turn20 {'Y' if has20 else '-'}  "
                  f"peak z={pkz if pkz is None else round(pkz, 2)}  "
                  f"prox {rel:.2f}h")
        # parity/cadence flags on the production chain
        sides = {}
        for ta, tid, wx, wy, gx, gy in c["anchors"]:
            pass
        # side per bound from claiming track floors: reuse claim then
        # derive side by nearest anchor's track floor
        tsides = br.track_sides(c["floors"])
        bsides = []
        for b in bounds:
            near = min(c["anchors"], key=lambda a: abs(a[0] - b))
            bsides.append(tsides.get(int(near[1]))
                          if abs(near[0] - b) <= 0.3 else None)
        flags = 0
        for i in range(1, len(bounds)):
            gap = bounds[i] - bounds[i - 1]
            same = (bsides[i] is not None and bsides[i] == bsides[i - 1])
            n_missed_in = sum(1 for t in missed
                              if bounds[i - 1] < t < bounds[i])
            if same or gap > 1.7:
                flags += 1
                print(f"   HOLE FLAG {bounds[i-1]:.2f}->{bounds[i]:.2f} "
                      f"(gap {gap:.2f}s{', same side' if same else ''})"
                      f" contains {n_missed_in} missed taps")
        # how many missed taps sit inside NO flagged hole?
        infl = 0
        for t in missed:
            ok = False
            for i in range(1, len(bounds)):
                gap = bounds[i] - bounds[i - 1]
                same = (bsides[i] is not None
                        and bsides[i] == bsides[i - 1])
                if bounds[i - 1] < t < bounds[i] and (same or gap > 1.7):
                    ok = True
            infl += ok
        print(f"   flagged holes cover {infl}/{len(missed)} missed taps")


if __name__ == "__main__":
    main()
