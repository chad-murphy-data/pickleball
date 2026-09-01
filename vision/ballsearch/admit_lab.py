"""Anchor ADMISSION lab: can a hybrid rule (z >= Z_MIN) OR
(z >= zlo AND pathdist <= rth) raise anchor recall without a fake
flood?  Pose channel only (blur channel unchanged).

Recall = taps with an admitted peak within 0.15 s. Fakes = admitted
peaks (deduped 0.35 s per side like production min-sep already is)
not within 0.15 s of any tap.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import hitter_chain as hc                              # noqa: E402
from claim_lab import (load, paddle_series, path_at,   # noqa: E402
                       paddle_at)
from hole_audit import pose_peaks_all                  # noqa: E402


def peak_rel(c, series, t, tid):
    px, py = path_at(c["timing_ref"], t)
    p = paddle_at(series, tid, t, tol=0.12)
    if p is None:
        return float("inf")
    return float(np.hypot(px - p[0], py - p[1]) / p[2])


def main():
    grand = {}
    for rally in [int(x) for x in sys.argv[1:]] or [7, 9, 10, 6]:
        c = load(rally)
        series = paddle_series(c["npz"])
        imps = c["imps"]
        peaks = [p for p in pose_peaks_all(c["npz"])
                 if c["serve"] - 0.3 <= p[0] < c["end"]]
        rels = [peak_rel(c, series, p[0], int(p[2])) for p in peaks]
        zs = [p[1] for p in peaks]
        print(f"== rally {rally}: {len(imps)} taps, {len(peaks)} "
              f"pose peaks in window")
        rules = [("z>=1.2 (prod)", lambda z, r: z >= 1.2)]
        for zlo in (0.5, 0.8):
            for rth in (0.4, 0.6):
                rules.append((f"z>=1.2 | z>={zlo}&rel<={rth}",
                              lambda z, r, zl=zlo, rt=rth:
                              z >= 1.2 or (z >= zl and r <= rt)))
        rules.append(("rel<=0.5 only", lambda z, r: r <= 0.5))
        for name, f in rules:
            adm = [(p[0], z) for p, z, r in zip(peaks, zs, rels)
                   if f(z, r)]
            rec = sum(1 for t in imps
                      if any(abs(t - a[0]) <= 0.15 for a in adm))
            fake = sum(1 for a in adm
                       if not any(abs(t - a[0]) <= 0.15 for t in imps))
            print(f"   {name:24s}: admitted {len(adm):3d}  "
                  f"tap recall {rec}/{len(imps)}  fakes {fake}")
            g = grand.setdefault(name, [0, 0, 0])
            g[0] += rec
            g[1] += len(imps)
            g[2] += fake
    print("\n== pooled ==")
    for name, (r, n, f) in grand.items():
        print(f"   {name:24s}: recall {r}/{n} = {r/n:.3f}   fakes {f}")


if __name__ == "__main__":
    main()
