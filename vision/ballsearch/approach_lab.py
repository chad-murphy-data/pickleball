"""Close-approach detection channel: local minima over time of
dist(decoded path, each track's paddle)/h as CONTACT candidates —
pose-free. Scored as recall vs taps / fake count, per rally.

Grid = timing_ref sample times; per track, rel(t) series; minima
below RTH with 0.35 s non-max suppression (same min-sep as pose).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, paddle_series, paddle_at   # noqa: E402

MIN_SEP = 0.35


def approach_events(c, series, rth):
    """(t, rel, tid) local minima of path-to-paddle distance."""
    evs = []
    for tid in series:
        ts, rels = [], []
        for t, x, y in c["timing_ref"]:
            p = paddle_at(series, tid, t, tol=0.12)
            if p is None:
                continue
            ts.append(t)
            rels.append(float(np.hypot(x - p[0], y - p[1]) / p[2]))
        for i in range(1, len(ts) - 1):
            if rels[i] <= rels[i - 1] and rels[i] <= rels[i + 1] \
                    and rels[i] <= rth:
                evs.append((ts[i], rels[i], tid))
    # non-max suppression: keep closest approach within MIN_SEP
    out = []
    for e in sorted(evs, key=lambda e: e[1]):
        if all(abs(e[0] - u[0]) >= MIN_SEP for u in out):
            out.append(e)
    return sorted(out)


def main():
    grand = {}
    for rally in [int(x) for x in sys.argv[1:]] or [7, 9, 10, 6]:
        c = load(rally)
        series = paddle_series(c["npz"])
        imps = [t for t in c["imps"]]
        print(f"== rally {rally}: {len(imps)} taps")
        for rth in (0.3, 0.5, 0.7):
            evs = [e for e in approach_events(c, series, rth)
                   if c["serve"] - 0.3 <= e[0] < c["end"]]
            rec = sum(1 for t in imps
                      if any(abs(t - e[0]) <= 0.15 for e in evs))
            fake = sum(1 for e in evs
                       if not any(abs(t - e[0]) <= 0.15 for t in imps))
            print(f"   rth={rth}: events {len(evs):3d}  recall "
                  f"{rec}/{len(imps)}  fakes {fake}")
            g = grand.setdefault(rth, [0, 0, 0])
            g[0] += rec
            g[1] += len(imps)
            g[2] += fake
    print("\n== pooled ==")
    for rth, (r, n, f) in sorted(grand.items()):
        print(f"   rth={rth}: recall {r}/{n} = {r/n:.3f}   fakes {f}")


if __name__ == "__main__":
    main()
