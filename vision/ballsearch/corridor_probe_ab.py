"""Plumbing probe for the ab-0 result: for every corridor the decode
cannot vouch for, print the A-only and B-only DP paths' sizes, overlap,
and median disagreement — distinguishes 'paths exist but disagree'
(mechanism) from 'paths empty' (possible anchor-code bug).

Usage: python3 corridor_probe_ab.py <rally> [--thr 14] [--rth 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, paddle_series               # noqa: E402
from corridor_lab import corridors                      # noqa: E402
from corridor_chain import frame_candidates             # noqa: E402
from corridor_dp import dp_path                         # noqa: E402
from corridor_fit import app_bounds, AGREE_N, AB_PX     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=14)
    ap.add_argument("--rth", type=float, default=0.5)
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cands = frame_candidates(a.rally, f_lo, f_hi, a.thr)
    bounds, _ = app_bounds(c, series, a.rth)
    cors = corridors(c, series, bounds)
    vis_by_f = {}
    for f, x, y in c["visited"]:
        vis_by_f.setdefault(int(f), []).append((x, y))
    print(f"rally {a.rally}: {len(cors)} corridors; unjudged ones:")
    for cor in cors:
        path = dp_path(cands, cor, t0)
        ds = []
        for f, (x, y) in path.items():
            near = [np.hypot(vx - x, vy - y) for df in (-1, 0, 1)
                    for vx, vy in vis_by_f.get(f + df, ())]
            if near:
                ds.append(min(near))
        if len(ds) >= AGREE_N:
            continue
        pA = dp_path(cands, cor, t0, anchor="A")
        pB = dp_path(cands, cor, t0, anchor="B")
        com = sorted(set(pA) & set(pB))
        dd = [float(np.hypot(pA[f][0] - pB[f][0],
                             pA[f][1] - pB[f][1])) for f in com]
        agr = sum(1 for d in dd if d <= AB_PX)
        print(f"  [{cor[0]:7.2f}-{cor[1]:7.2f}s] both={len(path):3d} "
              f"A={len(pA):3d} B={len(pB):3d} common={len(com):3d} "
              f"medAB={np.median(dd) if dd else float('nan'):6.1f}px "
              f"agree<= {AB_PX:.0f}px: {agr}")


if __name__ == "__main__":
    main()
