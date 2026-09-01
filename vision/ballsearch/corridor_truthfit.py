"""Observability autopsy for the corridor product FAIL: would the
split trigger fire even under PERFECT observations?

For each targeted candidate segment (same selection as corridor_fit
--targeted), fit fit_segment on (a) the current visited stream and
(b) the owner's V/S clicks alone — the best observations that can
ever exist — and print rms vs SPLIT_NEED, plausibility, and the
fitted arc count (bounces = arcs-1). Also the best gridded split on
the click stream. If click-rms sits under SPLIT_NEED, no observation
recovery can ever license a residual-triggered split there: the
one-arc lie fits the true path. Diagnostic use of labels only.

Usage: python3 corridor_truthfit.py <rally> [--rth 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br                     # noqa: E402
import court3d as c3                            # noqa: E402
from claim_lab import load, paddle_series       # noqa: E402
from corridor_lab import load_truth             # noqa: E402
from corridor_fit import app_bounds             # noqa: E402
from split_lab import (seg_fit, SPLIT_NEED, MIN_GAP, MARGIN,  # noqa
                       GRID_S, GRID_W)


def fit_line(P, obs, ta, tb, bevs, tag):
    seg, o = seg_fit(P, obs, ta, tb, bevs)
    if seg is None:
        print(f"    {tag:8s} n={len(o):3d}  (too few obs)")
        return None
    print(f"    {tag:8s} n={len(o):3d}  rms {seg['rms']:5.2f} px  "
          f"ok={seg['ok']}  arcs={len(seg['arcs'])}"
          f"  (bounces {len(seg['arcs']) - 1})")
    return seg


def best_split(P, obs, ta, tb, tc, bevs):
    out = None
    ts = tc - GRID_W
    while ts <= tc + GRID_W + 1e-9:
        t_s = round(ts, 3)
        ts += GRID_S
        if not (ta + 0.15 < t_s < tb - 0.15):
            continue
        L, oL = seg_fit(P, obs, ta, t_s, bevs)
        R, oR = seg_fit(P, obs, t_s, tb, bevs)
        if not (L and L["ok"] and R and R["ok"]):
            continue
        rms = ((len(oL) * L["rms"] + len(oR) * R["rms"])
               / max(1, len(oL) + len(oR)))
        arcs = len(L["arcs"]) + len(R["arcs"])
        if out is None or rms < out[0]:
            out = (rms, t_s, arcs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--rth", type=float, default=0.5)
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    P = c["P"]
    bounds, evs = app_bounds(c, series, a.rth)
    bevs = [e for e in c["turns"] if e not in set(bounds)]
    obs_v = [(c["t0"] + f / 60.0, x, y, 1.0) for f, x, y in c["visited"]]
    obs_t = [(t, x, y, 1.0) for t, x, y, v in load_truth(a.rally)]
    bl = sorted(bounds) + [c["end"]]
    segs = {}
    for tc in (e[0] for e in evs):
        if any(abs(tc - b) < MIN_GAP for b in bl) or tc <= bl[0]:
            continue
        k = max(i for i in range(len(bl) - 1) if bl[i] < tc)
        if bl[k] + MIN_GAP < tc < bl[k + 1] - MIN_GAP:
            segs.setdefault((bl[k], bl[k + 1]), []).append(tc)
    print(f"rally {a.rally}: {len(segs)} candidate segments  "
          f"(SPLIT_NEED {SPLIT_NEED} px, MARGIN {MARGIN})")
    n_blind = 0
    for (ta, tb), tcs in sorted(segs.items()):
        print(f"  seg {ta:7.2f}-{tb:7.2f}  dur {tb - ta:4.2f}s  "
              f"cands {['%.2f' % t for t in tcs]}")
        fit_line(P, obs_v, ta, tb, bevs, "visited")
        st = fit_line(P, obs_t, ta, tb, bevs, "clicks")
        if st is not None and st["rms"] < SPLIT_NEED:
            n_blind += 1
        for tc in tcs:
            bs = best_split(P, obs_t, ta, tb, tc, bevs)
            if bs:
                print(f"    clicks-split @{bs[1]:7.2f}: rms {bs[0]:5.2f}"
                      f" px  arcs={bs[2]}")
            else:
                print(f"    clicks-split @cand {tc:7.2f}: none plausible")
    print(f"  => {n_blind}/{len(segs)} segments where PERFECT obs fit "
          f"one segment under SPLIT_NEED (residual trigger blind)")


if __name__ == "__main__":
    main()
