"""Fit-validated contact splits: approach events with no nearby bound
propose splitting their containing segment; boundary time gridded
+/-0.15 s; accept only if both halves fit plausibly AND weighted rms
beats the unsplit segment by MARGIN px (fit_segment's own bounce-vs-arc
acceptance margin). The r7 orphan lesson: never place a bound on a raw
event time — let the fit choose within the grid.

Usage: python3 split_lab.py <rally> [--app RTH] [--rounds N]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br           # noqa: E402
import court3d as c3                  # noqa: E402
from claim_lab import (load, paddle_series, claim_production,  # noqa: E402
                       paddle_at)
from approach_lab import approach_events                       # noqa: E402
from fit_lab import score_c3                                   # noqa: E402

GRID_S = 1 / 15
GRID_W = 0.15
MARGIN = 0.5          # px: split must beat unsplit rms by this
MIN_GAP = 0.30        # candidate must sit this far from every bound
SPLIT_NEED = 2.0      # px: only rescue BROKEN segments (clean flight
                      # fits sit ~0.6-1.5; splitting those is never
                      # necessary and cost r7 a crossing)


def seg_fit(P, obs_all, t0, t1, events):
    obs = [o for o in obs_all
           if t0 + br.END_TRIM_S <= o[0] <= t1 - br.END_TRIM_S]
    if len(obs) < 5:
        return None, obs
    seg = c3.fit_segment(P, obs, t0, t1, events)
    seg["ok"] = br._plausible(seg)
    return seg, obs


def try_splits(c, bounds, bevs, cands, verbose=True):
    """one round: for each candidate inside a segment, accept the best
    gridded split that clears the margin. Returns (new_bounds, n_acc)."""
    P = c["P"]
    t0c = c["t0"]
    obs_all = [(t0c + f / 60.0, x, y, 1.0) for f, x, y in c["visited"]]
    bl = sorted(bounds) + [c["end"]]
    per_seg = {}                       # k -> (gain-sorted best split)
    for tc in cands:
        if any(abs(tc - b) < MIN_GAP for b in bl) or tc <= bl[0]:
            continue
        k = max(i for i in range(len(bl) - 1) if bl[i] < tc)
        ta, tb = bl[k], bl[k + 1]
        if not (ta + MIN_GAP < tc < tb - MIN_GAP):
            continue
        orig, obs = seg_fit(P, obs_all, ta, tb, bevs)
        orig_rms = orig["rms"] if orig else float("inf")
        if orig_rms < SPLIT_NEED:      # clean segment — leave it alone
            continue
        best = None
        ts = tc - GRID_W
        while ts <= tc + GRID_W + 1e-9:
            t_s = round(ts, 3)
            ts += GRID_S
            if not (ta + 0.15 < t_s < tb - 0.15):
                continue
            L, oL = seg_fit(P, obs_all, ta, t_s, bevs)
            R, oR = seg_fit(P, obs_all, t_s, tb, bevs)
            if not (L and L["ok"] and R and R["ok"]):
                continue
            rms = ((len(oL) * L["rms"] + len(oR) * R["rms"])
                   / max(1, len(oL) + len(oR)))
            if best is None or rms < best[0]:
                best = (rms, t_s)
        if best and best[0] < orig_rms - MARGIN:
            gain = orig_rms - best[0]
            if k not in per_seg or gain > per_seg[k][0]:
                per_seg[k] = (gain, tc, best[1], orig_rms, best[0])
    acc = [v for v in per_seg.values()]
    for gain, tc, t_s, r0, r1 in acc:
        if verbose:
            print(f"    split @{t_s:7.2f} (cand {tc:7.2f}): "
                  f"rms {r0:5.2f} -> {r1:5.2f} px")
    return sorted(set(bounds) | {t for _, _, t, _, _ in acc}), len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--app", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--base", choices=["prod", "app"], default="app")
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    evs = [e for e in approach_events(c, series, a.app)
           if c["serve"] - 0.3 <= e[0] < c["end"]]
    if a.base == "app":
        anc = list(c["anchors"])
        zs = list(c["zs"])
        for t, rel, tid in evs:
            p = paddle_at(series, tid, t, tol=0.12)
            if p is None:
                continue
            anc.append((t, tid, p[0], p[1], p[0], p[1]))
            zs.append(3.0 - rel)
        dd = br.dedupe_anchors(anc, zs, br.track_sides(c["floors"]),
                               c["turns"])
        bounds = br.claim_bounds(c["turns"], c["angs"], c["timing_ref"],
                                 dd)
    else:
        _, bounds = claim_production(c)
    bevs = [e for e in c["turns"] if e not in set(bounds)]
    print(f"rally {a.rally} base={a.base}: {len(bounds)} bounds")
    cands = [e[0] for e in evs]
    for rnd in range(a.rounds):
        bounds2, n = try_splits(c, bounds, bevs, cands)
        print(f"  round {rnd+1}: {n} splits accepted")
        if not n:
            break
        bounds = bounds2
    bevs = [e for e in c["turns"] if e not in set(bounds)]
    score_c3(c, sorted(bounds), bevs, quiet=False)


if __name__ == "__main__":
    main()
