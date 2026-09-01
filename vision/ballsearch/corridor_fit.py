"""Product test: corridor-DP observations -> check-3 fits.

Per corridor, the DP path is TRUSTED only if it shadows the existing
decode where the decode has points (median agreement <= AGREE_PX over
>= AGREE_N frames) — a label-free gate. Trusted paths inject their
BRIDGE points (frames where visited has nothing within +/-1 frame)
into c["visited"], then the app-base + fit-validated-splits flow runs
unchanged. Question: does restoring the missing observations make the
merged dink segments visibly broken (rms > SPLIT_NEED), so splits fire
and the bounce ledger moves?

Usage: python3 corridor_fit.py <rally> [--thr 14] [--rth 0.5]
                               [--agree 10] [--rounds 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br                     # noqa: E402
from claim_lab import load, paddle_series, paddle_at   # noqa: E402
from approach_lab import approach_events        # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors)  # noqa
from corridor_chain import frame_candidates     # noqa: E402
from corridor_dp import dp_path                 # noqa: E402
from fit_lab import score_c3                    # noqa: E402
from split_lab import try_splits                # noqa: E402

AGREE_PX = 10.0
AGREE_N = 5
EDGE_F = 2          # never inject this close to a corridor edge


def app_bounds(c, series, rth):
    evs = [e for e in approach_events(c, series, rth)
           if c["serve"] - 0.3 <= e[0] < c["end"]]
    anc, zs = list(c["anchors"]), list(c["zs"])
    for t, rel, tid in evs:
        p = paddle_at(series, tid, t, tol=0.12)
        if p is None:
            continue
        anc.append((t, tid, p[0], p[1], p[0], p[1]))
        zs.append(3.0 - rel)
    dd = br.dedupe_anchors(anc, zs, br.track_sides(c["floors"]),
                           c["turns"])
    return sorted(br.claim_bounds(c["turns"], c["angs"],
                                  c["timing_ref"], dd)), evs


AB_PX = 12.0        # A-only vs B-only agreement radius


def _ab_consensus(cands, cor, t0, body=None):
    """A-only/B-only agreement midpoints, or {} if too few."""
    pA = dp_path(cands, cor, t0, anchor="A", body=body)
    pB = dp_path(cands, cor, t0, anchor="B", body=body)
    path = {}
    for f in set(pA) & set(pB):
        d = np.hypot(pA[f][0] - pB[f][0], pA[f][1] - pB[f][1])
        if d <= AB_PX:
            path[f] = ((pA[f][0] + pB[f][0]) / 2,
                       (pA[f][1] + pB[f][1]) / 2)
    return path if len(path) >= 3 else {}


def injections(c, cands, cors, agree_px, ab=False, abo=False,
               body=None):
    """[(f, x, y, src)] bridge points. Decode-shadowing gate where the
    decode has points to vouch; ab=True: corridors the decode CANNOT
    vouch for fall back to the two-sided independence gate (A-only and
    B-only DP paths agreeing within AB_PX). abo=True: corridors where
    the path DISAGREES with a sparse decode are re-tried the same way —
    the autopsy showed decode points there are often arm centroids
    ~23 px off, so dual-anchor consensus outranks them."""
    vis_by_f = {}
    for f, x, y in c["visited"]:
        vis_by_f.setdefault(int(f), []).append((x, y))
    t0 = c["t0"]
    out, kept, tried, ab_kept = [], 0, 0, 0
    for cor in cors:
        path = dp_path(cands, cor, t0, body=body)
        if not path:
            continue
        tried += 1
        ds = []
        for f, (x, y) in path.items():
            near = [np.hypot(vx - x, vy - y)
                    for df in (-1, 0, 1)
                    for vx, vy in vis_by_f.get(f + df, ())]
            if near:
                ds.append(min(near))
        judged = len(ds) >= AGREE_N
        if judged and float(np.median(ds)) > agree_px:
            if not abo:
                continue                 # decode says junk; defer to it
            path = _ab_consensus(cands, cor, t0, body)
            if not path:
                continue
            ab_kept += 1
            src = "ab"
        elif not judged:
            if not ab:
                continue
            path = _ab_consensus(cands, cor, t0, body)
            if not path:
                continue
            ab_kept += 1
            src = "ab"
        else:
            kept += 1
            src = "sh"
        fa = int(round((cor[0] - t0) * 60))
        fb = int(round((cor[1] - t0) * 60))
        for f, (x, y) in path.items():
            if f - fa < EDGE_F or fb - f < EDGE_F:
                continue
            if any((f + df) in vis_by_f for df in (-1, 0, 1)):
                continue
            out.append((f, float(x), float(y), src))
    return out, kept, tried, ab_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=14)
    ap.add_argument("--rth", type=float, default=0.5)
    ap.add_argument("--agree", type=float, default=AGREE_PX)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--nosplit", action="store_true")
    ap.add_argument("--targeted", action="store_true",
                    help="inject only into segments holding a split "
                         "candidate (an approach event far from every "
                         "bound) — healthy segments stay untouched")
    ap.add_argument("--ab", action="store_true",
                    help="corridors the decode cannot vouch for fall "
                         "back to the A-only/B-only agreement gate")
    ap.add_argument("--skipbase", action="store_true",
                    help="skip the (already-known) baseline fit")
    ap.add_argument("--cands", choices=("cc", "peak"), default="cc")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--body", action="store_true")
    ap.add_argument("--abo", action="store_true",
                    help="A/B consensus overrides a disagreeing "
                         "sparse decode")
    a = ap.parse_args()
    import corridor_dp
    if a.k:
        corridor_dp.K = a.k
    c = load(a.rally)
    series = paddle_series(c["npz"])
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cands = frame_candidates(a.rally, f_lo, f_hi, a.thr, mode=a.cands)
    body = (corridor_dp.body_points(c, f_lo, f_hi) if a.body
            else None)
    bounds, evs = app_bounds(c, series, a.rth)
    cors = corridors(c, series, bounds)
    inj, kept, tried, ab_kept = injections(c, cands, cors, a.agree,
                                           ab=a.ab, abo=a.abo,
                                           body=body)
    if a.targeted:
        from split_lab import MIN_GAP
        bl = sorted(bounds) + [c["end"]]
        segs = []
        for tc in (e[0] for e in evs):
            if any(abs(tc - b) < MIN_GAP for b in bl) or tc <= bl[0]:
                continue
            k = max(i for i in range(len(bl) - 1) if bl[i] < tc)
            if bl[k] + MIN_GAP < tc < bl[k + 1] - MIN_GAP:
                segs.append((bl[k], bl[k + 1]))
        inj = [r for r in inj
               if any(ta <= t0 + r[0] / 60.0 <= tb for ta, tb in segs)]
        print(f"  [targeted: {len(segs)} candidate segments, "
              f"{len(inj)} obs kept]")
    # diagnostic only: injected-point accuracy vs the owner's clicks
    truth = {int(round((t - t0) * 60)): (x, y)
             for t, x, y, v in load_truth(a.rally)}

    def _acc(rows):
        dd = [np.hypot(x - truth[f][0], y - truth[f][1])
              for f, x, y, s in rows if f in truth]
        return (f"{np.median(dd):.1f}px on {len(dd)}" if dd else "n/a")

    n_ab = sum(1 for r in inj if r[3] == "ab")
    print(f"rally {a.rally}: corridors {tried} -> shadow {kept} "
          f"+ ab {ab_kept} (agree<={a.agree}px), injected {len(inj)} "
          f"obs ({n_ab} ab) [med-vs-truth {_acc(inj)}; "
          f"ab-only {_acc([r for r in inj if r[3] == 'ab'])}]")
    c2 = dict(c)
    c2["visited"] = sorted(list(c["visited"])
                           + [(f, x, y) for f, x, y, s in inj])
    bevs = [e for e in c["turns"] if e not in set(bounds)]
    if not a.skipbase:
        print("-- baseline (no injection):")
        score_c3(c, bounds, bevs, quiet=True)
    print("-- injected, no splits:")
    score_c3(c2, bounds, bevs, quiet=True)
    if not a.nosplit:
        b2 = list(bounds)
        cands_t = [e[0] for e in evs]
        for rnd in range(a.rounds):
            b2n, n = try_splits(c2, b2, bevs, cands_t)
            print(f"  round {rnd+1}: {n} splits accepted")
            if not n:
                break
            b2 = b2n
        bevs2 = [e for e in c["turns"] if e not in set(b2)]
        print("-- injected + splits:")
        score_c3(c2, sorted(b2), bevs2, quiet=False)


if __name__ == "__main__":
    main()
