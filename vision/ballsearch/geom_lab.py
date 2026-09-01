"""geom_lab — stratified grading of the incumbent corridor tracker, the
diagnostic the HANDOFF orders BEFORE any corridor-geometry change
(next-thread to-do #1, 2026-09-01).

Question: of the clicks the incumbent (dp-ccS+body, W_P_SOFT=25)
misses, how many are GEOMETRY (the search box was not where the ball
was) versus DETECTION (no candidate within 12 px anywhere) versus
SELECTION (a candidate existed, the chain took another)?

Per click, on each arm's corridors:
  nocor    no corridor covers the click time
  outwin   click outside the chord window  -> split by direction
           (above / below the chord in image y, or lateral), by whether
           a candidate sits within 12 px at f+-1 ("recoverable by
           geometry alone"), and by endpoint-region (< 0.15 s from a
           corridor edge) vs mid-flight
  nocand   in window, no candidate within 12 px at f+-1
  cand-hit in window, candidate existed, incumbent within 12 px
  cand-miss ... incumbent NOT within 12 px -> split by whether the
           nearest-true candidate was inside the K=14 nearest-center
           pool (POOL-EXCLUDED = the DP never saw it), and whether the
           incumbent had any point at that frame (wrong vs skipped)
Per corridor: duration T, chord L, window, click excursion above/below
the chord, endpoint error |A - first click| / |B - last click| (prod vs
oracle separates contact-TIME error from paddle-choice error).
Coverage counterfactuals (pure geometry, no DP): what a taller window
wy' = min(cap, 55 + 0.3 L + kT * T) would cover, and what a top-K-by-p
pool would admit, for a grid of kT / cap / K. These numbers pick the
KNOBS of the fix instrument; the fix itself is tuned under its own
frozen rule (geom_fix.py), r6/r7 only.

Diagnostic only: truth is used to ask WHY. r9/r10 may be run here
(grading + autopsy is their licensed use) once their caches exist.

Usage: python3 geom_lab.py <rally>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                                   # noqa: E402
import spaghetti as spag                                    # noqa: E402
from claim_lab import load, paddle_series                   # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          window_at, decode_recall)

W_P_SOFT = 25.0
R = cdp.R_MAIN
EDGE_S = 0.15
KT_GRID = (0.0, 40.0, 80.0, 120.0, 160.0)
CAP_GRID = (170.0, 260.0, 400.0)
K_GRID = (14, 20, 30)


def near_cands(cc, f, tx, ty, r=R):
    """candidates within r of (tx,ty) at f-1..f+1: [(dist, f, idx)]."""
    out = []
    for df in (-1, 0, 1):
        for j, c_ in enumerate(cc.get(f + df, ())):
            d = float(np.hypot(c_[0] - tx, c_[1] - ty))
            if d <= r:
                out.append((d, f + df, j))
    return sorted(out)


def pool_ranks(cc, cor, t0, f, j):
    """ranks of candidate j at frame f among the in-window candidates
    at f: by distance to window center (incumbent pool order) and by
    learned p (descending). None if it is not in the window."""
    ta, tb, A, B, wx, wy = cor
    t = t0 + f / 60.0
    cx, cy, _, _ = window_at(cor, min(max(t, ta), tb))
    cs = cc.get(f, ())
    inw = [k for k, c_ in enumerate(cs)
           if abs(c_[0] - cx) <= wx and abs(c_[1] - cy) <= wy]
    if j not in inw:
        return None
    dist = {k: float(np.hypot(cs[k][0] - cx, cs[k][1] - cy)) for k in inw}
    byd = sorted(inw, key=lambda k: dist[k])
    byp = sorted(inw, key=lambda k: -float(cs[k][4]))
    return byd.index(j) + 1, byp.index(j) + 1, len(inw)


def run_arm(name, cors, cc, truth, t0, body, dec):
    cdp.W_P_SOFT = W_P_SOFT
    track = cdp.build_track(cc, cors, t0, body=body)
    hits = spag.hits(track, truth, t0)
    rows = []
    for i, (t, tx, ty, vis) in enumerate(truth):
        f = int(round((t - t0) * 60))
        hit = hits[i] is not None and hits[i] <= R
        cor = next((co for co in cors if co[0] <= t <= co[1]), None)
        row = dict(i=i, t=t, x=tx, y=ty, vis=vis, hit=hit, dec=dec[i],
                   cor=cor, stratum=None)
        if cor is None:
            row["stratum"] = "nocor"
            rows.append(row)
            continue
        ta, tb, A, B, wx, wy = cor
        cx, cy, _, _ = window_at(cor, t)
        dx, dy = tx - cx, ty - cy
        row.update(dx=dx, dy=dy, edge=min(t - ta, tb - t) < EDGE_S)
        nc = near_cands(cc, f, tx, ty)
        row["hascand"] = bool(nc)
        if abs(dx) > wx or abs(dy) > wy:
            row["stratum"] = "outwin"
            row["dir"] = ("above" if dy < -wy else
                          "below" if dy > wy else "lateral")
            rows.append(row)
            continue
        if not nc:
            row["stratum"] = "nocand"
            rows.append(row)
            continue
        row["stratum"] = "cand-hit" if hit else "cand-miss"
        if not hit:
            d0, f0, j0 = nc[0]
            rk = pool_ranks(cc, cor, t0, f0, j0)
            row["rank_d"], row["rank_p"], row["n_inw"] = (
                rk if rk else (None, None, None))
            row["excluded"] = rk is not None and rk[0] > cdp.K
            row["skipped"] = all(track.get(f + df) is None
                                 for df in (-1, 0, 1))
        rows.append(row)

    n = len(rows)
    cnt = {}
    for r in rows:
        cnt[r["stratum"]] = cnt.get(r["stratum"], 0) + 1
    nh = sum(r["hit"] for r in rows)
    print(f"== {name}: {len(cors)} corridors, {n} clicks, incumbent "
          f"r@12 {nh}, misses {n - nh}")
    print("   stratum    n   (of misses)")
    for s in ("nocor", "outwin", "nocand", "cand-miss", "cand-hit"):
        k = cnt.get(s, 0)
        print(f"   {s:9s} {k:4d}   {100 * k / max(1, n - nh):5.1f}%"
              if s != "cand-hit" else f"   {s:9s} {k:4d}")
    ow = [r for r in rows if r["stratum"] == "outwin"]
    if ow:
        print(f"   outwin breakdown (n={len(ow)}): "
              + ", ".join(f"{d} {sum(1 for r in ow if r['dir'] == d)}"
                          for d in ("above", "below", "lateral"))
              + f"; edge-region {sum(1 for r in ow if r['edge'])}"
              + f"; has-candidate<=12px {sum(1 for r in ow if r['hascand'])}"
              + f" (V {sum(1 for r in ow if r['hascand'] and r['vis'] == 'V')})")
        exc = [abs(r["dy"]) - r["cor"][5] for r in ow if r["dir"] != "lateral"]
        if exc:
            print(f"   vertical overshoot beyond wy: median {np.median(exc):.0f}"
                  f" px, p90 {np.percentile(exc, 90):.0f}, max {max(exc):.0f}")
    cm = [r for r in rows if r["stratum"] == "cand-miss"]
    if cm:
        ex = sum(1 for r in cm if r["excluded"])
        sk = sum(1 for r in cm if r["skipped"])
        inp = sum(1 for r in cm if r["rank_p"] is not None
                  and r["rank_p"] <= cdp.K)
        print(f"   cand-miss breakdown (n={len(cm)}): POOL-EXCLUDED "
              f"(nearest-true rank>{cdp.K} by center) {ex}; of those "
              f"inside top-{cdp.K} by p {sum(1 for r in cm if r['excluded'] and r['rank_p'] <= cdp.K)}"
              f"; skipped-frame {sk}; wrong-pick {len(cm) - sk}; "
              f"true cand in top-{cdp.K}-by-p {inp}")
        rd = [r["rank_d"] for r in cm if r["rank_d"]]
        print(f"   nearest-true rank by center: median {np.median(rd):.0f}"
              f", p90 {np.percentile(rd, 90):.0f}; in-window cands/frame "
              f"median {np.median([r['n_inw'] for r in cm if r['n_inw']]):.0f}")
    # per-corridor table
    print("   corridor      T     L   wx  wy | n  hit | excursion up/down"
          " | endpt err A / B")
    for cor in cors:
        ta, tb, A, B, wx, wy = cor
        rr = [r for r in rows if r["cor"] is cor]
        if not rr:
            continue
        up = min(r["dy"] for r in rr)
        dn = max(r["dy"] for r in rr)
        fa_ = [r for r in rr if r["t"] - ta < 0.1]
        fb_ = [r for r in rr if tb - r["t"] < 0.1]
        ea = (np.hypot(fa_[0]["x"] - A[0], fa_[0]["y"] - A[1])
              if fa_ else np.nan)
        eb = (np.hypot(fb_[-1]["x"] - B[0], fb_[-1]["y"] - B[1])
              if fb_ else np.nan)
        L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
        print(f"   {ta:7.2f}-{tb:6.2f} {tb - ta:4.2f} {L:5.0f} {wx:4.0f}"
              f" {wy:4.0f} | {len(rr):2d} {sum(r['hit'] for r in rr):3d}"
              f" | {up:5.0f} / {dn:4.0f} | {ea:5.0f} / {eb:5.0f}")
    return rows


def coverage(rows):
    """geometry-only counterfactuals over the outwin + cand-miss
    clicks: taller window (kT, cap) coverage of outwin clicks that HAVE
    a candidate; top-K-by-p admission of pool-excluded true cands."""
    ow = [r for r in rows if r["stratum"] == "outwin" and r["hascand"]]
    print(f"   taller-window coverage of outwin-with-candidate "
          f"(n={len(ow)}): wy' = min(cap, 55 + 0.3L + kT*T)")
    for cap in CAP_GRID:
        line = []
        for kT in KT_GRID:
            k = 0
            for r in ow:
                ta, tb, A, B, wx, wy = r["cor"]
                L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
                wy2 = min(cap, 55 + 0.3 * L + kT * (tb - ta))
                if abs(r["dx"]) <= wx and abs(r["dy"]) <= wy2:
                    k += 1
            line.append(f"kT={kT:3.0f}: {k:3d}")
        print(f"     cap {cap:3.0f}  " + "  ".join(line))
    cm = [r for r in rows if r["stratum"] == "cand-miss"
          and r["rank_d"] is not None]
    print(f"   pool admission of cand-miss true candidates (n={len(cm)}):")
    for K in K_GRID:
        byd = sum(1 for r in cm if r["rank_d"] <= K)
        byp = sum(1 for r in cm if r["rank_p"] <= K)
        print(f"     K={K:2d}  by-center {byd:3d}   by-p {byp:3d}")


def main():
    rally = int(sys.argv[1])
    pxs = "_x" if rally in (6, 7) else ""
    c = load(rally)
    series = paddle_series(c["npz"])
    truth = load_truth(rally)
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cc = spag.cands_cached(rally, f_lo, f_hi, 14, "cc", lrn=True, pxs=pxs)
    body = cdp.body_points(c, f_lo, f_hi)
    dec = decode_recall(c, truth)
    print(f"rally {rally}: {len(truth)} V/S clicks, decode@12 "
          f"{sum(dec)}/{len(dec)}, p-cache '{pxs}', K={cdp.K}")
    for name, cors in (("prod", corridors(c, series,
                                          prod_contacts(c, series, 0.5))),
                       ("oracle", corridors(c, series, list(c["imps"])))):
        rows = run_arm(name, cors, cc, truth, t0, body, dec)
        coverage(rows)


if __name__ == "__main__":
    main()
