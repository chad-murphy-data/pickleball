"""Corridor v3 — global DP selection (the principled meet-in-middle).

Per corridor (contact A -> contact B): shortest path through the
per-frame detector candidates (windowed as v1), cost = velocity
CHANGE (accel — the ball is smooth, body fragments jitter, static
junk never moves like a flight) + gap penalties + soft anchoring to
both paddle endpoints. Viterbi with velocity carried on the
backpointer (standard tracking approximation). No training.

Usage: python3 corridor_dp.py <rally> [--thr 14] [--rth 0.5] [--fit]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, paddle_series           # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          window_at, decode_recall, R_MAIN)
from corridor_chain import frame_candidates          # noqa: E402

K = 14              # candidates kept per frame (nearest window center)
GAP = 6             # max frames bridged by one transition
W_ACC = 1.0         # cost per px/frame of velocity change
W_GAP = 9.0         # cost per skipped frame
W_END = 0.8         # cost per px of endpoint miss
END_R = 70.0        # must start/end this close to the paddles
END_F = 5           # ... within this many frames of the corridor edge
SPEED_MAX = 110.0
ACC_MAX = 70.0

# "humans aren't the ball" (owner-specced 2026-09-01): candidates near
# skeleton extremities — the measured impostors: shoes, wristbands,
# arm fragments — pay a soft DP cost. Soft, not a veto: the ball
# legitimately sits at the hitter's wrist at contact (endpoint frames
# are exempt) and passes near the far player's feet in image space on
# kitchen descents. ALL pose rows are used, refs included.
BODY_KPT = (7, 8, 9, 10, 13, 14, 15, 16)   # elbows wrists knees ankles
R_BODY = 16.0
W_BODY = 25.0

# Learned-emission SOFT term (2026-09-01, pre-registered in
# swing_explore_notes): taking a candidate costs W_P_SOFT * (1 - p)
# where p is emission.py's ball probability riding as the candidate's
# 5th field. 0.0 = off (and 4-tuple candidates are always exempt), so
# every existing caller is unchanged. Soft, not a cull: the hard
# p-filter killed whole chains on faint fast drives (r10 297.87 dp
# 20 -> 0); here a low-p node merely competes against W_GAP skips.
# W_P_SOFT is tuned on r6/r7 ONLY (cross-fold p), never on r9/r10.
W_P_SOFT = 0.0


def _pc(p):
    return W_P_SOFT * (1.0 - p) if (W_P_SOFT and p is not None) else 0.0


def body_points(c, f_lo, f_hi):
    """frame -> np.array of extremity (x, y) from the pose npz."""
    z = np.load(c["npz"])
    fr = np.round((np.asarray(z["t"]) - c["t0"]) * 60).astype(int)
    kpt, kpc = z["kpt"], z["kpc"]
    out = {}
    for i in np.where((fr >= f_lo) & (fr <= f_hi))[0]:
        pts = [kpt[i, j] for j in BODY_KPT if kpc[i, j] >= 0.3]
        if pts:
            out.setdefault(int(fr[i]), []).extend(
                (float(p[0]), float(p[1])) for p in pts)
    return {k: np.asarray(v) for k, v in out.items()}


def _body_costs(cf, fa, fb, body):
    """precomputed per-(frame, cand) soft cost; interior frames only."""
    bc = {}
    for f, cs in cf.items():
        if not cs or f - fa <= END_F or fb - f <= END_F:
            bc[f] = [0.0] * len(cs)
            continue
        arr = None
        for df in (0, -1, 1):
            arr = body.get(f + df)
            if arr is not None and len(arr):
                break
        if arr is None or not len(arr):
            bc[f] = [0.0] * len(cs)
            continue
        row = []
        for c_ in cs:
            x, y = c_[0], c_[1]
            d = float(np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)))
            row.append(W_BODY * max(0.0, 1.0 - d / R_BODY))
        bc[f] = row
    return bc


def corridor_cands(cands, cor, t0):
    ta, tb, A, B, wx, wy = cor
    fa, fb = int(round((ta - t0) * 60)), int(round((tb - t0) * 60))
    out = {}
    for f in range(fa, fb + 1):
        t = t0 + f / 60.0
        cx, cy, _, _ = window_at(cor, min(max(t, ta), tb))
        cs = [(c_[0], c_[1], c_[3], c_[4] if len(c_) > 4 else None)
              for c_ in cands.get(f, ())
              if abs(c_[0] - cx) <= wx and abs(c_[1] - cy) <= wy]
        cs.sort(key=lambda c: np.hypot(c[0] - cx, c[1] - cy))
        out[f] = cs[:K]
    return fa, fb, out


def dp_path(cands, cor, t0, anchor="both", body=None):
    """anchor: 'both' = position-tied at A and B; 'A' = tied at A only
    (far end free in position, corridor still covered in time); 'B' =
    mirror. A-only vs B-only agreement is the two-sided independence
    gate for corridors the decode cannot vouch for. body = frame ->
    extremity array; candidates near a body extremity pay W_BODY."""
    fa, fb, cf = corridor_cands(cands, cor, t0)
    A, B = cor[2], cor[3]
    frames = [f for f in range(fa, fb + 1) if cf[f]]
    if len(frames) < 3:
        return {}
    bc = (_body_costs(cf, fa, fb, body) if body is not None
          else {f: [0.0] * len(cf[f]) for f in cf})
    # state: (f, j) -> [cost, prev_state, vel]
    st = {}
    for f in frames:
        if f - fa > END_F:
            break
        for j, (x, y, pk, p) in enumerate(cf[f]):
            if anchor in ("both", "A"):
                d = float(np.hypot(x - A[0], y - A[1]))
                if d > END_R:
                    continue
                c0 = W_END * d
            else:
                c0 = 0.0
            st[(f, j)] = [c0 + W_GAP * (f - fa) + bc[f][j] + _pc(p),
                          None, None]
    if not st:
        return {}
    order = sorted(st) + []
    # forward sweep frame by frame
    fset = sorted(set(frames))
    idx = {f: i for i, f in enumerate(fset)}
    for f in fset:
        for j in range(len(cf[f])):
            s = st.get((f, j))
            if s is None:
                continue
            x, y, pk, _p = cf[f][j]
            i0 = idx[f]
            for g in fset[i0 + 1:]:
                if g - f > GAP:
                    break
                for k, (qx, qy, qpk, qp) in enumerate(cf[g]):
                    v = np.array([(qx - x) / (g - f),
                                  (qy - y) / (g - f)])
                    sp = float(np.hypot(*v))
                    if sp > SPEED_MAX:
                        continue
                    if s[2] is None:
                        acc = 0.0
                        c_extra = 0.03 * sp
                    else:
                        acc = float(np.hypot(*(v - s[2])))
                        if acc > ACC_MAX:
                            continue
                        c_extra = 0.0
                    c = (s[0] + W_ACC * acc + W_GAP * (g - f - 1)
                         + c_extra - 0.04 * min(qpk, 40.0)
                         + bc[g][k] + _pc(qp))
                    cur = st.get((g, k))
                    if cur is None or c < cur[0]:
                        st[(g, k)] = [c, (f, j), v]
    # terminate near B (or position-free when B is not the anchor)
    best, bc = None, float("inf")
    for (f, j), s in st.items():
        if fb - f > END_F:
            continue
        x, y, pk, _p = cf[f][j]
        if anchor in ("both", "B"):
            d = float(np.hypot(x - B[0], y - B[1]))
            if d > END_R:
                continue
            ce = W_END * d
        else:
            ce = 0.0
        c = s[0] + ce + W_GAP * (fb - f)
        if c < bc:
            bc, best = c, (f, j)
    if best is None:
        return {}
    path = {}
    cur = best
    while cur is not None:
        f, j = cur
        x, y = cf[f][j][:2]
        path[f] = (float(x), float(y))
        cur = st[cur][1]
    return path


def build_track(cands, cors, t0, disp=None, body=None):
    track = {}
    for cor in cors:
        if disp:
            ta, tb, A, B, wx, wy = cor
            cor = (ta, tb, (A[0] + disp[0], A[1] + disp[1]),
                   (B[0] + disp[0], B[1] + disp[1]), wx, wy)
        track.update(dp_path(cands, cor, t0, body=body))
    return track


def score(track, truth, t0, dec, tag):
    n = len(truth)
    hits = {R: 0 for R in (8, 12, 20)}
    added = have = 0
    for (t, tx, ty, vis), d in zip(truth, dec):
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        if p is None:
            continue
        have += 1
        dd = float(np.hypot(p[0] - tx, p[1] - ty))
        for R in hits:
            hits[R] += dd <= R
        added += dd <= R_MAIN and not d
    prec = hits[R_MAIN] / max(1, have)
    print(f"  {tag:12s} trackpts {len(track):4d}  at-click {have}/{n}"
          f"  r@8 {hits[8]}  r@12 {hits[12]}  r@20 {hits[20]}"
          f"  prec@12 {prec:.2f}  ADDED@12 {added}")


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=14)
    ap.add_argument("--rth", type=float, default=0.5)
    ap.add_argument("--cands", choices=("cc", "peak"), default="cc")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--body", action="store_true",
                    help="humans-aren't-the-ball soft cost from pose")
    a = ap.parse_args()
    K = a.k
    c = load(a.rally)
    series = paddle_series(c["npz"])
    truth = load_truth(a.rally)
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cands = frame_candidates(a.rally, f_lo, f_hi, a.thr, mode=a.cands)
    body = body_points(c, f_lo, f_hi) if a.body else None
    dec = decode_recall(c, truth)
    mc = float(np.mean([len(v) for v in cands.values()]))
    print(f"rally {a.rally}: {len(truth)} V/S clicks, "
          f"decode@12 {sum(dec)}/{len(dec)}  "
          f"[cands={a.cands} {mc:.0f}/frame, K={K}, "
          f"body={'on' if a.body else 'off'}]")
    rng = np.random.default_rng(20260901)
    for name, times in (("prod", prod_contacts(c, series, a.rth)),
                        ("oracle", list(c["imps"]))):
        cors = corridors(c, series, times)
        tr = build_track(cands, cors, t0, body=body)
        score(tr, truth, t0, dec, name)
        for vis in ("V", "S"):
            tt = [x for x in truth if x[3] == vis]
            dd = [d for x, d in zip(truth, dec) if x[3] == vis]
            score(tr, tt, t0, dd, f"  [{vis}]")
        for kk in range(2):
            d = (float(rng.uniform(160, 240)) * rng.choice([-1, 1]),
                 float(rng.uniform(80, 140)) * rng.choice([-1, 1]))
            score(build_track(cands, cors, t0, disp=d, body=body),
                  truth, t0, dec, f"{name}-null{kk}")


if __name__ == "__main__":
    main()
