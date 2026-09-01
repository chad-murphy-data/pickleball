"""Spaghetti trail matcher (owner-designed 2026-09-01: "a ball hit
from here can take the following approximate paths with the following
approximate likelihood — find paths that match; look for the ball and
say which trail the detections fit best").

Per corridor (contact A -> contact B): enumerate PHYSICAL trails —
drag-ballistic flights (court3d physics, k from the measured prior) —
and score each by how well the motion detections support it. No
frame-chaining: junk that lies along no physical trail supports
nothing; holes are bridged by the trail itself.

Trail families (all closed-form, no fitting at proposal time):
  DIRECT  pA on the camera ray through the A paddle pixel at height
          zA, pB likewise at zB; the drag arc connecting two 3D points
          in time T is unique: v0 = vt + (pB - pA - vt*T)*k/(1-e^-kT).
  BOUNCE  launch states sampled from the measured prior (speed x loft
          x lateral aim toward B) at pA; integrate to z=0 -> bounce
          point + time; arc 2 = the unique drag arc bounce -> pB.
          Restitution gates from the measured bounce physics.

Scoring: per frame, best candidate within R_SUP of the trail pixel,
weight (1 - d/R)*(1 + peak-term), near-extremity candidates damped
(the measured 50%-junk rule: judge the held point). Winner refined by
two local parameter-grid rounds ("given points 1..k, refine").

v3 (owner steer): the BOUNCE family proposes ONLY the shot book — the
~53 sanitized rms-validated click-fit launches (dink/drop/roll/drive/
smash/lob), height-matched to the contact; every trail pays W_MODE per
unit distance from its nearest real shot shape; corridors with no
strand above ABSTAIN net-evidence emit nothing. "Which trail do the
detections fit best" is now a choice among named real shapes.

Grading: owner clicks (V/S), identical metric to corridor_dp.score,
displaced-anchor nulls. Baseline = corridor_dp cc+body in-process.
Per-corridor table for the success/failure autopsy.

Usage: python3 spaghetti.py <rally> [--thr 14] [--rth 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                                  # noqa: E402
import corridor_dp as cdp                             # noqa: E402
from claim_lab import load, paddle_series, paddle_at  # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          window_at, decode_recall, R_MAIN)
from corridor_chain import frame_candidates           # noqa: E402

SP = Path(__file__).parent
G = c3.G
CAP = 120           # candidates kept per frame in the corridor window
R_SUP = 8.0         # support radius around the trail pixel (v2: was
                    # 14 — true supporters sit within a few px; the
                    # loose kernel let 14px junk pay)
W_PK = 0.02         # appearance/motion bonus per unit peak (cap 40)
BODY_FACT = 0.3     # near-extremity candidates damped (50% junk rule)
NEAR_BODY = 16.0
K_DRAG = 0.36       # measured median drag
MIN_COR_S = 0.15
N_PROBE = 8         # v2: random probes/frame — the per-frame null.
                    # score = support MINUS what a random window point
                    # would collect, so junk-dense frames carry no
                    # evidence (v1 objective rewarded junk density:
                    # sup 49/49 corridors with 0 hits)
PEN_BOUNCE = 1.5    # complexity penalty: bounce only when data demand
TOP_M = 6           # refine the top-M trails, not just the argmax

ABSTAIN = 2.0       # v3: emit nothing when no strand clears this net
                    # evidence (frames of above-null support)
P_LO, P_HI = 0.25, 1.75   # learned-emission appearance term:
                          # app = P_LO + P_HI * p  (p from emission.py,
                          # trained on r6/r7 clicks ONLY)
W_MODE = 0.7        # v3: penalty per unit distance from the nearest
                    # REAL shot shape (owner steer 2026-09-01: the
                    # sport has a limited repertoire — dinks, drops,
                    # rolls, drives, smashes, lobs; "any physics
                    # option that works" is too broad a family)

PRIOR = json.loads((SP / "launch_prior.json").read_text())
SP5, SP95 = PRIOR["speed"]["5"], PRIOR["speed"]["95"]
SP98 = PRIOR["speed"]["98"]
SPEEDS = (15.0, 21.0, 25.0, 32.0, 38.0, 47.0, 57.0, 70.0, 85.0, 95.0)
LOFTS = (-8.0, -2.0, 4.0, 10.0, 18.0, 30.0, 45.0, 60.0)
LATS = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)
LATS_REP = (-14.0, -7.0, 0.0, 7.0, 14.0)
ZS = (0.8, 2.0, 3.5, 5.5, 7.5)
KS = (0.15, 0.36, 0.8)


def _lib():
    """the shot book: every rms-validated click-fit launch, lightly
    sanitized (toward-opponent, sane height/speed — segment-edge fit
    artifacts out). Columns: speed, loft, z0."""
    rows = [(l["speed"], l["loft"], max(0.3, l["z0"]))
            for l in PRIOR["launches"]
            if l["fwd"] > 2.0 and -0.5 <= l["z0"] <= 9.0
            and 12.0 <= l["speed"] <= 105.0]
    return np.asarray(rows)


LIB = _lib()
LIB_SP, LIB_LF = LIB[:, 0], LIB[:, 1]


def mode_name(sp, loft):
    if loft >= 42:
        return "lob"
    if sp < 30:
        return "dink"
    if sp < 47:
        return "drop" if loft >= 14 else "roll"
    return "smash" if loft <= -6 else "drive"


# ------------------------------------------------------ ray geometry

def cam(P):
    C = c3.camera_center(P)
    Minv = np.linalg.inv(P[:, :3])
    Hg_inv = np.linalg.inv(P[:, [0, 1, 3]])
    return C, Minv, Hg_inv


def ray_dir(Minv, px):
    d = Minv @ np.array([px[0], px[1], 1.0])
    return d / np.linalg.norm(d)


def pt_at_z(C, d, z):
    if abs(d[2]) < 1e-9:
        return None
    s = (z - C[2]) / d[2]
    if s <= 0:
        return None
    return C + s * d


def ground_xy(Hg_inv, px):
    v = Hg_inv @ np.array([px[0], px[1], 1.0])
    return v[:2] / v[2]


def hitter_xy(c, series, npz, Hg_inv, t, A):
    """court (x, y) of the hitter's feet: track whose paddle is nearest
    A at t, ankle keypoints -> ground-plane inversion."""
    best = None
    for tid in series:
        p = paddle_at(series, tid, t, tol=0.12)
        if p is None:
            continue
        d = float(np.hypot(p[0] - A[0], p[1] - A[1]))
        if best is None or d < best[0]:
            best = (d, tid)
    if best is None or best[0] > 90:
        return None
    tid = best[1]
    fr = np.asarray(npz["t"])
    m = np.where((npz["track"] == tid) & (np.abs(fr - t) <= 0.15))[0]
    pts = []
    for i in m:
        for j in (15, 16):
            if npz["kpc"][i, j] >= 0.3:
                pts.append(npz["kpt"][i, j])
    if not pts:
        return None
    u = np.mean(pts, axis=0)
    xy = ground_xy(Hg_inv, u)
    if not (-6 <= xy[0] <= 26 and -6 <= xy[1] <= 50):
        return None
    return xy


# --------------------------------------------------- trail buildings

def drag_pos(p0, v0, k, tau):
    """p0 (n,3), v0 (n,3), k (n,), tau (F,) -> (n,F,3)."""
    k = np.asarray(k, float)[:, None, None]
    vt = np.zeros((1, 1, 3))
    vt = np.repeat(vt, p0.shape[0], 0) + np.array([0, 0, 1.0]) * (G / k)
    tau = np.asarray(tau, float)[None, :, None]
    ex = -np.expm1(-k * tau) / k
    return p0[:, None, :] + vt * tau + (v0[:, None, :] - vt) * ex


def drag_vel(v0, k, tau):
    """v0 (n,3), k (n,), tau (n,) -> (n,3)."""
    k = np.asarray(k, float)[:, None]
    vt = np.array([0, 0, 1.0]) * (G / k)
    e = np.exp(-k * np.asarray(tau, float)[:, None])
    return vt + (v0 - vt) * e


def _pos_at(p0, v0, k, tau):
    """p0, v0 (n,3), scalar k, PER-TRAIL tau (n,) -> (n,3)."""
    tau = np.asarray(tau, float)[:, None]
    vt = np.array([0, 0, G / k])
    return p0 + vt * tau + (v0 - vt) * (-np.expm1(-k * tau)) / k


def v0_between(p0, p1, T, k):
    """unique drag-arc launch velocity p0 -> p1 in time T (arrays ok:
    p0/p1 (n,3), T (n,) or scalar, k (n,) or scalar)."""
    p0, p1 = np.atleast_2d(p0), np.atleast_2d(p1)
    T = np.broadcast_to(np.asarray(T, float), (p0.shape[0],))
    k = np.broadcast_to(np.asarray(k, float), (p0.shape[0],))
    vt = np.array([0, 0, 1.0])[None, :] * (G / k)[:, None]
    A = (-np.expm1(-k * T) / k)[:, None]
    return vt + (p1 - p0 - vt * T[:, None]) / A


def build_direct(C, Minv, ta, tb, Apx, Bpx, hxy,
                 zas=ZS, zbs=ZS, ks=KS):
    """DIRECT family; returns list of meta dicts with 3D params."""
    T = tb - ta
    dA, dB = ray_dir(Minv, Apx), ray_dir(Minv, Bpx)
    out = []
    for za in zas:
        pA = pt_at_z(C, dA, za)
        if pA is None or not (-8 <= pA[0] <= 28 and -8 <= pA[1] <= 52):
            continue
        if hxy is not None and np.hypot(pA[0] - hxy[0],
                                        pA[1] - hxy[1]) > 9.0:
            continue
        for zb in zbs:
            pB = pt_at_z(C, dB, zb)
            if pB is None or not (-8 <= pB[0] <= 28
                                  and -8 <= pB[1] <= 52):
                continue
            for k in ks:
                v0 = v0_between(pA, pB, T, k)[0]
                sp = float(np.linalg.norm(v0))
                if sp > SP98 * 1.15:
                    continue
                loft = np.degrees(np.arctan2(
                    v0[2], np.hypot(v0[0], v0[1])))
                if not (-35 <= loft <= 80):
                    continue
                out.append(dict(kind="direct", pA=pA, pB=pB, v0=v0,
                                k=k, za=za, zb=zb, sp=sp,
                                loft=float(loft)))
    return out


def build_bounce(C, Minv, ta, tb, Apx, Bpx, hxy,
                 zas=ZS, speeds=SPEEDS, lofts=LOFTS, lats=LATS,
                 zbs=(0.8, 2.0, 3.5, 5.5), k=K_DRAG,
                 launch_pairs=None):
    """BOUNCE family; closed-form arc 2. launch_pairs (n,3 speed/loft/
    z0 — the shot book) replaces the speeds x lofts grid: only REAL
    shot shapes, height-matched to the launch (|z0 - za| <= 1.8)."""
    T = tb - ta
    dA, dB = ray_dir(Minv, Apx), ray_dir(Minv, Bpx)
    pB_ref = pt_at_z(C, dB, 2.5)
    if pB_ref is None:
        return []
    out = []
    tt = np.linspace(0.06, T - 0.05, 80)
    for za in zas:
        pA = pt_at_z(C, dA, za)
        if pA is None or not (-8 <= pA[0] <= 28 and -8 <= pA[1] <= 52):
            continue
        if hxy is not None and np.hypot(pA[0] - hxy[0],
                                        pA[1] - hxy[1]) > 9.0:
            continue
        u = pB_ref[:2] - pA[:2]
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            continue
        u = u / nu
        # launch grid: shot book if given, else speed x loft product
        if launch_pairs is not None:
            pairs = [(s_, f_) for (s_, f_, z_) in launch_pairs
                     if abs(z_ - za) <= 1.8]
        else:
            pairs = [(s_, f_) for s_ in speeds for f_ in lofts]
        vs = []
        for sp, lf in pairs:
            cl, sl = np.cos(np.radians(lf)), np.sin(np.radians(lf))
            for la in lats:
                ca, sa = (np.cos(np.radians(la)),
                          np.sin(np.radians(la)))
                ux = u[0] * ca - u[1] * sa
                uy = u[0] * sa + u[1] * ca
                vs.append([sp * cl * ux, sp * cl * uy, sp * sl])
        if not vs:
            continue
        v0 = np.asarray(vs)
        n = len(v0)
        p0 = np.repeat(pA[None, :], n, 0)
        Z = drag_pos(p0, v0, np.full(n, k), tt)[:, :, 2]
        below = Z <= 0.0
        hasb = below.any(axis=1)
        idx = np.argmax(below, axis=1)
        sel = np.where(hasb & (idx > 0))[0]
        if not len(sel):
            continue
        j = idx[sel]
        z0_, z1_ = Z[sel, j - 1], Z[sel, j]
        fr = z0_ / (z0_ - z1_ + 1e-12)
        ts = tt[j - 1] + fr * (tt[j] - tt[j - 1])
        ok = (ts >= 0.08) & (ts <= T - 0.06)
        sel, ts = sel[ok], ts[ok]
        if not len(sel):
            continue
        q = _pos_at(p0[sel], v0[sel], k, ts)
        q[:, 2] = 0.0
        inb = ((q[:, 0] >= -4) & (q[:, 0] <= 24)
               & (q[:, 1] >= -3) & (q[:, 1] <= 47))
        sel, ts, q = sel[inb], ts[inb], q[inb]
        if not len(sel):
            continue
        v_in = drag_vel(v0[sel], np.full(len(sel), k), ts)
        fall = v_in[:, 2] <= -0.5
        sel, ts, q, v_in = sel[fall], ts[fall], q[fall], v_in[fall]
        if not len(sel):
            continue
        sp_in = np.linalg.norm(v_in, axis=1)
        h_in = np.hypot(v_in[:, 0], v_in[:, 1])
        T2 = T - ts
        for zb in zbs:
            pB = pt_at_z(C, dB, zb)
            if pB is None:
                continue
            w0 = v0_between(q, pB, T2, k)
            ez = -w0[:, 2] / v_in[:, 2]
            h_out = np.hypot(w0[:, 0], w0[:, 1])
            mu = h_out / np.maximum(h_in, 1e-6)
            sp_out = np.linalg.norm(w0, axis=1)
            g = ((w0[:, 2] >= 0.2) & (ez >= 0.3) & (ez <= 1.4)
                 & (mu >= 0.05) & (mu <= 1.6)
                 & (sp_out <= 1.2 * sp_in + 3.0))
            for i in np.where(g)[0]:
                v_ = v0[sel[i]]
                out.append(dict(
                    kind="bounce", pA=pA, pB=pB, v0=v_.copy(),
                    k=k, ts=float(ts[i]), q=q[i].copy(),
                    w0=w0[i].copy(), za=za, zb=zb,
                    sp=float(np.linalg.norm(v_)),
                    loft=float(np.degrees(np.arctan2(
                        v_[2], np.hypot(v_[0], v_[1]))))))
    return out


def trail_pixels(P, trails, ta, times):
    """stacked pixel tracks (n, F, 2) for a list of trail metas."""
    n, F = len(trails), len(times)
    if n == 0:
        return np.zeros((0, F, 2)), np.zeros(0)
    tau = times - ta
    X = np.empty((n, F, 3))
    for i, tr in enumerate(trails):
        if tr["kind"] == "direct":
            X[i] = drag_pos(tr["pA"][None], tr["v0"][None],
                            [tr["k"]], tau)[0]
        else:
            x1 = drag_pos(tr["pA"][None], tr["v0"][None],
                          [tr["k"]], np.maximum(tau, 0))[0]
            t2 = np.maximum(tau - tr["ts"], 0)
            x2 = drag_pos(tr["q"][None], tr["w0"][None],
                          [tr["k"]], t2)[0]
            X[i] = np.where((tau < tr["ts"])[:, None], x1, x2)
    px = c3.project(P, X.reshape(-1, 3)).reshape(n, F, 2)
    zmin = X[:, :, 2].min(axis=1)
    return px, zmin


def prior_pen(trails):
    """v3: distance to the nearest REAL shot shape (speed/8, loft/10
    metric) beyond 1 unit is paid at W_MODE per unit; bounces pay the
    complexity toll. Shot-book members cost ~0 by construction."""
    if not trails:
        return np.zeros(0)
    sp = np.array([t["sp"] for t in trails])
    lf = np.array([t["loft"] for t in trails])
    dm = np.min(np.hypot((sp[:, None] - LIB_SP[None, :]) / 8.0,
                         (lf[:, None] - LIB_LF[None, :]) / 10.0),
                axis=1)
    pen = W_MODE * np.maximum(0.0, dm - 1.0)
    pen += np.array([PEN_BOUNCE if t["kind"] == "bounce" else 0.0
                     for t in trails])
    return pen


def frame_base(pool, cor, t0, fs, seed):
    """per-frame NULL support: what a random point in the window
    collects (N_PROBE Monte-Carlo probes, same kernel). A junk-dense
    frame supports every trail alike -> its evidence must be ~zero."""
    rng = np.random.default_rng(seed)
    base = {}
    for f in fs:
        pc = pool.get(f)
        if pc is None:
            continue
        arr, nb = pc
        t = t0 + f / 60.0
        cx, cy, wx, wy = window_at(cor, min(max(t, cor[0]), cor[1]))
        qx = rng.uniform(cx - wx, cx + wx, N_PROBE)
        qy = rng.uniform(cy - wy, cy + wy, N_PROBE)
        d = np.hypot(qx[:, None] - arr[None, :, 0],
                     qy[:, None] - arr[None, :, 1])
        w = kernel_w(d, arr, nb)
        base[f] = float(w.max(axis=1).mean())
    return base


# ------------------------------------------------------- the matcher

def kernel_w(d, arr, nb):
    """support kernel; column 3 = learned p when present (else the
    hand-built peak bonus)."""
    if arr.shape[1] >= 4:
        app = P_LO + P_HI * arr[None, :, 3]
    else:
        app = 1.0 + W_PK * np.minimum(arr[None, :, 2], 40.0)
    return np.clip(1.0 - d / R_SUP, 0.0, None) * app \
        * np.where(nb[None, :], BODY_FACT, 1.0)


def corridor_pool(cands, cor, t0, fa, fb, body):
    """per-frame candidate arrays inside the corridor window."""
    pool = {}
    for f in range(fa, fb + 1):
        t = t0 + f / 60.0
        cx, cy, wx, wy = window_at(cor, min(max(t, cor[0]), cor[1]))
        cs = [(c_[0], c_[1], c_[3]) + ((c_[4],) if len(c_) > 4 else ())
              for c_ in cands.get(f, ())
              if abs(c_[0] - cx) <= wx and abs(c_[1] - cy) <= wy][:CAP]
        if not cs:
            continue
        arr = np.asarray(cs)
        barr = body.get(f) if body else None
        if barr is not None and len(barr):
            d = np.hypot(arr[:, 0:1] - barr[None, :, 0],
                         arr[:, 1:2] - barr[None, :, 1]).min(axis=1)
            nb = d <= NEAR_BODY
        else:
            nb = np.zeros(len(arr), bool)
        pool[f] = (arr, nb)
    return pool


def score_trails(px, zmin, pen, pool, fs, base):
    n = px.shape[0]
    tot = np.zeros(n)
    supf = np.zeros(n, int)
    for j, f in enumerate(fs):
        pc = pool.get(f)
        if pc is None:
            continue
        arr, nb = pc
        d = np.hypot(px[:, j, 0:1] - arr[None, :, 0],
                     px[:, j, 1:2] - arr[None, :, 1])
        w = kernel_w(d, arr, nb)
        s = w.max(axis=1)
        tot += s - base.get(f, 0.0)
        supf += s > 0.15
    tot = tot - pen - np.where(zmin < -0.5, 25.0, 0.0)
    return tot, supf


def refine(P, C, Minv, best, ta, tb, times, pool, fs, hxy,
           Apx, Bpx, base, rounds=2):
    tr, sc = best
    for r in range(rounds):
        h = 0.6 / (r + 1)
        if tr["kind"] == "direct":
            zas = [tr["za"] + d for d in (-h, 0, h)]
            zbs = [tr["zb"] + d for d in (-h, 0, h)]
            ks = [tr["k"] * f for f in (0.75, 1.0, 1.35)]
            cands = build_direct(C, Minv, ta, tb, Apx, Bpx, hxy,
                                 zas, zbs, ks)
        else:
            sp0 = tr["sp"]
            v = tr["v0"]
            lf0 = float(np.degrees(np.arctan2(
                v[2], np.hypot(v[0], v[1]))))
            speeds = [sp0 * f for f in (0.95, 1.0, 1.05)]
            lofts = [lf0 + d for d in (-2.5 / (r + 1), 0,
                                       2.5 / (r + 1))]
            zas = [tr["za"] + d for d in (-h, 0, h)]
            zbs = [tr["zb"] + d for d in (-h, 0, h)]
            # center the lateral grid on the WINNER's actual heading
            # (its lat relative to the pA -> pB_ref aim), so refine
            # can reproduce and locally improve the winner
            pB_ref = pt_at_z(C, ray_dir(Minv, Bpx), 2.5)
            u = pB_ref[:2] - tr["pA"][:2]
            lat0 = float(np.degrees(
                np.arctan2(u[0] * v[1] - u[1] * v[0],
                           u[0] * v[0] + u[1] * v[1])))
            lats = tuple(lat0 + d for d in (-3.0 / (r + 1), 0.0,
                                            3.0 / (r + 1)))
            cands = build_bounce(C, Minv, ta, tb, Apx, Bpx, hxy,
                                 zas, speeds, lofts, lats, zbs,
                                 tr["k"])
        if not cands:
            return tr, sc
        px, zmin = trail_pixels(P, cands, ta, times)
        tot, _ = score_trails(px, zmin, prior_pen(cands), pool, fs,
                              base)
        i = int(np.argmax(tot))
        if tot[i] > sc:
            tr, sc = cands[i], float(tot[i])
    return tr, sc


def run_corridor(P, C, Minv, Hg_inv, c, series, npz, cands, cor, t0,
                 body, disp=None):
    ta, tb, A, B, wx, wy = cor
    if disp:
        A = (A[0] + disp[0], A[1] + disp[1])
        B = (B[0] + disp[0], B[1] + disp[1])
        cor = (ta, tb, A, B, wx, wy)
    if tb - ta < MIN_COR_S:
        return None
    fa, fb = int(round((ta - t0) * 60)), int(round((tb - t0) * 60))
    fs = list(range(fa, fb + 1))
    times = t0 + np.asarray(fs) / 60.0
    pool = corridor_pool(cands, cor, t0, fa, fb, body)
    if len(pool) < 3:
        return None
    hxy = hitter_xy(c, series, npz, Hg_inv, ta, A)
    trails = (build_direct(C, Minv, ta, tb, A, B, hxy)
              + build_bounce(C, Minv, ta, tb, A, B, hxy,
                             lats=LATS_REP, launch_pairs=LIB))
    if not trails:
        return None
    base = frame_base(pool, cor, t0, fs, 91000 + fa)
    px, zmin = trail_pixels(P, trails, ta, times)
    tot, supf = score_trails(px, zmin, prior_pen(trails), pool, fs,
                             base)
    best, bsc = None, -1e18
    for i in np.argsort(-tot)[:TOP_M]:
        tr_i, sc_i = refine(P, C, Minv,
                            (trails[int(i)], float(tot[int(i)])),
                            ta, tb, times, pool, fs, hxy, A, B, base)
        if sc_i > bsc:
            best, bsc = tr_i, sc_i
    abst = bsc < ABSTAIN
    bpx, _ = trail_pixels(P, [best], ta, times)
    bpx = bpx[0]
    full, snap = {}, {}
    nsup = 0
    for j, f in enumerate(fs):
        if not abst:
            full[f] = (float(bpx[j, 0]), float(bpx[j, 1]))
        pc = pool.get(f)
        if pc is None:
            continue
        arr, nb = pc
        d = np.hypot(arr[:, 0] - bpx[j, 0], arr[:, 1] - bpx[j, 1])
        m = int(np.argmin(d))
        if d[m] <= R_SUP:
            nsup += 1
            if not abst:
                snap[f] = (float(arr[m, 0]), float(arr[m, 1]))
    return dict(cor=cor, best=best, score=bsc, full=full, snap=snap,
                nsup=nsup, nfr=len(fs), n_trails=len(trails),
                abst=abst)


def build_tracks(P, C, Minv, Hg_inv, c, series, npz, cands, cors, t0,
                 body, disp=None):
    full, snap, recs = {}, {}, []
    for cor in cors:
        r = run_corridor(P, C, Minv, Hg_inv, c, series, npz, cands,
                         cor, t0, body, disp)
        if r is None:
            continue
        full.update(r["full"])
        snap.update(r["snap"])
        recs.append(r)
    return full, snap, recs


# -------------------------------------------------------- evaluation

def cands_cached(rally, f_lo, f_hi, thr, mode, lrn=False, pxs=""):
    """decode once, reuse across iterations (the decode dominates).
    lrn=True: attach the learned emission p (emission.py cache,
    row-aligned) -> 5-tuples (x, y, ar, pk, p). pxs: p-cache filename
    suffix ("_x" = cross-fold caches for the train rallies)."""
    p = SP / f"cands_r{rally}_{mode}_{thr}.npz"
    if not p.exists():
        cands = frame_candidates(rally, f_lo, f_hi, thr, mode=mode)
        rows = [(f, x, y, ar, pk) for f, cs in cands.items()
                for (x, y, ar, pk) in cs]
        np.savez_compressed(p, a=np.asarray(rows, np.float32))
    a = np.load(p)["a"]
    if lrn:
        pf = SP / f"p_r{rally}_{mode}_{thr}{pxs}.npz"
        if not pf.exists():
            raise SystemExit(f"missing {pf} — run emission.py cache "
                             f"{rally} first")
        z = np.load(pf)
        pv = z["p"]
        assert len(pv) == len(a) and np.allclose(z["fxy"], a[:, :3]), \
            "p-cache misaligned with cands cache"
        a = np.hstack([a, pv[:, None]])
    fs_ = a[:, 0].astype(int)
    order = np.argsort(fs_, kind="stable")
    a, fs_ = a[order], fs_[order]
    uniq, starts = np.unique(fs_, return_index=True)
    ends = np.append(starts[1:], len(a))
    return {int(f): [tuple(row) for row in a[s0:s1, 1:]]
            for f, s0, s1 in zip(uniq, starts, ends)}


def hits(track, truth, t0, R=R_MAIN):
    out = []
    for (t, tx, ty, vis) in truth:
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        out.append(None if p is None
                   else float(np.hypot(p[0] - tx, p[1] - ty)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=14)
    ap.add_argument("--rth", type=float, default=0.5)
    ap.add_argument("--lrn", action="store_true",
                    help="learned emission arms (needs emission.py "
                         "model + p-caches; trained on r6/r7 only)")
    ap.add_argument("--soft", type=float, default=None,
                    help="with --lrn: soft DP p-term W_P_SOFT instead "
                         "of the hard p-filter (weight tuned on r6/r7"
                         " cross-fold ONLY — softdp.py)")
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    npz = np.load(c["npz"])
    truth = load_truth(a.rally)
    t0 = c["t0"]
    P = c["P"]
    C, Minv, Hg_inv = cam(P)
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    dec = decode_recall(c, truth)
    body = cdp.body_points(c, f_lo, f_hi)
    print(f"rally {a.rally}: {len(truth)} V/S clicks, decode@12 "
          f"{sum(dec)}/{len(dec)}; prior: {PRIOR['n_launch']} launches"
          f" / {PRIOR['n_bounce']} bounces")
    cc = cands_cached(a.rally, f_lo, f_hi, a.thr, "cc", lrn=a.lrn)
    pk = cands_cached(a.rally, f_lo, f_hi, a.thr, "peak", lrn=a.lrn)
    if a.lrn and a.soft is not None:
        cdp.W_P_SOFT = a.soft
        cc_dp, dp_tag = cc, "dp-ccS+body"
        print(f"learned emission: SOFT dp p-term W_P_SOFT={a.soft:g} "
              f"(no hard cull)")
    elif a.lrn:
        mdl = json.loads((SP / "emission_model.json").read_text())
        kp = mdl["dp_keep_thr"]
        cc_dp = {f: [c_[:4] for c_ in cs if c_[4] >= kp]
                 for f, cs in cc.items()}
        dp_tag = "dp-ccL+body"
        print(f"learned emission: dp filter p >= {kp:.3f} keeps "
              f"{sum(len(v) for v in cc_dp.values())}/"
              f"{sum(len(v) for v in cc.values())} cc cands")
    else:
        cc_dp, dp_tag = cc, "dp-cc+body"
    sfx = "L" if a.lrn else ""
    rng = np.random.default_rng(20260901)
    for arm, times in (("prod", prod_contacts(c, series, a.rth)),
                       ("oracle", list(c["imps"]))):
        cors = corridors(c, series, times)
        print(f"== {arm}: {len(cors)} corridors")
        dp_tr = cdp.build_track(cc_dp, cors, t0, body=body)
        cdp.score(dp_tr, truth, t0, dec, dp_tag)
        results = {}
        for cname, cands in ((f"cc{sfx}", cc), (f"peak{sfx}", pk)):
            full, snap, recs = build_tracks(P, C, Minv, Hg_inv, c,
                                            series, npz, cands, cors,
                                            t0, body)
            results[cname] = (full, snap, recs)
            cdp.score(snap, truth, t0, dec, f"spag-{cname}")
            cdp.score(full, truth, t0, dec, f"spag-{cname}-F")
            for vis in ("V", "S"):
                tt = [x for x in truth if x[3] == vis]
                dd = [d for x, d in zip(truth, dec) if x[3] == vis]
                cdp.score(snap, tt, t0, dd, f"  [{vis}]")
            mh = {}
            nab = 0
            for r in recs:
                b = r["best"]
                mn = mode_name(b["sp"], b["loft"])
                mh[mn] = mh.get(mn, 0) + 1
                nab += bool(r.get("abst"))
            print(f"    modes[{cname}]: "
                  f"{dict(sorted(mh.items(), key=lambda x: -x[1]))}"
                  f"  abstained {nab}/{len(recs)}")
        for kk in range(2):
            d = (float(rng.uniform(160, 240)) * rng.choice([-1, 1]),
                 float(rng.uniform(80, 140)) * rng.choice([-1, 1]))
            fN, sN, _ = build_tracks(P, C, Minv, Hg_inv, c, series,
                                     npz, pk, cors, t0, body, disp=d)
            cdp.score(sN, truth, t0, dec, f"null{kk}")
            cdp.score(fN, truth, t0, dec, f"null{kk}-F")
        # ---- per-corridor autopsy table (successes AND failures)
        if arm != "prod":
            continue
        h_dp = hits(dp_tr, truth, t0)
        h_sn = hits(results[f"peak{sfx}"][1], truth, t0)
        h_fu = hits(results[f"peak{sfx}"][0], truth, t0)
        _, _, recs = results[f"peak{sfx}"]
        hums = [(s.get("ts") if s and s.get("kind") == "bounce"
                 else None, b0, b1)
                for s, b0, b1 in zip(c["h_segs"], c["hum"][1][:-1],
                                     c["hum"][1][1:])
                if s and s.get("ok")]
        print("-- corridors (prod, peak cands): t-span dur kind(spag)"
              " ts | human-kind | support | clicks: dp / snap / full")
        for r in recs:
            ta, tb = r["cor"][0], r["cor"][1]
            ov = [(min(tb, b1) - max(ta, b0), ts)
                  for ts, b0, b1 in hums if min(tb, b1) > max(ta, b0)]
            hk = "?"
            if ov:
                ts_h = max(ov)[1]
                hk = ("bounce" if ts_h is not None
                      and ta <= ts_h <= tb else "arc")
            idx = [i for i, (t, *_ ) in enumerate(truth)
                   if ta <= t <= tb]
            ndp = sum(1 for i in idx
                      if h_dp[i] is not None and h_dp[i] <= R_MAIN)
            nsn = sum(1 for i in idx
                      if h_sn[i] is not None and h_sn[i] <= R_MAIN)
            nfu = sum(1 for i in idx
                      if h_fu[i] is not None and h_fu[i] <= R_MAIN)
            b = r["best"]
            ts_s = f"{b['ts']+0:5.2f}" if b["kind"] == "bounce" else \
                "  -  "
            ab = "ABST" if r.get("abst") else "    "
            print(f"  {ta:7.2f}-{tb:7.2f} {tb-ta:4.2f}s "
                  f"{b['kind'][:3]}/{mode_name(b['sp'], b['loft']):5s}"
                  f"@{ts_s} sp{b['sp']:5.1f} {ab} "
                  f"| hum {hk:6s} | sup {r['nsup']:3d}/{r['nfr']:3d} "
                  f"| {len(idx):3d}: {ndp:3d} /{nsn:4d} /{nfu:4d}")


if __name__ == "__main__":
    main()
